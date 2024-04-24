#include "OmpRaceEngine.h"

#include <llvm/ADT/SCCIterator.h>

#include <sstream>

#include "BradPass.h"
#include "Graph/Trie.h"
#include "OMPModel.h"
#include "PTAModels/ExtFunctionManager.h"
#include "Races.h"
#include "aser/Util/Log.h"

using namespace aser;
using namespace llvm;

extern bool ENABLE_OLD_OMP_ALIAS_ANALYSIS;
extern bool CONFIG_NO_FILTER;
extern cl::opt<bool> CONFIG_NO_KEYWORD_FILTER;  // default set to true
extern cl::opt<bool> ConfigFlowFilter;
extern bool CONFIG_INCLUDE_ATOMIC;

extern bool DEBUG_RACE_EVENT;
extern bool DEBUG_OMP_RACE;
extern bool DEBUG_INDIRECT_CALL;
extern bool DEBUG_INDIRECT_CALL_ALL;

extern void debugEvent(PTA *pta, MemAccessEvent *e1, bool isWrite = false);

// must be static, used across different omp engine, a limitation due to handling nested forks
std::set<std::string> reductionVarNames;

// globally single-threaded basic blocks determined by omp_get_max_threads
// eg LULESH: if (numthreads > 1) {} else { single-thread}
std::set<const llvm::BasicBlock *> globalSingleThreadBlocks;

extern vector<string> DEBUG_FOCUS_VEC;  // .omp_outlined._debug GB_ATOMIC_WRITE_HX
void OMPEngine::addOMPReadOrWrite(TID tid, MemAccessEvent *e, bool isWrite) {
    // debug
    if (DEBUG_FOCUS_VEC.size() > 0) {
        if (isWrite && e->getInst()->getFunction()->getName().startswith(DEBUG_FOCUS_VEC[0])) {
            std::string line = getSourceLoc(e->getInst()).getSourceLine();
            if (line.find(DEBUG_FOCUS_VEC[1]) != string::npos)
                // && snippet.find("if (f == 2) continue")!=string::npos)
                // std::string snippet = getSourceLoc(e->getInst()).getSnippet();
                debugEvent(pta, e, true);
            else
                return;
        } else
            return;
    }

    // if (e->getInst()->getFunction()->getName().equals("httpAcceptHttpConnection")) {
    //     std::string line = getSourceLoc(e->getInst()).getSourceLine();
    //     if (line.find("pServer->status = HTTP_SERVER_RUNNING;") != string::npos)
    //         debugEvent(pta, e, true);
    //     else
    //         return;
    // } else
    //     return;  // only log numele write

    if (DEBUG_RACE_EVENT) {
        debugEvent(pta, e, isWrite);
    }
    static std::vector<const ObjTy *> pts;
    pts.clear();
    pta->getFSPointsTo(e->getContext(), e->getPointerOperand(), pts);
    //        LOG_DEBUG("thread {} write pts size {}: inst={}", tid, pts.size(), *(e->getInst()));
    //        LOG_DEBUG("inst source location: {}", getSourceLoc(e->getInst()).sig());

    // if (!e->getInst()->getFunction()->getName().equals("GraphContext_AttributeCount") &&
    //    !e->getInst()->getFunction()->getName().equals("raxGenericInsert"))
    {
        if (!CONFIG_NO_FILTER && pts.size() > 1) {
            // return;
            auto firstO = pts.front();
            auto lastO = pts.back();  // important do the last!
            pts.clear();
            pts.push_back(firstO);
            pts.push_back(lastO);
            // if (e->getInst()->getFunction()->getName().equals("raxGenericInsert")) {
            // std::string line = getSourceLoc(e->getInst()).getSourceLine();
            //     if (line.find("numele") != string::npos) {
            // llvm::outs() << "tid " << e->getTID() << " write push first pta object: " << firstO->getValue() << "\n"
            //              << line << "\n";
            //         // debugEvent(pta, e);
            //     }
            // }
        }
    }
    // if (CONFIG_FAST_MODE && pts.size() > 5) return;
    for (const ObjTy *o : pts) {
        if (o->isFIObject()) continue;
        if (!o->getValue() || isa<Function>(o->getValue())) {
            continue;
        }
        if (const GlobalVariable *GV = dyn_cast_or_null<GlobalVariable>(o->getValue())) {
            if (GV->isThreadLocal()) continue;
            if (GV->isConstant()) continue;
        }

        auto it = objIdxCache.find(o);
        unsigned int idx;
        if (it == objIdxCache.end()) {
            idx = objs.size();
            objs.push_back(o);
            objIdxCache.insert(std::make_pair(o, idx));
        } else {
            idx = it->second;
        }

        if (isWrite) {
            auto &writeSet = memWrites[idx][tid];
            writeSet.push_back(e);
            if (DEBUG_RACE_EVENT) llvm::outs() << "OMP writeSet add event: " << e->getID() << " idx: " << idx << "\n";
        } else {
            auto &readSet = memReads[idx][tid];
            readSet.push_back(e);
            if (DEBUG_RACE_EVENT) llvm::outs() << "OMP readSet add event: " << e->getID() << " idx: " << idx << "\n";
        }
    }
}

extern bool DEBUG_CALL_STACK;
static unsigned int call_stack_level = 0;
static string DEBUG_STRING_SPACE = "";
static aser::trie::TrieNode *cur_trie;

extern void addExploredFunction(const Function *f);

bool hasAllReductionPredecessor(const BasicBlock *B) {
    llvm::outs() << "B: " << B->getName() << "\n";
    if (B->getName().startswith(".omp.reduction.")) return true;

    if (B->getName() == "entry") return false;

    // the basic block B can be called from a reduction
    for (auto it = pred_begin(B), et = pred_end(B); it != et; ++it) {
        auto preB = *it;

        if (preB == B) continue;

        llvm::outs() << "preB: " << preB->getName() << "\n";

        if (!hasAllReductionPredecessor(preB)) return false;
    }

    return true;
}
void addTransitiveClosureSingleThreadBlocks(const BasicBlock *B) {
    globalSingleThreadBlocks.insert(B);

    for (auto it = succ_begin(B), et = succ_end(B); it != et; ++it) {
        auto sucB = *it;
        bool add = true;
        for (auto it2 = pred_begin(sucB), et2 = pred_end(sucB); it2 != et2; ++it2) {
            auto preB = *it2;
            if (preB != B && preB != sucB) {
                add = false;
                break;
            }
        }
        if (add) globalSingleThreadBlocks.insert(sucB);
    }
}

bool hasFortranReductionPredecessor(const BasicBlock *B) {
    // llvm::outs() << "B: " << B->getName() << "\n";
    // the basic block B can be called from a reduction
    auto preB = B->getSinglePredecessor();

    if (preB) {
        // llvm::outs() << "preB: " << preB->getName() << "\n";
        std::string line = getSourceLoc(preB->getTerminator()).getSourceLine();
        // llvm::outs() << "source line: " << line << "\n";

        if (line.find("!$omp section") != string::npos) {
            return true;
        }
    }

    return false;
}

bool isReductionBasicBlock(const BasicBlock *B) {
    if (B->hasName() && B->getName().startswith(".omp.sections."))
        return true;
    else if (hasFortranReductionPredecessor(B))
        return true;
    else
        return false;
}

// Build SHB graph, connectivity engine, and thread trace for this node
// If a new node is created the curEvent argument is updated to point to the new node
// input:
//   callNode - Node to be explored
//   thread - thread that is "executing" the callNode
//   curEvent - the most recently created node in the callstack, default value: nullptr
void OMPEngine::traverse(const CallGraphNodeTy *callNode, StaticThread *thread, Event *curEvent) {
    Function *func = const_cast<Function *>(callNode->getTargetFun()->getFunction());
    if (func->isDeclaration()) {
        return;
    }

    addExploredFunction(func);
    auto threadID = thread->getTID();

    // simulating call stack
    if (find(callStack.begin(), callStack.end(), callNode) == callStack.end()) {
        callStack.push_back(callNode);

        if (DEBUG_CALL_STACK) {  //+ " push "
            llvm::outs() << DEBUG_STRING_SPACE + "omp thread " << threadID
                         << " push " + demangle(func->getName().str()) + " level: " << call_stack_level << ""
                         << "\n";
            DEBUG_STRING_SPACE += "  ";
            call_stack_level++;
        }

    } else {
        // recursive call
        LOG_TRACE("omp thread skip recursive func call: {}", demangle(func->getName().str()));

        return;
    }
    int callDepth = callStack.size();
    auto context = callNode->getContext();

    LOG_TRACE("omp thread {} enter func : {}", threadID, demangle(func->getName().str()));
    cur_trie = aser::trie::getNode(cur_trie, func);
#ifdef TOPSORT_BASICBLOCK
    auto &LI = pass.getAnalysis<LoopInfoWrapperPass>(*func).getLoopInfo();
    vector<BasicBlock *> bbStack;  // in reverse topo order
    for (auto it = llvm::scc_begin<Function *>(func), ie = llvm::scc_end<Function *>(func); it != ie; ++it) {
        const vector<BasicBlock *> &SCC = *it;

        if (SCC.size() > 1) {
            // has loops
            Loop *L = LI.getLoopFor(SCC.back());
            if (L == nullptr) {
                for (auto bb : SCC) {
                    bbStack.push_back(bb);
                }
            } else {
                while (L->getParentLoop() != nullptr) {
                    // strip to the outermost loop
                    L = L->getParentLoop();
                }
                assert(L->getNumBlocks() == SCC.size());
                // sort the loop in topo order, but ignore all the back edges
                // according to the source code, seems like the order is already in topo order (ignore back edges)
                // TODO: confirm it
                auto bbList = L->getBlocks();
                for (int i = bbList.size() - 1; i >= 0; i--) {
                    bbStack.push_back(bbList[i]);
                }
            }
        } else {
            assert(SCC.size() == 1);
            bbStack.push_back(SCC[0]);
        }
    }

    for (int bbIdx = bbStack.size() - 1; bbIdx >= 0; bbIdx--) {
        BasicBlock &BB = *(bbStack[bbIdx]);
#else
    for (auto const &BB : *func) {
#endif  // if(BB.hasName() && BB.getName().startswith(".omp.reduction.")) continue;

        if (globalSingleThreadBlocks.count(&BB)) {
            if (DEBUG_OMP_RACE) llvm::outs() << "skipped BB: " << BB << "\n";

            continue;
        }

        for (llvm::BasicBlock::const_iterator BI = BB.begin(), BE = BB.end(); BI != BE; ++BI) {
            // traverse each instruction
            const Instruction *inst = dyn_cast<Instruction>(BI);

            // LOG_TRACE("Instruction inst={}", *inst);

            {  // THIS IS NEEDED BY SVF
                if (llvm::isa<llvm::CallBase>(inst)) {
                    // for inter-procedural array index analysis
                    // connecting tid with omp_outline argument
                    const Instruction *caller_inst = getEventCallerInstruction(callEventTraces, curEvent, threadID);
                    pass.svf->connectSCEVFunctionArgs(context, caller_inst, func, inst);
                }
            }
            // Test SVF SCEV
            if (false)
                if (isa<GetElementPtrInst>(inst)) {
                    const Instruction *caller_inst = getEventCallerInstruction(callEventTraces, curEvent, threadID);
                    auto scev = pass.svf->getGlobalSCEV(context, caller_inst, func, inst);
                }

            // Handle OpenMP specific functionality separately
            if (!visitOMP(inst, context, thread)) {
                if (LangModel::isReadORWrite(inst)) {
                    if (DEBUG_OMP_RACE)
                        llvm::outs() << "tid: " << threadID << " inst: " << *inst << " skipLoadStore: " << skipLoadStore
                                     << " isMasterOnly: " << isMasterOnly << " isSingleOnly: " << isSingleOnly
                                     << " isTheFirstThread: " << isTheFirstThread << "\n";

                    if (skipLoadStore) {
                        // if in the same basic block
                        if (inst->getFunction() == staticForFiniInst->getFunction()) {
                            bool shouldSkip = true;
                            // heuristic: if the same basic block contains __kmpc_reduce, do not skip
                            auto inst2 = inst->getNextNonDebugInstruction();
                            while (inst2 && !inst2->isTerminator()) {
                                // if (DEBUG_OMP_RACE) llvm::outs() << "explore inst2: " << *inst2 << "\n";
                                if (llvm::isa<llvm::CallBase>(inst2)) {
                                    const CallSite CS2(inst2);
                                    if (!CS2.isIndirectCall()) {
                                        auto targetFunc = CS2.getTargetFunction();
                                        if (OMPModel::isBarrier(targetFunc) || OMPModel::isMasterOrSingle(targetFunc) ||
                                            OMPModel::isReduce(targetFunc) || OMPModel::isReduceEnd(targetFunc)) {
                                            shouldSkip = false;
                                            break;
                                        }
                                    }
                                }
                                inst2 = inst2->getNextNonDebugInstruction();
                            }
                            if (shouldSkip) {
                                if (DEBUG_OMP_RACE)
                                    llvm::outs()
                                        << "skipped LoadStore in tid: " << threadID << " inst: " << *inst << "\n";
                                continue;
                            }
                        }
                    }
                    if ((isMasterOnly > 0 || isSingleOnly > 0) && !isTheFirstThread) continue;

                    if (onceOnlyBasicBlocks.find(inst->getParent()) != onceOnlyBasicBlocks.end() && !isTheFirstThread)
                        continue;

                    // only the second can record read/write in reduction
                    if (isTheFirstThread) {
                        if (inst->getParent()->hasName() && inst->getParent()->getName().startswith(".omp.reduction."))
                            continue;
                    }

                    // Handle Normal
                    if (auto *load = dyn_cast<LoadInst>(inst)) {
                        if (CONFIG_INCLUDE_ATOMIC || !load->isAtomic() && !load->isVolatile()) {
                            // heuristics: also skip address returned by call LULESH
                            // %call = call dereferenceable(8) double* @_ZN6Domain2fxEi(%class.Domain* nonnull %domain,
                            // i32 %.omp.iv.02), !dbg !11006 store double 0.000000e+00, double* %call, align 8, !dbg
                            // !11008, !tbaa !3011
                            auto ptrop = load->getPointerOperand();
                            if (auto call = dyn_cast<llvm::CallBase>(ptrop)) {
                                if (DEBUG_OMP_RACE)
                                    llvm::outs()
                                        << "skipped call load in tid: " << threadID << " inst: " << *inst << "\n";
                                continue;
                            }

                            auto e = graph.createReadEvent(context, inst, threadID);
                            addOMPReadOrWrite(threadID, e, false);
                        }
                    } else if (auto *store = dyn_cast<llvm::StoreInst>(inst)) {
                        if (CONFIG_INCLUDE_ATOMIC || !store->isAtomic() && !store->isVolatile()) {
                            auto ptrop = store->getPointerOperand();
                            if (auto call = dyn_cast<llvm::CallBase>(ptrop)) {
                                if (DEBUG_OMP_RACE)
                                    llvm::outs()
                                        << "skipped call store in tid: " << threadID << " inst: " << *inst << "\n";
                                continue;
                            }

                            auto e = graph.createWriteEvent(context, inst, threadID);
                            addOMPReadOrWrite(threadID, e, true);
                        }
                    }
                }
                // Handle callsites
                else if (llvm::isa<llvm::CallBase>(inst)) {
                    CallSite CS(inst);
                    if (CS.isIndirectCall()) {
                        // TODO: handle fortran
                        //  %30 = bitcast i32 (...)* @f90io_sc_i_fmt_write to i32 (i32, i32, ...)*, !dbg !35
                        //%31 = call i32 (i32, i32, ...) %30(i32 %29, i32 25), !dbg !35

                        auto cs = pta->getInDirectCallSite(context, inst);
                        if (!cs) continue;  // JEFF: workaround for PTA issue
                        auto indirectCalls = cs->getResolvedTarget();

                        if (DEBUG_INDIRECT_CALL) {
                            if (indirectCalls.empty()) {
                                llvm::outs()
                                    << "Failed to resolve any indirect call in func: "
                                    << demangle(func->getName().str()) << " at " << getSourceLoc(inst).sig() << "\n"
                                    << getSourceLoc(inst).getSnippet() << "\n";
                            } else if (indirectCalls.size() > 1) {
                                for (auto fun : indirectCalls)
                                    llvm::outs()
                                        << "Resolved indirect call: funcName=" << demangle(fun->getName().str())
                                        << "\n";
                                llvm::outs() << "Hitting an indirect call (" << indirectCalls.size()
                                             << " resolved calls in total) in func: " << demangle(func->getName().str())
                                             << " at " << getSourceLoc(inst).sig() << "\n"
                                             << getSourceLoc(inst).getSnippet() << "\n";
                            }
                        }

                        // NOTE: the number of indirect calls being resolved is already limited in PTA
                        for (auto fun : indirectCalls) {
                            LOG_TRACE("Traversing indirect call: funcName={}, tid={}", demangle(fun->getName().str()),
                                      threadID);
                            if (OMPModel::isReadWriteAPI(fun)) {
                                if (OMPModel::isWriteAPI(fun)) {
                                    auto e = graph.createApiWriteEvent(context, inst, threadID);
                                    addOMPReadOrWrite(threadID, e, true);
                                } else if (OMPModel::isReadAPI(fun)) {
                                    auto e = graph.createApiReadEvent(context, inst, threadID);
                                    addOMPReadOrWrite(threadID, e, false);
                                }
                            }
                        }

                        continue;
                    }
                    auto targetFunc = CS.getTargetFunction();
                    // if (ExtFunctionsManager::skip(targetFunc)) continue;

                    if (LangModel::isReadWriteAPI(targetFunc)) {
                        if ((isMasterOnly > 0 || isSingleOnly > 0) && !isTheFirstThread) continue;
                        if (LangModel::isWriteAPI(targetFunc)) {
                            auto e = graph.createApiWriteEvent(context, inst, threadID);
                            addOMPReadOrWrite(threadID, e, true);

                        } else if (LangModel::isReadAPI(targetFunc)) {
                            auto e = graph.createApiReadEvent(context, inst, threadID);
                            addOMPReadOrWrite(threadID, e, false);

                            // if (e->getInst()->getFunction()->getName().contains("canReach")) debugEvent(pta, e);
                            // JEFF: small hack for std::map...::operator[]
                            // if the next instruction is store then set it as write

                            auto nextInst = inst->getNextNonDebugInstruction();
                            if (nextInst && llvm::isa<llvm::StoreInst>(nextInst))
                                if (auto nextStore = llvm::cast<llvm::StoreInst>(nextInst)) {
                                    if (llvm::demangle(CS.getTargetFunction()->getName().str()).find("::operator[]") !=
                                            std::string::npos &&
                                        nextStore->getPointerOperand() == inst) {
                                        // let's do debug
                                        if (DEBUG_OMP_RACE)
                                            llvm::outs()
                                                << "OMP readAPI inst: " << *inst << "\n   nextInst: " << *nextInst
                                                << "\n"
                                                << "in function: " << func->getName()
                                                << "\n   demangled: " << demangle(func->getName().str()) << "\n\n";

                                        auto e2 = graph.createApiWriteEvent(context, inst, threadID);
                                        addOMPReadOrWrite(threadID, e2, true);
                                    }
                                }

                        } else if (StdStringFunctions::isStringCopy(targetFunc)) {
                            // LOG_DEBUG("String copy constructor: {}", demangle(CS.getTargetFunction()->getName()));
                            // TODO: may need to support more constructors
                            // Special handling for container constructors
                            // auto e1 = graph.createApiWriteEvent(callNode->getContext(), inst, threadID);
                            auto e2 = graph.createApiReadEvent(context, inst, threadID, true);
                            addOMPReadOrWrite(threadID, e2, false);

                        } else {
                            LOG_TRACE("unhandled container APIs in shb graph. api={}",
                                      demangle(CS.getTargetFunction()->getName().str()));
                        }
                        continue;
                    }

                    if (OMPModel::isReadWriteAPI(targetFunc)) {
                        if (OMPModel::isWriteAPI(targetFunc)) {
                            auto e = graph.createApiWriteEvent(context, inst, threadID);
                            addOMPReadOrWrite(threadID, e, true);

                        } else if (OMPModel::isReadAPI(targetFunc)) {
                            auto e = graph.createApiReadEvent(context, inst, threadID);
                            addOMPReadOrWrite(threadID, e, false);

                        } else {
                            LOG_TRACE("unhandled OMP APIs in shb graph. api={}", demangle(targetFunc->getName().str()));
                        }
                        continue;
                    }

                    // check call stack redundancy
                    if (aser::trie::willExceedBudget(cur_trie, CS.getCalledFunction()) &&
                        !LangModel::isCriticalAPI(CS.getCalledFunction())) {
                        LOG_TRACE("omp thread {} exceeded call stack budget for func: {}", thread->getTID(),
                                  demangle(CS.getCalledFunction()->getName().str()));
                        continue;
                    }

                    // Handle nested call
                    auto nextContext = CT::contextEvolve(callNode->getContext(), inst);
                    const CallGraphNodeTy *nextNode = pta->getDirectNodeOrNull(nextContext, CS.getCalledFunction());

                    if (nextNode == nullptr) {
                        // TODO: make an assertion to ensure it is a ignored function.
                        continue;
                    }

                    CallEvent *callEvent = nullptr;
                    // if (!OMPModel::isOmpDebug(targetFunc))
                    {
                        callEvent = graph.createCallEvent(callNode->getContext(), inst, nextNode, threadID);
                        callEventTraces[threadID].push_back(callEvent);
                    }
                    traverse(nextNode, thread, curEvent);
                    if (callEvent) {
                        callEvent->setEndID(Event::getLargestEventID());
                    }
                }
            }
        }
    }

    LOG_TRACE("omp thread {} exit func : {}", threadID, demangle(func->getName().str()));
    callStack.pop_back();
    if (cur_trie) cur_trie = cur_trie->parent;  // return to its parent

    if (DEBUG_CALL_STACK) {
        DEBUG_STRING_SPACE.pop_back();
        DEBUG_STRING_SPACE.pop_back();
        call_stack_level--;
        llvm::outs() << DEBUG_STRING_SPACE + "omp thread " << threadID
                     << " pop " + demangle(func->getName().str()) + " level: " << call_stack_level << "\n";
    }
}

std::map<const Value *, std::set<const Value *>> indirectUsesMap;
std::set<const Value *> getIndirectLoadStoreUses(const Value *value) {
    if (indirectUsesMap.find(value) != indirectUsesMap.end()) {
        return indirectUsesMap.at(value);
    }
    std::set<const Value *> ptrOps;

    if (auto inst = dyn_cast_or_null<Instruction>(value)) {  // through store-load chain
        auto nextInst = inst->getNextNonDebugInstruction();

        if (auto storeInst = dyn_cast_or_null<StoreInst>(nextInst)) {
            auto ptrop = storeInst->getPointerOperand();
            ptrOps.insert(ptrop);
            // llvm::outs() << "getOutlineArg store: " << *storeInst << "\n";
            for (auto U : ptrop->users()) {
                if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
                    ptrOps.insert(LI->getPointerOperand());
                    // llvm::outs() << "getOutlineArg load: " << *LI << "\n";
                    //%14 = load i32, i32* %numthreads, align 4, !dbg !3076, !tbaa !3027, !noalias !3019
                }
            }
        } else if (auto zext = dyn_cast_or_null<ZExtInst>(nextInst)) {
            ptrOps.insert(zext);
            // llvm::outs() << "getOutlineArg zext: " << *zext << "\n";

            for (auto U : zext->users()) {
                if (auto alloc = llvm::dyn_cast<llvm::AllocaInst>(U)) {
                    ptrOps.insert(alloc);
                    // llvm::outs() << "getOutlineArg alloc: " << *alloc << "\n";
                }
            }
        }
    }
    // TODO: consider alloca
    //%call = tail call i32 @omp_get_max_threads(), !dbg !5442
    //  %1 = zext i32 %call to i64, !dbg !5443
    //  %vla = alloca i32, i64 %1, align 16, !dbg !5443
    // ...
    //   %vla1 = alloca double, i64 %1, align 16, !dbg !5445

    //  @__kmpc_fork_call(%struct.ident_t* %.kmpc_loc.addr, i32 9, void (i32*, i32*, ...)* %5, double* %dthydro, i64
    //  %length.casted.sroa.0.0.insert.ext, i32** %regElemlist.addr, %class.Domain* %domain, i64 %3, i64 %1, double*
    //  %vla1, i64 %1, i32* %vla), !dbg !5449

    indirectUsesMap[value] = ptrOps;
    return ptrOps;
}
std::set<const llvm::Argument *> getOutlineArgs(const llvm::Instruction *const ompForkCall, const llvm::Value *value,
                                                std::set<const Value *> &ptrOps) {
    aser::CallSite CS(ompForkCall);
    // llvm::outs() << "try to get arg for use: " << *value << "\n";

    std::set<const llvm::Argument *> args;
    int i = 3;  // start_omp_fork_call meaningful arg starts from 3
    for (; i < CS.getNumArgOperands(); i++) {
        auto call_arg = CS.getArgOperand(i);
        // this is always a pointer eg %numthreads = alloca i32, align 4
        // llvm::outs() << "call_arg: " << *call_arg << "\n";

        if (call_arg == value) {  // direct use
                                  // go into outlined and save global single thread blocks
            auto outlined = OMPModel::getOutlinedFunction(*ompForkCall);
            auto arg = outlined->arg_begin() + (i - 1);
            args.insert(arg);
        } else {
            if (ptrOps.count(call_arg)) {
                auto outlined = OMPModel::getOutlinedFunction(*ompForkCall);
                auto arg = outlined->arg_begin() + (i - 1);
                // llvm::outs() << "found outlined arg: " << *arg << "\n";
                // todo: there can exist multiple!
                args.insert(arg);
            }
        }
    }
    // if (args.empty()) llvm::outs() << "did not find any outlined arg for ompForkCall: " << *ompForkCall << "\n";
    return args;
}

void traceGlobalSingleThreadBlocks(const llvm::Function *outlined, std::set<const llvm::Argument *> &args) {
    // it can be used by another call
    //  tail call fastcc void @.omp_outlined._debug__.25(i32 %numElem.addr.sroa.0.0.extract.trunc, %class.Domain*
    //  nonnull %domain, double** nonnull %determ, i32* nonnull %numthreads, double** nonnull %sigxx, double**
    //  nonnull %sigyy, double** nonnull %sigzz, double** nonnull %fx_elem, double** nonnull %fy_elem, double**
    //  nonnull %fz_elem, [8 x double]* nonnull %fx_local, [8 x double]* nonnull %fy_local, [8 x double]* nonnull
    //  %fz_local) #21, !dbg !10526
    // getArg
    //%17 = load i32, i32* %numthreads, align 4, !dbg !10654, !tbaa !3231
    //%cmp28 = icmp sgt i32 %17, 1, !dbg !10655
    //...
    // br i1 %cmp28, label %if.then, label %if.else, !dbg !10657

    for (auto arg : args) {
        // llvm::outs() << "omp outline: " << outlined->getName() << " arg: " << *arg << "\n";

        for (auto U : arg->users()) {
            // llvm::outs() << "use: " << *U << "\n";

            // case 1: use by load
            if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
                auto cmpInst = dyn_cast_or_null<ICmpInst>(LI->getNextNonDebugInstruction());
                // Check that the cmp instruction is checking that icmp sgt i32 %14, 1
                // i.e. the guarded block will be executed *only* on one thread

                // llvm::outs() << "maxthreads cmpInst: " << *cmpInst << "\n";

                if (cmpInst && cmpInst->getPredicate() == CmpInst::Predicate::ICMP_SGT) {
                    if (auto const_val = dyn_cast_or_null<ConstantInt>(cmpInst->getOperand(1))) {
                        if (const_val->isOneValue()) {
                            for (auto br : cmpInst->users()) {
                                if (auto brInst = dyn_cast_or_null<BranchInst>(br)) {
                                    if (brInst->getNumSuccessors() > 1) {
                                        auto candidate = brInst->getSuccessor(1);
                                        if (candidate->getUniquePredecessor()) {
                                            if (DEBUG_OMP_RACE)
                                                llvm::outs()
                                                    << "traceGlobalSingleThreadBlocks block: " << *candidate << "\n";
                                            // todo transitive closure if successor has unique predecessor or itself
                                            addTransitiveClosureSingleThreadBlocks(candidate);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (auto call = llvm::dyn_cast<CallBase>(U)) {
                aser::CallSite CS(call);  // case 2: use by omp fork call
                if (!CS.isIndirectCall()) {
                    std::set<const llvm::Argument *> args2;
                    auto outlined2 = CS.getTargetFunction();

                    if (OMPModel::isFork(CS.getTargetFunction())) {
                        // llvm::outs() << "nested call: " << *CS.getInstruction() << "\n";
                        std::set<const Value *> ptrOps = getIndirectLoadStoreUses(arg);
                        auto args2 = getOutlineArgs(CS.getInstruction(), arg, ptrOps);
                        outlined2 = OMPModel::getOutlinedFunction(*CS.getInstruction());
                    } else {
                        for (int i = 0; i < CS.getNumArgOperands(); i++) {
                            auto call_arg = CS.getArgOperand(i);
                            if (call_arg == arg) {  // direct use
                                auto arg2 = outlined2->arg_begin() + i;
                                args2.insert(arg2);
                            }
                        }
                    }
                    if (!args2.empty()) traceGlobalSingleThreadBlocks(outlined2, args2);
                }
            }
        }
    }
}

// inst should be a call to omp_get_max_threads
// returns the first block guarded by numthreads, or null if there is not one
void getMaxThreadsGuardedSingleBlock(const ctx *const context, const llvm::Instruction *inst,
                                     std::set<const llvm::BasicBlock *> &singleBlocks) {
    /* ---Searching for the pattern---
        %call = tail call i32 @omp_get_max_threads(), !dbg !8994
        store i32 %call, i32* %numthreads, align 4, !dbg !8993, !tbaa !3231, !noalias !3015

        %14 = load i32, i32* %numthreads, align 4, !dbg !9033, !tbaa !3231, !noalias !3015
        %cmp7 = icmp sgt i32 %14, 1, !dbg !9035
        br i1 %cmp7, label %if.then8, label %if.end10, !dbg !9036

        if.then8:

        ---which translates roughly to---
        numthreads = omp_get_max_threads()
        ...
        if (numthreads > 1) {
            ...
        } else {
            //single thread block
        }
    */
    // OK THERE ARE SEVERAL OTHER SCENARIOS IN LULESH

    // Case 1: direct comparison
    // Case 2: through load/store
    // Case 3: passed to  __kmpc_fork_call
}
// inst should be a call to omp_get_thread_num
// returns the first block guarded by a constant thread id, or null if there is not one
const llvm::BasicBlock *getFirstTIDGuardedBlock(const llvm::Instruction *inst) {
    /* ---Searching for the pattern---
        %call = call i32 @omp_get_thread_num(), !dbg !43
        %cmp = icmp eq i32 %call, 0, !dbg !46
        br i1 %cmp, label %if.then, label %if.end, !dbg !47

        if.then:

        ---which translates roughly to---
        if (omp_get_thread_num() == 0) {
            ...
        }
    */
    auto nextInst = inst->getNextNonDebugInstruction();
    auto cmpInst = dyn_cast_or_null<ICmpInst>(nextInst);

    // Check that the cmp instruction is checking that tid == some constant
    // i.e. the guarded block will be executed *only* on one thread
    if (!(cmpInst && cmpInst->isTrueWhenEqual() && cmpInst->getOperand(0) == inst &&
          isa<Constant>(cmpInst->getOperand(1)))) {
        return nullptr;
    }

    nextInst = nextInst->getNextNonDebugInstruction();
    auto brInst = dyn_cast_or_null<BranchInst>(nextInst);
    if (!brInst || brInst->isUnconditional() || brInst->getCondition() != cmpInst) {
        return nullptr;
    }

    return brInst->getSuccessor(0);
}

// return true if this funciton was able to handle the callnode
bool OMPEngine::visitOMP(const llvm::Instruction *inst, const ctx *const context, StaticThread *thread) {
    if (llvm::isa<llvm::CallBase>(inst)) {
        const CallSite CS(inst);
        if (CS.isIndirectCall()) {
            return false;
        }
        auto targetFunc = CS.getTargetFunction();
        int callDepth = callStack.size();

        if (OMPModel::isFork(targetFunc)) {
            // TODO: Unhandled nested omp region
            // JEFF:  a better way is to create new threads here -- it would be a large change

            // OMPRaceDetect(pass).analyze(context, inst);
            // Skip duplicated omp fork instruction

            return true;
        } else if (OMPModel::isReduce(targetFunc)) {
            // get reduction names
            auto reductionVal = CS.getArgOperand(CS.getNumArgOperands() - 1);
            auto var = reductionVal->getName();

            auto varName = var.substr(var.find_last_of('.') + 1);
            // llvm::outs() << "reduction var: " << varName << "\n";

            reductionVarNames.insert(varName.str());
            return true;
        } else if (OMPModel::isBarrier(targetFunc)) {
            // TODO: - compute unqiue id for each omp region
            graph.createBarrierEvent(context, inst, thread->getTID(), 0);
            return true;
        } else if (OMPModel::isMaster(targetFunc)) {
            skipLoadStore = false;
            isMasterOnly++;
            // llvm::outs() << "isMasterOnly true: " << *inst << "\n";
            return true;
        } else if (OMPModel::isMasterEnd(targetFunc)) {
            isMasterOnly--;
            // llvm::outs() << "isMasterOnly false: " << *inst << "\n";
            return true;
        } else if (OMPModel::isSingle(targetFunc)) {
            skipLoadStore = false;
            isSingleOnly = true;
            return true;
        } else if (OMPModel::isSingleEnd(targetFunc)) {
            isSingleOnly = false;
            return true;
        } else if (OMPModel::isPushNumThreads(targetFunc)) {
            //__kmpc_push_num_threads
            // llvm::outs() << "in isPushNumThreads: " << *inst << "\n";

            isTeams = false;
        } else if (OMPModel::isPushNumTeams(targetFunc)) {
            //__kmpc_push_num_teams
            // llvm::outs() << "in isPushNumTeams: " << *inst << "\n";
            // auto number_teams = CS.getArgOperand(2);
            if (auto constant = dyn_cast<Constant>(CS.getArgOperand(2))) {
                if (constant->isOneValue()) {
                    isTeams = false;
                    // llvm::outs() << "setting isTeams false: " << *constant << "\n";
                }
            } else {
                // llvm::outs() << "number_teams is not constant: " << *CS.getArgOperand(2) << "\n";
            }

        } else if (OMPModel::isCriticalStart(targetFunc)) {
            if (!isTeams) {
                // openmp critical 3rd arg is pointer to lock
                auto lock = CS.getArgOperand(2);
                graph.createLockEvent(context, inst, thread->getTID(), lock);
            } else {
                // llvm::outs() << "is in teams skipping critical start: " << *inst << "\n";
            }
            return true;
        } else if (OMPModel::isCriticalEnd(targetFunc)) {
            if (!isTeams) {
                // openmp critical 3rd arg is pointer to lock
                auto lock = CS.getArgOperand(2);
                graph.createUnlockEvent(context, inst, thread->getTID(), lock);
            } else {
                // llvm::outs() << "is in teams skipping critical end: " << *inst << "\n";
            }
            return true;

        } else if (OMPModel::isSetLock(targetFunc)) {
            if (!isTeams) {
                // omp_set_lock's only arg is the lock
                auto lock = CS.getArgOperand(0);
                graph.createLockEvent(context, inst, thread->getTID(), lock);
            } else {
                // llvm::outs() << "is in teams skipping lock: " << *inst << "\n";
            }
            return true;
        } else if (OMPModel::isUnsetLock(targetFunc)) {
            if (!isTeams) {  // omp_unset_lock's only arg is the lock
                auto lock = CS.getArgOperand(0);
                graph.createUnlockEvent(context, inst, thread->getTID(), lock);
            } else {
                // llvm::outs() << "is in teams skipping unlock: " << *inst << "\n";
            }
            return true;
        } else if (LangModel::isMutexLock(inst)) {
            // omp atomic
            auto lock = CS.getArgOperand(0);
            graph.createLockEvent(context, inst, thread->getTID(), lock);

            return true;
        } else if (LangModel::isMutexUnLock(inst)) {
            auto lock = CS.getArgOperand(0);
            graph.createUnlockEvent(context, inst, thread->getTID(), lock);

            return true;
        } else if (OMPModel::isOrderedStart(targetFunc)) {
            // for race detection, the semantics of ordered is similar to critical?
            // openmp ordered handled in lockUnlockFunctions

            return true;
        } else if (OMPModel::isOrderedEnd(targetFunc)) {
            return true;

        } else if (OMPModel::isTask(targetFunc)) {
            // TODO： || OMPModel::isTaskAlloc(targetFunc) not supported for now
            // reason:  pta->getDirectNode(newCtx, calledFunc); crash

            if (isTheFirstThread || isMasterOnly == 0 && isSingleOnly == 0) {
                // the same fork instruction, for at most once
                if (onceOnlyTaskInstructions.find(inst) == onceOnlyTaskInstructions.end()) {
                    onceOnlyTaskInstructions.insert(inst);
                    auto e = graph.createForkEvent(context, inst, thread->getTID());
                    thread->addForkSite(e);
                    // llvm::outs() << "fork task: " << *inst << "\n";
                }
            }
            return true;
        } else if (OMPModel::isTaskWait(targetFunc)) {
            if (isTheFirstThread || isMasterOnly == 0 && isSingleOnly == 0) {
                // TODO: - join all direct child tasks
                auto e = graph.createJoinEvent(context, inst, thread->getTID());
                thread->addJoinSite(e);
            }
        } else if (OMPModel::isTaskDepend(targetFunc)) {
            if (isTheFirstThread || isMasterOnly == 0 && isSingleOnly == 0) {
                // TODO: model task dependencies
            }
        } else if (OMPModel::isStaticForInit(targetFunc)) {  // stub here
            // fortran/DRB152
            //  start tracing load/store after this
            skipLoadStore = false;
            return true;
        } else if (OMPModel::isStaticForFini(targetFunc)) {
            // graph.createBarrierEvent(context, inst, thread->getTID(), 0);
            // fortran/DRB059 lastprivate
            //  model __kmpc_for_static_fini: skip load/store after it

            skipLoadStore = true;
            staticForFiniInst = inst;
            // if this basic block is in the middle, we may miss real races..
            return true;
        } else if (OMPModel::isGetThreadNum(targetFunc)) {
            auto trueBlock = getFirstTIDGuardedBlock(inst);
            if (!trueBlock) {
                return true;
            }
            // llvm::outs() << "isGetThreadNum trueBlock: " << *trueBlock << "\n";

            onceOnlyBasicBlocks.insert(trueBlock);

            return true;
        } else if (OMPModel::isGetMaxThreadsNum(targetFunc)) {
            return true;

        } else if (OMPModel::isTaskGroupStart(targetFunc)) {
            auto startEvent = Event::getLargestEventID();
            // If the end of the last taskgroup was not reached, ignore the previous group
            if (!taskgroups.empty() && taskgroups.back().second != nullptr) {
                LOG_WARN("Ignoring unmatched taskgroup start. inst={}", *inst);
                taskgroups.back().first = startEvent;
            } else {
                taskgroups.push_back(std::make_pair(startEvent, nullptr));
            }
            return true;
        } else if (OMPModel::isTaskGroupEnd(targetFunc)) {
            // Check that there is an unmatched taskgroup start
            if (taskgroups.empty() || taskgroups.back().second != nullptr) {
                LOG_WARN("Ignoring unmatched taskgroup end. inst={}", *inst);
            } else {
                auto e = graph.createJoinEvent(context, inst, thread->getTID());
                taskgroups.back().second = e;
            }
            return true;
        } else if (OMPModel::isAnyOpenMPCall(targetFunc)) {
            return true;
        }
    }

    return false;
}

std::vector<unsigned int> OMPEngine::collectShared() const {
    std::vector<unsigned int> sharedIdxes;
    // Guesstimate that 1/5 of all objects are shared
    sharedIdxes.reserve(memWrites.size() / 5);

    for (auto const &[idx, writemap] : memWrites) {
        assert(!writemap.empty() && "Shouldnt be in map if there are no writes");

        // Is written by two threads
        if (writemap.size() > 1) {
            sharedIdxes.push_back(idx);
            continue;
        }

        // If it is written by 1 thread and never read it is not shared
        auto it = memReads.find(idx);
        if (it == memReads.end()) continue;

        // We know only one thread writes to this object
        // If multiple threads read atleast one must be different than the writing thread
        // Therefore it is written by one thread and read by another
        auto const &readmap = it->second;
        if (readmap.size() > 1) {
            sharedIdxes.push_back(idx);
            continue;
        }

        // We know one thread writes and one thread reads
        // If they are not the same thread the object is shared
        auto writingThreadID = writemap.begin()->first;
        auto readingThreadID = readmap.begin()->first;
        if (writingThreadID != readingThreadID) {
            sharedIdxes.push_back(idx);
            continue;
        }
    }

    sharedIdxes.shrink_to_fit();
    return sharedIdxes;
}

void OMPEngine::shrinkShared(std::vector<unsigned int> &sharedIdxes) {
    for (auto idx : sharedIdxes) {
        auto &tidToWrites = memWrites[idx];
        auto &tidToReads = memReads[idx];
        auto &tidToWritesMask = writesMask[idx];
        auto &tidToReadsMask = readsMask[idx];
        for (auto &[wtid, wset] : tidToWrites) {
            auto &wsetMask = tidToWritesMask[wtid];
            MemAccessEvent *prev = nullptr;
            std::string prevLastIn;
            std::string prevNextOut;
            for (auto e : wset) {
                if (prev == nullptr) {
                    wsetMask.push_back(e);
                    prev = e;
                    prevLastIn = graph.findLastIn(wtid, e->getID());
                    prevNextOut = graph.findNextOut(wtid, e->getID());
                } else {
                    auto lastIn = graph.findLastIn(wtid, e->getID());
                    auto nextOut = graph.findNextOut(wtid, e->getID());
                    if (prev->getLocksetID() != e->getLocksetID() || prevLastIn != lastIn || prevNextOut != nextOut) {
                        wsetMask.push_back(e);
                    } else {
                        // do not shrink events across different omp.sections
                        if (e->getInst()->getParent() != prev->getInst()->getParent())
                            if (isReductionBasicBlock(e->getInst()->getParent())) {
                                wsetMask.push_back(e);
                            }
                    }
                    prev = e;
                    prevLastIn = lastIn;
                    prevNextOut = nextOut;
                }
            }
        }
        for (auto &[rtid, rset] : tidToReads) {
            auto &rsetMask = tidToReadsMask[rtid];
            MemAccessEvent *prev = nullptr;
            std::string prevLastIn;
            std::string prevNextOut;
            for (auto e : rset) {
                if (prev == nullptr) {
                    rsetMask.push_back(e);
                    prev = e;
                    prevLastIn = graph.findLastIn(rtid, e->getID());
                    prevNextOut = graph.findNextOut(rtid, e->getID());
                } else {
                    auto lastIn = graph.findLastIn(rtid, e->getID());
                    auto nextOut = graph.findNextOut(rtid, e->getID());
                    if (prev->getLocksetID() != e->getLocksetID() || prevLastIn != lastIn || prevNextOut == nextOut) {
                        rsetMask.push_back(e);
                    } else {
                        // do not shrink events across different omp.sections
                        if (e->getInst()->getParent() != prev->getInst()->getParent())
                            if (isReductionBasicBlock(e->getInst()->getParent())) {
                                rsetMask.push_back(e);
                            }
                    }
                    prev = e;
                    prevLastIn = lastIn;
                    prevNextOut = nextOut;
                }
            }
        }
    }
}

void OMPEngine::detectRace(const CallGraphNodeTy *callNode1, const CallGraphNodeTy *callNode2) {
    // Create two dummy threads
    StaticThread thread1(callNode1);
    StaticThread thread2(callNode2);

    // TODO: set scev for omp_outlined parameters
    callEventTraces[thread1.getTID()].clear();
    callEventTraces[thread2.getTID()].clear();

    // JEFF: for context-sensitive SVF, initialize calleventtrace for these two threads
    auto *callEvent1 = graph.createCallEvent(callNode1->getContext(), ompRegion, callNode1, thread1.getTID());
    callEventTraces[thread1.getTID()].push_back(callEvent1);
    auto *callEvent2 = graph.createCallEvent(callNode2->getContext(), ompRegion, callNode2, thread2.getTID());
    callEventTraces[thread2.getTID()].push_back(callEvent2);

    std::set<StaticThread *> threadSet;
    std::queue<StaticThread *> threadList;
    std::map<const CallGraphNodeTy *, int> forkTimesMap;

    threadList.push(&thread1);
    threadList.push(&thread2);

    auto start = std::chrono::steady_clock::now();

    // Build SHB/Connectivity and trace for each thread
    while (!threadList.empty()) {
        auto currentThread = threadList.front();

        threadList.pop();

        auto entry = currentThread->getEntryNode();

        // LOG_DEBUG("omp traverse. tid={}, func={}", currentThread->getTID(), entry->getTargetFun()->getName());
        if (DEBUG_OMP_RACE)
            llvm::outs() << "omp traverse tid: " << currentThread->getTID()
                         << " func: " << entry->getTargetFun()->getName() << "\n";

        skipLoadStore = false;
        isMasterOnly = 0;
        isSingleOnly = 0;
        taskgroups.clear();
        traverse(entry, currentThread);
        if (isTheFirstThread) isTheFirstThread = false;  // not the first thread right after this

        for (auto forkEvent : currentThread->getForkSites()) {
            aser::CallSite forkSite(forkEvent->getInst());
            assert(forkSite.isCallOrInvoke());
            // assert(OMPModel::isTask(forkSite.getCalledFunction()) && "Only support task threads for now");

            auto newCtx = CT::contextEvolve(forkEvent->getContext(), forkEvent->getInst());
            auto calledFunc = OMPModel::getTaskFunction(forkSite);
            if (!calledFunc) {
                // LOG_DEBUG("Skipping omp task. task={}", *forkSite.getInstruction());
                forkEvent->setSpawnedThread(nullptr);
                continue;
            }

            auto directNode = pta->getDirectNode(newCtx, calledFunc);

            auto taskThread = new StaticThread(directNode);

            // Needed by SVFPass?
            auto const tid = taskThread->getTID();
            callEventTraces[tid].clear();
            auto callEvent = graph.createCallEvent(directNode->getContext(), forkEvent->getInst(), directNode, tid);
            callEventTraces[tid].push_back(callEvent);

            forkEvent->setSpawnedThread(taskThread);
            threadList.push(taskThread);

            for (auto taskWaitEvent : currentThread->getJoinSites()) {
                graph.addThreadJoinEdge(taskWaitEvent, taskThread->getTID());
            }

            for (auto taskgroup : taskgroups) {
                auto thisID = forkEvent->getID();
                auto startID = taskgroup.first;
                if (thisID <= startID) continue;
                auto end = taskgroup.second;
                if (thisID >= end->getID()) continue;

                graph.addThreadJoinEdge(end, forkEvent->getSpawnedThread()->getTID());
            }
        }

        threadSet.insert(currentThread);
    }

    for (auto thread : threadSet) {
        for (auto forkEvent : thread->getForkSites()) {
            if (forkEvent->getSpawnedThread()) {
                graph.addThreadForkEdge(forkEvent, forkEvent->getSpawnedThread()->getTID());
            }
        }
    }

    // NOTE: for now we assume the threads will not create new threads, but that will change when we support tasks
    // cur_trie = aser::trie::getNode(nullptr, nullptr);
    // traverse(callNode1, &thread1);
    // cur_trie = aser::trie::getNode(nullptr, nullptr);
    // traverse(callNode2, &thread2);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedTime = end - start;
    LOG_TRACE("OpenMP Region Visited. time={}s", elapsedTime.count());

    start = std::chrono::steady_clock::now();
    auto sharedIdxes = collectShared();
    shrinkShared(sharedIdxes);
    graph.initializeReachEngine();
    bool found;
    for (auto idx : sharedIdxes) {
        found = false;
        // auto const &readset = reads[obj];
        // auto const &writeset = writes[obj];
        auto const &readset = readsMask[idx];
        auto const &writeset = writesMask[idx];

        for (auto &[wtid, wSet] : writeset) {
            // write <-> read race
            for (auto &[xtid, xSet] : readset) {
                if (wtid != xtid) {
                    found = enumerateRacePair(wSet, xSet, objs[idx]);
                    if (!CONFIG_NO_FILTER && found) break;
                }
            }
            if (!CONFIG_NO_FILTER && found) break;

            // write <-> write race
            if (writeset.size() > 1) {
                for (auto &[xtid, xSet] : writeset) {
                    // if t1 has compared with t2,
                    // then we don't want t2 to compare with t1 again
                    // therefore each thread only comparesto a thread with
                    // larger tid
                    if (wtid < xtid) {
                        found = enumerateRacePair(wSet, xSet, objs[idx]);
                        if (!CONFIG_NO_FILTER && found) break;
                    }
                }
            }
            if (!CONFIG_NO_FILTER && found) break;
        }
    }
}

extern void setDebugAllSCEV(bool);

// Simple data-flow analysis to see if this instruction flows from a gep instruction
bool isLoadedFromArray(const Instruction *inst) {
    if (inst == nullptr) {
        return false;
    }

    const Instruction *source = nullptr;
    const Instruction *next = inst;

    while (next != nullptr) {
        source = next;

        if (auto sext = dyn_cast<SExtInst>(source)) {
            next = dyn_cast<Instruction>(sext->getOperand(0));
        } else if (auto load = dyn_cast<LoadInst>(source)) {
            next = dyn_cast<Instruction>(load->getPointerOperand());
        } else if (isa<GetElementPtrInst>(source)) {
            return true;
        } else {
            // llvm::outs() << "UNKN: " << *source << "\n";
            next = nullptr;
        }
    }

    return false;
}

bool isIndexFromArray(const llvm::GetElementPtrInst *gep) {
    // gep 1st arg is base ptr, all others are indexes
    // loop skips base ptr by starting i=1 and checks if any index loads from array
    auto const nOps = gep->getNumOperands();
    for (unsigned int i = 1; i < nOps; i++) {
        auto indexValue = gep->getOperand(i);
        if (isLoadedFromArray(dyn_cast<Instruction>(indexValue))) {
            return true;
        }
    }

    return false;
}

// Get instructions that store to this location
std::vector<const StoreInst *> getModifiers(const Value *val) {
    std::vector<const StoreInst *> modifiers;

    for (auto const &user : val->users()) {
        if (auto store = dyn_cast<StoreInst>(user)) {
            // Check if a store is storing to val
            if (store->getPointerOperand() == val) {
                modifiers.push_back(store);
            }
            // Check if val is passed to a function which then stores to val
            else if (auto call = dyn_cast<CallBase>(user)) {
                auto it = std::find(call->operands().begin(), call->operands().end(), val);
                if (it == call->operands().end()) continue;
                auto n = it->getOperandNo();
                auto arg = call->getCalledFunction()->arg_begin() + n;
                auto mods = getModifiers(arg);
                modifiers.insert(modifiers.end(), mods.begin(), mods.end());
            }
        }
    }

    return modifiers;
}

std::vector<const Use *> getFlowsFrom(const Value *val) {
    // logger::endPhase();
    std::vector<const Use *> worklist;
    std::set<const Use *> visited;

    // If v is a User, add all non-visited uses to worklist to be traversed
    std::function<void(const Value *v)> appendUses;
    appendUses = [&visited, &worklist, &appendUses](const Value *v) {
        auto user = dyn_cast<User>(v);
        if (!user) {
            return;
        }

        for (auto const &use : user->operands()) {
            // push each use to be visited
            if (visited.find(&use) == visited.end()) {
                worklist.push_back(&use);
            }

            // Find store insts that might modify this value and push thier dependencies as well
            auto modifiers = getModifiers(use.get());
            for (auto store : modifiers) {
                auto &innerUse = store->getOperandUse(0);
                assert(innerUse.get() == store->getValueOperand());

                if (visited.find(&innerUse) == visited.end()) {
                    worklist.push_back(&innerUse);
                }
            }
            // llvm::outs() << "2nd: " << *use << "\n";
            // for (auto user2 : use->users()) {
            //     if (user2 == user) {
            //         continue;
            //     }
            //     llvm::outs() << "\t" << *user2 << "\n";

            //     // Check if a store changes the value used
            //     if (auto store = dyn_cast<StoreInst>(user2)) {
            //         if (store->getPointerOperand() == use.get()) {
            //             appendUses(store->getValueOperand());
            //         }
            //     }
            //     // Check if a function call modifies the value
            //     else if (auto call = dyn_cast<CallBase>(user2)) {
            //         auto it = std::find(call->operands().begin(), call->operands().end(), use.get());
            //         if (it == call->operands().end()) continue;
            //         auto n = it->getOperandNo();
            //         auto arg = call->getCalledFunction()->arg_begin() + n;
            //         llvm::outs() << *arg << "\n";
            //         for (auto use : arg->uses()) {
            //             if (auto store = dyn_cast<StoreInst>(use)) {
            //                 if (store->getPointerOperand() == arg) {
            //                 }
            //             }
            //         }
            //     }
            // }
        }
    };

    appendUses(val);
    while (!worklist.empty()) {
        const Use *use = worklist.back();
        worklist.pop_back();

        visited.insert(use);

        appendUses(use->get());
    }

    return std::vector<const Use *>(visited.begin(), visited.end());
}

bool flowsFromArrayInterProc(const Value *val, std::vector<CallEvent *> callStack) {
    std::vector<const Value *> functionArgs;
    if (auto arg = dyn_cast<Argument>(val)) {
        auto n = arg->getArgNo();
        auto call = dyn_cast<CallBase>(callStack.back()->getInst());
        functionArgs.push_back(call->getArgOperand(n));
    }

    auto sources = getFlowsFrom(val);
    for (auto use : sources) {
        if (isa<GetElementPtrInst>(use)) {
            return true;
        } else if (auto arg = dyn_cast<Argument>(use)) {
            if (callStack.empty()) {
                continue;
            }
            auto n = arg->getArgNo();
            auto call = dyn_cast<CallBase>(callStack.back()->getInst());
            if (!call) {
                continue;
            }
            functionArgs.push_back(call->getArgOperand(n));
        }
    }

    callStack.pop_back();
    for (auto arg : functionArgs) {
        if (flowsFromArrayInterProc(arg, callStack)) return true;
    }

    return false;
}

bool indexFlowsFromArrayInterProc(const GetElementPtrInst *gep, std::vector<CallEvent *> callStack) {
    if (!gep) return false;

    // first operand of GEP is the array base ptr. All others are indexes
    // we only care about indexes so i starts from 1
    auto uses = gep->getOperandList();
    for (auto i = 1; i < gep->getNumOperands(); i++) {
        if (flowsFromArrayInterProc(uses[i], callStack)) return true;
    }

    return false;
}

bool flowsFromKey(const Value *val, std::vector<std::string> keys) {
    std::vector<const Use *> worklist;
    std::set<const Use *> visited;
    auto append = [&worklist, &visited](const Value *val) {
        if (auto user = dyn_cast<User>(val)) {
            for (auto const &use : user->operands()) {
                if (visited.find(&use) == visited.end()) {
                    worklist.push_back(&use);
                }
            }
        }
    };

    auto matchesKey = [&keys](StringRef &name) {
        for (auto const &key : keys) {
            if (name.find(key) != std::string::npos) {
                return true;
            }
        }
        return false;
    };

    append(val);
    while (!worklist.empty()) {
        auto use = worklist.back();
        worklist.pop_back();
        visited.insert(use);

        auto name = use->get()->getName();
        if (matchesKey(name)) {
            return true;
        }

        append(use->get());
    }

    return false;
}

bool OMPEngine::enumerateRacePair(std::vector<MemAccessEvent *> wset, std::vector<MemAccessEvent *> xset,
                                  const ObjTy *obj) {
    if (obj->isFIObject()) return false;

    for (MemAccessEvent *e1 : wset) {
        for (MemAccessEvent *e2 : xset) {
            // TBAA filtering
            auto ml1 = MemoryLocation::getOrNone(e1->getInst());
            auto ml2 = MemoryLocation::getOrNone(e2->getInst());
            if (ml1.hasValue() && ml2.hasValue()) {
                auto isAlias = tbaa.alias(ml1.getValue(), ml2.getValue(), aaqi);
                if (isAlias == AliasResult::NoAlias) {
                    LOG_TRACE("Filtered out by TBAA. e1={}, e2={}", *e1->getInst(), *e2->getInst());
                    continue;
                }
            }

            // special handling of .omp.sections.
            if (e1->getInst()->getFunction() == e2->getInst()->getFunction()) {
                auto BB1 = e1->getInst()->getParent();
                auto BB2 = e2->getInst()->getParent();
                if (BB1 == BB2) {
                    if (BB1->hasName() && BB2->hasName()) {
                        auto bbName1 = BB1->getName();
                        auto bbName2 = BB2->getName();
                        // DRB023
                        // llvm::outs()<<"bbName1: "<<bbName1<<" bbName2: "<<bbName2<<"\n";

                        //                debugEvent(pta, e1, true);
                        //                debugEvent(pta, e2);
                        if (bbName1.startswith(".omp.sections.") && bbName2.startswith(".omp.sections.")) {
                            continue;  // skip events from the same section
                        } else if (isReductionBasicBlock(BB1)) {
                            // leverage source level info  -- !$omp section
                            continue;
                        }
                    }
                }
            }
            // NOTE: actual analysis starts here!
            int p = 1;
            auto op1 = e1->getPointerOperand();
            auto op2 = e2->getPointerOperand();
            if (op1 != op2)
            // if (gep1 && gep2)
            {
                if (e2->getType() == EventType::APIRead || e2->getType() == EventType::Read) {
                    {
                        continue;
                    }
                    // if (false)
                    //     if (pta->alias(e1->getContext(), op1, e2->getContext(), op2)) {
                    //         llvm::outs() << "why alias: "
                    //                      << " op1: " << *op1 << " op2: " << *op2 << "\n";
                    //         debug = true;
                    //     }
                }
            }

            if (graph.checkHappensBefore(e1, e2)) continue;

            // check lock set
            if (graph.sharesLock(e1, e2)) {
                ignoreRaceLocations(e1, e2);
                continue;
            }
            auto *e1_caller = getEventCallerInstruction(callEventTraces, e1, e1->getTID());
            auto *e2_caller = getEventCallerInstruction(callEventTraces, e2, e2->getTID());

            if (isNestedFork && e1->getInst() == e2->getInst()) {  // for nested fork, do not check array indices

                // skip reduction
                auto varName = getSourceLoc(obj->getValue()).getName();
                // llvm::outs()<<"inst: "<<*e1->getInst()<<" varName: "<<varName<<" bbName:
                // "<<e1->getInst()->getParent()->getName()<<"\n"; for(auto var: reductionVarNames)
                //     llvm::outs()<<"reductionVarNames: "<<var<<"\n";

                if (reductionVarNames.count(varName) == 0)
                // if(nestedForkInLoop)
                {
                    // llvm::outs()<<"nested fork: adding omp race "<<varName<<"\n";
                    // debugEvent(pta, e1, true);
                    // debugEvent(pta, e2);
                    p = pass.svf->mayAlias(ompEntryFunc, e1->getContext(), e1_caller, e1->getInst(), op1,
                                           e2->getContext(), e2_caller, e2->getInst(), op2);
                    if (p>0 )
                        DataRace::collectOMP(e1, e2, obj, callEventTraces, callingCtx, ompRegion, p);
                    if (!CONFIG_NO_FILTER)
                        return true;
                    else
                        continue;
                }
            }

            // if this is array accesses
            // do array index analysis
            auto gep1 = llvm::dyn_cast<llvm::GetElementPtrInst>(e1->getPointerOperand()->stripPointerCasts());
            auto gep2 = llvm::dyn_cast<llvm::GetElementPtrInst>(e2->getPointerOperand()->stripPointerCasts());

            if (ConfigFlowFilter) {
                if (gep1) {
                    auto callStack = getCallEventStack(e1, callEventTraces);
                    if (indexFlowsFromArrayInterProc(gep1, callStack)) continue;
                }

                if (gep2) {
                    auto callStack = getCallEventStack(e2, callEventTraces);
                    if (indexFlowsFromArrayInterProc(gep2, callStack)) continue;
                }
            }

            // bool disableOldOMPAlias = true;
            if (ENABLE_OLD_OMP_ALIAS_ANALYSIS) {
                if (gep1 && gep2) {  // two array index
                    auto &AI = pass.getAnalysis<ArrayIndexAnalysisPass>(
                        const_cast<llvm::Function &>(*(e1->getInst()->getFunction())));
                    p = AI.canOmpIndexAlias(e1->getContext(), gep1, gep2,
                                            *(pass.getAnalysis<PointerAnalysisPass<PTA>>().getPTA()));

                    if (p>0) {
                        continue;
                    }

                    if (DEBUG_OMP_RACE) {
                        debugEvent(pta, e1, true);
                        if (e2->getType() == EventType::APIRead || e2->getType() == EventType::Read)
                            debugEvent(pta, e2);
                        else
                            debugEvent(pta, e2, true);
                    }
                    // let's check if the base pointers can alias, if not return Priority::NORACE;
                    if (!pta->alias(e1->getContext(), gep1->getPointerOperand(), e2->getContext(),
                                    gep2->getPointerOperand())) {
                        continue;
                    }
                }
                // if (p != Priority::NORACE)
                // llvm::outs() << "\n!!! The original canOmpIndexAlias implementation reports race !!!\n";
            }

            // Filter out different constant array index
            if (gep1 || gep2) {
                int64_t offset1 = 0, offset2 = 0;
                if (gep1) {
                    if (gep1->hasAllConstantIndices()) {
                        auto &DL = gep1->getFunction()->getParent()->getDataLayout();
                        APInt constOffset(DL.getIndexSizeInBits(gep1->getAddressSpace()), 0);
                        gep1->accumulateConstantOffset(DL, constOffset);
                        offset1 = constOffset.getSExtValue();
                    } else {
                        offset1 = -1;
                    }
                }

                if (gep2) {
                    if (gep2->hasAllConstantIndices()) {
                        auto &DL = gep2->getFunction()->getParent()->getDataLayout();
                        APInt constOffset(DL.getIndexSizeInBits(gep2->getAddressSpace()), 0);
                        gep2->accumulateConstantOffset(DL, constOffset);
                        offset2 = constOffset.getSExtValue();
                    } else {
                        offset2 = -1;
                    }
                }

                if (offset1 >= 0 && offset2 >= 0 && offset1 != offset2) {
                    LOG_WARN("different constant array index got filtered out!");
                    continue;
                }
            }
            // Filter array indirection
            // if (gep1 && isIndexFromArray(gep1)) continue;
            // if (gep2 && isIndexFromArray(gep2)) continue;

            // // Filter flowing from "first" "last"
            // if (gep1 && flowsFromKey(gep1, {"first", "last"})) continue;
            // if (gep2 && flowsFromKey(gep2, {"first", "last"})) continue;

            // if (gep1) {
            //     auto &AI = pass.getAnalysis<BradPass>(const_cast<llvm::Function &>(*(gep1->getFunction())));
            //     if (AI.bradtesting(gep1)) continue;
            // }

            // if (gep2) {
            //     auto &AI = pass.getAnalysis<BradPass>(const_cast<llvm::Function &>(*(gep2->getFunction())));
            //     if (AI.bradtesting(gep2)) continue;
            // }

            // TEST SVF
            // if (e1->getInst()->getFunction()->getName().contains("CalcElemShapeFunctionDerivatives"))
            if (false) {
                auto scev1 = pass.svf->getGlobalSCEV(e1->getContext(), nullptr, e1->getInst()->getFunction(),
                                                     e1->getPointerOperand());
                auto scev2 = pass.svf->getGlobalSCEV(e2->getContext(), nullptr, e2->getInst()->getFunction(),
                                                     e2->getPointerOperand());
                // exit(0);  // debug
            }

            if (!CONFIG_NO_KEYWORD_FILTER) {
                if (pass.svf->hasSyntacticalSignal(e1->getInst()->getFunction(), e1->getPointerOperand())
                    // JEFF: I would expect check e1 is enough as e2 is similar most of the time
                    || pass.svf->hasSyntacticalSignal(e2->getInst()->getFunction(), e2->getPointerOperand())) {
                    // llvm::outs() << "\nskipping due to syntactical filtering\n";
                    continue;
                }
            }
            // will they be alias?
            // TODO: to make it context sensitive, we need the info of this function's caller!
            // CallEvent *call_e1 = callEventTraces[e1->getTID()].back();
            // CallEvent *call_e2 = callEventTraces[e2->getTID()].back();

            // JEFF: debug
            bool debug = false;
            // if (getSourceLoc(e1->getInst()).sig().find("Mesh.cc") != string::npos) {
            //     // if (e1->getInst()->getFunction()->getName().contains("DoCritical")) {
            //     debug = true;
            // }
            // else
            //    continue;
            setDebugAllSCEV(DEBUG_OMP_RACE || debug);

            // if (ConfigSVFFilter) {
            //     // llvm::outs() << "SVF FILTER RUNNING\n";
            //     if (gep1 && pass.svf->flowsFromAny(const_cast<llvm::GetElementPtrInst *>(gep1),
            //                                        {"start", "end", "first", "last"})) {
            //         continue;
            //     }

            //     if (gep2 && pass.svf->flowsFromAny(const_cast<llvm::GetElementPtrInst *>(gep2),
            //                                        {"start", "end", "first", "last"})) {
            //         continue;
            //     }
            // }

            // Hx [i] = t
            if (false)
                if (e1->getInst() == e2->getInst() && gep1 && gep2) {
                    op1 = gep1->getPointerOperand();
                    op2 = gep2->getPointerOperand();
                    if (gep1->getNumOperands() > 1) {
                        op1 = gep1->getOperand(1);
                        op2 = gep2->getOperand(1);
                    }
                }
            bool aliasChecked = false;
            // if e1/e2 is writeAPI operator[], do an additional check for the index
            if (e1->getType() == EventType::APIWrite || e2->getType() == EventType::APIWrite) {
                // thread_to_core[tid]
                //_ZNSt6vectorIiSaIiEEixEm
                // std::vector<int, std::allocator<int> >::operator[](unsigned long)
                if (auto call1 = llvm::dyn_cast<llvm::CallBase>(e1->getInst()))
                    if (auto call2 = llvm::dyn_cast<llvm::CallBase>(e2->getInst())) {
                        if (!CONFIG_NO_KEYWORD_FILTER) {
                            if (call1->arg_size() > 1 &&
                                pass.svf->hasSyntacticalSignal(e1->getInst()->getFunction(), call1->getArgOperand(1))) {
                                continue;
                            }
                            // JEFF: I would expect check e1 is enough as e2 is similar most of the time
                            if (call2->arg_size() > 1 &&
                                pass.svf->hasSyntacticalSignal(e2->getInst()->getFunction(), call2->getArgOperand(1))) {
                                // llvm::outs() << "\nskipping due to syntactical filtering\n";
                                continue;
                            }
                        }
                        CallSite CS1(e1->getInst());
                        CallSite CS2(e2->getInst());
                        auto targetFunc1 = CS1.getTargetFunction();
                        auto targetFunc2 = CS2.getTargetFunction();
                        if (demangle(targetFunc1->getName().str()).find("::operator[]") != std::string::npos &&
                            demangle(targetFunc2->getName().str()).find("::operator[]") != std::string::npos) {
                            p = pass.svf->mayAlias(ompEntryFunc, e1->getContext(), e1_caller, e1->getInst(),
                                                   call1->getArgOperand(1), e2->getContext(), e2_caller, e2->getInst(),
                                                   call2->getArgOperand(1));
                            aliasChecked = true;
                        } else if (e1->getType() == EventType::APIRead &&
                                       demangle(targetFunc2->getName().str()).find("::operator[]") !=
                                           std::string::npos ||
                                   e2->getType() == EventType::APIRead &&
                                       demangle(targetFunc1->getName().str()).find("::operator[]") !=
                                           std::string::npos) {
                            // get rid of c++ FP: one is a write to vector[], and the other is vector.size()
                            // activeNodes.size()
                            // activeNodes[v] = 0;
                            p = 0;
                            aliasChecked = true;
                        }
                    }
            }
            if (!aliasChecked)
                p = pass.svf->mayAlias(ompEntryFunc, e1->getContext(), e1_caller, e1->getInst(), op1, e2->getContext(),
                                       e2_caller, e2->getInst(),
                                       op2);  // do they always have the same context?

            if (p >0) {  //
                DataRace::collectOMP(e1, e2, obj, callEventTraces, callingCtx, ompRegion, p);

                if (DEBUG_OMP_RACE || debug) {
                    llvm::outs() << "\n\nSVF CHECKING OPENMP RACE (f1: "
                                 << demangle(e1->getInst()->getFunction()->getName().str())
                                 << "\n                          f2: "
                                 << demangle(e2->getInst()->getFunction()->getName().str())
                                 << "\n                          caller1: "
                                 << demangle(e1_caller->getFunction()->getName().str())
                                 << "\n                          caller2: "
                                 << demangle(e2_caller->getFunction()->getName().str())
                                 << "\n                          e1: " << *e1->getInst() << " tid: " << e1->getTID()
                                 << " id: " << e1->getID() << " locksetId: " << e1->getLocksetID()
                                 << "\n                          e2: " << *e2->getInst() << " tid: " << e2->getTID()
                                 << " id: " << e2->getID() << " locksetId: " << e2->getLocksetID() << ")\n\n";
                    setDebugAllSCEV(false);
                }
            } else {
                if (debug) llvm::outs() << "\n!!!!!! SVF determines it is not a race !!!!!!!\n";
                continue;
            }
            if (!CONFIG_NO_FILTER) return true;
        }
    }
    return false;
}

std::string OMPEngine::getRaceSig(const llvm::Instruction *inst1, const llvm::Instruction *inst2) {
    std::stringstream ss;
    std::string sig1, sig2;
    ss << (void *)inst1;
    sig1 = ss.str();
    // set the content of the stringstream to empty
    ss.str("");
    ss << (void *)inst2;
    sig2 = ss.str();
    // set the content of the stringstream to empty
    ss.str("");
    if (sig1.compare(sig2) <= 0) {
        ss << sig1 << "|" << sig2;
    } else {
        ss << sig2 << "|" << sig1;
    }
    return ss.str();
}

void OMPEngine::analyze(const llvm::Instruction *const ompForkCall, const ctx *const context) {
    if (DEBUG_OMP_RACE) {
        llvm::outs() << "isTeams: " << isTeams << " isNestedForkLoop: " << isNestedFork << "\n";
    }

    ompRegion = ompForkCall;
    auto firstFork = ompForkCall;
    auto secondFork = ompForkCall->getNextNode();
    // JEFF: crash on GraphBLAS openmp_demo
    // assert(OMPModel::isFork(firstFork) && "OMPEngine must analyze kmpc_fork call");
    // assert(OMPModel::isFork(secondFork) && "Missing duplicate fork call");
    if (!OMPModel::isFork(secondFork)) return;
    auto getOutlineCallNode = [&](const llvm::Instruction *const forkInst) {
        // Get omp outlined function
        auto outlined = OMPModel::getOutlinedFunction(*forkInst);
        assert(outlined && "Could not find outlined function");

        // Compute context of callnode
        auto callNodeContext = CT::contextEvolve(context, forkInst);

        // Get the CallNode
        const CallGraphNodeTy *callNode = pta->getDirectNode(callNodeContext, outlined);
        assert(callNode && "Could not find callnode");
        return callNode;
    };

    const CallGraphNodeTy *firstCallNode = getOutlineCallNode(firstFork);
    const CallGraphNodeTy *secondCallNode = getOutlineCallNode(secondFork);

    ompEntryFunc = firstCallNode->getTargetFun()->getFunction();

    auto fname = firstCallNode->getTargetFun()->getName();
    LOG_TRACE("Analyzing OpenMP Region. region={}", fname);
    auto timerStart = std::chrono::steady_clock::now();

    // debug
    // if (fname != ".omp_outlined..3.611") return;

    //    // llvm::outs() << "OMP SCEV initializing callEventTraces for: " << fname << "\n";
    //    // TODO: set scev for omp_outlined parameters
    //    callEventTraces[1].clear();
    //    callEventTraces[2].clear();
    //
    //    // JEFF: for context-sensitive SVF, initialize calleventtrace for these two threads
    //    auto *callEvent1 = graph.createCallEvent(firstCallNode->getContext(), ompForkCall, firstCallNode, 1);
    //    callEventTraces[1].push_back(callEvent1);
    //    auto *callEvent2 = graph.createCallEvent(secondCallNode->getContext(), ompForkCall, secondCallNode, 2);
    //    callEventTraces[2].push_back(callEvent2);

    if (DEBUG_OMP_RACE) llvm::outs() << "OMPRaceEngine ompEntryFunc: " << fname << "\n";

    pass.svf->connectSCEVOMPEntryFunctionArgs(context, ompForkCall, ompEntryFunc);

    detectRace(firstCallNode, secondCallNode);

    auto timerEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedTime = timerEnd - timerStart;
    LOG_TRACE("Analyzed OpenMP Region. region={}, time={}s", fname, elapsedTime.count());

    static std::chrono::duration<double> totalTime;
    totalTime += elapsedTime;
    LOG_TRACE("OMP Engine total time. time={}s", totalTime.count());
}

void OMPRaceDetect::addMaxThreadInstructions(const llvm::Instruction *inst) { maxThreadInsts.insert(inst); }
void OMPRaceDetect::analyze(const ctx *const context, const llvm::Instruction *const ompForkCall,
                            CallingCtx &callingCtx) {
    // check if we have analyzed this omp_outlined function before;
    auto outlined = OMPModel::getOutlinedFunction(*ompForkCall);
    //    if(!outlined->getName().startswith(".omp_outlined..1377"))
    //        return;

    auto result = cache.insert(outlined);
    if (!result.second) {
        LOG_TRACE("Skipping redundant omp region. region={}", *outlined);
        return;
    }
    // check global max threads here
    for (auto maxinst : maxThreadInsts) {
        // llvm::outs() << "maxinst: " << *maxinst << "\n";
        // llvm::outs() << "ompForkCall: " << *ompForkCall << "\n";
        if (maxinst->getFunction() == ompForkCall->getFunction()) {
            std::set<const Value *> ptrOps = getIndirectLoadStoreUses(maxinst);
            // llvm::outs() << "getOutlineArg function: " << ompForkCall->getFunction()->getName() << "\n";
            auto args = getOutlineArgs(ompForkCall, maxinst, ptrOps);
            traceGlobalSingleThreadBlocks(outlined, args);
        }
    }

    auto ompcheck = new OMPEngine(pass, callingCtx, teams, nestedFork);
    ompcheck->analyze(ompForkCall, context);
    delete ompcheck;
}
