#include "CUDARaceEngine.h"

#include <sstream>

#include "CUDAModel.h"
#include "RDUtil.h"
#include "Races.h"

using namespace aser;
using namespace llvm;

extern bool CONFIG_NO_FILTER;

void CUDAEngine::addRead(TID threadID, MemAccessEvent *event) {
    static std::vector<const ObjTy *> pts;
    pts.clear();

    assert(event && "Cannot accept nullptr");

    pta.getPointsTo(event->getContext(), event->getPointerOperand(), pts);
    if (!CONFIG_NO_FILTER && pts.size() > 10) {
        return;
    }
    for (auto const obj : pts) {
        // Skip thread local object such as @sum0 = dso_local thread_local global i32 0, align 4, !dbg !0
        if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(obj->getValue()))
            if (GV->isThreadLocal()) continue;

        auto &readset = reads[obj][threadID];
        readset.insert(event);
    }
}

void CUDAEngine::addWrite(TID threadID, MemAccessEvent *event) {
    static std::vector<const ObjTy *> pts;
    pts.clear();

    assert(event && "Cannot accept nullptr");

    pta.getPointsTo(event->getContext(), event->getPointerOperand(), pts);
    if (!CONFIG_NO_FILTER && pts.size() > 10) {
        return;
    }
    for (auto const obj : pts) {
        if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(obj->getValue()))
            if (GV->isThreadLocal()) continue;
        auto &writeset = writes[obj][threadID];
        writeset.insert(event);
    }
}

// Build SHB graph, connectivity engine, and thread trace for this node
// If a new node is created the curEvent argument is updated to point to the new node
// input:
//   callNode - Node to be explored
//   thread - thread that is "executing" the callNode
//   curEvent - the most recently created node in the callstack, default value: nullptr
void CUDAEngine::traverse(const CallGraphNodeTy *callNode, StaticThread *thread, Event *curEvent) {
    // simulating call stack
    if (find(callStack.begin(), callStack.end(), callNode) == callStack.end()) {
        callStack.push_back(callNode);
    } else {
        // recursive call
        return;
    }
    int callDepth = callStack.size();
    auto context = callNode->getContext();
    const llvm::Function *func = callNode->getTargetFun()->getFunction();
    auto threadID = thread->getTID();

    for (auto const &BB : *func) {
        for (llvm::BasicBlock::const_iterator BI = BB.begin(), BE = BB.end(); BI != BE; ++BI) {
            // traverse each instruction
            const Instruction *inst = dyn_cast<Instruction>(BI);

            {  // THIS IS NEEDED BY SVF
                if (llvm::isa<llvm::CallBase>(inst)) {
                    // for inter-procedural array index analysis
                    // connecting tid with omp_outline argument
                    const Instruction *caller_inst =
                        aser::getEventCallerInstruction(callEventTraces, curEvent, threadID);
                    pass.svf->connectSCEVFunctionArgs(context, caller_inst, func, inst);
                }
            }

            // Handle CUDA specific functionality separately
            if (!visitCUDA(inst, context, thread)) {
                // Handle Normal
                if (llvm::isa<llvm::LoadInst>(inst) && !inst->isAtomic()) {
                    auto e = graph.createReadEvent(context, inst, threadID);
                    addRead(threadID, e);
                } else if (llvm::isa<llvm::StoreInst>(inst) && !inst->isAtomic()) {
                    auto e = graph.createWriteEvent(context, inst, threadID);
                    addWrite(threadID, e);
                }
                // Handle callsites
                else if (llvm::isa<llvm::CallBase>(inst)) {
                    CallSite CS(inst);
                    if (CS.isIndirectCall()) continue;

                    auto targetFunc = CS.getTargetFunction();

                    if (LangModel::isReadWriteAPI(targetFunc)) {
                        if (LangModel::isWriteAPI(targetFunc)) {
                            auto e = graph.createApiWriteEvent(context, inst, threadID);
                            addWrite(threadID, e);
                        } else if (LangModel::isReadAPI(targetFunc)) {
                            auto e = graph.createApiReadEvent(context, inst, threadID);
                            addRead(threadID, e);
                        } else if (StdStringFunctions::isStringCopy(CS.getTargetFunction())) {
                            // LOG_DEBUG("String copy constructor: {}", demangle(CS.getTargetFunction()->getName()));
                            // TODO: may need to support more constructors
                            // Special handling for container constructors
                            // auto e1 = graph.createApiWriteEvent(callNode->getContext(), inst, threadID);
                            // addWrite(threadID, e1);
                            auto e2 = graph.createApiReadEvent(callNode->getContext(), inst, threadID, true);
                            addRead(threadID, e2);
                        } else {
                            LOG_TRACE("unhandled container APIs in shb graph. api={}",
                                      demangle(CS.getTargetFunction()->getName().str()));
                            continue;
                        }
                    }

                    // Handle nested call
                    auto nextContext = CT::contextEvolve(callNode->getContext(), inst);
                    LOG_DEBUG("CUDA calling function. func={}", demangle(targetFunc->getName().str()));

                    const CallGraphNodeTy *nextNode = pta.getDirectNodeOrNull(nextContext, CS.getCalledFunction());

                    if (nextNode == nullptr) {
                        // TODO: make an assertion to ensure it is a ignored function.
                        continue;
                    }

                    CallEvent *callEvent = nullptr;
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

    callStack.pop_back();
}

// return true if this funciton was able to handle the callnode
bool CUDAEngine::visitCUDA(const llvm::Instruction *inst, const ctx *const context, StaticThread *thread) {
    if (llvm::isa<llvm::CallBase>(inst)) {
        const CallSite CS(inst);
        if (CS.isIndirectCall()) {
            return false;
        }
        auto targetFunc = CS.getTargetFunction();
        int callDepth = callStack.size();

        const StringRef func_name = (targetFunc->getName());  // demangle

        LOG_DEBUG("CUDA call function. func={}", targetFunc->getName());

        if (func_name.startswith("llvm.")) {
            if (func_name.equals("llvm.nvvm.barrier0")) {
                //__syncthreads barrier
                // TODO: - compute unqiue id for each __syncthreads region
                // llvm::outs() << "CUDA block barrier function: " << targetFunc->getName() << "\n";
                graph.createBarrierEvent(context, inst, thread->getTID(), 0);
            } else if (func_name.equals("llvm.nvvm.bar.warp.sync")) {  //_Z10__syncwarpj
                // llvm::outs() << "CUDA warp barrier function: " << targetFunc->getName() << "\n";
                // TODO: support warp level barrier
                graph.createBarrierEvent(context, inst, thread->getTID(), 0);
            }
            // get rid of llvm.dbg.value ...
            return true;
        } else if (CUDAModel::isFork(func_name)) {
            // Unhandled nested cuda kernel call?
            return true;

        } else if (CUDAModel::isAnyCUDACall(func_name)) {
            return true;
        }
    }

    return false;
}

std::vector<const ObjTy *> CUDAEngine::collectShared() const {
    std::vector<const ObjTy *> shared;
    // Guesstimate that 1/5 of all objects are shared
    shared.reserve(writes.size() / 5);

    for (auto const &[obj, writemap] : writes) {
        assert(!writemap.empty() && "Shouldnt be in map if there are no writes");

        // Is written by two threads
        if (writemap.size() > 1) {
            shared.push_back(obj);
            continue;
        }

        // If it is written by 1 thread and never read it is not shared
        auto it = reads.find(obj);
        if (it == reads.end()) continue;

        // We know only one thread writes to this object
        // If multiple threads read atleast one must be different than the writing thread
        // Therefore it is written by one thread and read by another
        auto const &readmap = it->second;
        if (readmap.size() > 1) {
            shared.push_back(obj);
            continue;
        }

        // We know one thread writes and one thread reads
        // If they are not the same thread the object is shared
        auto writingThreadID = writemap.begin()->first;
        auto readingThreadID = readmap.begin()->first;
        if (writingThreadID != readingThreadID) {
            shared.push_back(obj);
            continue;
        }
    }

    shared.shrink_to_fit();
    return shared;
}

void CUDAEngine::shrinkShared(std::vector<const ObjTy *> &shared) {
    for (auto o : shared) {
        auto &tidToWrites = writes[o];
        auto &tidToReads = reads[o];
        auto &tidToWritesMask = writesMask[o];
        auto &tidToReadsMask = readsMask[o];
        for (auto &[wtid, wset] : tidToWrites) {
            auto &wsetMask = tidToWritesMask[wtid];
            MemAccessEvent *prev = nullptr;
            std::string prevLastIn;
            std::string prevNextOut;
            for (auto e : wset) {
                if (prev == nullptr) {
                    wsetMask.insert(e);
                    prev = e;
                    prevLastIn = graph.findLastIn(wtid, e->getID());
                    prevNextOut = graph.findNextOut(wtid, e->getID());
                } else {
                    auto lastIn = graph.findLastIn(wtid, e->getID());
                    auto nextOut = graph.findNextOut(wtid, e->getID());
                    if (prev->getLocksetID() != e->getLocksetID() || prevLastIn != lastIn || prevNextOut != nextOut) {
                        wsetMask.insert(e);
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
                    rsetMask.insert(e);
                    prev = e;
                    prevLastIn = graph.findLastIn(rtid, e->getID());
                    prevNextOut = graph.findNextOut(rtid, e->getID());
                } else {
                    auto lastIn = graph.findLastIn(rtid, e->getID());
                    auto nextOut = graph.findNextOut(rtid, e->getID());
                    if (prev->getLocksetID() != e->getLocksetID() || prevLastIn != lastIn || prevNextOut == nextOut) {
                        rsetMask.insert(e);
                    }
                    prev = e;
                    prevLastIn = lastIn;
                    prevNextOut = nextOut;
                }
            }
        }
    }
}

void CUDAEngine::detectRace(const CallGraphNodeTy *callNode1, const CallGraphNodeTy *callNode2) {
    // Create two dummy threads
    StaticThread thread1(callNode1);
    StaticThread thread2(callNode2);

    auto start = std::chrono::steady_clock::now();

    // Build SHB/Connectivity and trace for each thread
    // NOTE: for now we assume the threads will not create new threads, but that will change when we support tasks
    traverse(callNode1, &thread1);
    traverse(callNode2, &thread2);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedTime = end - start;
    SPDLOG_INFO("Traversal: {}s", elapsedTime.count());

    start = std::chrono::steady_clock::now();
    auto sharedObjs = collectShared();
    shrinkShared(sharedObjs);
    graph.initializeReachEngine();
    bool found;
    for (auto const &obj : sharedObjs) {
        found = false;
        // auto const &readset = reads[obj];
        // auto const &writeset = writes[obj];
        auto const &readset = readsMask[obj];
        auto const &writeset = writesMask[obj];

        for (auto &[wtid, wSet] : writeset) {
            // write <-> read race
            for (auto &[xtid, xSet] : readset) {
                if (wtid != xtid) {
                    found = enumerateRacePair(wSet, xSet, obj);
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
                        found = enumerateRacePair(wSet, xSet, obj);
                        if (!CONFIG_NO_FILTER && found) break;
                    }
                }
            }
            if (!CONFIG_NO_FILTER && found) break;
        }
    }
}

bool CUDAEngine::enumerateRacePair(std::set<MemAccessEvent *> wset, std::set<MemAccessEvent *> xset, const ObjTy *obj) {
    
    for (MemAccessEvent *e1 : wset) {
        for (MemAccessEvent *e2 : xset) {
            // TBAA filtering
            auto ml1 = MemoryLocation::getOrNone(e1->getInst());
            auto ml2 = MemoryLocation::getOrNone(e2->getInst());
            if (ml1.hasValue() && ml2.hasValue()) {
                auto isAlias = tbaa.alias(ml1.getValue(), ml2.getValue(), aaqi);
                if (isAlias == AliasResult::NoAlias) {
                    // SPDLOG_CRITICAL("Filtered out by TBAA");
                    continue;
                }
            }

            if (graph.checkHappensBefore(e1, e2)) continue;

            // check lock set
            if (graph.sharesLock(e1, e2)) continue;

            // NOTE: actual analysis starts here!
            int p = 2;
            // if this is array accesses
            // do array index analysis
            auto gep1 = llvm::dyn_cast<llvm::GetElementPtrInst>(e1->getPointerOperand()->stripPointerCasts());
            auto gep2 = llvm::dyn_cast<llvm::GetElementPtrInst>(e2->getPointerOperand()->stripPointerCasts());
            bool disableOldOMPAlias = true;
            if (!disableOldOMPAlias) {
                if (gep1 && gep2) {  // two array index
                    auto &AI = pass.getAnalysis<ArrayIndexAnalysisPass>(
                        const_cast<llvm::Function &>(*(e1->getInst()->getFunction())));
                    p = AI.canOmpIndexAlias(e1->getContext(), gep1, gep2,
                                            *pass.getAnalysis<PointerAnalysisPass<PTA>>().getPTA());
                    if (p == 0) {
                        continue;
                    }
                }
            }

            auto *e1_caller = getEventCallerInstruction(callEventTraces, e1, e1->getTID());
            auto *e2_caller = getEventCallerInstruction(callEventTraces, e2, e2->getTID());
            if (false)
                llvm::outs() << "\n\n  SVF CHECKING CUDA RACE (f1: " << e1->getInst()->getFunction()->getName()
                             << "\n                          f2: " << e2->getInst()->getFunction()->getName()
                             << "\n                          caller1: " << e1_caller->getFunction()->getName()
                             << "\n                          caller2: " << e2_caller->getFunction()->getName()
                             << "\n                          e1: " << *e1->getInst()
                             << "\n                          e2: " << *e2->getInst() << ")\n";

            p = pass.svf->mayAlias(nullptr,e1->getContext(), e1_caller, e1->getInst(), e1->getPointerOperand(),
                                   e2->getContext(), e2_caller, e2->getInst(),
                                   e2->getPointerOperand());  // do they always have the same context?
            if (p >0)                        //
                DataRace::collectOMP(e1, e2, obj, callEventTraces, callingCtx, cudaRegion, p);
            else {
                // llvm::outs() << "\n!!!!!! SVF determines it is not a race !!!!!!!\n";
                continue;
            }

            DataRace::collect(e1, e2, obj, callEventTraces,p);

            if (!CONFIG_NO_FILTER) return true;
        }
    }
    return false;
}

std::string CUDAEngine::getRaceSig(const llvm::Instruction *inst1, const llvm::Instruction *inst2) {
    std::stringstream ss;
    std::string sig1, sig2;
    ss << (void *)inst1;
    sig1 = ss.str();
    // flush the stream
    std::stringstream().swap(ss);
    ss << (void *)inst2;
    sig2 = ss.str();
    std::stringstream().swap(ss);
    if (sig1.compare(sig2) <= 0) {
        ss << sig1 << "|" << sig2;
    } else {
        ss << sig2 << "|" << sig1;
    }
    return ss.str();
}

void CUDAEngine::analyze(const llvm::Instruction *const cudaKernelCall, const ctx *const context) {
    cudaRegion = cudaKernelCall;

    auto getKernelCallNode = [&](const llvm::Instruction *const forkInst) {
        // Compute context of callnode
        auto callNodeContext = CT::contextEvolve(context, forkInst);

        // Get the CallNode
        CallSite CS(cudaKernelCall);
        auto kernel_func = CS.getTargetFunction();
        const CallGraphNodeTy *callNode = pta.getDirectNode(callNodeContext, kernel_func);
        assert(callNode && "Could not find callnode");
        return callNode;
    };

    const CallGraphNodeTy *firstCallNode = getKernelCallNode(cudaKernelCall);
    const CallGraphNodeTy *secondCallNode = getKernelCallNode(cudaKernelCall);

    auto fname = firstCallNode->getTargetFun()->getName();
    SPDLOG_INFO("Analyzing CUDA Kernel Function: {}", demangle(fname.str()));
    auto timerStart = std::chrono::steady_clock::now();

    // llvm::outs() << "OMP SCEV initializing callEventTraces for: " << fname << "\n";
    // TODO: set scev for omp_outlined parameters
    callEventTraces[1].clear();
    callEventTraces[2].clear();

    // JEFF: for context-sensitive SVF, initialize calleventtrace for these two threads
    auto *callEvent1 = graph.createCallEvent(firstCallNode->getContext(), cudaKernelCall, firstCallNode, 1);
    callEventTraces[1].push_back(callEvent1);
    auto *callEvent2 = graph.createCallEvent(secondCallNode->getContext(), cudaKernelCall, secondCallNode, 2);
    callEventTraces[2].push_back(callEvent2);

    detectRace(firstCallNode, secondCallNode);

    auto timerEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedTime = timerEnd - timerStart;
    SPDLOG_INFO("Analyzed {} in {} seconds", demangle(fname.str()), elapsedTime.count());

    static std::chrono::duration<double> totalTime;
    totalTime += elapsedTime;
    SPDLOG_INFO("CUDA Engine total time spent: {}s", totalTime.count());
}

void CUDARaceDetect::analyze(const ctx *const context, const llvm::Instruction *const cudaKernelCall,
                             CallingCtx &callingCtx) {
    // check if we have analyzed this omp_outlined function before;
    CallSite CS(cudaKernelCall);
    auto kernel_func = CS.getTargetFunction();
    auto result = cache.insert(kernel_func);
    if (!result.second) {
        // SPDLOG_INFO("Skipping a kernel function we have seen before");
        return;
    }

    auto cudacheck = new CUDAEngine(pass, callingCtx);
    cudacheck->analyze(cudaKernelCall, context);
    delete cudacheck;
}
