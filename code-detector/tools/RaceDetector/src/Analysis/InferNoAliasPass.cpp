//
// Created by peiming on 4/17/20.
//

#include "Analysis/InferNoAliasPass.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

using namespace aser;
using namespace llvm;

namespace {

class InferNoAliasPass : public llvm::ModulePass {
private:
public:
    static char ID;
    InferNoAliasPass() : llvm::ModulePass(ID) {}

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
        AU.setPreservesAll();

        AU.addRequired<LoopInfoWrapperPass>();
        AU.addRequired<DominatorTreeWrapperPass>();
    }

    bool runOnModule(llvm::Module &M) override;
};

}  // namespace

static bool isEscapeSite(const llvm::Instruction *I, const llvm::Value *V) {
    if (auto SI = dyn_cast<StoreInst>(I)) {
        // store the src into another memory is a escape (more accurate, store into global? need alias information)
        // However, store into the src is not a escape.
        if (SI->getPointerOperand() != V) {
            // do not count store self into self
            if (SI->getPointerOperand()->stripInBoundsOffsets() != V) {
                // a potential escape location
                return true;
            }
        }
    } else if (auto call = dyn_cast<CallBase>(I)) {
        if (auto fun = call->getCalledFunction()) {
            int argNo = 0;
            for (; argNo < call->getNumArgOperands(); argNo++) {
                auto arg = call->getArgOperand(argNo);
                if (arg == V) {
                    break;
                }
            }

            if (argNo < fun->arg_size()) {
                auto arg = fun->arg_begin() + argNo;
                if (!arg->hasNoCaptureAttr()) {
                    // if arg can be captured by the callee
                    // consider as escape
                    return true;
                }
            }
        }
    }
    // phi node is not a escape site
    // it does not cause side effects and leak the reference
//    else if (auto phi = dyn_cast<PHINode>(I)) {
//        // Conservatively treat phi as escape,
//        return false;
//    }

    return false;
}

// can src escaped before used in dst
// src is the pointer
// dst is the instruction that uses the pointer
// TODO: this need more accurate CFG analysis. Should not be too hard
static bool hasEscapeBetween(Value *src, Instruction *dst, const DominatorTree *DT, const LoopInfo *LI) {
    SmallVector<Instruction *, 16> worklist;
    SmallVector<Use *, 16> useChain;

    std::set<Use *> visited;
    for (Use &use : src->uses()) {
        useChain.push_back(&use);
        visited.insert(&use);
    }

    while (!useChain.empty()) {
        auto use = useChain.pop_back_val();

        if (auto SI = dyn_cast<StoreInst>(use->getUser())) {
            // store the src into another memory is a escape (more accurate, store into global? need alias information)
            // However, store into the src is not a escape.
            if (SI->getPointerOperand() != use->get()) {
                // do not count store self into self
                if (SI->getPointerOperand()->stripInBoundsOffsets() != src) {
                    // a potential escape location
                    worklist.push_back(SI);
                }
            }
        } else if (auto call = dyn_cast<CallBase>(use->getUser())) {
            if (auto fun = call->getCalledFunction()) {
                int argNo = 0;
                for (; argNo < call->getNumArgOperands(); argNo++) {
                    auto arg = call->getArgOperand(argNo);
                    if (arg == use->get()) {
                        break;
                    }
                }

                if (argNo < fun->arg_size()) {
                    auto arg = fun->arg_begin() + argNo;
                    if (!arg->hasNoCaptureAttr()) {
                        // if arg can be captured by the callee
                        // consider as escape
                        // pthread_mutex_init has no side effect but it is not marked as nocapture
                        // TODO: collect more common APIs which does not capture the passed in pointer
                        if (!fun->getName().startswith("pthread_mutex")) {
                            worklist.push_back(call);
                        }
                    }

                }
            }
        } else if (isa<GetElementPtrInst>(use->getUser()) ||
                   isa<BitCastInst>(use->getUser()) ||
                   isa<PHINode>(use->getUser())) {
            for (auto &moreUse : use->getUser()->uses()) {
                if (visited.find(&moreUse) == visited.end()) {
                    // the use chain might have circles due to PhiNode
                    useChain.push_back(&moreUse);
                    visited.insert(&moreUse);
                }
            }
        }
        // load is not considerred as escape

        // TODO: other memory operation need to be handled as well
        // but they are rarely used, maybe address them in the feature.
    }

    for (auto escape : worklist) {
        if (escape != dst && isPotentiallyReachable(escape, dst, nullptr, DT, LI)) {
            // might escape
            return true;
        }
    }

    return false;
}

static bool isNoAliasPtrRec(Value *ptr, std::vector<Value *> &visited) {
    if (std::find(visited.begin(), visited.end(), ptr) != visited.end()) {
        // circle
        return true;
    }

    if (isa<AllocaInst>(ptr) ||  // a stack variable
        isa<CallBase>(ptr) ||    // a heap allocation
        isa<Argument>(ptr)) {    // a noalias argument

        if (auto heapObj = dyn_cast<CallBase>(ptr)) {
            if (!heapObj->hasRetAttr(Attribute::NoAlias)) {
                // make sure the returned ptr has noalias attribute
                return false;
            }
        } else if (auto argObj = dyn_cast<Argument>(ptr)) {
            if (!argObj->hasNoAliasAttr()) {
                // make sure the argument is marked as no alias.
                return false;
            }
        }  // else alloca is by default no alias
        return true;
    } else if (auto phi = dyn_cast<PHINode>(ptr)) {
        visited.push_back(ptr);
        bool isAllSrcNoAlias = true;
        for (Value *op : phi->operand_values()) {
            if (!isNoAliasPtrRec(op->stripInBoundsOffsets(), visited)) {
                isAllSrcNoAlias = false;
                break;
            }
        }
        visited.pop_back();
        // all the operand from phi node is a no alias pointer
        return isAllSrcNoAlias;
    }

    return false;
}

static bool isNoAliasPtr(Value *ptr) {
    std::vector<Value *> visited;
    return isNoAliasPtrRec(ptr, visited);
}

static bool inferInterProcNoAlias(llvm::Function &F, const DominatorTree *DT, const LoopInfo *LI) {
    // TODO: we should have a fix point algorithm
    if (F.isDeclaration()) {
        return false;
    }

    BitVector ptrArgs(F.arg_size());
    for (int i = 0; i < F.arg_size(); i++) {
        Argument *arg = F.arg_begin() + i;
        if (arg->getType()->isPointerTy() &&
            !arg->hasNoAliasAttr()) {  // no need to re-infer the argument that already has noalias information
            ptrArgs.set(i);
        }
    }

    // this is a context-insensitive pass
    // so we can infer the noalias attribute iff all the callsites pass a noalias object
    SmallVector<GlobalValue *, 8> workList;
    workList.push_back(&F);

    while (!workList.empty()) {
        if (ptrArgs.empty()) {
            break;
        }

        auto GV = workList.pop_back_val();
        for (auto user : GV->users()) {
            if (auto callSite = dyn_cast<CallBase>(user)) {
                // make sure it is calling the function (not used as callback)
                if (callSite->getCalledOperand() != GV) {
                    ptrArgs.clear();
                }

                for (unsigned ptrArg : ptrArgs.set_bits()) {
                    auto arg = callSite->getArgOperand(ptrArg);
                    auto allocSite = arg->stripInBoundsOffsets();

                    if (!isNoAliasPtr(allocSite)) {
                        ptrArgs.reset(ptrArg);
                        continue;
                    }

                    // THEN: make sure the noalias won't escape before passed to the function
                    if (hasEscapeBetween(allocSite, callSite, DT, LI)) {
                        ptrArgs.reset(ptrArg);
                        continue;
                    }
                }
            } else if (auto alias = dyn_cast<GlobalAlias>(user)) {
                    // global alias
                    workList.push_back(alias);
            } else {
                // we might have a indirect call to the function
                // most conservatively, we should not progagate noalias information
                ptrArgs.clear();
                break;
            }
        }
    }

    // the remaining ptr args are no alias
    if (ptrArgs.any()) {
        for (auto index : ptrArgs.set_bits()) {
            F.addParamAttr(index, Attribute::NoAlias);
            return true;
        }
    }

    return false;
}

static void inferIntraProcNoAlias(llvm::Function &F, MDNode *globalScopeMD,
                                  const DominatorTree *DT, const LoopInfo *LI) {
    if (F.isDeclaration()) {
        return;
    }

    for (auto &BB : F) {
        for (auto &I : BB) {
            if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
                Value *ptr = nullptr;
                if (auto load = dyn_cast<LoadInst>(&I)) {
                    ptr = load->getPointerOperand();
                } else {
                    ptr = cast<StoreInst>(I).getPointerOperand();
                }

                AAMDNodes AANodes;
                I.getAAMetadata(AANodes);
                ptr = ptr->stripInBoundsOffsets();

                if (isNoAliasPtr(ptr)) {
                    // strip off casting and offset
                    if (auto inst = dyn_cast<Instruction>(ptr);
                        inst != nullptr && inst->getParent() == I.getParent()) {

                        // simple case, in the same basic block
                        inst = inst->getNextNode();
                        bool hasEscape = false;
                        while (inst != &I) {
                            if (isEscapeSite(&I, ptr)) {
                                hasEscape = true;
                                break;
                            }
                            inst = inst->getNextNode();
                        }

                        if (!hasEscape) {
                            AANodes.NoAlias = globalScopeMD;
                        } else {
                            AANodes.Scope = globalScopeMD;
                        }
                    } else {
                        if (!hasEscapeBetween(ptr, &I, DT, LI)) {
                            AANodes.NoAlias = globalScopeMD;
                        } else {
                            AANodes.Scope = globalScopeMD;
                        }
                    }
                }

                I.setAAMetadata(AANodes);
            }
        }
    }
}

// TODO: maybe change it to context sensitive, but on demand querying if it does help a lot
bool InferNoAliasPass::runOnModule(llvm::Module &M) {
    for (auto &F : M) {
        if (F.returnDoesNotAlias()) {
            for (auto user : F.users()) {
                if (auto call = dyn_cast<CallBase>(user)) {
                    if (call->getCalledFunction() == &F && !call->returnDoesNotAlias()) {
                        call->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
                    }
                }
            }
        }
    }

    bool changed = false;
    // This should be a fixed point algorithm as newly computed noalias argument can result in more noalias argument
    // in the callee
    // FIXME: the fixed point algorithm can be improved! we can use a worklist instead of restarting all over again!
    do {
    for (auto &F : M) {
        // first, propagate no alias into parameter
        changed = inferInterProcNoAlias(F, nullptr, nullptr);
    }
    } while (changed);

    MDNode *globalScope = MDNode::get(M.getContext(), MDString::get(M.getContext(), "aser.global.AAScope"));
    for (auto &F : M) {
        // then, proprogate noalias to loads and stores within each function
        inferIntraProcNoAlias(F, globalScope, nullptr, nullptr);
    }

    // changed |= inferIntraProcNoAlias(M);

    return false;
}

ModulePass *aser::createInferNoAliasPass() { return new InferNoAliasPass(); }

char InferNoAliasPass::ID = 0;
static RegisterPass<InferNoAliasPass> INAP("Infer NoAlias Metadata",
                                           "Infer NoAlias Metadata",
                                           true, /*CFG only*/
                                           false /*is analysis*/);
