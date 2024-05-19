//
// Created by peiming on 12/12/19.
//

#include "ConstantPropagationPass.h"

#include "aser/Util/Log.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <llvm/IR/AbstractCallSite.h>

using namespace llvm;

DEBUG_COUNTER(CPCounter, "constprop-transform", "Controls which instructions are killed");

bool OMPConstantPropagation::runOnFunction(Function &F) {
    // Initialize the worklist to all of the instructions ready to process...
    SmallPtrSet<Instruction *, 16> WorkList;
    // The SmallVector of WorkList ensures that we do iteration at stable order.
    // We use two containers rather than one SetVector, since remove is
    // linear-time, and we don't care enough to remove from Vec.
    SmallVector<Instruction *, 16> WorkListVec;
    for (Instruction &I : instructions(&F)) {
        WorkList.insert(&I);
        WorkListVec.push_back(&I);
    }

    bool Changed = false;
    const DataLayout &DL = F.getParent()->getDataLayout();
    TargetLibraryInfo *TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);

    while (!WorkList.empty()) {
        SmallVector<Instruction *, 16> NewWorkListVec;
        for (auto *I : WorkListVec) {
            WorkList.erase(I);  // Remove element from the worklist...

            if (!I->use_empty())  // Don't muck with dead instructions...
                if (Constant *C = ConstantFoldInstruction(I, DL, TLI)) {
                    if (!DebugCounter::shouldExecute(CPCounter)) continue;

                    // Add all of the users of this instruction to the worklist,
                    // they might be constant propagatable now...
                    for (User *U : I->users()) {
                        // If user not in the set, then add it to the vector.
                        if (WorkList.insert(cast<Instruction>(U)).second)
                            NewWorkListVec.push_back(cast<Instruction>(U));
                    }

                    // Replace all of the uses of a variable with uses of the
                    // constant.
                    I->replaceAllUsesWith(C);

                    if (isInstructionTriviallyDead(I, TLI)) {
                        I->eraseFromParent();
                    }

                    // We made a change to the function...
                    Changed = true;
                }
        }
        WorkListVec = std::move(NewWorkListVec);
    }
    return Changed;
}

static bool PropagateConstantsIntoArguments(Function &F) {
    if (F.arg_empty() || F.use_empty()) return false;  // No arguments? Early exit.

    // For each argument, keep track of its constant value and whether it is a
    // constant or not.  The bool is driven to true when found to be
    // non-constant.
    SmallVector<std::pair<Constant *, bool>, 16> ArgumentConstants;
    ArgumentConstants.resize(F.arg_size());

    unsigned NumNonconstant = 0;
    for (Use &U : F.uses()) {
        User *UR = U.getUser();
        // Ignore blockaddress uses.
        if (isa<BlockAddress>(UR)) continue;

        // If no abstract call site was created we did not understand the use,
        // bail.
        AbstractCallSite ACS(&U);
        if (!ACS) return false;

        // Mismatched argument count is undefined behavior. Simply bail out to
        // avoid handling of such situations below (avoiding asserts/crashes).
        unsigned NumActualArgs = ACS.getNumArgOperands();
        if (F.isVarArg() ? ArgumentConstants.size() > NumActualArgs : ArgumentConstants.size() != NumActualArgs)
            return false;

        // Check out all of the potentially constant arguments.  Note that we
        // don't inspect varargs here.
        Function::arg_iterator Arg = F.arg_begin();
        for (unsigned i = 0, e = ArgumentConstants.size(); i != e; ++i, ++Arg) {
            // If this argument is known non-constant, ignore it.
            if (ArgumentConstants[i].second) continue;

            Value *V = ACS.getCallArgOperand(i);
            Constant *C = dyn_cast_or_null<Constant>(V);

            // Mismatched argument type is undefined behavior. Simply bail out
            // to avoid handling of such situations below (avoiding
            // asserts/crashes).
            if (C && Arg->getType() != C->getType()) return false;

            // We can only propagate thread independent values through
            // callbacks. This is different to direct/indirect call sites
            // because for them we know the thread executing the caller and
            // callee is the same. For callbacks this is not guaranteed, thus a
            // thread dependent value could be different for the caller and
            // callee, making it invalid to propagate.
            if (C && ACS.isCallbackCall() && C->isThreadDependent()) {
                // Argument became non-constant. If all arguments are
                // non-constant now, give up on this function.
                if (++NumNonconstant == ArgumentConstants.size()) return false;

                ArgumentConstants[i].second = true;
                continue;
            }

            if (C && ArgumentConstants[i].first == nullptr) {
                ArgumentConstants[i].first = C;  // First constant seen.
            } else if (C && ArgumentConstants[i].first == C) {
                // Still the constant value we think it is.
            } else if (V == &*Arg) {
                // Ignore recursive calls passing argument down.
            } else {
                // Argument became non-constant.  If all arguments are
                // non-constant now, give up on this function.
                if (++NumNonconstant == ArgumentConstants.size()) return false;
                ArgumentConstants[i].second = true;
            }
        }
    }

    // If we got to this point, there is a constant argument!
    assert(NumNonconstant != ArgumentConstants.size());
    bool MadeChange = false;
    Function::arg_iterator AI = F.arg_begin();
    for (unsigned i = 0, e = ArgumentConstants.size(); i != e; ++i, ++AI) {
        // Do we have a constant argument?
        if (ArgumentConstants[i].second || AI->use_empty() || AI->hasInAllocaAttr() ||
            (AI->hasByValAttr() && !F.onlyReadsMemory()))
            continue;

        Value *V = ArgumentConstants[i].first;
        if (!V) V = UndefValue::get(AI->getType());
        AI->replaceAllUsesWith(V);
        MadeChange = true;
    }
    return MadeChange;
}

ModulePass *createOMPConstantPropagationPass() { return new OMPConstantPropagation(); }

bool OMPConstantPropagation::runOnModule(Module &M) {
    LOG_INFO("Propagating Constants");

    if (skipModule(M)) return false;

    bool Changed = false;
    bool LocalChange = true;
    bool functionChanged = true;

    // propagate constant into function arguement
    SmallSet<Function *, 8> changedFunction;
    while (functionChanged) {
        while (LocalChange) {
            LocalChange = false;
            for (Function &F : M) {
                if (!F.isDeclaration()) {
                    // Delete any klingons.
                    F.removeDeadConstantUsers();
                    if (PropagateConstantsIntoArguments(F)) {
                        changedFunction.insert(&F);
                        LocalChange = true;
                    }
                }
            }
            Changed |= LocalChange;
        }

        functionChanged = false;
        // propagate constant inside the function
        if (!changedFunction.empty()) {
            for (Function *F : changedFunction) {
                functionChanged |= runOnFunction(*F);
            }
        }
    }
    return Changed;
}

char OMPConstantPropagation::ID = 0;
static RegisterPass<OMPConstantPropagation> OCP("Constant Propagation for OMP callbacks",
                                                "Constant Propagation for OMP callbacks", true, /*CFG only*/
                                                false /*is analysis*/);
