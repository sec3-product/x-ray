//
// Created by peiming on 1/6/20.
//
#include "aser/PreProcessing/Passes/CanonicalizeGEPPass.h"

//#include <llvm/IR/CallSite.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/IR/Operator.h>

#include "aser/Util/Log.h"

using namespace llvm;
using namespace aser;

static bool expandNestedGEP(Function &F, IRBuilder<NoFolder> &builder) {
    bool fixPoint;
    bool changed = false;  // whether we changed IR

    // 1st, get rid of nested instructions (using constexpr as their operand)
    // TODO: this might influence the performance because of the fix point
    // TODO: it is probably fine, but optimize it when we do find it is an issue
    do {
        fixPoint = true;
        for (auto &BB : F) {
            for (auto &I : BB) {
                // for every instruction in the Function, try
                for (int i = 0; i < I.getNumOperands(); i++) {
                    Value *op = I.getOperand(i);
                    if (auto CE = dyn_cast<ConstantExpr>(op)) {
                        changed = true;
                        fixPoint = false;
                        builder.SetInsertPoint(&I);
                        Instruction *inst = builder.Insert(CE->getAsInstruction());
                        I.replaceUsesOfWith(op, inst);
                    }
                }
            }
        }
    } while (!fixPoint);  // we need a fixed point algorithm here because constant expression can be recursive

    return changed;
}

static bool splitVariableGEP(Function &F, IRBuilder<NoFolder> &builder) {
    bool changed = false;
    // the queue that stores all the constant indices
    std::vector<Value *> consIndices;
    std::vector<GetElementPtrInst *> erasedGEP;
    ConstantInt *zero = ConstantInt::get(IntegerType::getInt32Ty(F.getContext()), 0);

    // 2nd, split instructions with that uses variable to index
    for (auto &BB : F) {
        for (auto &I : BB) {
            if (auto GEP = dyn_cast<GetElementPtrInst>(&I)) {
                if (GEP->hasAllConstantIndices()) {
                    continue;
                }
                LOG_TRACE("replacing: {} with", I);
                builder.SetInsertPoint(GEP);
                Value *lastBasePtr = GEP->getPointerOperand();

                gep_type_iterator GTI = gep_type_begin(GEP);
                if (!isa<Constant>(GTI.getOperand())) {
                    // the first idx is not a constant
                    lastBasePtr = builder.CreateGEP(nullptr, lastBasePtr, GTI.getOperand());
                    LOG_TRACE("{}", *lastBasePtr);
                    // since we skip the first index, we now start from gep 0,
                    consIndices.push_back(zero);
                    GTI++;
                } else if (auto *idx = cast<Constant>(GTI.getOperand()); !idx->isZeroValue()) {
                    // the first index is constant, but is not a zero
                    // getelementptr %ptr, 4, %idx1, ...
                    lastBasePtr = builder.CreateGEP(nullptr, lastBasePtr, idx);
                    LOG_TRACE("{}", *lastBasePtr);
                    consIndices.push_back(zero);
                    GTI++;
                }

                for (gep_type_iterator GTE = gep_type_end(GEP); GTI != GTE; GTI++) {
                    Value *op = GTI.getOperand();

                    if (isa<Constant>(op)) {
                        consIndices.push_back(op);
                    } else {
                        // we have a variable index
                        // 1st, emit stored constant index if we have any
                        // 2nd, emit the variable index.
                        if (consIndices.size() != 1) {
                            // if we have the const index, emit them
                            // we are now handling a variable index, emit all the constant indices
                            lastBasePtr = builder.CreateGEP(nullptr, lastBasePtr, consIndices);
                            LOG_TRACE("{}", *lastBasePtr);
                        }
                        // emit current variable index
                        lastBasePtr = builder.CreateGEP(nullptr, lastBasePtr, {zero, op});
                        LOG_TRACE("{}", *lastBasePtr);
                        // clear stored indices.
                        consIndices.clear();
                        consIndices.push_back(zero);
                    }
                }

                if (consIndices.size() != 1) {
                    // consIndices.insert(consIndices.begin(), zero);
                    lastBasePtr = builder.CreateGEP(nullptr, lastBasePtr, consIndices);
                    LOG_TRACE("{}", *lastBasePtr);
                } else {
                    // the first should be a zero
                    assert(consIndices.front() == zero);
                }

                consIndices.clear();
                GEP->replaceAllUsesWith(lastBasePtr);
                erasedGEP.push_back(GEP);
            }
        }
    }

    for (auto GEP : erasedGEP) {
        changed = true;
        GEP->eraseFromParent();
    }

    return changed;
}

bool CanonicalizeGEPPass::doInitialization(Module &M) {
    LOG_INFO("Canonicalizing Loops");
    return false;
}

bool CanonicalizeGEPPass::runOnFunction(Function &F) {
    IRBuilder<NoFolder> builder(F.getContext());
    bool changed = false;
    // for field sensitive
    // 1st, expanded nested GEP instructio
    changed |= expandNestedGEP(F, builder);
    // 2nd, split GEP instruction that uses variable indices
    changed |= splitVariableGEP(F, builder);

    return changed;
}

char CanonicalizeGEPPass::ID = 0;
static RegisterPass<CanonicalizeGEPPass> CIP("", "Canonicalize GetElementPtr instruction", true, /*CFG only*/
                                              false /*is analysis*/);
