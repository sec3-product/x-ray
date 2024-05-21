//
// Created by peiming on 12/9/19.
// TODO: this component need dramatic refactoring later!

#ifndef ASER_PTA_BRADPASS_H
#define ASER_PTA_BRADPASS_H

#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Pass.h>

#include "PTAModels/GraphBLASModel.h"

namespace aser {

class BradPass : public llvm::FunctionPass {
private:
    static const uint64_t INVALID_LOOP_BOUND = UINT64_MAX;

    llvm::LoopInfo *LI = nullptr;
    llvm::ScalarEvolution *SE = nullptr;
    llvm::DominatorTree *DT = nullptr;

public:
    static char ID;

    BradPass() : llvm::FunctionPass(ID) {}

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
    bool runOnFunction(llvm::Function &F) override;

    bool bradtesting(const llvm::GetElementPtrInst *inst);
};

}  // namespace aser

#endif  // ASER_PTA_BRADPASS_H
