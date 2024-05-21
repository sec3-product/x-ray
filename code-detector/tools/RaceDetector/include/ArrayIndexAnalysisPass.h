//
// Created by peiming on 12/9/19.
// TODO: this component need dramatic refactoring later!

#ifndef ASER_PTA_ARRAYINDEXANALYSISPASS_H
#define ASER_PTA_ARRAYINDEXANALYSISPASS_H

#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Pass.h>

#include "PTAModels/GraphBLASModel.h"
#include "RDUtil.h"

namespace aser {

class ArrayIndexAnalysisPass : public llvm::FunctionPass {
private:
    static const uint64_t INVALID_LOOP_BOUND = UINT64_MAX;

    llvm::LoopInfo *LI = nullptr;
    llvm::ScalarEvolution *SE = nullptr;
    llvm::DominatorTree *DT = nullptr;

    //    llvm::BasicBlock *initForBlock = nullptr;
    //    llvm::CallInst *initForLoopInst = nullptr;

    llvm::SmallDenseMap<llvm::BasicBlock *, llvm::CallInst *, 4> ompStaticInitForBlocks;
    llvm::SmallDenseMap<llvm::BasicBlock *, llvm::CallInst *, 4> ompDispatchInitForBlocks;
    llvm::SmallDenseMap<const llvm::Value *, const llvm::ConstantInt *, 8> resolvedConstant;

    bool isOMPForLoop(const llvm::Loop *);
    bool isInOMPForLoop(const llvm::Instruction *);
    llvm::CallInst *getOMPStaticInitCall(const llvm::Loop *);
    llvm::CallInst *getOMPDispatchInitCall(const llvm::Loop *);

    const llvm::ConstantInt *resolveSingleStoreInt(const PTA &pta, const ctx *context, const llvm::Value *v);

    void resolveConstantParameter(const PTA &pta, const ctx *context, const llvm::Argument *v);

    void resolveConstantGlobal(const llvm::GlobalVariable *g);

    uint64_t resolveBound(const PTA &pta, const ctx *context, const llvm::Value *v,
                          const llvm::Instruction *initForCall);

public:
    static char ID;

    ArrayIndexAnalysisPass() : llvm::FunctionPass(ID) {}

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
    bool runOnFunction(llvm::Function &F) override;

    const llvm::SCEVAddRecExpr *getOMPLoopSCEV(const llvm::SCEV *root);

    static bool SCEVContainsUnknown(const llvm::SCEV *root);

    // PTA is required to resolve up and lower bound
    int canOmpIndexAlias(const ctx *context, const llvm::GetElementPtrInst *gep1,
                              const llvm::GetElementPtrInst *gep2, const PTA &pta);
};

}  // namespace aser

#endif  // ASER_PTA_ARRAYINDEXANALYSISPASS_H
