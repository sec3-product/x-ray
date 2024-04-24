//
// Created by peiming on 12/12/19.
//

#ifndef ASER_PTA_CONSTANTPROPAGATIONPASS_H
#define ASER_PTA_CONSTANTPROPAGATIONPASS_H

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

struct OMPConstantPropagation : public llvm::ModulePass {
    static char ID;  // Pass identification, replacement for typeid
    OMPConstantPropagation() : ModulePass(ID) {
        // llvm::initializeTargetLibraryInfoWrapperPassPass(*llvm::PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
        AU.setPreservesCFG();
        AU.addRequired<llvm::TargetLibraryInfoWrapperPass>();
    }

    bool runOnModule(llvm::Module &M) override;

    // constant propagation intra-procedurally
    bool runOnFunction(llvm::Function &F);
};

#endif  // ASER_PTA_CONSTANTPROPAGATIONPASS_H
