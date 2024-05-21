//
// Created by Yanze on 7/14/20.
//

#ifndef ASER_PTA_UNROLLTHREADCREATELOOPPASS_H
#define ASER_PTA_UNROLLTHREADCREATELOOPPASS_H

#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/LoopPass.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Pass.h>

namespace aser {

class UnrollThreadCreateLoopPass : public llvm::LoopPass {
public:
    static char ID;
    static std::set<llvm::StringRef> THREAD_CREATE_API;
    UnrollThreadCreateLoopPass() : llvm::LoopPass(ID) {}

    bool runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) override;

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
        AU.addRequired<llvm::AssumptionCacheTracker>();
        AU.addRequired<llvm::DominatorTreeWrapperPass>();
        AU.addRequired<llvm::LoopInfoWrapperPass>();
        AU.addRequired<llvm::ScalarEvolutionWrapperPass>();
        AU.addRequired<llvm::TargetTransformInfoWrapperPass>();
    }
};

}  // namespace aser

#endif  // ASER_PTA_UNROLLTHREADCREATELOOPPASS_H