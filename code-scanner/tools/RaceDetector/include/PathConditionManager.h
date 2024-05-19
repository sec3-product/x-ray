#ifndef PATHCONDITIONMANAGER_H
#define PATHCONDITIONMANAGER_H

#include <llvm/IR/Instructions.h>

#include <map>

#include "StaticThread.h"
#include "aser/PointerAnalysis/Program/CallSite.h"

namespace aser {
class PathCondition {
private:
    static std::set<const llvm::BasicBlock *> skipBBs;

    inline static void selectTruePath(const llvm::BranchInst *br);
    inline static void selectFalsePath(const llvm::BranchInst *br);
    // `br`: the branch inst which we alreday solved its condition
    // `isTrue`: if we want to skip the true-branch (the branch condition evaluated to be false)
    // skipBranch(br, true) means we want to skip the true-branch of br
    static void skipBranch(const llvm::BranchInst *br, bool isTrueBr);

    // evaluate ICmpInst
    static bool evalICmp(const llvm::BranchInst *br, const llvm::ICmpInst *icmp,
                         std::map<uint8_t, const llvm::Constant *> *valMap);
    static const llvm::Constant *solveICmpOpVal(const llvm::ICmpInst *icmp, unsigned opNo,
                                                std::map<uint8_t, const llvm::Constant *> *valMap);

public:
    static const std::map<uint8_t, const llvm::Constant *> collectArgVals(
        aser::CallSite &cs, const std::map<uint8_t, const llvm::Constant *> *preValMap = nullptr);
    static void collectPThreadCreate(aser::TID tid, aser::CallSite &cs,
                                     const std::map<uint8_t, const llvm::Constant *> *preValMap = nullptr);
    static bool shouldSkipBB(const llvm::BasicBlock *BB);
    static void checkBranchCondition(const llvm::BranchInst *br, std::map<uint8_t, const llvm::Constant *> *valMap);
    static void checkSwitch(const llvm::SwitchInst *si, std::map<uint8_t, const llvm::Constant *> *valMap);
    static void skipSwitchPaths(const llvm::SwitchInst *si, const llvm::ConstantInt *caseVal);
};

}  // namespace aser

#endif