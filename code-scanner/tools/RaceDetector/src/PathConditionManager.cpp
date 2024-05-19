#include "PathConditionManager.h"

#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>

#include <map>
#include <vector>

#include "Graph/Event.h"
#include "CustomAPIRewriters/ThreadAPIRewriter.h"

using namespace std;
using namespace llvm;
using namespace aser;

// FIXME: we may bug under the following situtation
// Recursive Call:
//  if we found a false branch to skip,
//  yet when we are traversing the true branch (assuming true branch comes first),
//  there's a recursive function call whose branch (the same branch) cannot be skipped.
//  But since we do not distinguish the callsite, we will skip the inner BB (which should not be skipped)
// NOTE: we use a set to store BBs which we should skip because:
// 1. if there are complex branches (which involves multiple BBs), we need to skip more than one BB
// 2. we want to support switch inst, because there are usually multiple BBs to skip
// 3. we want to handle nested skippable branches (if we only store the next BB to skip, it may get overwritten)
set<const BasicBlock *> PathCondition::skipBBs;

// Decide the BBs we need to skip traversing in race detection
// if a branch condition is solvable.
// Algorithm:
// Traverse along the BB from the branch entry
// if we meet a BB with >= 1 predecessor, check if all its predecessor are skipped
// if so, we skip this BB, and keep traversing
// if not, we stop
void PathCondition::skipBranch(const BranchInst *br, bool isTrueBr) {
    static std::queue<const BasicBlock *> worklist;
    // the index for br->getSuccessor()
    // 0 is true-branch, 1 is false-branch
    unsigned brIdx = isTrueBr ? 0 : 1;
    auto entry = br->getSuccessor(brIdx);
    worklist.push(entry);

    unsigned count = 0;
    const BasicBlock *next;
    while (!worklist.empty()) {
        count++;
        next = worklist.front();
        worklist.pop();

        bool shouldSkip = true;
        // more than one predecessors
        if (next->getSinglePredecessor() == nullptr) {
            for (auto pred : predecessors(next)) {
                // if any of the predecessor we should not skip
                // then we should not skip this BB either
                if (!skipBBs.count(pred)) {
                    shouldSkip = false;
                    break;
                }
            }
        }
        // we skip a BB under either conditions:
        // 1. only one predecessor
        // 2. all predecessors should be skipped
        if (shouldSkip) {
            auto result = skipBBs.insert(next);
            if (result.second) {
                // push all the successors of current BB into worklist
                auto term = next->getTerminator();
                if (term != nullptr) {
                    auto numSucc = term->getNumSuccessors();
                    for (int i = 0; i < numSucc; i++) {
                        auto succ = term->getSuccessor(i);
                        worklist.push(succ);
                    }
                }
            }
        }
        // to capture anomalies
        if (count > 10000) {
            LOG_ERROR("count > 10000. br = {}", *br);
            break;
        }
    }
}

bool PathCondition::shouldSkipBB(const BasicBlock *BB) {
    if (skipBBs.count(BB)) {
        skipBBs.erase(BB);
        return true;
    }
    return false;
}

const map<uint8_t, const Constant *> PathCondition::collectArgVals(CallSite &cs,
                                                                   const map<uint8_t, const Constant *> *preValMap) {
    map<uint8_t, const Constant *> valMap;
    unsigned argn = cs.getNumArgOperands();
    for (int i = 0; i < argn; i++) {
        auto arg = cs.getArgOperand(i);
        // if this is a constant argument, collect it
        if (isa<Constant>(arg)) {
            const Constant *c = cast<Constant>(arg);
            LOG_TRACE("constant collected. no = {}, constant = {}", i, *c);
            valMap[i] = c;
        }
        // if it is an argument passed from its caller
        // we keep propagate it
        else if (preValMap != nullptr && isa<Argument>(arg)) {
            auto argNo = cast<Argument>(arg)->getArgNo();
            if (preValMap->count(argNo)) {
                valMap[i] = preValMap->at(argNo);
            }
        }
    }
    return valMap;
}

// TODO: there are mainly 2 things needs to be addressed
// 1. the arg passed to the thread routine is the 4-th arg
// 2. there's a type casting to (void *), so it's harder to know if it's a constant
void PathCondition::collectPThreadCreate(TID tid, CallSite &cs, const map<uint8_t, const Constant *> *preValMap) {
    map<uint8_t, const Constant *> valMap;
    int argStartIdx = 3;

    if (cs.getCalledFunction()->getName().startswith(ThreadAPIRewriter::getCanonicalizedAPIPrefix())) {
        argStartIdx = 2;
    }

    for (int argIdx = argStartIdx, i = 0; argIdx < cs.getNumArgOperands(); argIdx++, i++) {
        auto arg = cs.getArgOperand(argIdx);
        // llvm::outs() << "arg param: " << *arg << "\n";
        if (!isa<Instruction>(arg)) return;

        auto var = cast<Instruction>(arg)->getOperand(0);
        // FIXME: right now we only handle the simplest case
        for (auto U : var->users()) {
            if (U == arg) break;
            // llvm::outs() << "user: " << *U << "\n";
            if (isa<StoreInst>(U)) {
                auto store = cast<StoreInst>(U);
                auto storeVal = store->getOperand(0);
                if (isa<Constant>(storeVal)) {
                    // llvm::outs() << "stored val: " << *storeVal << "\n";
                    // NOTE: although the argument for pthread subroutine is the 4-th arg in pthread_create
                    // it is the only arg for pthread subroutine
                    valMap[i] = cast<Constant>(storeVal);
                }
            }
        }
    }

    if (!valMap.empty()) StaticThread::setThreadArg(tid, valMap);
}

// This is the main logic for path-sensitive analysis
// 1. we try using the valMap to compute the value of a path-condition
// 2. try solving the path-condition (decide true/false)
// 3. based on the constraint-solving result, decide which BB to skip
void PathCondition::checkBranchCondition(const llvm::BranchInst *br,
                                         std::map<uint8_t, const llvm::Constant *> *valMap) {
    assert(br->isConditional() && "the argument br must be a conditional branch");

    auto cond = br->getCondition();
    // direct const propagation
    if (isa<Argument>(cond)) {
        auto argNo = cast<Argument>(cond)->getArgNo();
        if (valMap->count(argNo)) {
            auto val = valMap->at(argNo);
            // llvm::outs() << "isZero?: " << val->isOneValue() << "\n";
            if (val->isOneValue()) {
                // llvm::outs() << "===========================\n";
                // llvm::outs() << "SELECT an always TRUE conditional branch: " << *br << "\n"
                //              << "Found condition value propagate from constant: " << *val
                //              << "\n"
                //                 "EVALUATE value to be: true\n";
                // auto dbgloc = br->getDebugLoc();
                // llvm::outs() << dbgloc->getFilename() << "@" << dbgloc->getLine() << "\n";

                // only traverse true branch
                PathCondition::skipBranch(br, false);
            } else {
                // llvm::outs() << "===========================\n";
                // llvm::outs() << "SELECT an always FALSE conditional branch: " << *br << "\n"
                //              << "Found condition value propagate from constant: " << *val
                //              << "\n"
                //                 "EVALUATE value to be: false\n";
                // auto dbgloc = br->getDebugLoc();
                // llvm::outs() << dbgloc->getFilename() << "@" << dbgloc->getLine() << "\n";

                // only traverse false branch
                PathCondition::skipBranch(br, true);
            }
        }
    }
    // complex cases, such as if (arg == 1) is involved
    else if (isa<ICmpInst>(cond)) {
        auto icmp = cast<ICmpInst>(cond);
        // llvm::outs() << "Found complex condition: " << *icmp << "\n";
        evalICmp(br, icmp, valMap);
    }
}

void PathCondition::skipSwitchPaths(const SwitchInst *si, const ConstantInt *caseVal) {
    static std::queue<const BasicBlock *> worklist;

    // solve the switch path
    // `dest` is the selected case entry
    auto dest = si->findCaseValue(caseVal)->getCaseSuccessor();

    for (auto c : si->cases()) {
        auto succ = c.getCaseSuccessor();
        // multiple cases can share a path
        // we need to make sure the selected path are not skipped
        // NOTE: do we also need to check the predecessor number here?
        if (succ != dest) {
            // save to worklist for later traversal
            worklist.push(succ);
        }
    }
    // the default case is not within the `si->cases()`
    auto defaultCase = si->getDefaultDest();
    // if default case only has 1 predecessor
    // this means we should definitely traverse it
    // because we are sure it needs to be skipped
    if (defaultCase != dest && defaultCase->hasNPredecessors(1)) {
        worklist.push(defaultCase);
    }

    int count = 0;
    const BasicBlock *next;
    while (!worklist.empty()) {
        count++;
        next = worklist.front();
        worklist.pop();

        // NOTE: the logic is similar to `skipBranch`
        bool shouldSkip = true;
        // more than one predecessors
        if (next->getSinglePredecessor() == nullptr) {
            if (next == dest) {
                // NOTE: this is a tricky case for switch
                // if a case is not end with `break`,
                // it is possible to enter other switch entries.
                // so we should not skip this BB if it's the selected case entry
                shouldSkip = false;
            } else {
                // the BB where the switch inst belongs to
                auto switchBB = si->getParent();
                for (auto pred : predecessors(next)) {
                    if (pred == switchBB) continue;
                    // if any of the predecessor we should not skip
                    // then we should not skip this BB either
                    if (!skipBBs.count(pred)) {
                        shouldSkip = false;
                        break;
                    }
                }
            }
        }
        // we skip a BB under either conditions:
        // 1. only one predecessor
        // 2. if a BB is not the selected case entry, and all of it's predecessors are
        //  either from the switch inst's BB or a previously skipped BB
        if (shouldSkip) {
            auto result = skipBBs.insert(next);
            if (result.second) {
                // push all the successors of current BB into worklist
                auto term = next->getTerminator();
                if (term != nullptr) {
                    auto numSucc = term->getNumSuccessors();
                    for (int i = 0; i < numSucc; i++) {
                        auto succ = term->getSuccessor(i);
                        worklist.push(succ);
                    }
                }
            }
        }
        // to capture anomalies
        if (count > 10000) {
            LOG_ERROR("count>10000. case value = {}, switch = {}", *caseVal, *si);
            break;
        }
    }
}

void PathCondition::checkSwitch(const llvm::SwitchInst *si, std::map<uint8_t, const llvm::Constant *> *valMap) {
    auto cond = si->getCondition();
    if (isa<Argument>(cond)) {
        auto argNo = cast<Argument>(cond)->getArgNo();
        if (valMap->count(argNo)) {
            auto val = valMap->at(argNo);
            if (auto caseVal = dyn_cast<ConstantInt>(val)) {
                skipSwitchPaths(si, caseVal);
            } else {
                LOG_ERROR("wrong const value being propagated. switch inst = {}", *si);
            }
        }
    }
}

// TODO: right now we only support ICmpInst, maybe more later?
// this function abstracts the condition solving process
// later we could apply different backend to this (like z3)
bool PathCondition::evalICmp(const llvm::BranchInst *br, const ICmpInst *icmp,
                             std::map<uint8_t, const llvm::Constant *> *valMap) {
    auto predicate = icmp->getPredicate();
    const Constant *val1, *val2;
    val1 = solveICmpOpVal(icmp, 0, valMap);
    val2 = solveICmpOpVal(icmp, 1, valMap);
    if (val1 == nullptr || val2 == nullptr) return false;

    if (predicate == CmpInst::Predicate::ICMP_EQ) {
        if (val1 == val2) {
            // llvm::outs() << "===========================\n";
            // llvm::outs() << "SELECT an always TRUE conditional branch: " << *br << "\n"
            //              << "ICMP EQ: " << *icmp << "\n"
            //              << "Found condition value propagate from constant: \n"
            //              << "value 1: " << *val1 << "\n"
            //              << "value 2: " << *val2 << "\n"
            //              << "EVALUATE value to be: value 1 == value 2\n";
            // auto dbgloc = br->getDebugLoc();
            // llvm::outs() << dbgloc->getFilename() << "@" << dbgloc->getLine() << "\n";

            // PathCondition::selectTruePath(br);
            PathCondition::skipBranch(br, false);
        } else {
            // llvm::outs() << "===========================\n";
            // llvm::outs() << "SELECT an always False conditional branch: " << *br << "\n"
            //              << "ICMP EQ: " << *icmp << "\n"
            //              << "Found condition value propagate from constant: \n"
            //              << "value 1: " << *val1 << "\n"
            //              << "value 2: " << *val2 << "\n"
            //              << "EVALUATE value to be: value 1 != value 2\n";
            // auto dbgloc = br->getDebugLoc();
            // llvm::outs() << dbgloc->getFilename() << "@" << dbgloc->getLine() << "\n";

            // PathCondition::selectFalsePath(br);
            PathCondition::skipBranch(br, true);
        }
        return true;
    } else if (predicate == CmpInst::Predicate::ICMP_NE) {
        if (val1 != val2) {
            // llvm::outs() << "===========================\n";
            // llvm::outs() << "SELECT an always True conditional branch: " << *br << "\n"
            //              << "ICMP NE: " << *icmp << "\n"
            //              << "Found condition value propagate from constant: \n"
            //              << "value 1: " << *val1 << "\n"
            //              << "value 2: " << *val2 << "\n"
            //              << "EVALUATE value to be: value 1 != value 2\n";
            // auto dbgloc = br->getDebugLoc();
            // llvm::outs() << dbgloc->getFilename() << "@" << dbgloc->getLine() << "\n";

            // PathCondition::selectTruePath(br);
            PathCondition::skipBranch(br, false);
        } else {
            // llvm::outs() << "===========================\n";
            // llvm::outs() << "SELECT an always False conditional branch: " << *br << "\n"
            //              << "ICMP NE: " << *icmp << "\n"
            //              << "Found condition value propagate from constant: \n"
            //              << "value 1: " << *val1 << "\n"
            //              << "value 2: " << *val2 << "\n"
            //              << "EVALUATE value to be: value 1 == value 2\n";
            // auto dbgloc = br->getDebugLoc();
            // llvm::outs() << dbgloc->getFilename() << "@" << dbgloc->getLine() << "\n";

            // PathCondition::selectFalsePath(br);
            PathCondition::skipBranch(br, true);
        }
        return true;
    }
    return false;
}

const Constant *PathCondition::solveICmpOpVal(const llvm::ICmpInst *icmp, unsigned opNo,
                                              std::map<uint8_t, const llvm::Constant *> *valMap) {
    auto op = icmp->getOperand(opNo);
    if (isa<Constant>(op)) {
        return cast<Constant>(op);
    } else if (isa<Argument>(op->stripPointerCasts())) {
        auto argNo = cast<Argument>(op->stripPointerCasts())->getArgNo();
        if (valMap->count(argNo)) return valMap->at(argNo);
    } else if (isa<LoadInst>(op)) {
        // FIXME: this part is buggy
        // need also check the def-use chain
        auto rawArg = cast<LoadInst>(op)->getPointerOperand()->stripPointerCasts();
        if (isa<Argument>(rawArg)) {
            auto argNo = cast<Argument>(rawArg)->getArgNo();
            if (valMap->count(argNo)) return valMap->at(argNo);
        }
    }
    return nullptr;
}

/**********************************************************

       Below are deprecated functions

***********************************************************/

bool findInst(const BasicBlock *BB, Instruction *target) {
    for (BasicBlock::const_iterator BI = BB->begin(), BE = BB->end(); BI != BE; ++BI) {
        const Instruction *inst = dyn_cast<Instruction>(BI);
        if (inst == target) {
            return true;
        }
    }
    return false;
}

// Fact 1: default case is always the last case
// Fact 2: default is optional
// Algorithm: the exit should post-dominate all cases
// FIXME: This function may be expensive, can try using cache
// FIXME: A corner case that this function is unsound:
// ```
// while (cond) {
//     switch(x) {
//     case 1:
//         // BB1
//     case 2:
//         // BB2
//         continue;
//     }
//     // BB.exit
// }
// ```
// The LLVM IR will put the conditional check for while loop at the end (after BB.exit),
// so the `continue` in the IR will be transformed as jump to the very end first,
// then jump back to the start of the while loop.
// Hence, the BB.exit no longer post-dominates BB2.
// This will result in a later BB being recognized as exit, and our SHB traversal
// will miss some BBs.
const BasicBlock *findSwitchExit(const SwitchInst *si) {
    auto defaultDest = si->getDefaultDest();
    const BasicBlock *firstDest = si->cases().begin()->getCaseSuccessor();
    Function &F = *const_cast<Function *>(si->getFunction());
    PostDominatorTree pdt(F);

    const BasicBlock *exit;
    exit = defaultDest;
    for (auto c : si->cases()) {
        // llvm::outs() << "find post-dom BB for: " << *c.getCaseSuccessor() << "\n";
        do {
            assert(exit != nullptr && "exit should not be a nullptr!");
            // llvm::outs() << "exit: " << *exit << "\n";
            // llvm::outs() << "dominates?: " << pdt.dominates(exit, c.getCaseSuccessor()) << "\n";
            if (pdt.dominates(exit, c.getCaseSuccessor())) {
                break;
            }
            exit = exit->getNextNode();
        } while (exit != nullptr);
    }
    // llvm::outs() << "exit: " << *exit << "\n";
    return exit;
}

// NOTE: the two functions below are too conservative (and wrong)
// they only skip the first BB of a branch
// we should delete them if `skipBranch` is stable enough
void PathCondition::selectTruePath(const BranchInst *br) { skipBBs.insert(br->getSuccessor(1)); }
void PathCondition::selectFalsePath(const BranchInst *br) { skipBBs.insert(br->getSuccessor(0)); }
