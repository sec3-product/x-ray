//
// Created by peiming on 1/8/20.
//


#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/CallSite.h>
#include <llvm/Pass.h>
#include <llvm/IR/Module.h>

#include "aser/PointerAnalysis/PointerAnalysisPass.h"
#include "aser/PointerAnalysis/Context/NoCtx.h"

#ifndef ASER_PTA_PTAVERIFICATIONPASS_H
#define ASER_PTA_PTAVERIFICATIONPASS_H

// check the correctness of the PTA by checking when it is true for
// __aser_alias__ and __aser_no_alias__
#define CHECK_NO_ALIAS_FUN "__aser_no_alias__"
#define CHECK_ALIAS_FUN "__aser_alias__"

#define KNRM "\x1B[1;0m"
#define KRED "\x1B[1;31m"
#define KGRN "\x1B[1;32m"
#define KYEL "\x1B[1;33m"
#define KBLU "\x1B[1;34m"
#define KPUR "\x1B[1;35m"
#define KCYA "\x1B[1;36m"
#define KWHT "\x1B[1;37m"

namespace aser {

class AserMarkerCallSite : public llvm::CallSite {
private:
    inline bool isFunNameEqualsTo(llvm::StringRef funName) const {
        if (this->isCall()) {
            if (auto fun = llvm::dyn_cast<llvm::Function>(this->getCalledValue())) {
                return fun->getName().contains(funName);
            }
        }
        return false;
    }

public:
    explicit AserMarkerCallSite(llvm::Instruction *II)
        : CallSite(II) {}

    [[nodiscard]] inline bool isNoAliasCheck() const {
        return isFunNameEqualsTo(CHECK_NO_ALIAS_FUN);
    }

    [[nodiscard]] inline bool isAliasCheck() const {
        return isFunNameEqualsTo(CHECK_ALIAS_FUN);
    }

};

template <typename Solver>
class PTAVerificationPass : public llvm::ModulePass {

public:
    using ctx = NoCtx;

    static char ID;
    PTAVerificationPass() : llvm::ModulePass(ID) {
        //static_assert(std::is_same<typename Solver::ctx, NoCtx>::value && "Only support context insensitive");
    }

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
        AU.addRequired<PointerAnalysisPass<Solver>>();
        AU.setPreservesAll();  // does not transform the LLVM module
    }

    bool runOnModule(llvm::Module &M) override {
        this->getAnalysis<PointerAnalysisPass<Solver>>().analyze(&M, "_Z5entry1BPS_");

        auto &pta = *(this->getAnalysis<PointerAnalysisPass<Solver>>().getPTA());

//        for (auto &F : M) {
//            for (auto &BB : F) {
//                for (auto &I : BB) {
//                    AserMarkerCallSite CS(&I);
//                    if (CS.isNoAliasCheck()) {
//                        auto ptr1 = CS.getArgOperand(0);
//                        auto ptr2 = CS.getArgOperand(1);
//
//                        if (!pta.alias(nullptr, ptr1, nullptr, ptr2)) {
//                            llvm::outs() << KGRN "no alias check succeed!" KNRM "\n";
//                        } else {
//                            llvm::outs() << KRED "no alias check fail!" KNRM "\n";
//                            llvm::outs() << *ptr1->stripPointerCasts() << "\n";
//                            llvm::outs() << *ptr2->stripPointerCasts() << "\n";
//                        }
//                    } else if (CS.isAliasCheck()) {
//                        auto ptr1 = CS.getArgOperand(0);
//                        auto ptr2 = CS.getArgOperand(1);
//
//                        if (pta.alias(nullptr, ptr1, nullptr, ptr2)) {
//                            llvm::outs() << KGRN "alias check succeed!" KNRM "\n";
//                        } else {
//                            llvm::outs() << KRED "alias check fail!" KNRM "\n";
//                            llvm::outs() << *ptr1->stripPointerCasts() << "\n";
//                            llvm::outs() << *ptr2->stripPointerCasts() << "\n";
//                        }
//                    }
//                }
//            }
//        }
//
//        size_t ptrNum = 0;
//        double ptsSize = 0;
//        for (auto &F : M) {
//            for (auto &BB : F) {
//                for (auto &I : BB) {
//                    if (I.getType()->isPointerTy()) {
//                        std::vector<const typename Solver::ObjTy *> result;
//                        pta.getPointsTo(nullptr, &I, result);
//                        if (result.size() == 0) {
//                            //SPDLOG_WARN("empty PTS for {} \n\t @@ {}", I, llvm::demangle(F.getName()));
//                            continue;
//                        } else if (result.size() >= 3) {
//                            //SPDLOG_WARN("BIG PTS ({}) for {} \n\t @@ {}", result.size(), I, llvm::demangle(F.getName()));
//                        }
//                        ptrNum++;
//                        ptsSize += result.size();
//                    }
//                }
//            }
//        }

        //SPDLOG_INFO("PTS size: {}", ptsSize / ptrNum);
        return false;
    }
};

}

template <typename PTA>
char aser::PTAVerificationPass<PTA>::ID = 0;

template <typename PTA>
static llvm::RegisterPass<aser::PTAVerificationPass<PTA>>
    PVP("", "", true, true);

#endif  // ASER_PTA_PTAVERIFICATIONPASS_H
