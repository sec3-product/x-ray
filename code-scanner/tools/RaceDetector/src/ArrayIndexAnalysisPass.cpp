//
// Created by peiming on 12/9/19.
//
#include <ArrayIndexAnalysisPass.h>
#include <aser/PointerAnalysis/Program/CallSite.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>

using namespace aser;
using namespace llvm;

/// Return true if any node in \p Root satisfies the predicate \p Pred.
template <typename PredTy>
const llvm::SCEV *FindSCEVExpr(const llvm::SCEV *Root, PredTy Pred) {
    struct FindClosure {
        const llvm::SCEV *Found = nullptr;
        PredTy Pred;

        FindClosure(PredTy Pred) : Pred(Pred) {}

        bool follow(const llvm::SCEV *S) {
            if (!Pred(S)) return true;

            Found = S;
            return false;
        }

        bool isDone() const { return Found != nullptr; }
    };

    FindClosure FC(Pred);
    visitAll(Root, FC);
    return FC.Found;
}

// move add operation out the (sext ) SCEV
class BitExtSCEVRewriter : public llvm::SCEVRewriteVisitor<BitExtSCEVRewriter> {
public:
    using super = SCEVRewriteVisitor<BitExtSCEVRewriter>;

    explicit BitExtSCEVRewriter(llvm::ScalarEvolution &SE) : super(SE) {}

    const llvm::SCEV *visit(const llvm::SCEV *S) {
        auto result = super::visit(S);
        while (result != S) {
            S = result;
            result = super::visit(S);
        }
        return result;
    }

    const llvm::SCEV *visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *Expr) {
        const llvm::SCEV *Operand = super::visit(Expr->getOperand());
        if (auto add = llvm::dyn_cast<llvm::SCEVNAryExpr>(Operand)) {
            llvm::SmallVector<const llvm::SCEV *, 2> Operands;
            for (auto op : add->operands()) {
                Operands.push_back(SE.getZeroExtendExpr(op, Expr->getType()));
            }
            switch (add->getSCEVType()) {
                case llvm::scMulExpr:
                    return SE.getMulExpr(Operands);
                case llvm::scAddExpr:
                    return SE.getAddExpr(Operands);
                case llvm::scAddRecExpr:
                    auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(add);
                    return SE.getAddRecExpr(Operands, addRec->getLoop(), addRec->getNoWrapFlags());
            }
        }

        return Operand == Expr->getOperand() ? Expr : SE.getZeroExtendExpr(Operand, Expr->getType());
    }

    const llvm::SCEV *visitSignExtendExpr(const llvm::SCEVSignExtendExpr *Expr) {
        const llvm::SCEV *Operand = Expr->getOperand();

        if (auto add = llvm::dyn_cast<llvm::SCEVNAryExpr>(Operand)) {
            llvm::SmallVector<const llvm::SCEV *, 2> Operands;
            for (auto op : add->operands()) {
                Operands.push_back(SE.getSignExtendExpr(op, Expr->getType()));
            }
            switch (add->getSCEVType()) {
                case llvm::scMulExpr:
                    return SE.getMulExpr(Operands);
                case llvm::scAddExpr:
                    return SE.getAddExpr(Operands);
                case llvm::scAddRecExpr:
                    auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(add);
                    return SE.getAddRecExpr(Operands, addRec->getLoop(), addRec->getNoWrapFlags());
            }
        }

        return Operand == Expr->getOperand() ? Expr : SE.getSignExtendExpr(Operand, Expr->getType());
    }
};

// move add operation out the (sext) SCEV
class SCEVSimplifier : public llvm::SCEVRewriteVisitor<SCEVSimplifier> {
private:
    using super = SCEVRewriteVisitor<SCEVSimplifier>;
    using ResolvedUnknowns = llvm::SmallDenseMap<const llvm::Value *, const llvm::ConstantInt *, 8>;

    const ResolvedUnknowns &resolvedUnknowns;

public:
    SCEVSimplifier(const ResolvedUnknowns &resloved, llvm::ScalarEvolution &SE)
        : resolvedUnknowns(resloved), super(SE) {}

    const llvm::SCEV *visitUnknown(const llvm::SCEVUnknown *Expr) {
        auto it = resolvedUnknowns.find(Expr->getValue());
        if (it != resolvedUnknowns.end()) {
            return SE.getConstant(const_cast<llvm::ConstantInt *>(it->getSecond()));
        }
        return Expr;
    }
};

// move add operation out the (sext) SCEV
class SCEVApplyBound : public llvm::SCEVRewriteVisitor<SCEVApplyBound> {
private:
    using super = SCEVRewriteVisitor<SCEVApplyBound>;
    using ResolvedUnknowns = llvm::SmallDenseMap<const llvm::Value *, const llvm::ConstantInt *, 8>;
    const ResolvedUnknowns &resolvedUnknowns;
    const llvm::Loop *ompLoop;

public:
    SCEVApplyBound(const llvm::Loop *ompLoop, const ResolvedUnknowns &resloved, llvm::ScalarEvolution &SE)
        : resolvedUnknowns(resloved), ompLoop(ompLoop), super(SE) {}

    const llvm::SCEV *visitAddRecExpr(const llvm::SCEVAddRecExpr *Expr) {
        if (Expr->getLoop() == ompLoop) {
            return Expr;
        }

        if (Expr->isAffine()) {
            auto op = visit(Expr->getOperand(0));
            auto step = Expr->getOperand(1);

            auto backEdgeCount = SE.getBackedgeTakenCount(Expr->getLoop());
            SCEVSimplifier simplifier(resolvedUnknowns, SE);
            auto simplifiedCount = simplifier.visit(backEdgeCount);

            if (llvm::isa<SCEVCouldNotCompute>(simplifiedCount) || llvm::isa<SCEVCouldNotCompute>(op) ||
                llvm::isa<SCEVCouldNotCompute>(step)) {
                return Expr;
            }

            auto bounded = SE.getAddExpr(op, SE.getMulExpr(simplifiedCount, step));

            return bounded;
        } else {
            return Expr;
        }
    }
};

void ArrayIndexAnalysisPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
}

bool ArrayIndexAnalysisPass::runOnFunction(llvm::Function &F) {
    this->SE = nullptr;
    this->LI = nullptr;
    this->ompStaticInitForBlocks.clear();
    this->ompDispatchInitForBlocks.clear();
    this->resolvedConstant.clear();

    for (auto &BB : F) {
        for (auto &I : BB) {
            if (isa<CallInst>(I)) {
                aser::CallSite CS(&I);
                if (CS.getCalledFunction() != nullptr && CS.getCalledFunction()->hasName()) {
                    auto funcName = CS.getCalledFunction()->getName();
                    if (funcName.contains("kmpc_for_static_init")) {
                        this->ompStaticInitForBlocks.insert(std::make_pair(&BB, dyn_cast<CallInst>(&I)));
                    } else if (funcName.contains("kmpc_dispatch_init")) {
                        this->ompDispatchInitForBlocks.insert(std::make_pair(&BB, dyn_cast<CallInst>(&I)));
                    }
                }
            }
        }
    }

    if (!ompStaticInitForBlocks.empty() || !ompDispatchInitForBlocks.empty()) {
        this->SE = &this->getAnalysis<ScalarEvolutionWrapperPass>().getSE();
        this->LI = &this->getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
        this->DT = &this->getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    }
    return false;
}

CallInst *ArrayIndexAnalysisPass::getOMPStaticInitCall(const Loop *L) {
    if (L->getLoopPreheader() == nullptr) {
        return nullptr;
    }

    auto forInitBlock = L->getLoopPreheader()->getUniquePredecessor();
    if (forInitBlock == nullptr) {
        return nullptr;
    }

    auto it = this->ompStaticInitForBlocks.find(forInitBlock);
    if (it == this->ompStaticInitForBlocks.end()) {
        if (auto predecessor = forInitBlock->getUniquePredecessor()) {
            // fortran will have another bb before static
            // init block to substract 1 as array is starting from 1 for fortran
            it = this->ompStaticInitForBlocks.find(predecessor);
            if (it == this->ompStaticInitForBlocks.end()) {
                return nullptr;
            }
        } else {
            return nullptr;
        }
    }

    return it->second;
}

CallInst *ArrayIndexAnalysisPass::getOMPDispatchInitCall(const Loop *L) {
    auto predBlock = L->getLoopPredecessor();
    if (predBlock == nullptr) {
        return nullptr;
    }
    auto it = this->ompDispatchInitForBlocks.find(predBlock);
    if (it == this->ompDispatchInitForBlocks.end()) {
        return nullptr;
    }

    return it->second;
}

bool ArrayIndexAnalysisPass::isOMPForLoop(const Loop *L) {
    if (getOMPStaticInitCall(L) != nullptr || getOMPDispatchInitCall(L) != nullptr) {
        return true;
    } else if (L->getParentLoop() != nullptr) {
        return getOMPDispatchInitCall(L->getParentLoop()) != nullptr;
    }
    return false;
}

bool ArrayIndexAnalysisPass::isInOMPForLoop(const llvm::Instruction *I) {
    Loop *L = LI->getLoopFor(I->getParent());
    if (!L) {
        // not even in a loop
        return false;
    }

    while (L->getParentLoop() != nullptr) {
        L = L->getParentLoop();
    }

    // spdlog::info("Instruction in Loop: {}",
    // L->getLoopPreheader()->getName().str()); NOTE: openmp for loop's
    // preheader always has a unique predecessor that calls
    // "__kmpc_for_static_init"
    // TODO: any other patterns?
    return isOMPForLoop(L);
}

const llvm::ConstantInt *ArrayIndexAnalysisPass::resolveSingleStoreInt(const PTA &pta, const ctx *context,
                                                                       const llvm::Value *v) {
    if (!v->getType()->isPointerTy()) {
        return nullptr;
    }

    std::vector<const typename PTA::ObjTy *> pts;
    pta.getPointsTo(context, v, pts);

    if (pts.size() != 1) {
        return nullptr;
    }

    auto obj = pts[0];
    const llvm::Value *allocSite = obj->getValue();

    const llvm::StoreInst *storeInst = nullptr;
    for (auto user : allocSite->users()) {
        if (auto SI = llvm::dyn_cast<llvm::StoreInst>(user)) {
            // simple cases, only has one store instruction
            if (storeInst == nullptr) {
                storeInst = SI;
            } else {
                return nullptr;
            }
        }
    }
    if (storeInst == nullptr) {
        return nullptr;
    }
    if (auto bound = llvm::dyn_cast<llvm::ConstantInt>(storeInst->getOperand(0))) {
        return bound;
    }

    return nullptr;
}

void ArrayIndexAnalysisPass::resolveConstantParameter(const PTA &pta, const ctx *context, const llvm::Argument *v) {
    if (auto constInt = resolveSingleStoreInt(pta, context, v)) {
        // this->resolvedConstant.insert(std::make_pair(v, constInt));
        if (v->hasAttribute(llvm::Attribute::AttrKind::ReadOnly)) {
            for (auto user : v->users()) {
                if (auto LI = llvm::dyn_cast<llvm::LoadInst>(user)) {
                    resolvedConstant.insert(std::make_pair(LI, constInt));
                }
            }
        }
    }
}

void ArrayIndexAnalysisPass::resolveConstantGlobal(const llvm::GlobalVariable *g) {
    if (g->hasInitializer()) {
        if (auto constant = llvm::dyn_cast<llvm::ConstantInt>(g->getInitializer())) {
            for (auto user : g->users()) {
                if (auto LI = llvm::dyn_cast<llvm::LoadInst>(user)) {
                    resolvedConstant.insert(std::make_pair(LI, constant));
                }
            }
        }
    }
}

uint64_t ArrayIndexAnalysisPass::resolveBound(const PTA &pta, const ctx *context, const llvm::Value *v,
                                              const llvm::Instruction *initForCall) {
    // omp.lb and omp.ub should always be alloca instruction?
    assert(!ompStaticInitForBlocks.empty() || !ompDispatchInitForBlocks.empty());
    auto allocaSite = llvm::dyn_cast<llvm::AllocaInst>(v);

    if (!allocaSite) {
        return INVALID_LOOP_BOUND;
    }

    const llvm::StoreInst *storeInst = nullptr;
    for (auto user : allocaSite->users()) {
        if (auto SI = llvm::dyn_cast<llvm::StoreInst>(user)) {
            // simple cases, only has one store instruction
            if (storeInst == nullptr) {
                if (this->DT->dominates(SI, initForCall)) {
                    storeInst = SI;
                }
            } else {
                if (this->DT->dominates(SI, initForCall)) {
                    // LOG_DEBUG("omp bound has one than one store!!"); // TODO: Peiming clarify
                    return INVALID_LOOP_BOUND;
                }
            }
        }
    }

    if (storeInst) {
        // get the stored value into the SCEV
        auto boundSCEV = this->SE->getSCEV(const_cast<llvm::Value *>(storeInst->getValueOperand()));

        SCEVSimplifier rewriter(this->resolvedConstant, *SE);
        auto result = rewriter.visit(boundSCEV);

        if (auto constBound = llvm::dyn_cast<llvm::SCEVConstant>(result)) {
            // FIXME: take care of the sign
            return constBound->getAPInt().getLimitedValue();
        }
        return INVALID_LOOP_BOUND;
    } else {
        // LOG_DEBUG("omp bound has no store??");
        return INVALID_LOOP_BOUND;
    }
}

const llvm::SCEVAddRecExpr *ArrayIndexAnalysisPass::getOMPLoopSCEV(const llvm::SCEV *root) {
    // get the outter-most loop (omp loop should always be the outter-most
    // loop
    auto omp = FindSCEVExpr(root, [this](const llvm::SCEV *S) -> bool {
        if (auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(S)) {
            if (isOMPForLoop(addRec->getLoop())) {
                return true;
            }
        }
        return false;
    });

    return llvm::dyn_cast_or_null<llvm::SCEVAddRecExpr>(omp);
}

bool ArrayIndexAnalysisPass::SCEVContainsUnknown(const llvm::SCEV *root) {
    return llvm::SCEVExprContains(root, [](const llvm::SCEV *S) -> bool { return llvm::isa<llvm::SCEVUnknown>(S); });
}
// PTA is required to resolve up and lower bound
int ArrayIndexAnalysisPass::canOmpIndexAlias(const ctx *context, const llvm::GetElementPtrInst *gep1,
                                                  const llvm::GetElementPtrInst *gep2, const PTA &pta) {
    // llvm::outs() << "checking omp index in func: " << gep1->getFunction()->getName() << "\n";
    // llvm::outs() << "gep1: " << *gep1 << "\n";
    // llvm::outs() << "gep2: " << *gep2 << "\n";
    if (ompStaticInitForBlocks.empty() && ompDispatchInitForBlocks.empty()) {
        return 2;
    }

    // should be in the same function
    if (gep1->getFunction() != gep2->getFunction()) {
        return 2;
    }

    assert(LI != nullptr && SE != nullptr);

    // not in OMP for loop
    if (!isInOMPForLoop(gep1) || !isInOMPForLoop(gep2)) {
        // can not determine
        return 2;
    }

    BitExtSCEVRewriter rewriter(*SE);
    // bit extension will block the constant evaluation
    auto scev1 = rewriter.visit(SE->getSCEV(const_cast<llvm::GetElementPtrInst *>(gep1)));
    auto scev2 = rewriter.visit(SE->getSCEV(const_cast<llvm::GetElementPtrInst *>(gep2)));
    // llvm::outs() << "scev1: " << *scev1 << "\n";
    // llvm::outs() << "scev2: " << *scev2 << "\n";

    // NOTE: if the program directly accessing elements using induction variables
    // then we can directly return false here (meaning there's no race)
    // because for two different i and j, the same computation on them won't results
    // in the same index value
    // However, if the elemnts in the array is accessed in an indirect way
    // for example:
    //  idx = idx_array[i]; // the real index in stored in an array
    //  x[idx] = ?;
    // then it is still possible to have races, even if their scev is the same
    // Therefore, we cannot directly return false.

    // FIXME: commenting this out will fail on DRB094
    // because the loop structure in llvm bitcode is not what we assume
    // if (scev1 == scev2) return false;

    if (resolvedConstant.empty()) {
        for (auto &arg : gep1->getFunction()->args()) {
            resolveConstantParameter(pta, context, &arg);
        }

        for (auto &global : gep1->getModule()->globals()) {
            resolveConstantGlobal(&global);
        }
    }

    auto omp1 = getOMPLoopSCEV(scev1);
    auto omp2 = getOMPLoopSCEV(scev2);

    // one is not in omp1 loop
    if (omp1 == nullptr || omp2 == nullptr) {
        return 2;
    }

    // not the same omp1 loop
    if (omp1->getLoop() != omp2->getLoop()) {
        return 2;
    }

    bool isFortranCode = true;
    if (isFortranCode) {
        auto diff = this->SE->getMinusSCEV(scev1, scev2);
        auto result = rewriter.visit(diff);

        if (auto gap = llvm::dyn_cast<llvm::SCEVConstant>(result)) {
            if (gap->isZero()) {
                return 0;
            }
        }
    } else {  // cpp code
        // llvm::outs() << "omp1: " << *omp1 << "\n";
        // llvm::outs() << "omp2: " << *omp2 << "\n";
        auto diff = this->SE->getMinusSCEV(omp1, omp2);
        auto result = rewriter.visit(diff);

        // TODO: dirty fix, we may have offset (%x + {i, +, 8}<loop>) with the loop, simply remove the offset
        scev1 = FindSCEVExpr(scev1, [this](const llvm::SCEV *S) -> bool {
            if (auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(S)) {
                return true;
            }
            return false;
        });

        scev2 = FindSCEVExpr(scev2, [this](const llvm::SCEV *S) -> bool {
            if (auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(S)) {
                return true;
            }
            return false;
        });

        if (auto gap = llvm::dyn_cast<llvm::SCEVConstant>(result)) {
            // The distance between two different thread can be computed
            if (omp1 == scev1 && omp2 == scev2) {
                // there is only one level loop, and the loop in the open mp
                // model
                if (gap->isZero()) {
                    return 0;
                }
                // the gap distance between two access
                uint64_t distance = gap->getAPInt().abs().getLimitedValue();
                auto addRecOmp = omp1;
                if (addRecOmp->isAffine()) {
                    auto step = addRecOmp->getOperand(1);
                    if (auto constStep = llvm::dyn_cast<llvm::SCEVConstant>(step)) {
                        // here try to resolve the upper bound and lower
                        // bound
                        uint64_t step = constStep->getAPInt().getLimitedValue();
                        // spdlog::info("distance: {}, stride: {}", distance, step);
                        // spdlog::info("races if iteration number >= {}", distance / step);

                        // assume we iterate at least one time
                        if (distance == step) {
                            return 1;
                        }

                        llvm::CallInst *initForCall = getOMPStaticInitCall(addRecOmp->getLoop());
                        uint64_t lowerBound, upperBound;

                        if (initForCall != nullptr) {
                            lowerBound = this->resolveBound(pta, context, initForCall->getArgOperand(4), initForCall);
                            upperBound = this->resolveBound(pta, context, initForCall->getArgOperand(5), initForCall);
                        } else {
                            initForCall = getOMPDispatchInitCall(addRecOmp->getLoop());
                            lowerBound = this->resolveBound(pta, context, initForCall->getArgOperand(3), initForCall);
                            upperBound = this->resolveBound(pta, context, initForCall->getArgOperand(4), initForCall);
                        }

                        if (upperBound < (distance / step)) {
                            return 0;
                        }
                    }
                }
            } else {
                // the openmp for loop has nested loops
                auto wholeGap = SE->getMinusSCEV(scev1, scev2);
                // 1st, try to evaluate a constant value if possible
                SCEVSimplifier simplifier(this->resolvedConstant, *SE);
                wholeGap = rewriter.visit(wholeGap);
                // llv,::outs() << *wholeGap << "\n";

                SCEVApplyBound boundApplier(omp1->getLoop(), resolvedConstant, *SE);
                auto b1 = boundApplier.visit(scev1);

                SCEVApplyBound boundApplier2(omp2->getLoop(), resolvedConstant, *SE);
                auto b2 = boundApplier2.visit(scev2);

                // TODO: dirty fix, we may have offset (%x + {i, +, 8}<loop>) with the loop, simply remove the offset
                b1 = FindSCEVExpr(b1, [this](const llvm::SCEV *S) -> bool {
                    if (auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(S)) {
                        return true;
                    }
                    return false;
                });

                b2 = FindSCEVExpr(b2, [this](const llvm::SCEV *S) -> bool {
                    if (auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(S)) {
                        return true;
                    }
                    return false;
                });

                // one is not in omp1 loop
                if (b1 == nullptr || b2 == nullptr) {
                    return 2;
                }

                auto r1 = SE->getMinusSCEV(b1, omp2);
                auto r2 = SE->getMinusSCEV(b2, omp1);

                if (auto c1 = llvm::dyn_cast<llvm::SCEVConstant>(r1)) {
                    if (auto c2 = llvm::dyn_cast<llvm::SCEVConstant>(r2)) {
                        if (auto constStep = llvm::dyn_cast<llvm::SCEVConstant>(omp1->getOperand(1))) {
                            uint64_t v1 = c1->getAPInt().abs().getLimitedValue();
                            uint64_t v2 = c2->getAPInt().abs().getLimitedValue();

                            if (v1 < constStep->getAPInt().abs().getLimitedValue() &&
                                v2 < constStep->getAPInt().abs().getLimitedValue()) {
                                return 0;
                            }
                        }
                    }
                }
            }
        }
    }
    return 1;
}

char ArrayIndexAnalysisPass::ID = 0;
static RegisterPass<ArrayIndexAnalysisPass> AIA("OpenMP Array Index Analysis", "OpenMp Array Index Analysis",
                                                true, /*CFG only*/
                                                true /*is analysis*/);
