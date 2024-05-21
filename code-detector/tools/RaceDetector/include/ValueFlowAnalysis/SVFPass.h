//
// Created by peiming on 3/12/20.
//

// a sparse value flow analysis pass
#ifndef ASER_PTA_SVFPASS_H
#define ASER_PTA_SVFPASS_H

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/Pass.h>

#include <map>

#include "PTAModels/GraphBLASModel.h"
#include "RDUtil.h"
#include "aser/PointerAnalysis/PointerAnalysisPass.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"

using namespace std;
using namespace llvm;

namespace aser {
namespace SVFModel {
void simplifySCEVExpression(string &expr);
}
class SCEVDataItem {
private:
    const llvm::SCEV *scev;  // the scev of v

    using ValueSCEVDataMap = DenseMap<const Value *, SCEVDataItem *>;
    ValueSCEVDataMap unknownMap;

public:
    const llvm::Function *f;  // the current func f
    llvm::Value *v;           // the value v within f
    SCEVDataItem(const llvm::SCEV *scev, const llvm::Function *f, llvm::Value *v) : scev(scev), f(f), v(v) {}
    void addUnknownValue(const Value *v, SCEVDataItem *data) {
        // when data is ***COULDNOTCOMPUTE***, v is an argument
        unknownMap[v] = data;
    }
    const llvm::SCEV *getSCEV() { return scev; }
    ValueSCEVDataMap &getUnknownMap() { return unknownMap; }

    // void updateSCEV(const llvm::SCEV *newValue) { scev = newValue; }
    bool hasUnknown(llvm::Value *un) {
        if (unknownMap.find(un) != unknownMap.end())
            return true;
        else {
            for (auto [v, data] : unknownMap) {
                if (data->hasUnknown(un)) return true;
            }
        }
        return false;
    }
    void dump();
};

class SVFPass : public llvm::ModulePass {  // TODO: if we want it in OnDemand way, An immutable pass works better
private:
    map<const llvm::Function *, llvm::ScalarEvolution *> seMap;
    // map<const llvm::Function *, std::set<const llvm::SCEV *>> retScevMap;
    // context sensitive using instruction as the context
    DenseMap<const Instruction *, DenseMap<const Value *, SCEVDataItem *>> callArgScevMap;
    DenseMap<const Value *, SCEVDataItem *> callArgOmpForkScevMap;

    DenseMap<const Instruction *, DenseMap<const Value *, SCEVDataItem *>> unresolvedUnknowns;
    DenseMap<const Value *, SCEVDataItem *> callRetScevMap;

    // for maintaining lower and upper bounds of loop induction
    DenseMap<const Function *, std::pair<const Value *, const Value *>> boundsFunctionMap;
    DenseMap<const Function *, std::pair<SCEVDataItem *, SCEVDataItem *>> boundsFunctionScevDataMap;
    DenseMap<const Function *, const Instruction *> boundsFunctionStaticInitInstMap;

    // cache from value to scev
    map<const Value *, SCEVDataItem *> scevCache;
    set<const llvm::Instruction *> callInstCache;
    map<const Value *, std::string> scevStringResultCache;
    map<const Value *, bool> syntacticalCache;

    PTA *pta;
    void ConstructSVFG(llvm::Module &M) {}
    SCEVDataItem *getGlobalSCEVInternal(const ctx *context, const llvm::Instruction *caller_inst,
                                        const llvm::Function *F, const llvm::Value *v);

public:
    static char ID;
    SVFPass() : llvm::ModulePass(ID) {}

    llvm::ScalarEvolution *getOrCreateFunctionSCEV(const llvm::Function *);
    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
        AU.addRequired<PointerAnalysisPass<PTA>>();      // rely on pointer analysis
        AU.addRequired<ScalarEvolutionWrapperPass>();    // rely on scalar evolution
        AU.addRequired<TargetLibraryInfoWrapperPass>();  // rely on scalar evolution
        AU.addRequired<AssumptionCacheTracker>();        // rely on scalar evolution
        AU.addRequired<DominatorTreeWrapperPass>();      // rely on scalar evolution
        AU.addRequired<LoopInfoWrapperPass>();           // rely on scalar evolution

        AU.setPreservesAll();  // does not transform the LLVM module
    }

    bool runOnModule(llvm::Module &M) override {
        // llvm::outs() << "hello from SVF: " << M.getName() << "\n";

        getAnalysis<PointerAnalysisPass<PTA>>().analyze(&M);

        this->pta = getAnalysis<PointerAnalysisPass<PTA>>().getPTA();

        // auto se = &this->getAnalysis<ScalarEvolutionWrapperPass>().getSE();  // scalar evolution
        // seMap[&F] = se;

        // actually logic
        // construct the svfg first
        ConstructSVFG(M);
        return false;
    }

    bool hasSyntacticalSignal(const llvm::Function *F, const llvm::Value *v);
    int mayAlias(const llvm::Function *ompEntry, const ctx *ctx1, const llvm::Instruction *f1_caller,
                      const llvm::Instruction *inst1, const Value *v1, const ctx *ctx2, const llvm::Instruction *f2_caller,
                      const llvm::Instruction *inst2, const Value *v2);

    void connectSCEVOMPEntryFunctionArgs(const ctx *context, const llvm::Instruction *const ompForkCall,
                                         const llvm::Function *ompEntryFunc);

    void connectSCEVFunctionArgs(const ctx *context, const llvm::Instruction *caller_inst,
                                 const llvm::Function *F_caller, const llvm::Instruction *call);
    // APIs for client application
    // probably need a context
    const llvm::SCEV *getGlobalSCEV(const ctx *context, const llvm::Instruction *F_caller, const llvm::Function *F,
                                    const llvm::Value *v);
    void printSCEVItem(const llvm::Function *F, DenseMap<const Value *, SCEVDataItem *> &unknownMap);

    // combine two scev
    llvm::SCEV *combineSCEVs(llvm::SCEV *, llvm::SCEV *);

    // can a llvm::Value (source) flow to another llvm::Value (sink)
    bool canFlowTo(llvm::Value *, llvm::Value *) {
        // Need done by Yanze
        return true;
    }

    bool flowsFromAny(llvm::Instruction *inst, std::vector<std::string> keys);
};

// static llvm::RegisterPass<SVFPass> AN("SVFPass", "Sparse Value Flow Analysis", true /*CFG only*/, true /*is
// analysis*/);
}  // namespace aser

#endif  // ASER_PTA_SVFPASS_H
