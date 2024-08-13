//
// Created by peiming on 11/15/19.
//

#ifndef RACEDETECTOR_GRAPHBLASMODEL_H
#define RACEDETECTOR_GRAPHBLASMODEL_H

#include "GraphBLASHeapModel.h"
#include "aser/PointerAnalysis/Context/HybridCtx.h"
#include "aser/PointerAnalysis/Context/KCallSite.h"
#include "aser/PointerAnalysis/Context/KOrigin.h"
#include "aser/PointerAnalysis/Context/NoCtx.h"
#include "aser/PointerAnalysis/Models/LanguageModel/LangModelBase.h"
#include "aser/PointerAnalysis/Models/MemoryModel/CppMemModel/CppMemModel.h"
#include "aser/PointerAnalysis/Program/CtxModule.h"
#include "aser/PointerAnalysis/Solver/AndersenWave.h"
#include "aser/PointerAnalysis/Solver/PartialUpdateSolver.h"
#include "aser/PointerAnalysis/Solver/PointsTo/BitVectorPTS.h"

namespace aser {

// comment this out for hybrid ctx, still experimental feature
// using ctx = HybridCtx<KOrigin<3>, KCallSite<3>>;
using ctx = KOrigin<3>;
using MemModel = cpp::CppMemModel<ctx>;
using PtsTy = BitVectorPTS;

// TODO: vector could also use the partial demangler
// this class deals with conventions that specific to different programming
// language e.g., virtual pointers, the default one uses no convention
class GraphBLASModel : public LangModelBase<ctx, MemModel, PtsTy, GraphBLASModel> {
public:
    using PT = PTSTrait<PtsTy>;
    using CT = CtxTrait<ctx>;
    using MMT = MemModelTrait<MemModel>;

    using CallGraphTy = typename Super::CallGraphTy;
    using ConsGraphTy = typename Super::ConsGraphTy;
    using CallNodeTy = typename Super::CallNodeTy;

    using ObjNode = CGObjNode<ctx, typename MMT::ObjectTy>;
    using PtrNode = CGPtrNode<ctx>;

private:
    using Self = GraphBLASModel;
    using Super = LangModelBase<ctx, MemModel, PtsTy, Self>;

    GraphBLASHeapModel heapModel;

    std::map<StringRef, ObjNode *> mapObjects;
    std::map<std::pair<StringRef, const llvm::Value *>, ObjNode *> mapParentObjects;

public:
    GraphBLASModel(llvm::Module *M, llvm::StringRef entry) : Super(M, entry) {
        ctx::setOriginRules([&](const ctx *context, const llvm::Instruction *I) -> bool {
            return false;
        });
    }

    // callbacks called by ConsGraphBuilder
    // whether the resolved indirect call is compatible with API
    bool isCompatible(const llvm::Instruction *callsite, const llvm::Function *target);

    // callsite is supplied optional, in case in the future we might need it
    bool isHeapAllocAPI(const llvm::Function *F, const llvm::Instruction *callsite = nullptr) {
        return heapModel.isHeapAllocFun(F);
    }

    // modelling the heap allocation
    void interceptHeapAllocSite(const CtxFunction<ctx> *caller, const CtxFunction<ctx> *callee,
                                const llvm::Instruction *callsite);

    // return true if the function need to be expanded in callgraph.
    InterceptResult interceptFunction(const ctx *callerCtx, const ctx *calleeCtx, const llvm::Function *F,
                                      const llvm::Instruction *callsite);

    IndirectResolveOption onNewIndirectTargetResolvation(const llvm::Function *F, const llvm::Instruction *callsite);

    bool interceptCallSite(const CtxFunction<ctx> *caller, const CtxFunction<ctx> *callee,
                           const llvm::Function *originalTarget, const llvm::Instruction *callsite);

    static const llvm::Function *findSignalHandlerFunc(const llvm::Instruction *inst);
    static const llvm::Function *findThreadStartFunc(PartialUpdateSolver<GraphBLASModel> *pta, const ctx *context,
                                                     const llvm::Instruction *inst);
    static const llvm::StringRef findGlobalString(const llvm::Value *value);
    static bool isPreviledgeAccount(llvm::StringRef accountName);
    static bool isAuthorityAccount(llvm::StringRef accountName);
    static bool isUserProvidedAccount(llvm::StringRef accountName);

    static bool isRegisterSignal(const llvm::Instruction *inst);
    static bool isRegisterSignalAction(const llvm::Instruction *inst);
    static bool isRead(const llvm::Instruction *inst);
    static bool isWrite(const llvm::Instruction *inst);
    static bool isReadORWrite(const llvm::Instruction *inst);

    static bool isRustModelAPI(const llvm::Function *func);
    static bool isRustAPI(const llvm::Function *func);
    static bool isRustNormalCall(const llvm::Instruction *inst);
    static bool isRustNormalCall(const llvm::Function *func);

    friend Super;
    friend LangModelTrait<Super>;
};

// Specialization of LangModelTrait<DefaultModel>
template <>
struct LangModelTrait<GraphBLASModel> : public LangModelTrait<LangModelBase<ctx, MemModel, PtsTy, GraphBLASModel>> {};

using LangModel = GraphBLASModel;
using PTA = PartialUpdateSolver<LangModel>;
using LMT = LangModelTrait<LangModel>;
using CallGraphNodeTy = GraphBLASModel::CallNodeTy;
using ObjTy = PTA::ObjTy;
using CT = CtxTrait<ctx>;
using GT = llvm::GraphTraits<const GraphBLASModel::CallGraphTy>;
using CallGraphTy = GraphBLASModel::CallGraphTy;

}  // namespace aser

#endif
