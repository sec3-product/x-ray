//
// Created by peiming on 11/15/19.
//

#ifndef RACEDETECTOR_GRAPHBLASMODEL_H
#define RACEDETECTOR_GRAPHBLASMODEL_H

#include <llvm/Demangle/Demangle.h>

#include "GraphBLASHeapModel.h"
#include "aser/PointerAnalysis/Context/HybridCtx.h"
#include "aser/PointerAnalysis/Context/KCallSite.h"
#include "aser/PointerAnalysis/Context/KOrigin.h"
#include "aser/PointerAnalysis/Context/NoCtx.h"
#include "aser/PointerAnalysis/Models/LanguageModel/LangModelBase.h"
//#include "aser/PointerAnalysis/Models/MemoryModel/FieldInsensitive/FIMemModel.h"
#include "aser/PointerAnalysis/Models/MemoryModel/CppMemModel/CppMemModel.h"
#include "aser/PointerAnalysis/Models/MemoryModel/FieldSensitive/FSMemModel.h"
#include "aser/PointerAnalysis/Program/CtxModule.h"
#include "aser/PointerAnalysis/Solver/AndersenWave.h"
#include "aser/PointerAnalysis/Solver/PartialUpdateSolver.h"
#include "aser/PointerAnalysis/Solver/PointsTo/BitVectorPTS.h"

extern bool CONFIG_CTX_INSENSITIVE_PTA;
extern bool isInCheckingMissedOmp;

namespace aser {

// comment this out for hybrid ctx, still experimental feature
// using ctx = HybridCtx<KOrigin<3>, KCallSite<3>>;
using ctx = KOrigin<3>;
using MemModel = cpp::CppMemModel<ctx>;
using PtsTy = BitVectorPTS;

static bool containsAny(llvm::StringRef src, const std::set<llvm::StringRef> &list) {
    for (auto e : list) {
        if (src.contains(e)) {
            return true;
        }
    }
    return false;
}

// TODO: vector could also use the partial demangler
class StdVectorFunctions {
private:
    static const std::set<llvm::StringRef> VECTOR_READS;
    static const std::set<llvm::StringRef> VECTOR_WRITES;

public:
    static bool isVectorRead(const llvm::StringRef name) {
        if (isVectorAccess(name)) {
            return containsAny(name, VECTOR_READS);
        }
        return false;
    }

    static bool isVectorRead(const llvm::Function *func) { return isVectorRead(llvm::demangle(func->getName().str())); }

    static bool isVectorWrite(const llvm::StringRef name) {
        if (isVectorAccess(name)) {
            return containsAny(name, VECTOR_WRITES);
        }
        return false;
    }

    static bool isVectorWrite(const llvm::Function *func) {
        return isVectorWrite(llvm::demangle(func->getName().str()));
    }

    static bool isVectorAccess(const llvm::StringRef name) {
        return name.startswith("std::vector") || name.startswith("std::set") || name.startswith("std::map");
    }

    static bool isVectorAccess(const llvm::Function *func) {
        return isVectorAccess(llvm::demangle(func->getName().str()));
    }
};

// TODO: right now we only handle copy constructor for string
// we may need to handle other constructors
class StdStringFunctions {
private:
    static const llvm::StringRef STRING_CTOR;
    static const std::set<llvm::StringRef> STRING_READS;
    static const std::set<llvm::StringRef> STRING_WRITES;

public:
    static bool isStringRead(const llvm::StringRef name) {
        llvm::ItaniumPartialDemangler demangler;
        if (!demangler.partialDemangle(name.begin())) {
            if (isStringAccess(demangler.getFunctionName(nullptr, nullptr))) {
                return containsAny(demangler.getFunctionBaseName(nullptr, nullptr), STRING_READS);
            }
        }
        return false;
    }

    static bool isStringRead(const llvm::Function *func) { return isStringRead(func->getName()); }

    static bool isStringWrite(const llvm::StringRef name) {
        llvm::ItaniumPartialDemangler demangler;
        if (!demangler.partialDemangle(name.begin())) {
            if (isStringAccess(demangler.getFunctionName(nullptr, nullptr))) {
                return containsAny(demangler.getFunctionBaseName(nullptr, nullptr), STRING_WRITES);
            }
        }
        return false;
    }

    static bool isStringWrite(const llvm::Function *func) { return isStringWrite(func->getName()); }

    static bool isStringAccess(const llvm::StringRef name) {
        return name.startswith("std::__cxx11::basic_string") || name.startswith("std::basic_string");
    }

    static bool isStringAccess(const llvm::Function *func) {
        return isStringAccess(llvm::demangle(func->getName().str()));
    }

    // NOTE: string copy is tricky
    // std::string str1 = str2
    // it actually calls the basic_string constructor
    // std::basic_string(&str1, &str2) (template parameter is omitted here)
    static bool isStringCopy(const llvm::StringRef name) {
        llvm::ItaniumPartialDemangler demangler;
        demangler.partialDemangle(name.begin());
        if (isStringAccess(demangler.getFunctionName(nullptr, nullptr))) {
            auto base = demangler.getFunctionBaseName(nullptr, nullptr);
            if (base == STRING_CTOR) {
                // FIXME: this implementation is limited
                // it only handles std::string copy
                std::string param = demangler.getFunctionParameters(nullptr, nullptr);
                // the one below is the constructor for string literals
                // return param == "(char const*, std::allocator<char> const&)";
                return param ==
                       "(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)";
            }
        }
        return false;
    }

    static bool isStringCopy(const llvm::Function *func) { return isStringCopy(func->getName()); }
};

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

    GraphBLASModel(llvm::Module *M, llvm::StringRef entry) : Super(M, entry) {
        ctx::setOriginRules([&](const ctx *context, const llvm::Instruction *I) -> bool {
            if (CONFIG_CTX_INSENSITIVE_PTA) return false;
            return this->isInvokingAnOrigin(context, I);
        });
    }
    static const llvm::Function *findSignalHandlerFunc(const llvm::Instruction *inst);
    static const llvm::Function *findThreadStartFunc(PartialUpdateSolver<GraphBLASModel> *pta, const ctx *context,
                                                     const llvm::Instruction *inst);
    static bool isLibEventSetCallBack(const llvm::Instruction *inst);
    static bool isLibEventSetCallBack(const Function *F);
    static bool isLibEventDispath(const llvm::Instruction *inst);
    static bool isLibEventDispath(const Function *F);
    static const llvm::StringRef findGlobalString(const llvm::Value *value);
    static bool isPreviledgeAccount(llvm::StringRef accountName);
    static bool isAuthorityAccount(llvm::StringRef accountName);
    static bool isUserProvidedAccount(llvm::StringRef accountName);

    static bool isStdThreadCreate(const llvm::Instruction *inst);
    static bool isStdThreadJoin(const llvm::Instruction *inst);
    static bool isStdThreadAssign(const llvm::Instruction *inst);
    static bool isThreadCreate(const llvm::Instruction *inst);
    static bool isThreadCreate(const Function *F);
    static bool isThreadJoin(const llvm::Instruction *inst);
    static bool isRegisterSignal(const llvm::Instruction *inst);
    static bool isRegisterSignalAction(const llvm::Instruction *inst);
    static bool isMutexTryLock(const llvm::Instruction *inst);
    static bool isMutexLock(const llvm::Instruction *inst);
    static bool isMutexUnLock(const llvm::Instruction *inst);
    static bool isRead(const llvm::Instruction *inst);
    static bool isWrite(const llvm::Instruction *inst);
    static bool isReadORWrite(const llvm::Instruction *inst);
    static bool isCondWait(const llvm::Instruction *inst);
    static bool isCondSignal(const llvm::Instruction *inst);
    static bool isCondBroadcast(const llvm::Instruction *inst);

    static bool isSemaphoreWait(const llvm::Instruction *inst);
    static bool isSemaphorePost(const llvm::Instruction *inst);
    // NOTE: unify read/write API for container classes such as string/vector/set/...
    static bool isReadAPI(const llvm::Function *func);
    static bool isWriteAPI(const llvm::Function *func);
    static bool isMemoryFree(const llvm::Function *func);
    static bool isMemoryFree(const llvm::Instruction *inst);
    static bool isCriticalAPI(const llvm::Function *func);
    static bool isCriticalCallStack(std::vector<const llvm::Function *> &callStack);
    static bool isReadWriteAPI(const llvm::Function *func);
    static bool isRWLock(const llvm::Instruction *inst);
    static bool isRWUnLock(const llvm::Instruction *inst);
    static bool isRDLock(const llvm::Instruction *inst);
    static void addLockCallStackFunction(std::vector<const Function *> &callStack);
    static bool isSyncCall(const llvm::Instruction *inst);
    static bool isSyncCall(const llvm::Function *func);

    static bool isRustModelAPI(const llvm::Function *func);
    static bool isRustAPI(const llvm::Function *func);
    static bool isRustNormalCall(const llvm::Instruction *inst);
    static bool isRustNormalCall(const llvm::Function *func);

    bool isInvokingAnOrigin(const ctx *prevCtx, const llvm::Instruction *I);

    friend Super;
    friend LangModelTrait<Super>;
};

// Specialization of LangModelTrait<DefaultModel>
template <>
struct LangModelTrait<GraphBLASModel> : public LangModelTrait<LangModelBase<ctx, MemModel, PtsTy, GraphBLASModel>> {};

//    using Super = LangModelTrait<LangModelBase<ctx, MemModel, PtsTy, GraphBLASModel>>;
//
//    static inline void addDependentPasses(llvm::legacy::PassManager &passes) {
//        // Super::addDependentPasses(passes);
//        // passes.add(new InsertKMPCPass());
//    }
//
//    // set analysis usage
//    static void getAnalysisUsage(llvm::AnalysisUsage &AU) {
//        AU.addRequired<InsertKMPCPass>();
//        Super::getAnalysisUsage(AU);
//    }
//};

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
