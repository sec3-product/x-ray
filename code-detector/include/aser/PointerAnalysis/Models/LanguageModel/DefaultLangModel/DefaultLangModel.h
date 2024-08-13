//
// Created by peiming on 10/22/19.
//
#ifndef ASER_PTA_DEFAULTLANGMODEL_H
#define ASER_PTA_DEFAULTLANGMODEL_H

#include "DefaultExtFunctions.h"
#include "aser/PointerAnalysis/Models/DefaultHeapModel.h"
#include "aser/PointerAnalysis/Models/LanguageModel/LangModelBase.h"
#include "aser/PointerAnalysis/Models/MemoryModel/FieldInsensitive/FIMemModel.h"
#include "aser/PointerAnalysis/Program/CtxModule.h"
#include "aser/PointerAnalysis/Solver/PointsTo/BitVectorPTS.h"
#include "aser/Util/Log.h"

namespace aser {

// this class deals with conventions that specific to different programming
// language e.g., virtual pointers, the default one uses no convention
template <typename ctx, typename MemModel, typename PtsTy = BitVectorPTS>
class DefaultLangModel
    : public LangModelBase<ctx, MemModel, PtsTy,
                           DefaultLangModel<ctx, MemModel, PtsTy>> {
private:
  DefaultHeapModel heapModel;

protected:
  using Self = DefaultLangModel<ctx, MemModel, PtsTy>;
  using Super = LangModelBase<ctx, MemModel, PtsTy, Self>;

  using PT = typename Super::PT;
  using CT = typename Super::CT;
  using MMT = typename Super::MMT;

  using CallGraphTy = typename Super::CallGraphTy;
  using ConsGraphTy = typename Super::ConsGraphTy;

  using ObjNode = typename Super::ObjNode;
  using PtrNode = typename Super::PtrNode;
  using CGNodeTy = typename Super::CGNodeTy;

public:
  // determine whether the API is a heap allocation API
  inline bool isExtHeapAllocAPI(const llvm::Function *F,
                                const llvm::Instruction *callSite) {
    return heapModel.isHeapAllocFun(F);
  }

  bool isHeapAllocAPI(const llvm::Function *F,
                      const llvm::Instruction *callSite = nullptr) {
    return heapModel.isHeapAllocFun(F);
  }

  // determine whether the resolved indirect call is compatible
  inline bool isCompatible(const llvm::Instruction *callsite,
                           const llvm::Function *target) {
    auto call = llvm::cast<llvm::CallBase>(callsite);
    // only pthread will override to indirect call in default language model
    auto pthread = call->getCalledFunction();
    assert(pthread && pthread->getName().equals("pthread_create"));

    // pthread call back type -> i8* (*) (i8*)
    if (target->arg_size() != 1) {
      return false;
    }

    // pthread's callback's return type does not matter.
    return target->arg_begin()->getType() ==
           llvm::Type::getInt8PtrTy(callsite->getContext());
  }

  // modelling the heap allocation
  inline void interceptHeapAllocSite(const CtxFunction<ctx> *caller,
                                     const CtxFunction<ctx> *callee,
                                     const llvm::Instruction *allocSite) {
    llvm::Type *heapObjType = nullptr;
    if constexpr (MMT::NEED_TYPE_INFO) {
      heapObjType =
          heapModel.inferHeapAllocType(callee->getFunction(), allocSite);
      if (heapObjType != nullptr) {
        LOG_TRACE("Infer Heap Object type. obj={}, type={}", *allocSite,
                  *heapObjType);
      } else {
        LOG_TRACE("Infer Heap Object to be field-insensitive. obj={}",
                  *allocSite);
      }
    }

    ObjNode *obj =
        this->allocHeapObj(caller->getContext(), allocSite, heapObjType);
    PtrNode *ptr = this->getPtrNode(caller->getContext(), allocSite);

    this->consGraph->addConstraints(obj, ptr, Constraints::addr_of);
  }

  // determine whether the function need to be modelled
  inline InterceptResult interceptFunction(const ctx *callerCtx,
                                           const ctx *calleeCtx,
                                           const llvm::Function *F,
                                           const llvm::Instruction *callsite) {
    const llvm::Function *fun = F;
    if (F->isIntrinsic()) {
      return {nullptr, InterceptResult::Option::IGNORE};
    }
    if (DefaultExtFunctions::isThreadCreation(fun)) {
      // thread creation site is intercepted, and the body of the thread
      // need to be handled. dbg_os() << *callsite << "\n";
      aser::CallSite CS(callsite);
      assert(CS.isCallOrInvoke());
      const llvm::Value *v = CS.getArgOperand(2);
      if (auto threadFun =
              llvm::dyn_cast<llvm::Function>(v->stripPointerCasts())) {
        fun = threadFun; // replace call to pthread_create to the thread
                         // starting routine
      } else {
        // the callback for pthread_create is a indirect call!
        return {v, InterceptResult::Option::EXPAND_BODY};
      }
      // return InterceptResult::EXPAND_BODY;
    }

    // does not get handled by the language model, it is most conservative
    return {fun, InterceptResult::Option::EXPAND_BODY};
  }

  // modelling a callsite
  inline bool interceptCallSite(const CtxFunction<ctx> *caller,
                                const CtxFunction<ctx> *callee,
                                const llvm::Function *originalTarget,
                                const llvm::Instruction *callsite) {
    // the rule of context evolution should be obeyed.
    aser::CallSite CS(callsite);
    assert(CS.isCallOrInvoke());

    // we need to use callsite to identify the thread creation as the
    // ctxfunction might be intercepted before
    // TODO: what about the indirect call resolved to pthread_create?
    if (auto fun = CS.getCalledFunction()) {
      // if (auto fun = callee->getFunction()) {
      if (DefaultExtFunctions::isThreadCreation(fun)) {
        // TODO: HANDLE MORE APIs
        // dbg_os() << *callee->getFunction() << "\n";
        assert(callee->getFunction()->arg_size() == 1);

        // link the parameter
        PtrNode *formal = this->getPtrNode(
            callee->getContext(), &*callee->getFunction()->arg_begin());
        PtrNode *actual =
            this->getPtrNode(caller->getContext(), CS.getArgOperand(3));
        this->consGraph->addConstraints(actual, formal, Constraints::copy);
        return true;
      }
    }

    return false;
  }

protected:
  DefaultLangModel(llvm::Module *M, llvm::StringRef entry) : Super(M, entry) {}

  friend Super;
  friend LangModelTrait<Super>;
};

// Specialization of LangModelTrait<DefaultModel>
template <typename ctx, typename MemModel, typename PtsTy>
class LangModelTrait<DefaultLangModel<ctx, MemModel, PtsTy>>
    : public LangModelTrait<LangModelBase<
          ctx, MemModel, PtsTy, DefaultLangModel<ctx, MemModel, PtsTy>>> {};

} // namespace aser

#endif