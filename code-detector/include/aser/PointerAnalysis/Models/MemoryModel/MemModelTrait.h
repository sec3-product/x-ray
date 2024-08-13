//
// Created by peiming on 10/22/19.
//
#ifndef ASER_PTA_MEMMODELTRAIT_H
#define ASER_PTA_MEMMODELTRAIT_H

#include <llvm/IR/IntrinsicInst.h>

#include "aser/PointerAnalysis/Program/InterceptResult.h"

namespace llvm::legacy {
class PassManager;
}
namespace aser {
template <typename ctx> class CtxFunction;

// this class handles static object modelling,
// e.g., field sensitive
template <typename MemModel> struct MemModelTrait {
  // context type
  using CtxTy = typename MemModel::UnknownTypeError;
  // object type
  using ObjectTy = typename MemModel::UnknownTypeError;
  // canonializer

  using Canonicalizer = typename MemModel::UnknownTypeError;
  /*
  //  ** required fields begin
  // whether *all* GEPs will be collapse
  static const bool COLLAPSE_GEP = false;
  // whether all BitCast will be collapse
  static const bool COLLAPSE_BITCAST = false;
  // whether type information is necessary, we need type information to build
  the memory layout static const bool NEED_TYPE_INFO = false;
  // ** required fields end
  */

  /* ****required methods begin
  //
  template <typename PT>
  inline static CGObjNode<MemModel> *allocateNullObj(MemModel &model, const
  llvm::Module *module) { return model.template allocNullObj<PT>(module);
  }

  template <typename PT>
  inline static CGObjNode<MemModel> *allocateUniObj(MemModel &model, const
  llvm::Module *module) { return model.template allocUniObj<PT>(module);
  }

  template <typename PT>
  inline static CGObjNode<MemModel> *allocateFunction(MemModel &model, const
  llvm::Function *fun) { return model.template allocFunction<PT>(fun);
  }

  template <typename PT>
  inline static CGObjNode<MemModel> *allocateGlobalVariable(MemModel &model,
                                                                   const
  llvm::GlobalVariable *gVar, const llvm::DataLayout &DL) { return
  model.template allocGlobalVariable<PT>(gVar, DL);
  }

  template <typename PT>
  inline static CGObjNode<MemModel> *allocateStackObj(MemModel &model, const
  CtxTy *context, const llvm::AllocaInst *gVar, const llvm::DataLayout &DL) {
      return model.template allocStackObj<PT>(context, gVar, DL);
  }

  template <typename PT>
  inline static CGObjNode<MemModel> *allocateHeapObj(MemModel &model, const
  CtxTy *context, const llvm::Instruction *callsite, const llvm::DataLayout &DL,
  llvm::Type *T) { return model.template allocHeapObj<PT>(context, callsite, DL,
  T);
  }

  template <typename PT>
  inline static CGObjNode<MemModel> *allocateAnonObj(MemModel &model, const
  CtxTy *context, const llvm::DataLayout &DL, llvm::Type *T, const llvm::Value
  *tag = nullptr, bool recursive = true) { return model.template
  allocAnonObj<PT>(context, DL, T, tag, recursive);
  }

  template <typename PT>
  inline static CGObjNode<MemModel> *indexObject(MemModel &model, const ObjectTy
  *obj, const llvm::GetElementPtrInst *gep) { return model.template
  indexObject<PT>(obj, gep);
  }

  template <typename PT>
  inline static void handleMemCpy(MemModel &model, const CtxTy *C, const
  llvm::MemCpyInst *memCpy, CGPtrNode<CtxTy> *src, CGPtrNode<CtxTy> *dst) {
      return model.template handleMemCpy<PT>(C, memCpy, src, dst);
  }

  template <typename PT>
  inline static void initializeGlobal(MemModel &memModel, const
  llvm::GlobalVariable *gVar, const llvm::DataLayout &DL) { memModel.template
  initializeGlobal<PT>(gVar, DL);
  }
  **** required method end */
};

// a helper class to implement a MemModelTrait
// basically delegate the function to the underlying memory model implementation
template <typename MemModel> struct MemModelHelper {
  // context type
  using CtxTy = typename MemModel::CtxTy;
  // object type
  using ObjectTy = typename MemModel::ObjectTy;

  using Canonicalizer = typename MemModel::Canonicalizer;

  inline static void
  addPreProcessingPass(llvm::legacy::PassManagerBase &passes) {
    return MemModel::addPreProcessingPass(passes);
  }

  template <typename PT>
  inline static CGObjNode<CtxTy, ObjectTy> *
  allocateNullObj(MemModel &model, const llvm::Module *module) {
    return model.template allocNullObj<PT>(module);
  }

  template <typename PT>
  inline static CGObjNode<CtxTy, ObjectTy> *
  allocateUniObj(MemModel &model, const llvm::Module *module) {
    return model.template allocUniObj<PT>(module);
  }

  template <typename PT>
  inline static CGObjNode<CtxTy, ObjectTy> *
  allocateFunction(MemModel &model, const llvm::Function *fun) {
    return model.template allocFunction<PT>(fun);
  }

  template <typename PT>
  inline static CGObjNode<CtxTy, ObjectTy> *
  allocateGlobalVariable(MemModel &model, const llvm::GlobalVariable *gVar,
                         const llvm::DataLayout &DL) {
    return model.template allocGlobalVariable<PT>(gVar, DL);
  }

  template <typename PT>
  inline static CGObjNode<CtxTy, ObjectTy> *
  allocateStackObj(MemModel &model, const CtxTy *context,
                   const llvm::AllocaInst *gVar, const llvm::DataLayout &DL) {
    return model.template allocStackObj<PT>(context, gVar, DL);
  }

  template <typename PT>
  inline static CGObjNode<CtxTy, ObjectTy> *
  allocateHeapObj(MemModel &model, const CtxTy *context,
                  const llvm::Instruction *callsite, const llvm::DataLayout &DL,
                  llvm::Type *T) {
    return model.template allocHeapObj<PT>(context, callsite, DL, T);
  }

  template <typename PT>
  inline static CGObjNode<CtxTy, ObjectTy> *
  allocateAnonObj(MemModel &model, const CtxTy *context,
                  const llvm::DataLayout &DL, llvm::Type *T,
                  const llvm::Value *tag = nullptr, bool recursive = true) {
    return model.template allocAnonObj<PT>(context, DL, T, tag, recursive);
  }

  template <typename PT>
  inline static CGObjNode<CtxTy, ObjectTy> *
  indexObject(MemModel &model, const ObjectTy *obj,
              const llvm::Instruction *gep) {
    return model.template indexObject<PT>(obj, gep);
  }

  template <typename PT>
  inline static void
  handleMemCpy(MemModel &model, const CtxTy *C, const llvm::MemCpyInst *memCpy,
               CGPtrNode<CtxTy> *src, CGPtrNode<CtxTy> *dst) {
    return model.template handleMemCpy<PT>(C, memCpy, src, dst);
  }

  template <typename PT>
  inline static void initializeGlobal(MemModel &memModel,
                                      const llvm::GlobalVariable *gVar,
                                      const llvm::DataLayout &DL) {
    memModel.template initializeGlobal<PT>(gVar, DL);
  }

  // return *true* when the callsite handled by the
  template <typename PT>
  inline static constexpr bool
  interceptCallSite(MemModel &memModel, const CtxFunction<CtxTy> *caller,
                    const CtxFunction<CtxTy> *callee,
                    const llvm::Instruction *callSite) {
    return memModel.template interceptCallSite<PT>(caller, callee, callSite);
  }

  inline static InterceptResult
  interceptFunction(MemModel &memModel, const llvm::Function *F,
                    const llvm::Instruction *callSite) {
    return memModel.interceptFunction(F, callSite);
  }
};

} // namespace aser

#endif