//
// Created by peiming on 1/16/20.
//

#ifndef ASER_PTA_LANGMODELBASE_H
#define ASER_PTA_LANGMODELBASE_H

#include "ConsGraphBuilder.h"
#include "aser/Util/Util.h"

namespace aser {

// TODO: what about the function subroutine is an indirect call
// TODO: what about the indirect call resolved to pthread_create?
// CRTP
template <typename ctx, typename MemModel, typename PtsTy, typename SubClass>
class LangModelBase : public ConsGraphBuilder<ctx, MemModel, PtsTy, SubClass> {
public:
  // **** start ****
  // five functions need to be overriden by the subclass

  // determine whether the resolved indirect call is compatible
  inline bool isCompatible(const llvm::Instruction *callsite,
                           const llvm::Function *target) {
    llvm_unreachable("SubClass should override the function!");
  }

  //    // determine whether the API is a heap allocation API
  //    inline bool isExtHeapAllocAPI(const llvm::Function *F, const
  //    llvm::Instruction *callSite) {
  //        llvm_unreachable("SubClass should override the function!");
  //    }

  // modelling the heap allocation
  inline void interceptHeapAllocSite(const CtxFunction<ctx> *caller,
                                     const CtxFunction<ctx> *callee,
                                     const llvm::Instruction *callsite) {
    llvm_unreachable("SubClass should override the function!");
  }

  // determine whether the function need to be modelled
  inline InterceptResult interceptFunction(const llvm::Function *F,
                                           const llvm::Instruction *callsite) {
    llvm_unreachable("SubClass should override the function!");
  }

  // modelling a callsite
  inline bool interceptCallSite(const CtxFunction<ctx> *caller,
                                const CtxFunction<ctx> *callee,
                                const llvm::Instruction *callsite) {
    llvm_unreachable("SubClass should override the function!");
  }

  bool isHeapAllocAPI(const llvm::Function *fun) {
    llvm_unreachable("SubClass should override the function!");
  }
  // **** end ****

  // **** optional start ****
  inline void interceptEntryFun(const llvm::Function *entry) {
    // empty, by default do not intercept Entry function for now
    auto it = entry->arg_begin();
    auto ie = entry->arg_end();

    for (; it != ie; it++) {
      if (it->getType()->isPointerTy()) {
        auto objNode = MMT::template allocateAnonObj<PT>(
            this->getMemModel(),                    // memory model
            CT::getInitialCtx(),                    // initial context
            this->getLLVMModule()->getDataLayout(), // data layout
            // model pointer as array, it is more conservative
            // as it allows indexing on the object
            // int * --> int []
            getUnboundedArrayTy(it->getType()->getPointerElementType()),
            &(*it), // use the argument as tag
            true);  // do it recursively

        auto ptrNode = this->getPtrNode(CT::getInitialCtx(), &(*it));
        this->consGraph->addConstraints(objNode, ptrNode, Constraints::addr_of);
      }
    }
  }

  inline IndirectResolveOption
  onNewIndirectTargetResolvation(const llvm::Function *F,
                                 const llvm::Instruction *callsite) {
    // by default
    return IndirectResolveOption::WITH_LIMIT;
  }

  // **** optional end ****
protected:
  using Super = ConsGraphBuilder<ctx, MemModel, PtsTy, SubClass>;
  using Self = LangModelBase<ctx, MemModel, PtsTy, SubClass>;

  using PT = PTSTrait<PtsTy>;
  using CT = CtxTrait<ctx>;
  using MMT = MemModelTrait<MemModel>;

  using CallGraphTy = typename Super::CallGraphTy;
  using ConsGraphTy = typename Super::ConsGraphTy;

  using CGNodeTy = typename Super::CGNodeTy;
  using ObjNode = typename Super::ObjNode;
  using PtrNode = typename Super::PtrNode;

  template <typename KeyT>
  [[nodiscard]] inline MapObject<KeyT, Self> *
  getOrAllocMapObj(const ctx *context, const void *tag) {
    using MapObjKeyT = std::pair<const ctx *, const void *>;
    static llvm::DenseMap<MapObjKeyT, MapObject<KeyT, Self>> mapObjMap;

    return &mapObjMap.try_emplace(std::make_pair(context, tag), *this)
                .first->second;
  }

  // return true if the function need to be expanded in callgraph.
  inline InterceptResult overrideFunction(const ctx *callerCtx,
                                          const ctx *calleeCtx,
                                          const llvm::Function *F,
                                          const llvm::Instruction *callsite) {
    if (static_cast<SubClass *>(this)->isHeapAllocAPI(F, callsite)) {
      // we need the callsite of a heap allocation,
      // but we do not care about the implementation detail of heap allocation
      // we do not alter the Function for malloc function
      return {F, InterceptResult::Option::ONLY_CALLSITE};
    }
    if (callsite == nullptr) {
      static_cast<SubClass *>(this)->interceptEntryFun(F);
      return {F, InterceptResult::Option::EXPAND_BODY};
    }

    return static_cast<SubClass *>(this)->interceptFunction(
        calleeCtx, callerCtx, F, callsite);
  }

  // return true if the call site is handled by the language model.
  inline bool overrideCallSite(const CtxFunction<ctx> *caller,
                               const CtxFunction<ctx> *callee,
                               const llvm::Function *originalTarget,
                               const llvm::Instruction *callsite) {
    // context should match
    assert(CT::contextEvolve(caller->getContext(), callsite) ==
           callee->getContext());

    if (static_cast<SubClass *>(this)->isHeapAllocAPI(callee->getFunction(),
                                                      callsite)) {
      // handle different heap allocation APIs.
      static_cast<SubClass *>(this)->interceptHeapAllocSite(caller, callee,
                                                            callsite);
      return true;
    }

    return static_cast<SubClass *>(this)->interceptCallSite(
        caller, callee, originalTarget, callsite);
  }

  LangModelBase(llvm::Module *M, llvm::StringRef entry) : Super(M, entry) {}

  friend Super;
  friend LangModelTrait<LangModelBase<ctx, MemModel, PtsTy, SubClass>>;
};

template <typename ctx, typename MemModel, typename PtsTy, typename SubClass>
struct LangModelTrait<LangModelBase<ctx, MemModel, PtsTy, SubClass>> {
  using self = LangModelBase<ctx, MemModel, PtsTy, SubClass>;

  using CtxTy = ctx;
  // constraint graph type
  using ConsGraphTy = typename self::ConsGraphTy;
  // call callgraph type
  using CallGraphTy = typename self::CallGraphTy;
  // call callgraph node type
  using CallNodeTy = typename self::CallNodeTy;
  // constraint callgraph node type
  using CGNodeTy = typename self::CGNodeTy;
  using PT = PTSTrait<PtsTy>;
  using CT = CtxTrait<ctx>;

  using ObjNodeTy = typename self::ObjNode;
  using PtrNode = typename self::PtrNode;
  // data structure for points-to set
  using PointsToTy = PtsTy;
  // LangModel type
  using LangModelTy = SubClass;
  // memory langModel type
  using MemModelTy = MemModel;

  // memory model trait
  using MMT = MemModelTrait<MemModelTy>;
  // object type (depend on which memory trait we are using
  using ObjectTy = typename MMT::ObjectTy;

  // build initial langModel from a llvm module
  static inline LangModelTy *buildInitModel(llvm::Module *M,
                                            llvm::StringRef entry) {
    return new LangModelTy(M, entry);
  }

  static inline void
  addPreProcessingPass(llvm::legacy::PassManagerBase &passes) {
    MMT::addPreProcessingPass(passes);
  }

  static inline void constructConsGraph(LangModelTy *model) {
    model->constructConsGraph();
  }

  static inline CGNodeTy *indexObject(LangModelTy *model, ObjNodeTy *objNode,
                                      const llvm::Instruction *idx) {
    return model->indexObject(objNode, idx);
  }

  // get constructed constraint callgraph
  static inline const ConsGraphTy *getConsGraph(const LangModelTy *model) {
    return model->getConsGraph();
  }

  // get constructed constraint callgraph
  static inline ConsGraphTy *getConsGraph(LangModelTy *model) {
    return model->getConsGraph();
  }

  // get constructed call callgraph
  static inline const CallGraphTy *getCallGraph(LangModelTy *model) {
    return model->getCallGraph();
  }

  // true if at least one indirect call site is updated.
  static inline bool updateFunPtrs(LangModelTy *model,
                                   const llvm::SparseBitVector<> &resolved) {
    return model->updateFunPtrs(resolved);
  }

  // get corresponding pointer nodes that represent v (in all different
  // contexts)
  static inline void getNodesForValue(LangModelTy *model, const llvm::Value *v,
                                      std::set<CGNodeTy *> &result) {
    // langModel->getNodesForValue(v, result);
  }

  // get corresponding pointer nodes that represent v
  static inline void getPointedBy(LangModelTy *model, const ctx *context,
                                  const llvm::Value *v,
                                  std::set<Pointer<ctx>> &result) {}

  [[nodiscard]] static inline llvm::StringRef
  getEntryName(const LangModelTy *model) {
    return model->getEntryName();
  }

  [[nodiscard]] static inline const llvm::Module *
  getLLVMModule(const LangModelTy *model) {
    return model->getLLVMModule();
  }

  [[nodiscard]] static inline const CallGraphNode<ctx> *
  getDirectNode(LangModelTy *model, const ctx *C, const llvm::Function *F) {
    return model->getDirectNode(C, F);
  }

  [[nodiscard]] static inline const CallGraphNode<ctx> *
  getDirectNodeOrNull(LangModelTy *model, const ctx *C,
                      const llvm::Function *F) {
    return model->getDirectNodeOrNull(C, F);
  }

  [[nodiscard]] [[deprecated("use getInDirectCallSite(ctx, instruction) "
                             "instead")]] static inline const CallGraphNode<ctx>
      *getInDirectNode(LangModelTy *model, const ctx *C,
                       const llvm::Instruction *I) {
    return model->getInDirectNode(C, I);
  }

  [[nodiscard]] static inline const InDirectCallSite<ctx> *
  getInDirectCallSite(LangModelTy *model, const ctx *C,
                      const llvm::Instruction *I) {
    return model->getInDirectNode(C, I)->getTargetFunPtr();
  }

  static inline NodeID getSuperNodeIDForValue(LangModelTy *model, const ctx *C,
                                              const llvm::Value *V) {
    auto result = model->getPtrNodeOrNull(C, V);
    if (result) {
      return result->getSuperNode()->getNodeID();
    }
    return INVALID_NODE_ID;
  }

  static inline PtrNode *getPtrNode(LangModelTy *model, const ctx *C,
                                    const llvm::Value *V) {
    return model->getPtrNode(C, V);
  }

  static inline bool isHeapAllocAPI(LangModelTy *model,
                                    const llvm::Function *fun) {
    return model->isHeapAllocAPI(fun);
  }

  // this is a special method only used for LockSetManager
  // To handle empty pts on lock object
  // FIXME: this method should be removed, create a wrapper class for lock
  // object, and assign an unique fake identifier for those empty pts lock ptr
  static inline ObjNodeTy *allocSpecialAnonObj(LangModelTy *model,
                                               const llvm::Instruction *I,
                                               const llvm::Value *V) {
    auto objNode = MMT::template allocateHeapObj<PT>(
        model->getMemModel(), // memory model
        CT::getInitialCtx(),  // initial context
        I,
        model->getLLVMModule()->getDataLayout(), // data layout
        V->getType()->getPointerElementType());
    return objNode;
  }

  template <typename L, typename S> friend class SolverBase; // every SolverBase

  template <typename Key, typename PT> friend class MapObject;
};

} // namespace aser
#endif // ASER_PTA_LANGMODELBASE_H
