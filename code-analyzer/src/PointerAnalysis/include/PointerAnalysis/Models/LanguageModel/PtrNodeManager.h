#pragma once

#include <unordered_map>

#include "PointerAnalysis/Program/CtxFunction.h"
#include "PointerAnalysis/Program/Pointer.h"
#include "PointerAnalysis/Util/SingleInstanceOwner.h"

extern bool DEBUG_PTA;
extern bool DEBUG_PTA_VERBOSE;

namespace xray {

template <typename ctx>
class PtrNodeManager : public SingleInstanceOwner<Pointer<ctx>> {
private:
  using CT = CtxTrait<ctx>;
  using ConsGraph = ConstraintGraph<ctx>;
  using BaseNode = CGNodeBase<ctx>;
  using PtrNode = CGPtrNode<ctx>;

  ConsGraph *consGraph;

  PtrNode *nullPtrNode; // special node for null pointer
  PtrNode *uniPtrNode;  // special node that can point to every object.

protected:
  explicit PtrNodeManager() : consGraph(nullptr) {}

  template <typename PT> void init(ConsGraph *CG, llvm::LLVMContext &context) {
    this->consGraph = CG;

    // create null ptr node
    const llvm::Value *nullValue =
        llvm::ConstantPointerNull::get(llvm::Type::getInt8PtrTy(context));
    auto nullPtr = this->create(CT::getGlobalCtx(), nullValue);
    nullPtrNode = consGraph->template addCGNode<PtrNode, PT>(nullPtr);
    const_cast<Pointer<ctx> *>(nullPtr)->setPtrNode(nullPtrNode);

    // create universal ptr node
    const llvm::Value *uniValue =
        llvm::UndefValue::get(llvm::Type::getInt8PtrTy(context));
    auto uniPtr = this->create(CT::getGlobalCtx(), uniValue);
    uniPtrNode = consGraph->template addCGNode<PtrNode, PT>(uniPtr);
    const_cast<Pointer<ctx> *>(uniPtr)->setPtrNode(uniPtrNode);
  }

public:
  // return map does not managed by singleinstanceowner as it may conflict
  // with function pointers (when initialCtx and globalCtx are the same)
  std::unordered_map<std::pair<const ctx *, const llvm::Function *>,
                     Pointer<ctx>>
      retPtrMap;
  //    an anonoymous pointer but can be indexed
  //    std::unordered_map<std::pair<const ctx *, const void *>, PtrNode *>
  //    taggedAnonPtrMap;
  inline PtrNode *getUniPtr() const { return uniPtrNode; }

  inline PtrNode *getNullPtr() const { return nullPtrNode; }

  template <typename PT>
  inline PtrNode *createRetNode(const CtxFunction<ctx> *fun) {
    auto result = retPtrMap.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(fun->getContext(), fun->getFunction()),
        std::forward_as_tuple(fun->getContext(), fun->getFunction()));

    assert(result.second);
    auto retPtr = &result.first->second;

    auto ret = consGraph->template addCGNode<PtrNode, PT>(retPtr);
    retPtr->setPtrNode(ret);

    if (DEBUG_PTA_VERBOSE) {
      llvm::outs() << "createRetNode: " << ret->getNodeID()
                   << " fun: " << fun->getFunction()->getName() << "\n"; // JEFF
    } else if (DEBUG_PTA)
      llvm::outs() << "createRetNode: " << ret->getNodeID() << "\n";

    return ret;
  }

  inline PtrNode *getRetNode(const ctx *C, const llvm::Function *F) {
    auto it = retPtrMap.find(std::make_pair(C, F));
    assert(it != retPtrMap.end());
    return it->second.getPtrNode();
  }

  // anonoyous ptr node should never be indexed, just a place holder
  // logically exist, but no corresponding llvm::Value
  template <typename PT> inline PtrNode *createAnonPtrNode() {
    auto ret = consGraph->template addCGNode<PtrNode, PT>();
    return ret;
  }

  template <typename PT> inline PtrNode *createAnonPtrNode(PtrNodeTag *tag) {
    auto ret = consGraph->template addCGNode<PtrNode, PT>(tag);
    return ret;
  }

  //    // a tagged anonmynous pointer
  //    inline PtrNode *getOrCreateTaggedAnonPtrNode(const ctx *C, const void
  //    *V) {
  //        auto it = taggedAnonPtrMap.find(std::make_pair(C, V));
  //        if (it != taggedAnonPtrMap.end()) {
  //            return it->second;
  //        }
  //
  //        // create the anonomyous pointer
  //        auto ret = consGraph->template addCGNode<PtrNode, PT>();
  //        auto result =
  //        taggedAnonPtrMap.insert(std::make_pair(std::make_pair(C, V), ret));
  //        assert(result.second);
  //        return ret;
  //    }

  template <typename Canonicalizer>
  inline PtrNode *getPtrNode(const ctx *C, const llvm::Value *V) {
    V = Canonicalizer::canonicalize(V);
    if (LLVM_UNLIKELY(V == uniPtrNode->getPointer()->getValue())) {
      return uniPtrNode;
    }
    if (LLVM_UNLIKELY(V == nullPtrNode->getPointer()->getValue())) {
      return nullPtrNode;
    }
    if (llvm::isa<llvm::GlobalValue>(V)) {
      C = CT::getGlobalCtx();
    }
    // bool isCE = llvm::dyn_cast<llvm::ConstantExpr>(V);
    auto ptrNode_ = this->getOrNull(C, V);
    if (ptrNode_)
      return ptrNode_->getPtrNode();
    else
      return nullptr;
  }

  template <typename Canonicalizer>
  inline PtrNode *getPtrNodeOrNull(const ctx *C, const llvm::Value *V) {
    V = Canonicalizer::canonicalize(V);
    if (LLVM_UNLIKELY(V == uniPtrNode->getPointer()->getValue())) {
      return uniPtrNode;
    }
    if (LLVM_UNLIKELY(V == nullPtrNode->getPointer()->getValue())) {
      return nullPtrNode;
    }
    if (llvm::isa<llvm::GlobalValue>(V)) {
      C = CT::getGlobalCtx();
    }

    auto result = this->getOrNull(C, V);
    if (result != nullptr) {
      return result->getPtrNode();
    }
    return nullptr;
  }

  template <typename Canonicalizer, typename PT>
  inline PtrNode *getOrCreatePtrNode(const ctx *C, const llvm::Value *V) {
    V = Canonicalizer::canonicalize(V);
    if (LLVM_UNLIKELY(V == uniPtrNode->getPointer()->getValue())) {
      return uniPtrNode;
    }
    if (LLVM_UNLIKELY(V == nullPtrNode->getPointer()->getValue())) {
      return nullPtrNode;
    }
    if (llvm::isa<llvm::GlobalValue>(V)) {
      C = CT::getGlobalCtx();
    }

    std::pair<const Pointer<ctx> *, bool> result = this->getOrCreate(C, V);
    const Pointer<ctx> *ptr = result.first;
    if (result.second) {
      auto ret = consGraph->template addCGNode<PtrNode, PT>(ptr);
      const_cast<Pointer<ctx> *>(ptr)->setPtrNode(ret);
      if (DEBUG_PTA_VERBOSE) {
        llvm::outs() << "getOrCreatePtrNode: " << ret->getNodeID()
                     << " value: " << *V << "\n"; // JEFF
      } else if (DEBUG_PTA)
        llvm::outs() << "getOrCreatePtrNode: " << ret->getNodeID() << "\n";
    }
    return ptr->getPtrNode();
  }

  template <typename Canonicalizer, typename PT>
  inline PtrNode *createPtrNode(const ctx *C, const llvm::Value *V) {
    V = Canonicalizer::canonicalize(V);
    if (LLVM_UNLIKELY(V == uniPtrNode->getPointer()->getValue())) {
      return uniPtrNode;
    }
    if (LLVM_UNLIKELY(V == nullPtrNode->getPointer()->getValue())) {
      return nullPtrNode;
    }
    if (llvm::isa<llvm::GlobalValue>(V)) {
      C = CT::getGlobalCtx();
    }

    auto ptr = this->create(C, V);
    auto ret = consGraph->template addCGNode<PtrNode, PT>(ptr);
    const_cast<Pointer<ctx> *>(ptr)->setPtrNode(ret);

    if (DEBUG_PTA_VERBOSE) {
      llvm::outs() << "createPtrNode: " << ret->getNodeID() << " value: " << *V
                   << "\n"; // JEFF
    } else if (DEBUG_PTA)
      llvm::outs() << "createPtrNode: " << ret->getNodeID() << "\n";

    return ret;
  }
};

} // namespace xray
