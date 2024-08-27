#pragma once

#include <llvm/IR/Type.h>

#include <memory>

#include "PointerAnalysis/Graph/ConstraintGraph/CGPtrNode.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/FSObject.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/MemBlock.h"
#include "PointerAnalysis/Program/CtxFunction.h"
#include "PointerAnalysis/Util/Demangler.h"

// we use a new namespace in case in the future we want to extend the PTA on
// other language which has the same type of containers

namespace llvm {
class CallBase;
}

namespace xray {
namespace cpp {

// forward declaration
template <typename ctx> class CppMemModel;

class VectorAPI {
public:
  enum class APIKind {
    UNKNOWN, // have not modelled API
    PUSH_BACK,
    IT_BEGIN,
    IT_END,
    OPERATOR_INDEX, // operator[]
    CTOR,           // constructor
    DTOR,           // destructor
  };

  static std::map<llvm::StringRef, APIKind> VecAPIMap;

private:
  APIKind kind;
  Demangler demangler;
  const llvm::CallBase *APICall;
  const llvm::Type *vecElemType; // the type of the vectors' element

  // return nullptr if the type passed in is not a std::vector type
  void init(const llvm::Function *call);

public:
  explicit VectorAPI(const llvm::Instruction *call)
      : APICall(llvm::dyn_cast_or_null<llvm::CallBase>(call)),
        vecElemType(nullptr), kind(APIKind::UNKNOWN) {
    if (APICall != nullptr) {
      init(APICall->getCalledFunction());
    }
  }

  explicit VectorAPI(const llvm::Function *F)
      : APICall(nullptr), vecElemType(nullptr), kind(APIKind::UNKNOWN) {
    init(F);
  }

  inline const llvm::Value *getArgOperand(unsigned int i) const {
    return APICall->getArgOperand(i);
  }

  inline const llvm::CallBase *getCallSite() const { return APICall; }

  inline APIKind getAPIKind() const { return kind; }

  inline const llvm::Type *getVecElemType() const { return vecElemType; }

  static const llvm::Type *resolveVecElemType(const llvm::Type *T);

  static inline bool isSupportedElementType(const llvm::Type *T) {
    return T->isFloatingPointTy() || T->isIntegerTy() || T->isPointerTy();
  }
};

template <typename ctx> struct VecAPITag : public PtrNodeTag {
  using PtrNode = CGPtrNode<ctx>;
  const std::vector<PtrNode *> params;
  const llvm::CallBase *callsite;

  VecAPITag(const llvm::CallBase *I, std::vector<PtrNode *> &&v)
      : params(std::move(v)), callsite(I) {}

  std::string toString() override {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << *callsite;
    return os.str();
  }
};

// TODO: we only handle vectors that stores scalar variables for now
// e.g., vector<int>, vector<A *>, etc
template <typename ctx> class Vector : public FSObject<ctx> {
private:
  using PtrNode = CGPtrNode<ctx>;
  using ConsGraph = ConstraintGraph<ctx>;

  FSObject<ctx> theElem; // the elements
  // type of the element
  const llvm::Type *elemType;

  Vector(const llvm::Type *elemType)
      : FSObject<ctx>(nullptr, ObjectKind::Special),
        theElem(nullptr, ObjectKind::Special, false), elemType(elemType) {}

  Vector(size_t pOffset, const llvm::Type *elemType)
      : FSObject<ctx>(nullptr, pOffset, ObjectKind::Special),
        theElem(nullptr, ObjectKind::Special, false), elemType(elemType) {}

  Vector(MemBlock<ctx> *block, size_t pOffset, const llvm::Type *elemType)
      : FSObject<ctx>(block, pOffset, ObjectKind::Special),
        theElem(block, ObjectKind::Special, false), elemType(elemType) {}

  void setMemBlock(MemBlock<ctx> *memBlock) {
    assert(this->memBlock == nullptr);

    FSObject<ctx>::setMemBlock(memBlock);
    theElem.setMemBlock(memBlock);
  }

  template <typename PT> void initWithNode(ConstraintGraph<ctx> *CG) {
    FSObject<ctx>::template initWithNode<PT>(CG);
    theElem.template initWithNode<PT>(CG);
  }

public:
  // *src* can points to theVec
  bool processSpecial(CGNodeBase<ctx> *src,
                      CGNodeBase<ctx> *dst) const override {
    bool changed = false;

    auto consGraph = static_cast<ConsGraph *>(dst->getGraph());
    PtrNode *dstPtr = llvm::cast<PtrNode>(dst);
    // assert(dstPtr->isAnonNode());
    const llvm::Instruction *apiCall;
    if (dstPtr->isAnonNode()) {
      auto tag = static_cast<VecAPITag<ctx> *>(dstPtr->getTag());
      apiCall = tag->callsite;
    } else {
      apiCall = cast<Instruction>(dstPtr->getPointer()->getValue());
    }

    VectorAPI calledAPI(apiCall);
    switch (calledAPI.getAPIKind()) {
    case VectorAPI::APIKind::PUSH_BACK: {
      auto tag = static_cast<VecAPITag<ctx> *>(dstPtr->getTag());
      auto params = tag->params;

      assert(calledAPI.getVecElemType()->isPointerTy());
      assert(params.size() == 1);

      changed = consGraph->addConstraints(params.front(), theElem.getObjNode(),
                                          Constraints::copy);
      break;
    }
    case VectorAPI::APIKind::IT_BEGIN:
    case VectorAPI::APIKind::OPERATOR_INDEX: {
      changed = consGraph->addConstraints(theElem.getObjNode(), dst,
                                          Constraints::addr_of);
      break;
    }
    case VectorAPI::APIKind::IT_END:
      break;
    case VectorAPI::APIKind::CTOR: // copy/move/initializer-list
      break;
    case VectorAPI::APIKind::DTOR:
      break;
    case VectorAPI::APIKind::UNKNOWN:
      return false;
    }

    return changed;
  }

  friend CppMemModel<ctx>;
};

} // namespace cpp
} // namespace xray
