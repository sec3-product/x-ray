#pragma once

#include "PointerAnalysis/Models/MemoryModel/CppMemModel/RewriteModeledAPIPass.h"
#include "PointerAnalysis/Models/MemoryModel/CppMemModel/SpecialObject/VTablePtr.h"
#include "PointerAnalysis/Models/MemoryModel/CppMemModel/SpecialObject/Vector.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/FSCanonicalizer.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/FSMemModel.h"
#include "PointerAnalysis/Program/InterceptResult.h"
#include "PointerAnalysis/Util/Demangler.h"
#include "PointerAnalysis/Util/TypeMetaData.h"

// TODO: there are a lot of things to do to model C++'s memory model accurately
// besides vtable e.g., runtime type information + class hirachy, so we put it
// into a CppMemModel this will make it easier for future extension when we need
// to handle more languages such as Rust/Fortran

extern cl::opt<bool> CONFIG_VTABLE_MODE;

namespace xray {

template <typename ctx> class FSMemModel;

namespace cpp {

// CppMemModel is extended on field-sensitive model
template <typename ctx> class CppMemModel : public FSMemModel<ctx> {
private:
#define super Super::template
  using Super = FSMemModel<ctx>;
  using Self = CppMemModel<ctx>;
  using CT = typename Super::CT;
  using ObjNode = typename Super::ObjNode;
  using PtrNode = typename Super::PtrNode;
  using BaseNode = typename Super::BaseNode;
  using PtrManager = typename Super::PtrManager;
  using ConsGraphTy = typename Super::ConsGraphTy;
  using ContainerAllocator = llvm::BumpPtrAllocator;

  ContainerAllocator Allocator;

public:
  using CtxTy = typename Super::CtxTy;
  using ObjectTy = typename Super::ObjectTy;
  using Canonicalizer = FSCanonicalizer;

  explicit CppMemModel(ConsGraphTy &consGraph, PtrManager &owner,
                       llvm::Module &M)
      : Super(consGraph, owner, M, Super::MemModelKind::CPP) {
    TypeMDinit(&M);
  }

private:
  PtrNode *getPtrNode(const ctx *C, const llvm::Value *V) {
    return this->ptrManager.template getPtrNode<Canonicalizer>(C, V);
  }

  template <typename PT> PtrNode *createAnonNode() {
    return this->ptrManager.template createAnonPtrNode<PT>();
  }

  template <typename PT>
  PtrNode *createAnonNode(const llvm::CallBase *apiCall,
                          std::vector<PtrNode *> &&params) {
    auto tag = new (Allocator) VecAPITag<ctx>(apiCall, std::move(params));
    return this->ptrManager.template createAnonPtrNode<PT>(tag);
  }

  static bool isSpecialTypeImpl(const llvm::Type *T) {
    auto vectorElemT = VectorAPI::resolveVecElemType(T);
    bool isVector =
        vectorElemT && VectorAPI::isSupportedElementType(vectorElemT);
    // is type equivalience accurate enough? i.e., How often is it when user use
    // the same type but not for vtable should be rare, the vtable ptr type is
    // "i32 (...) **"
    // FIXME: if there might be user defined type that is identical to vtable
    // pointer, identify vtable pointer using more accurate predicate
    bool isVptr = CONFIG_VTABLE_MODE && isVTablePtrType(T);
    return isVptr || isVector;
  }

  template <typename PT>
  void modelVectorAPIs(const ctx *C, VectorAPI &calledAPI) {
    switch (calledAPI.getAPIKind()) {
    case VectorAPI::APIKind::PUSH_BACK:
      if (calledAPI.getVecElemType()->isPointerTy()) {
        // only care when a pointer is pushed in
        auto vec = getPtrNode(
            C, calledAPI.getArgOperand(0)); // first argument is *this*
        auto elem = getPtrNode(
            C, calledAPI.getArgOperand(
                   1)); // second argument is a pointer to the element.
        auto loadedElem = createAnonNode<PT>();

        this->consGraph.addConstraints(elem, loadedElem, Constraints::load);

        std::vector<PtrNode *> params{loadedElem};
        auto fake =
            createAnonNode<PT>(calledAPI.getCallSite(), std::move(params));

        // add a special constraints
        this->consGraph.addConstraints(vec, fake, Constraints::special);
      }
      break;
    case VectorAPI::APIKind::IT_BEGIN:
    case VectorAPI::APIKind::OPERATOR_INDEX: {
      auto vec = getPtrNode(C, calledAPI.getArgOperand(0));
      auto idx = getPtrNode(C, calledAPI.getCallSite());

      this->consGraph.addConstraints(vec, idx, Constraints::special);
      break;
    }
    case VectorAPI::APIKind::CTOR:
      break;
    case VectorAPI::APIKind::DTOR:
    case VectorAPI::APIKind::IT_END:
      // need not handle?
      break;
    case VectorAPI::APIKind::UNKNOWN:
      llvm_unreachable("");
      break;
    }
  }

  template <typename PT>
  inline ObjNode *allocStructArrayObjImpl(const ctx *C, const llvm::Value *V,
                                          AllocKind T, llvm::Type *type,
                                          const llvm::DataLayout &DL) {
    if (T != AllocKind::Anonymous) {
      if (auto GV = llvm::dyn_cast<llvm::GlobalVariable>(V)) {
        if (isVTableVariable(GV) && !GV->isDeclaration()) {
          // if this is a vtable, handle it specially (do not collapse the array
          // when building the memory layout)
          const MemLayout *layout = this->layoutManager.getLayoutForType(
              GV->getType()->getPointerElementType(), DL, false);
          auto block =
              super allocMemBlock<AggregateMemBlock<ctx>>(C, GV, T, layout);
          return super createNode<PT>(block->getObjectAt(0));
        }
      }
    }

    auto vectorElemT = VectorAPI::resolveVecElemType(type);
    if (vectorElemT && VectorAPI::isSupportedElementType(vectorElemT)) {
      auto vector = new Vector<ctx>(vectorElemT);
      vector->template initWithNode<PT>(&this->consGraph);

      // hold by a scalar memblock so that it can not be indexed,
      // the ownership of the object is managed by the scalar memory block
      auto block = super allocMemBlock<ScalarMemBlock<ctx>>(C, V, T, vector);
      vector->setMemBlock(block);

      return vector->getObjNode();
    }

    auto elemType = type;
    while (auto AT = llvm::dyn_cast<llvm::ArrayType>(elemType)) {
      elemType = AT->getArrayElementType(); // strip array
    }

    auto collectSpecialObj = [&](llvm::Type *T, MemLayout *L, size_t &lOffset,
                                 size_t &pOffset) -> bool {
      if (isSpecialTypeImpl(T)) {
        L->setElementOffset(lOffset);
        L->setSpecialOffset(pOffset);
        // vectorOffset.emplace_back(pOffset, vectorElemT);
        lOffset += DL.getTypeAllocSize(T);
        pOffset += DL.getTypeAllocSize(T);
        return true;
      }

      return false;
    };

    std::vector<std::pair<size_t, const llvm::Type *>> vectorOffset;

    auto layout =
        this->layoutManager.getLayoutForType(type, DL, true, collectSpecialObj);
    auto block = static_cast<AggregateMemBlock<ctx> *>(
        super allocMemBlock<AggregateMemBlock<ctx>>(C, V, T, layout));

    for (auto offset : layout->getSpecialLayout()) {
      auto elem = getTypeAtOffset(type, offset, DL);
      assert(elem != nullptr);

      while (auto AT = llvm::dyn_cast<llvm::ArrayType>(elem)) {
        elem = AT->getArrayElementType(); // strip array
      }

      llvm::Type *vecElemType = nullptr;
      while (auto ST = llvm::dyn_cast<llvm::StructType>(elem)) {
        auto result = VectorAPI::resolveVecElemType(ST);
        if (result != nullptr) {
          vectorElemT = result;
          break;
        } else {
          elem = ST->getElementType(0);
          while (auto AT = llvm::dyn_cast<llvm::ArrayType>(elem)) {
            elem = AT->getArrayElementType(); // strip array
          }
        }
      }

      if (vectorElemT != nullptr) {
        auto vector = new Vector<ctx>(block, offset, vectorElemT);
        vector->template initWithNode<PT>(&this->consGraph);
        // the memory block will take the ownership of the object
        block->initializeOffsetWith(offset, vector);
      } else {
        // this is a vtable pointer
        assert(isVTablePtrType(elem) && CONFIG_VTABLE_MODE);

        // allocate the vtable pointer object
        auto vptr = new VTablePtr<ctx>(block, offset, type);
        vptr->template initWithNode<PT>(&this->consGraph);

        block->initializeOffsetWith(offset, vptr);
      }
    }

    auto result = block->getObjectAt(0);
    if (result->getObjNodeOrNull() == nullptr) {
      super createNode<PT>(result);
    }

    return result->getObjNode();
  }

  template <typename PT>
  void initializeGlobal(const llvm::GlobalVariable *gVar,
                        const llvm::DataLayout &DL) {
    if (gVar->getType()->isPointerTy()) { // should always be true?
      auto vectorElemT = VectorAPI::resolveVecElemType(
          gVar->getType()->getPointerElementType());
      if (vectorElemT && VectorAPI::isSupportedElementType(vectorElemT)) {
        return;
      }
    }

    super initializeGlobal<PT>(gVar, DL);
  }

  inline InterceptResult interceptFunction(const llvm::Function *F,
                                           const llvm::Instruction *callSite) {
    // does not matter as they function body should be deleted by
    // RewriteModeledAPIPass
    return {F, InterceptResult::Option::EXPAND_BODY};
  }

  // return *true* when the callsite handled by the
  template <typename PT>
  inline bool interceptCallSite(const CtxFunction<CtxTy> *caller,
                                const CtxFunction<CtxTy> *callee,
                                const llvm::Instruction *callSite) {
    if (CONFIG_VTABLE_MODE) {
      if (auto call = dyn_cast<CallBase>(callSite)) {
        if (call->getCalledFunction() != nullptr &&
            call->getCalledFunction()->getName().equals(
                ".coderrect.vtable.init")) {
          // the vtable init function
          const ctx *context = caller->getContext();
          // call void @.coderrect.vtable.init(i32 (...)*** %2, i32 (...)** %4),
          PtrNode *vptr = this->getPtrNode(context, call->getArgOperand(1));
          PtrNode *src = this->getPtrNode(context, call->getArgOperand(0));
          // src is set to vptr
          this->consGraph.addConstraints(src, vptr, Constraints::special);
        }
      }
    }

    // here handle the different constainer API
    VectorAPI vecAPI(callSite);
    if (vecAPI.getAPIKind() != VectorAPI::APIKind::UNKNOWN) {
      modelVectorAPIs<PT>(callee->getContext(), vecAPI);
      return true;
    }
    return false;
  }

  static inline void
  addPreProcessingPass(llvm::legacy::PassManagerBase &passes) {
    passes.add(new RewriteModeledAPIPass());
  }

  friend MemModelHelper<Self>;
  friend Super;
};

} // namespace cpp

template <typename ctx>
struct MemModelTrait<cpp::CppMemModel<ctx>>
    : MemModelHelper<cpp::CppMemModel<ctx>> {
  // Maybe better to put into Canonicalizer classes?
  // whether *all* GEPs will be collapse by the canonicalizer
  static const bool COLLAPSE_GEP = false;
  // whether *all* BitCast will be collapse by the canonicalizer
  static const bool COLLAPSE_BITCAST = true;

  // whether type information is necessary
  // we need type information to build the memory layout
  static const bool NEED_TYPE_INFO = true;
};

#undef super

} // namespace xray
