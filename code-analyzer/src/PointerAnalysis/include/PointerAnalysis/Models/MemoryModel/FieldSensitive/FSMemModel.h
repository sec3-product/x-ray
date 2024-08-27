#pragma once

#include <Logger/Logger.h>
#include <PreProcessing/Passes/CanonicalizeGEPPass.h>
#include <PreProcessing/Passes/LoweringMemCpyPass.h>
#include <llvm/Support/Allocator.h>

#include "PointerAnalysis/Models/LanguageModel/PtrNodeManager.h"
#include "PointerAnalysis/Models/MemoryModel/CppMemModel/CppMemModel.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/FSCanonicalizer.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/FSObject.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/Layout/MemLayoutManager.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/MemBlock.h"
#include "PointerAnalysis/Models/MemoryModel/SpecialObjects/MapObject.h"
#include "PointerAnalysis/Program/InterceptResult.h"
#include "PointerAnalysis/Util/ConstExprVisitor.h"
#include "PointerAnalysis/Util/Util.h"

extern bool DEBUG_PTA;
extern bool DEBUG_PTA_VERBOSE;
extern size_t PTAAnonLimit;

namespace xray {

size_t getGEPStepSize(const llvm::GetElementPtrInst *GEP,
                      const llvm::DataLayout &DL);

// namespace cpp {
// template <typename ctx>
// class CppMemModel;
// }

template <typename ctx> class FSMemModel {
protected:
  using CT = CtxTrait<ctx>;
  using Self = FSMemModel<ctx>;
  using ObjNode = CGObjNode<ctx, FSObject<ctx>>;
  using PtrNode = CGPtrNode<ctx>;
  using BaseNode = CGNodeBase<ctx>;
  using ConsGraphTy = ConstraintGraph<ctx>;
  using PtrManager = PtrNodeManager<ctx>;
  using MemBlockAllocator = llvm::BumpPtrAllocator;
  using CtxAllocPair = std::pair<const ctx *, const llvm::Value *>;

  enum class MemModelKind {
    FS,  // normal field sensitive
    CPP, // cpp memory model
  };

  const MemModelKind kind;
  MemBlockAllocator Allocator;
  PtrManager &ptrManager; // the manager for pointer
  ConsGraphTy &consGraph;
  llvm::Module &module;

  ObjNode *nullObjNode = nullptr;
  ObjNode *uniObjNode = nullptr;

  MemLayoutManager layoutManager;
  // map from a allocation site to the Memory Block
  llvm::DenseMap<CtxAllocPair, MemBlock<ctx> *> memBlockMap;

  inline llvm::Module &getLLVMModule() const { return module; }

  template <typename PT> inline ObjNode *createNode(const FSObject<ctx> *obj) {
    assert(obj != nullptr);
    auto ret = consGraph.template addCGNode<ObjNode, PT>(obj);
    const_cast<FSObject<ctx> *>(obj)->setObjNode(ret);

    if (DEBUG_PTA_VERBOSE) {
      llvm::outs() << "createObjNode: " << ret->getNodeID()
                   << " obj: " << *obj->getValue() << "\n"; // JEFF
    } else if (DEBUG_PTA)
      llvm::outs() << "createObjNode: " << ret->getNodeID() << "\n";

    return ret;
  }

  inline MemBlock<ctx> *getMemBlock(const ctx *c, const llvm::Value *v) {
    // v is the allocation site of the memory block
    if (llvm::isa<llvm::GlobalVariable>(v)) {
      c = CT::getGlobalCtx();
    }
    auto it = memBlockMap.find(std::make_pair(c, v));
    assert(it != memBlockMap.end() && "can not find the memory block");
    return it->second;
  }

  template <typename BlockT, typename... Args>
  MemBlock<ctx> *allocMemBlock(const ctx *c, const llvm::Value *v,
                               Args &&...args) {
    bool result;
    MemBlock<ctx> *block =
        new (Allocator) BlockT(c, v, std::forward<Args>(args)...);
    // we do not to put anonymous object into the map
    if (block->getAllocSite().getAllocType() != AllocKind::Anonymous) {
      std::tie(std::ignore, result) =
          this->memBlockMap.insert(std::make_pair(std::make_pair(c, v), block));
      // must be adding a new memory block
      assert(result && "allocating a existing memory block");
    }
    return block;
  }

  // The llvm::type does not have to be consistent with the type of the
  // llvm::value as the heap allocation type might be inferred using heuristics.
  template <typename PT>
  inline ObjNode *allocStructArrayObjImpl(const ctx *C, const llvm::Value *V,
                                          AllocKind T, llvm::Type *type,
                                          const llvm::DataLayout &DL) {
    auto layout = layoutManager.getLayoutForType(type, DL);
    auto block = this->allocMemBlock<AggregateMemBlock<ctx>>(C, V, T, layout);
    return createNode<PT>(block->getObjectAt(0));
  }

  template <typename PT>
  inline ObjNode *allocStructArrayObj(const ctx *C, const llvm::Value *V,
                                      AllocKind T, llvm::Type *type,
                                      const llvm::DataLayout &DL) {
    switch (kind) {
    case MemModelKind::FS: {
      return this->allocStructArrayObjImpl<PT>(C, V, T, type, DL);
    }
    case MemModelKind::CPP: {
      return static_cast<cpp::CppMemModel<ctx> *>(this)
          ->template allocStructArrayObjImpl<PT>(C, V, T, type, DL);
    }
    default:
      return nullptr;
    }
  }

  template <typename PT>
  inline ObjNode *allocFIObject(const ctx *C, const llvm::Value *V,
                                AllocKind T) {
    // we can not infer the type of the allocating heap object,
    // conservatively treat it field insensitively.
    MemBlock<ctx> *block = this->allocMemBlock<FIMemBlock<ctx>>(C, V, T);
    return createNode<PT>(block->getObjectAt(0));
  }

  template <typename PT>
  ObjNode *allocValueWithType(const ctx *C, const llvm::Value *V, AllocKind T,
                              llvm::Type *type, const llvm::DataLayout &DL) {
    assert(type && "llvm::Type can not be null");
    LOG_TRACE("allocate object. type={}", *type);

    MemBlock<ctx> *block;
    switch (type->getTypeID()) {
    case llvm::Type::StructTyID:
    case llvm::Type::ArrayTyID: {
      if (auto ST = llvm::dyn_cast<llvm::StructType>(type)) {
        if (ST->isOpaque()) {
          // do not know the type layout for a opaque type,
          // simply treat it as field-insensitive object
          LOG_DEBUG("Value has opaque type. value={}", *V);
          block = this->allocMemBlock<FIMemBlock<ctx>>(C, V, T);
          break;
        }
      }
      return allocStructArrayObj<PT>(C, V, T, type, DL);
    }
    case llvm::Type::FixedVectorTyID:
    case llvm::Type::ScalableVectorTyID: {
      // case llvm::Type::VectorTyID: {
      LOG_DEBUG("Vector Type not handled. type={}", *V);
      block = this->allocMemBlock<FIMemBlock<ctx>>(C, V, T);
      break;
      // llvm_unreachable("vector type not handled");
    }
    default: {
      // TODO: when will i8* be a FIObject?
      block = this->allocMemBlock<ScalarMemBlock<ctx>>(C, V, T);
      break;
    }
    }

    assert(block && "MemBlock can not be NULL!");
    // get the object at 0 offset
    return createNode<PT>(block->getObjectAt(0));
  }

public:
  using CtxTy = ctx;
  // object type
  using ObjectTy = FSObject<ctx>;
  using Canonicalizer = FSCanonicalizer;

  FSMemModel(ConsGraphTy &consGraph, PtrManager &owner, llvm::Module &M,
             MemModelKind kind = MemModelKind::FS)
      : consGraph(consGraph), ptrManager(owner), module(M), kind(kind) {}

protected:
  template <typename PT>
  inline ObjNode *allocNullObj(const llvm::Module *module) {
    assert(!nullObjNode && "recreating a null object!");

    auto v = llvm::ConstantPointerNull::get(
        llvm::Type::getInt8PtrTy(module->getContext()));
    auto *block = this->allocMemBlock<FIMemBlock<ctx>>(CT::getGlobalCtx(), v,
                                                       AllocKind::Null);
    return this->template createNode<PT>(block->getObjectAt(0));
  }

  template <typename PT>
  inline ObjNode *allocUniObj(const llvm::Module *module) {
    assert(!uniObjNode && "recreating a universal object!");

    auto v =
        llvm::UndefValue::get(llvm::Type::getInt8PtrTy(module->getContext()));
    auto *block = this->allocMemBlock<FIMemBlock<ctx>>(CT::getGlobalCtx(), v,
                                                       AllocKind::Universal);
    return this->template createNode<PT>(block->getObjectAt(0));
  }

  template <typename PT>
  inline void handleMemCpy(const ctx *C, const llvm::MemCpyInst *memCpy,
                           PtrNode *src, PtrNode *dst) {
    // TODO: memcpy whose cpy size is constant should already be lowered before
    // PTA, for the remaining memcpy, we do not handle it for now. if we handle
    // it, it more introduce too many false positive, as we have to do it
    // conservatively
    LOG_TRACE("unhandled memcpy instruction. inst={}", *memCpy);
  }

  bool isSpecialType(const llvm::Type *T) {
    switch (kind) {
    case MemModelKind::FS: {
      return false;
    }
    case MemModelKind::CPP: {
      return cpp::CppMemModel<ctx>::isSpecialTypeImpl(T);
    }
    }
    return false;
  }

  template <typename PT>
  inline ObjNode *allocFunction(const llvm::Function *f) {
    // create a function object (function pointer can not be indexed as well)
    MemBlock<ctx> *block = this->allocMemBlock<ScalarMemBlock<ctx>>(
        CT::getGlobalCtx(), f, AllocKind::Functions);
    return createNode<PT>(block->getObjectAt(0));
  }

  template <typename PT>
  inline ObjNode *allocGlobalVariable(const llvm::GlobalVariable *g,
                                      const llvm::DataLayout &DL) {
    LOG_TRACE("allocating global. global={}", *g);
    auto pointedType = g->getType()->getPointerElementType();
    return this->allocValueWithType<PT>(CT::getGlobalCtx(), g,
                                        AllocKind::Globals, pointedType, DL);
  }

  template <typename PT>
  inline ObjNode *allocStackObj(const ctx *C, const llvm::AllocaInst *I,
                                const llvm::DataLayout &DL) {
    LOG_TRACE("allocating Stack Object. inst={}", *I);

    const llvm::Value *arraySize = I->getArraySize();
    auto elementType = I->getType()->getPointerElementType();

    llvm::Type *type;
    if (auto constSize = llvm::dyn_cast<llvm::ConstantInt>(arraySize)) {
      size_t elementNum = constSize->getSExtValue();
      if (elementNum == 1) {
        type = elementType;
      } else {
        type = llvm::ArrayType::get(elementType, elementNum);
      }
    } else {
      type =
          llvm::ArrayType::get(elementType, std::numeric_limits<size_t>::max());
    }

    return this->allocValueWithType<PT>(C, I, AllocKind::Stack, type, DL);
  }

  template <typename PT>
  inline ObjNode *allocHeapObj(const ctx *C, const llvm::Instruction *I,
                               const llvm::DataLayout &DL, llvm::Type *T) {
    if (T != nullptr) {
      return this->allocValueWithType<PT>(C, I, AllocKind::Heap, T, DL);
    }

    // we can not infer the type of the allocating heap object,
    // conservatively treat it field insensitively.
    MemBlock<ctx> *block =
        this->allocMemBlock<FIMemBlock<ctx>>(C, I, AllocKind::Heap);
    return createNode<PT>(block->getObjectAt(0));
  }

  template <typename PT>
  void initAnonAggregateObj(
      const ctx *C, const llvm::DataLayout &DL, llvm::Type *T,
      MemBlock<ctx> *block, std::vector<const llvm::Type *> &typeTree,
      const llvm::Value *tag,
      size_t &globOffset) { // the offset of the field currently being accessed
    // must be sized type
    //        llvm::outs() <<
    //        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << "\n"; for
    //        (auto T : typeTree) {
    //            llvm::outs() << *T << "\n";
    //        }
    assert(T->isSized());

    // flatten the array/structure type
    auto structTy = llvm::cast<llvm::StructType>(T);
    auto layout = DL.getStructLayout(structTy);

    for (int i = 0; i < structTy->getNumElements(); i++) {
      auto elemTy = stripArray(structTy->getElementType(i));

      size_t offset = globOffset + layout->getElementOffset(i);
      if (auto structElem = llvm::dyn_cast<llvm::StructType>(elemTy)) {
        // recursive into the type tree.
        // since the type tree is normally not so deep, probably okay
        if (!isSpecialType(structElem)) {
          initAnonAggregateObj<PT>(C, DL, elemTy, block, typeTree, tag, offset);
        }
      } else if (auto ptrElem = llvm::dyn_cast<llvm::PointerType>(elemTy)) {
        // skipping the element type if the type is not valid for array
        // e.g., function, void type
        ObjNode *child = allocAnonObjRec<PT>(
            C, DL, getUnboundedArrayTy(elemTy->getPointerElementType()), tag,
            typeTree);
        if (child != nullptr) {
          const FSObject<ctx> *object = block->getPtrObjectAt(offset);
          // assert(object && "No Pointer Element at the offset");
          if (object != nullptr) {
            // we can have the following types
            // type <{ i8, [0 x %struct.RSValue*] }>
            if (object->getObjNodeOrNull() == nullptr) {
              createNode<PT>(object);
            } else {
              assert(offset == 0);
            }
            consGraph.addConstraints(child, object->getObjNode(),
                                     Constraints::addr_of);
          }
        }
      }

      // skip other type
    }

    // accumulated the global offset
    globOffset += DL.getTypeAllocSize(structTy);
  }

  int allocatedCount;
  const int ANON_REC_LIMIT = PTAAnonLimit;
  const int ANON_REC_DEPTH_LIMIT = 10;
  template <typename PT>
  ObjNode *allocAnonObjRec(const ctx *C, const llvm::DataLayout &DL,
                           llvm::Type *T, const llvm::Value *tag,
                           std::vector<const llvm::Type *> &typeTree) {

    // limit the depth of anonmyous object
    if (typeTree.size() > ANON_REC_DEPTH_LIMIT) {
      return nullptr;
    }

    if (std::find(typeTree.begin(), typeTree.end(), T) != typeTree.end() ||
        T == nullptr || (ANON_REC_LIMIT && allocatedCount > ANON_REC_LIMIT)) {
      // recursive type
      // i.e., link_list {link_list *next};
      return nullptr;
    }

    if (!T->isSized()) { // opaque type, can not infer the memory layout
      auto block =
          this->allocMemBlock<FIMemBlock<ctx>>(C, tag, AllocKind::Anonymous);
      auto objNode = createNode<PT>(block->getObjectAt(0));
      return objNode;
    }

    // DFS over the type tree
    typeTree.push_back(T);

    allocatedCount++;
    ObjNode *objNode =
        this->allocValueWithType<PT>(C, tag, AllocKind::Anonymous, T, DL);
    MemBlock<ctx> *block = objNode->getObject()->memBlock;

    if (!objNode->getObject()->isSpecialObject()) {
      // first allocate the root memoryblock and create the memory block
      switch (T->getTypeID()) {
      case llvm::Type::StructTyID: {
        // flatten the aggregated type and allocate their child
        // result = allocAnonAggregateObj(C, DL, T, typeTree);
        size_t offset = 0;
        initAnonAggregateObj<PT>(C, DL, T, block, typeTree, tag, offset);
        break;
      }
      case llvm::Type::ArrayTyID: {
        llvm::Type *elemTy = T;
        do {
          // strip all the array to get the inner most type
          elemTy = elemTy->getArrayElementType();
        } while (llvm::isa<llvm::ArrayType>(elemTy));
        assert(!llvm::isa<llvm::ArrayType>(elemTy));

        if (auto structElem = llvm::dyn_cast<llvm::StructType>(elemTy)) {
          size_t offset = 0;
          // the element is a structure, try to initialize it recursively
          initAnonAggregateObj<PT>(C, DL, structElem, block, typeTree, tag,
                                   offset);
        } else if (auto ptrElem = llvm::dyn_cast<llvm::PointerType>(elemTy)) {
          // treat ptr as array to allow indexing, that is int * -> int []
          ObjNode *child = allocAnonObjRec<PT>(
              C, DL, getUnboundedArrayTy(elemTy->getPointerElementType()), tag,
              typeTree);
          if (child != nullptr) {
            consGraph.addConstraints(child, objNode, Constraints::addr_of);
          }
        }
        break;
      }
      case llvm::Type::PointerTyID: {
        ObjNode *child = allocAnonObjRec<PT>(
            C, DL, getUnboundedArrayTy(T->getPointerElementType()), tag,
            typeTree);
        if (child != nullptr) {
          consGraph.addConstraints(child, objNode, Constraints::addr_of);
        }
        break;
      }
      default:
        break;
      }
    }

    // pop the type tree
    typeTree.pop_back();
    return objNode;
  }

  template <typename PT>
  inline ObjNode *allocAnonObj(const ctx *C, const llvm::DataLayout &DL,
                               llvm::Type *T, const llvm::Value *tag,
                               bool recursive) {
    // the anonymous object can not be access and looked up
    // or the T is not sized (e.g., opaque type due to incomplete bc file)
    if (T != nullptr) {
      if (recursive) {
        // if we are request to build the object recursively...
        // initialize it according to the type tree
        allocatedCount = 0;
        std::vector<const llvm::Type *> DFSStack;
        return allocAnonObjRec<PT>(C, DL, T, tag, DFSStack);
      }
      return this->allocValueWithType<PT>(C, tag, AllocKind::Anonymous, T, DL);
    }

    MemBlock<ctx> *block =
        this->allocMemBlock<FIMemBlock<ctx>>(C, tag, AllocKind::Anonymous);
    return createNode<PT>(block->getObjectAt(0));
  }

  template <typename PT>
  inline void processScalarGlobals(MemBlock<ctx> *memBlock,
                                   const llvm::Constant *C, size_t &offset,
                                   const llvm::DataLayout &DL) {
    if (C->getType()->isPointerTy() && !llvm::isa<llvm::BlockAddress>(C)) {
      /* FIXME: what if the constexpr is a complicated expression or global
         alias? need to evaluate it statically. */
      // now assume it point to a global variable
      // llvm::APInt constOffset(64, 0);
      C = C->stripPointerCasts();
      while (auto GA = dyn_cast<GlobalAlias>(C)) {
        C = GA->getAliasee();
      }

      C = llvm::dyn_cast<llvm::Constant>(Canonicalizer::canonicalize(C));
      // auto V = C->stripAndAccumulateConstantOffsets(DL, constOffset, true);

      auto *ptr = memBlock->getObjectAt(offset);

      if (ptr->getObjNodeOrNull() == nullptr) {
        createNode<PT>(ptr);
      }

      auto *ptrNode = ptr->getObjNode();
      ObjNode *objNode = nullptr;
      if (auto GEP = llvm::dyn_cast<llvm::GEPOperator>(C)) {
        auto off = llvm::APInt(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        auto baseValue = GEP->stripAndAccumulateConstantOffsets(DL, off, true);
        // objNode can be none, when it is a external symbol, which does not
        // have initializers.
        auto FSobj = getMemBlock(CT::getGlobalCtx(), baseValue)
                         ->getObjectAt(off.getSExtValue());
        // FIXME: Field-sensitive object can be nullptr, because we handle i8*
        // as scalar object however, it should be the most conservative type in
        // LLVM (void *) probably should handle it as a field-insensitive
        // object.
        if (FSobj != nullptr) {
          objNode = FSobj->getObjNodeOrNull();
          if (objNode == nullptr) {
            // it might depends on the globals that we have not met yet.
            objNode = createNode<PT>(FSobj);
          }
        }
      } else {
        objNode =
            getMemBlock(CT::getGlobalCtx(), C)->getObjectAt(0)->getObjNode();
      }

      if (objNode != nullptr) {
        if (DEBUG_PTA_VERBOSE) {
          llvm::outs() << "processScalarGlobals: " << *C
                       << "\noffset: " << offset << "\n"
                       << "objNode: " << objNode->getNodeID()
                       << " ptrNode: " << ptrNode->getNodeID() << "\n";
        } else if (DEBUG_PTA)
          llvm::outs() << "processScalarGlobals: offset: " << offset << "\n"
                       << "objNode: " << objNode->getNodeID()
                       << " ptrNode: " << ptrNode->getNodeID() << "\n";

        consGraph.addConstraints(objNode, ptrNode, Constraints::addr_of);
      }
    }

    // accumulate the physical offset
    offset += DL.getTypeAllocSize(C->getType());
  }

  template <typename PT>
  inline void processAggregateGlobals(MemBlock<ctx> *memBlock,
                                      const llvm::Constant *C, size_t &offset,
                                      const llvm::DataLayout &DL) {
    assert(llvm::isa<llvm::ConstantArray>(C) ||
           llvm::isa<llvm::ConstantStruct>(C) ||
           llvm::isa<llvm::ConstantDataArray>(C));

    size_t initOffset = offset;
    if (llvm::isa<llvm::ConstantArray>(C) ||
        llvm::isa<llvm::ConstantStruct>(C)) {
      for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i) {
        // make up the padding if it is a structure type.
        if (auto structType = llvm::dyn_cast<llvm::StructType>(C->getType())) {
          auto structLayout = DL.getStructLayout(structType);
          if (structLayout->hasPadding()) {
            assert(offset >= initOffset);
            size_t padding =
                structLayout->getElementOffset(i) - (offset - initOffset);
            offset += padding;
          }
        }

        // recursively traverse the initializer
        processInitializer<PT>(
            memBlock, llvm::cast<llvm::Constant>(C->getOperand(i)), offset, DL);
      }
      // make up the padding if it is a struct type
      if (auto structType = llvm::dyn_cast<llvm::StructType>(C->getType())) {
        // if (initOffset + DL.getTypeAllocSize(C->getType()) - offset != 0) {
        offset = initOffset + DL.getTypeAllocSize(C->getType());
        //}
      }
    } else if (llvm::isa<llvm::ConstantDataArray>(C)) {
      // Constant Data Array does not have operand, as they store
      // the value directly instead of as Value*s
      // it does not contains pointers, simply skip it.
      // For something like:
      // private unnamed_addr constant [8 x i8] c"abcdefg\00", align 1
      offset += DL.getTypeAllocSize(C->getType());
    }
  }

  // TODO: change it to non-recursive version if the initializer is deep
  template <typename PT>
  void processInitializer(MemBlock<ctx> *memBlock,
                          const llvm::Constant *initializer, size_t &offset,
                          const llvm::DataLayout &DL) {
    if (initializer->isNullValue() ||
        llvm::isa<llvm::UndefValue>(initializer)) {
      // skip zero initializer and undef initializer
      offset += DL.getTypeAllocSize(initializer->getType());
    } else if (initializer->getType()->isSingleValueType()) {
      processScalarGlobals<PT>(memBlock, initializer, offset, DL);
    } else {
      processAggregateGlobals<PT>(memBlock, initializer, offset, DL);
    }
  }

  template <typename PT>
  void initializeGlobal(const llvm::GlobalVariable *gVar,
                        const llvm::DataLayout &DL) {
    if (gVar->hasInitializer()) {
      size_t offset = 0;

      // 1st, get the memory block of the global variable
      MemBlock<ctx> *memBlock = getMemBlock(CT::getGlobalCtx(), gVar);
      // 2nd, recursively initialize the global variable
      processInitializer<PT>(memBlock, gVar->getInitializer(), offset, DL);
      if (gVar->isConstant()) {
        // the memory block is immutable, i.e., the content of the memory block
        // can not be updated later.
        memBlock->setImmutable();
      }
    }
    // else an extern symbol, conservatively can point to anything, simply skip
    // it for now
  }

  template <typename PT>
  ObjNode *indexObject(const FSObject<ctx> *obj, const llvm::Instruction *idx) {
    const llvm::DataLayout &DL = idx->getModule()->getDataLayout();
    auto result = obj->indexObject(idx, DL);

    // can not index
    if (result == nullptr) {
      return nullptr;
    }

    if (result->getObjNodeOrNull() == nullptr) {
      // the lazily instantiated field object
      createNode<PT>(result);
    }

    return result->getObjNode();
  }

  inline InterceptResult interceptFunction(const llvm::Function *F,
                                           const llvm::Instruction *callSite) {
    return {F, InterceptResult::Option::EXPAND_BODY};
  }

  // return *true* when the callsite handled by the
  template <typename PT>
  inline constexpr bool
  interceptCallSite(const CtxFunction<CtxTy> *caller,
                    const CtxFunction<CtxTy> *callee,
                    const llvm::Instruction *callSite) const {
    return false;
  }

  friend MemModelTrait<FSMemModel<ctx>>;
  friend MemModelHelper<FSMemModel<ctx>>;
};

template <typename ctx>
struct MemModelTrait<FSMemModel<ctx>> : MemModelHelper<FSMemModel<ctx>> {
  //    using CtxTy = ctx;
  //    using ObjectTy = FSObject<ctx>;
  using Canonicalizer = FSCanonicalizer;

  // whether *all* GEPs will be collapse
  static const bool COLLAPSE_GEP = false;
  // whether all BitCast will be collapse
  static const bool COLLAPSE_BITCAST = true;
  // whether type information is necessary
  // we need type information to build the memory layout
  static const bool NEED_TYPE_INFO = true;
};

} // namespace xray
