//
// Created by peiming on 12/18/19.
//

#ifndef ASER_PTA_MEMBLOCK_H
#define ASER_PTA_MEMBLOCK_H

#include <llvm/ADT/IndexedMap.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/CommandLine.h>

#include "PointerAnalysis/Models/MemoryModel/AllocSite.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/FSObject.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/Layout/MemLayout.h"
#include "Util/Util.h"

extern cl::opt<bool> USE_MEMLAYOUT_FILTERING;

namespace xray {

namespace cpp {
template <typename ctx> class CppMemModel;
}
// forward declarations
template <typename ctx> class FSMemModel;
template <typename ctx> class FIMemBlock;
template <typename ctx> class ScalarMemBlock;
template <typename ctx> class AggregateMemBlock;

size_t getGEPStepSize(const llvm::GetElementPtrInst *GEP,
                      const llvm::DataLayout &DL);

bool isArrayExistAtOffset(const std::map<size_t, ArrayLayout *> &arrayMap,
                          size_t pOffset, size_t elementSize);

enum class MemBlockKind {
  // Array, Structure
  Aggregate = 0,
  // field-insensitive blocks (maybe a heap allocation site but we can not
  // determine the type)
  FIBlock = 1,
  // primitive type (can not be indexed)
  // e.g., float, integer or pointer
  Scalar = 2,
};

// each memory block corresponding to one allocation site
template <typename ctx> class MemBlock {
private:
  const MemBlockKind kind;
  const AllocSite<ctx> allocSite;

protected:
  MemBlock(const ctx *c, const llvm::Value *v, const AllocKind t,
           const MemBlockKind kind)
      : allocSite(c, v, t), kind(kind){};

public:
  // nullptr if the memory block can not be indexed (scalar memory object) and
  // the offset is non-zero else the corresponding object
  [[nodiscard]] inline const FSObject<ctx> *getObjectAt(size_t offset) {
    switch (kind) {
    case MemBlockKind::Aggregate:
      return static_cast<AggregateMemBlock<ctx> *>(this)->indexMemoryBlock(
          offset, false);
    case MemBlockKind::FIBlock:
      return &static_cast<FIMemBlock<ctx> *>(this)->object;
    case MemBlockKind::Scalar: {
      if (offset == 0) {
        return static_cast<ScalarMemBlock<ctx> *>(this)->object.get();
      }
      return nullptr;
    }
    }
    llvm_unreachable("does not support yet");
  }

  [[nodiscard]] inline const FSObject<ctx> *getPtrObjectAt(size_t offset) {
    switch (kind) {
    case MemBlockKind::Aggregate:
      return static_cast<AggregateMemBlock<ctx> *>(this)->indexMemoryBlock(
          offset, true);
    case MemBlockKind::FIBlock:
      return &static_cast<FIMemBlock<ctx> *>(this)->object;
    case MemBlockKind::Scalar: {
      if (offset == 0) {
        return static_cast<ScalarMemBlock<ctx> *>(this)->object.get();
      }
    }
    default:
      return nullptr;
    }
  }

  [[nodiscard]] inline const llvm::Type *
  getOffsetType(size_t pOffset, const llvm::DataLayout &DL) {
    switch (kind) {
    case MemBlockKind::Aggregate:
      return static_cast<AggregateMemBlock<ctx> *>(this)->getOffsetType(pOffset,
                                                                        DL);
    case MemBlockKind::FIBlock:
    case MemBlockKind::Scalar: {
      // scalar object and field-insensitive object can not be queried
      return nullptr; //
    }
    }
    llvm_unreachable("does not support yet");
  }

  void setImmutable() {
    switch (kind) {
    case MemBlockKind::Aggregate:
      return static_cast<AggregateMemBlock<ctx> *>(this)->setImmutable();
    case MemBlockKind::FIBlock:
      return static_cast<FIMemBlock<ctx> *>(this)->setImmutable();
    case MemBlockKind::Scalar:
      return static_cast<ScalarMemBlock<ctx> *>(this)->setImmutable();
    }
  }

  [[nodiscard]] FSObject<ctx> *indexObject(const FSObject<ctx> *obj,
                                           const llvm::Instruction *idx,
                                           const llvm::DataLayout &DL,
                                           bool ensurePtr) {
    assert(obj->memBlock == this);
    switch (kind) {
    case MemBlockKind::Aggregate:
      return static_cast<AggregateMemBlock<ctx> *>(this)->indexObject(
          obj, idx, DL, ensurePtr);
    case MemBlockKind::FIBlock:
      return &static_cast<FIMemBlock<ctx> *>(this)->object;
    case MemBlockKind::Scalar: {
      // scalar object can not be indexed
      return nullptr;
    }
    default:
      return nullptr;
    }
  }

  [[nodiscard]] inline const ctx *getContext() const {
    return allocSite.getContext();
  }

  [[nodiscard]] inline const llvm::Module *getLLVMModule() const {
    return this->allocSite.getLLVMModule();
  }

  [[nodiscard]] inline const llvm::Value *getValue() const {
    return allocSite.getValue();
  }

  [[nodiscard]] inline AllocKind getAllocKind() const {
    return allocSite.getAllocType();
  }

  [[nodiscard]] inline const AllocSite<ctx> &getAllocSite() const {
    return allocSite;
  }

  [[nodiscard]] inline bool validateStepSize(size_t pOffset, size_t stepSize) {
    switch (kind) {
    case MemBlockKind::Aggregate:
      return static_cast<AggregateMemBlock<ctx> *>(this)->validateStepSize(
          pOffset, stepSize);
    case MemBlockKind::FIBlock:
      return true;
    case MemBlockKind::Scalar: {
      // can not index Scalar Object.
      return false;
    }
    }
  }

  [[nodiscard]] inline bool isFIBlock() {
    return kind == MemBlockKind::FIBlock;
  }

  [[nodiscard]] inline std::string
  getFieldAccessPath(size_t pOffset, const llvm::StringRef separator) const {
    if (kind == MemBlockKind::Aggregate) {
      return static_cast<const AggregateMemBlock<ctx> *>(this)
          ->getFieldAccessPath(pOffset, separator);
    }
    return "";
  }

  friend FSObject<ctx>;
};

// memory block with unknown type
template <typename ctx> class FIMemBlock : public MemBlock<ctx> {
private:
  FSObject<ctx> object;

public:
  FIMemBlock(const ctx *c, const llvm::Value *v, const AllocKind t)
      : MemBlock<ctx>(c, v, t, MemBlockKind::FIBlock), object(this) {}

  inline void setImmutable() { object.setImmutable(); }

  friend MemBlock<ctx>;
};

// memory block that stores primitive type (single pointer or a single
// integer..)
template <typename ctx> class ScalarMemBlock : public MemBlock<ctx> {
private:
  std::unique_ptr<FSObject<ctx>> object;

public:
  ScalarMemBlock(const ctx *c, const llvm::Value *v, const AllocKind t)
      : MemBlock<ctx>(c, v, t, MemBlockKind::Scalar) {
    object = std::make_unique<FSObject<ctx>>(this);
  }

  ScalarMemBlock(const ctx *c, const llvm::Value *v, const AllocKind t,
                 FSObject<ctx> *obj)
      : MemBlock<ctx>(c, v, t, MemBlockKind::Scalar) {
    object.reset(obj);
  }

  inline void setImmutable() { object->setImmutable(); }

  friend MemBlock<ctx>;
};

// array, structure
template <typename ctx> class AggregateMemBlock : public MemBlock<ctx> {
private:
  bool isImmutable;
  // the allocation site of the memory block
  const MemLayout *layout;
  // this vector is indexed by logical indices
  std::vector<std::unique_ptr<FSObject<ctx>>> fieldObjs;
  std::vector<const llvm::Type *> fieldType;

  const llvm::Type *getOffsetType(size_t pOffset, const llvm::DataLayout &DL) {
    size_t lOffset = layout->indexPhysicalOffset(pOffset);
    int fieldNum = layout->getLogicalOffset(lOffset);
    if (fieldType[fieldNum]) {
      // simply return cached type
      return fieldType[fieldNum];
    }

    const llvm::Type *rootType = this->layout->getType();

    if (pOffset == 0) {
      assert(lOffset == 0 && fieldNum == 0);
      fieldType[0] = rootType;
      return rootType;
    } else {
      auto type = getTypeAtOffset(rootType, pOffset, DL, false);
      assert(type != nullptr);
      fieldType[fieldNum] = type;
      return type;
    }
  }

  FSObject<ctx> *indexObject(const FSObject<ctx> *obj,
                             const llvm::Instruction *idx,
                             const llvm::DataLayout &DL, bool ensurePtr) {
    auto gep = llvm::dyn_cast<llvm::GetElementPtrInst>(idx);
    if (gep == nullptr) {
      // only gep can be used to index AggregateMemBlock
      return nullptr;
    }

    if (gep->hasAllConstantIndices()) {
      // simple case, indexing object using constant offset.
      const llvm::Type *fieldTy = obj->getOffsetType(DL);
      auto offset = llvm::APInt(DL.getIndexTypeSizeInBits(gep->getType()), 0);

      gep->accumulateConstantOffset(DL, offset);
      if (!USE_MEMLAYOUT_FILTERING) {
        // TODO: configuration for this
        // type filtering on the indexed object
        if (offset.getSExtValue() >= 0) {
          // do not filter out backward index
          if (!gep->getPointerOperandType()
                   ->getPointerElementType()
                   ->isIntegerTy(8)) {
            // do not filter out gep i8 *? which is void * in C
            if (!isZeroOffsetTypeInRootType(
                    fieldTy,
                    gep->getPointerOperandType()->getPointerElementType(),
                    DL)) {
              return nullptr;
            }
          }
        }
      }

      int64_t off = obj->getPOffset() + offset.getSExtValue();
      if (off < 0) {
        return nullptr;
      }

      auto result = this->indexMemoryBlock(off, ensurePtr);
      if (result == nullptr) {
        return nullptr;
      }
      return result;

    } else {
      size_t stepSize = getGEPStepSize(gep, DL);

      if (stepSize == std::numeric_limits<size_t>::max()) {
        // corner case, whether index variable is undef value
        return nullptr;
      }
      // when we encounter an object that indexed by variable, we need to ensure
      // that we are indexing an array although C++/C allows using variable to
      // index structure, but it is rare in practices, and we ignore it at the
      // current stage.
      if (this->validateStepSize(obj->getPOffset(), stepSize)) {
        return const_cast<FSObject<ctx> *>(obj);
      } else {
        return nullptr;
      }
    }
  }

  // force the pOffset be initialized with the given object (this object can be
  // a special object such as containers)
  void initializeOffsetWith(size_t pOffset, const FSObject<ctx> *cobj) {
    auto *obj = const_cast<FSObject<ctx> *>(cobj);
    if (pOffset == 0) {
      // fast path
      assert(fieldObjs[0].get() == nullptr);
      fieldObjs[0].reset(obj);
      return;
    }

    // 1st, convert physical offset to layout offset.
    size_t lOffset = layout->indexPhysicalOffset(pOffset);
    int fieldNum = layout->getLogicalOffset(lOffset);
    assert(fieldNum > 0 && fieldObjs[fieldNum].get() == nullptr);
    fieldObjs[fieldNum].reset(obj);
  }

  // offset is the physical offset
  FSObject<ctx> *indexMemoryBlock(size_t pOffset, bool ensurePtr = false) {
    if (pOffset == 0) {
      // fast path
      if (ensurePtr ? layout->offsetIsPtr(0) : true) {
        if (fieldObjs[0].get() == nullptr) {
          fieldObjs[0] =
              std::unique_ptr<FSObject<ctx>>(new FSObject<ctx>(this));
        }
        return fieldObjs[0].get();
      }
      return nullptr;
    }

    // 1st, convert physical offset to layout offset.
    size_t lOffset = layout->indexPhysicalOffset(pOffset);
    if (lOffset != std::numeric_limits<size_t>::max()) {
      int fieldNum = layout->getLogicalOffset(lOffset);
      if (fieldNum >= 0 && (ensurePtr ? layout->offsetIsPtr(lOffset) : true)) {
        // TODO: this is the bottle neck
        // 2nd, index the memory block, return cached object or create a new
        // object.
        if (fieldObjs[fieldNum].get() == nullptr) {
          fieldObjs[fieldNum] =
              std::unique_ptr<FSObject<ctx>>(new FSObject<ctx>(this, pOffset));
          if (isImmutable) {
            fieldObjs[fieldNum]->setImmutable();
          }
        }
        return fieldObjs[fieldNum].get();
      }
    }
    // the computed layout offset can not be indexed
    return nullptr;
  }

  [[nodiscard]] inline std::string
  getFieldAccessPath(size_t pOffset, llvm::StringRef separator) const {
    return this->layout->getFieldAccessPath(this->getLLVMModule(), pOffset,
                                            separator);
  }

  [[nodiscard]] inline bool validateStepSize(size_t pOffset,
                                             size_t stepSize) const {
    return isArrayExistAtOffset(layout->getSubArrayMap(), pOffset, stepSize);
  }

public:
  AggregateMemBlock(const ctx *c, const llvm::Value *v, const AllocKind t,
                    const MemLayout *layout)
      : MemBlock<ctx>(c, v, t, MemBlockKind::Aggregate), layout(layout),
        isImmutable(false) {
    // insert the first objects, other object are lazily initialized
    // objectMap.try_emplace(0 /*key*/, this, 0, 0);
    if (layout->getNumIndexableElem() == 0) {
      fieldObjs.resize(
          1); // model zero-sized object as if it has at least one element
      fieldType.resize(1);
    } else {
      fieldObjs.resize(layout->getNumIndexableElem());
      fieldType.resize(layout->getNumIndexableElem());
    }
  }

  void setImmutable() {
    isImmutable = true;
    for (std::unique_ptr<FSObject<ctx>> &obj : fieldObjs) {
      if (obj.get() != nullptr) {
        obj->setImmutable();
      }
    }
  }

  friend MemBlock<ctx>;
  friend FSObject<ctx>;
  friend cpp::CppMemModel<ctx>;
};

} // namespace xray

#endif // ASER_PTA_MEMBLOCK_H
