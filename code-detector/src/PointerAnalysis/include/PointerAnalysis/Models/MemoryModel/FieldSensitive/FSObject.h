//
// Created by peiming on 12/19/19.
//

#ifndef ASER_PTA_FSOBJECT_H
#define ASER_PTA_FSOBJECT_H

#include <llvm/IR/GlobalValue.h>
#include <llvm/Support/Casting.h>

#include "PointerAnalysis/Context/CtxTrait.h"
#include "PointerAnalysis/Graph/ConstraintGraph/ConstraintGraph.h"
#include "PointerAnalysis/Models/MemoryModel/AllocSite.h"
#include "PointerAnalysis/Models/MemoryModel/Object.h"

namespace aser {

template <typename ctx, typename ObjT> class CGObjNode;
template <typename ctx> class FSMemModel;
template <typename ctx> class MemBlock;

enum class ObjectKind {
  Normal, // normal objects
  Special // special objects
};

template <typename ctx> class FSObject : public Object<ctx, FSObject<ctx>> {
protected:
  MemBlock<ctx> *memBlock;

private:
  using CT = CtxTrait<ctx>;
  using Super = Object<ctx, FSObject<ctx>>;
  // the physical offset (relative to the first byte of the allocated memory)
  size_t pOffset;

  bool allowIndex;
  const ObjectKind kind;

public:
  using ObjNode = CGObjNode<ctx, FSObject<ctx>>;

  explicit FSObject(MemBlock<ctx> *memBlock,
                    ObjectKind kind = ObjectKind::Normal,
                    bool allowIndex = true)
      : Super(), memBlock(memBlock), pOffset(0), kind(kind),
        allowIndex(allowIndex){};

  FSObject(MemBlock<ctx> *memBlock, size_t pOffset,
           ObjectKind kind = ObjectKind::Normal, bool allowIndex = true)
      : Super(), memBlock(memBlock), pOffset(pOffset), kind(kind),
        allowIndex(allowIndex){};

  template <typename PT> void initWithNode(ConstraintGraph<ctx> *CG) {
    auto ret = CG->template addCGNode<ObjNode, PT>(this);
    this->setObjNode(ret);
  }

  inline void setMemBlock(MemBlock<ctx> *block) {
    assert(memBlock == nullptr);
    this->memBlock = block;
  }

  // can not be moved/copied
  FSObject(const FSObject<ctx> &) = delete;
  FSObject(FSObject<ctx> &&) = delete;
  FSObject<ctx> &operator=(const FSObject<ctx> &) = delete;
  FSObject<ctx> &operator=(FSObject<ctx> &&) = delete;

  inline const FSObject<ctx> *indexObject(const llvm::Instruction *idx,
                                          const llvm::DataLayout &DL) const {
    if (allowIndex) {
      return this->memBlock->indexObject(this, idx, DL, false);
    }
    return nullptr;
  }

  inline const FSObject<ctx> *indexPtrObject(const llvm::Instruction *idx,
                                             const llvm::DataLayout &DL) const {
    if (allowIndex) {
      return this->memBlock->indexObject(this, idx, DL, true);
    }
    return nullptr;
  }

  virtual bool processSpecial(CGNodeBase<ctx> *src,
                              CGNodeBase<ctx> *dst) const {
    return false;
  }

  [[nodiscard]] inline const AllocSite<ctx> &getAllocSite() const {
    return this->memBlock->getAllocSite();
  }

  [[nodiscard]] inline const ctx *getContext() const {
    return this->getAllocSite().getContext();
  }

  [[nodiscard]] inline const llvm::Value *getValue() const {
    return this->getAllocSite().getValue();
  }

  [[nodiscard]] inline AllocKind getAllocType() const {
    return this->getAllocSite().getAllocType();
  }

  [[nodiscard]] inline const llvm::Type *
  getOffsetType(const llvm::DataLayout &DL) const {
    return this->memBlock->getOffsetType(pOffset, DL);
  }

  [[nodiscard]] inline bool isFunction() const {
    return this->getAllocType() == AllocKind::Functions;
  }

  [[nodiscard]] inline const llvm::Type *getType() const {
    return this->getAllocSite().getValue()->getType();
  }

  [[nodiscard]] inline bool isGlobalObj() const {
    return this->getAllocType() == AllocKind::Globals;
  }

  [[nodiscard]] inline bool isStackObj() const {
    return this->getAllocType() == AllocKind::Stack;
  }

  [[nodiscard]] inline bool isHeapObj() const {
    return this->getAllocType() == AllocKind::Heap;
  }

  [[nodiscard]] inline bool isAnonObj() const {
    return this->getAllocType() == AllocKind::Anonymous;
  }

  [[nodiscard]] inline bool isFIObject() const {
    return this->memBlock->isFIBlock();
  }

  [[nodiscard]] inline bool isSpecialObject() const {
    return this->kind == ObjectKind::Special;
  }

  inline size_t getPOffset() const { return pOffset; }

  [[nodiscard]] inline std::string
  getFieldAccessPath(const llvm::StringRef separator = "->") const {
    return this->memBlock->getFieldAccessPath(pOffset, separator);
  }

  [[nodiscard]] inline std::string toString(bool detailed = true) const {
    if (detailed) {
      std::string ctxStr = CT::toString(getContext(), detailed);
      llvm::raw_string_ostream os(ctxStr);
      if (getAllocType() == AllocKind::Anonymous) {
        os << "\n Anonymous";
      } else if (getValue()->hasName()) {
        os << CtxTrait<ctx>::toString(getContext(), true) << "\n";
        if (llvm::isa<llvm::GlobalValue>(getValue())) {
          os << getValue()->getName();
        } else {
          os << *getValue() << "\n"; //->getName();
        }
      } else {
        os << CtxTrait<ctx>::toString(getContext(), true) << "\n";
        os << *getValue() << "\n";
      }
      if (auto inst = llvm::dyn_cast_or_null<Instruction>(getValue())) {
        os << "@" << inst->getFunction()->getName();
      }
      os << "\nOffset:" << pOffset << getFieldAccessPath();

      // os << "\n" << getPTASourceLocSnippet(getValue());  // JEFF

      return os.str();
    } else {
      if (getValue()->hasName()) {
        return getValue()->getName().str();
      }
      return "";
    }
  }

  friend FSMemModel<ctx>;
  friend MemBlock<ctx>;
  friend ObjNode;
};

} // namespace aser

#endif // ASER_PTA_FSOBJECT_H
