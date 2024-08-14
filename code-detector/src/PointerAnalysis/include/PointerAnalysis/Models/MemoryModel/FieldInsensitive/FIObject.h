//
// Created by peiming on 11/3/19.
//
#ifndef ASER_PTA_FIOBJECT_H
#define ASER_PTA_FIOBJECT_H

#include "PointerAnalysis/Models/MemoryModel/AllocSite.h"

namespace aser {

template <typename ctx, typename ObjT> class CGObjNode;
template <typename ctx> class FIMemModel;

template <typename ctx> class FIObject {
private:
  using CT = CtxTrait<ctx>;
  using ObjNode = CGObjNode<ctx, FIObject<ctx>>;

  // the allocation site
  // for field-insentive PTA, each allocation site correponding to one object.
  const AllocSite<ctx> allocSite;
  ObjNode *objNode = nullptr;

  inline void setObjNode(ObjNode *node) {
    assert(objNode == nullptr);
    objNode = node;
  }

public:
  FIObject(const ctx *c, const llvm::Value *v, const AllocKind t)
      : allocSite(c, v, t){};
  // can not be moved/copied
  FIObject(const FIObject<ctx> &) = delete;
  FIObject(FIObject<ctx> &&) = delete;
  FIObject<ctx> &operator=(const FIObject<ctx> &) = delete;
  FIObject<ctx> &operator=(FIObject<ctx> &&) = delete;

  [[nodiscard]] inline const AllocSite<ctx> &getAllocSite() const {
    return this->allocSite;
  }

  [[nodiscard]] inline const ctx *getContext() const {
    return this->getAllocSite().getContext();
  }

  [[nodiscard]] inline const llvm::Value *getValue() const {
    return this->getAllocSite().getValue();
  }

  [[nodiscard]] inline ObjNode *getObjNode() const { return objNode; }

  [[nodiscard]] inline AllocKind getAllocType() const {
    return this->getAllocSite().getAllocType();
  }

  [[nodiscard]] inline bool isFunction() const {
    return this->getAllocType() == AllocKind::Functions;
  }

  [[nodiscard]] inline const llvm::Type *getType() const {
    return this->allocSite.getValue()->getType();
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

  [[nodiscard]] inline std::string toString(bool detailed = true) const {
    if (detailed) {
      std::string ctxStr = CT::toString(getContext(), detailed);
      llvm::raw_string_ostream os(ctxStr);
      os << "\n" << *getValue();
      return os.str();
    } else {
      if (getValue()->hasName()) {
        return getValue()->getName();
      }
      return "";
    }
  }

  friend FIMemModel<ctx>;
  friend CGObjNode<ctx, FIObject<ctx>>;
};

template <typename ctx>
bool operator==(const FIObject<ctx> &lhs, const FIObject<ctx> &rhs) {
  return lhs.getValue() == rhs.getValue() &&
         lhs.getContext() == rhs.getContext();
}

} // namespace aser

// for container operation
namespace std {

template <typename ctx> struct hash<aser::FIObject<ctx>> {
  size_t operator()(const aser::FIObject<ctx> &obj) const {
    llvm::hash_code seed = llvm::hash_value(obj.getContext());
    llvm::hash_code hash = llvm::hash_combine(obj.getValue(), seed);
    return hash_value(hash);
  }
};

} // namespace std

#endif