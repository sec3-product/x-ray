//
// Created by peiming on 11/1/19.
//
#ifndef ASER_PTA_POINTER_H
#define ASER_PTA_POINTER_H

#include "aser/Util/Util.h"

// represent pointers in programs
#include <llvm/ADT/Hashing.h>

namespace aser {

template <typename ctx> class CGPtrNode;

template <typename ctx> class Pointer {
private:
  const ctx *context;
  const llvm::Value *value; // can be an instruction or global variables
  CGPtrNode<ctx> *ptrNode = nullptr;

public:
  Pointer(const ctx *context, const llvm::Value *value)
      : context(context), value(value) {}

  Pointer(const Pointer<ctx> &) = delete;
  Pointer(Pointer<ctx> &&) = delete; // can not be moved! as the address is held
                                     // by the corresponding callgraph node
  Pointer<ctx> &operator=(const Pointer<ctx> &) = delete;
  Pointer<ctx> &operator=(Pointer<ctx> &&) = delete;

  [[nodiscard]] inline const ctx *getContext() const { return context; }

  [[nodiscard]] inline const llvm::Value *getValue() const { return value; }

  inline void setPtrNode(CGPtrNode<ctx> *node) {
    assert(ptrNode == nullptr); // can only be set once
    this->ptrNode = node;
  }

  [[nodiscard]] inline CGPtrNode<ctx> *getPtrNode() const { return ptrNode; }
};

// for container operation
template <typename ctx>
bool operator==(const Pointer<ctx> &lhs, const Pointer<ctx> &rhs) {
  return lhs.getContext() == rhs.getContext() &&
         lhs.getValue() == rhs.getValue();
}

template <typename ctx>
bool operator<(const Pointer<ctx> &lhs, const Pointer<ctx> &rhs) {
  if (lhs.getValue() == rhs.getValue()) {
    return lhs.getContext() < rhs.getContext();
  }
  return lhs.getValue() < rhs.getValue();
}

} // namespace aser

namespace std {
// only hash context and value
template <typename ctx> struct hash<aser::Pointer<ctx>> {
  size_t operator()(const aser::Pointer<ctx> &ptr) const {
    llvm::hash_code seed = llvm::hash_value(ptr.getContext());
    llvm::hash_code hash = llvm::hash_combine(ptr.getValue(), seed);
    return hash_value(hash);
  }
};

} // namespace std

#endif