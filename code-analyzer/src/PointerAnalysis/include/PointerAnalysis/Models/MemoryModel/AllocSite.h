//
// Created by peiming on 10/22/19.
//
#ifndef ASER_PTA_ALLOCSITE_H
#define ASER_PTA_ALLOCSITE_H

#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Value.h>

namespace xray {

// forward declaration
template <typename ctx> class FIObject;

template <typename ctx> class MemBlock;

enum class AllocKind : uint8_t {
  Stack = 0,
  Heap = 1,
  Functions = 2,
  Globals = 3,
  Anonymous = 4, // logically exist, but does not have a concrete allocation
                 // site (e.g., argument passed to main)
  Null = 5,
  Universal = 6,
};

template <typename ctx> class AllocSite {
private:
  // allocated context
  const ctx *const context;
  // the llvm value that allocates the memory block
  const llvm::Value *const value;
  // allocate type
  const AllocKind type;
  // can only be created by allocating a new memory block
  // there is only one instance of AllocSite per ctx+value
  AllocSite(const ctx *c, const llvm::Value *v, const AllocKind t)
      : context(c), value(v), type(t) {
    assert(value != nullptr || type == AllocKind::Anonymous);
  }

public:
  // making a copy of an allocation site is not allowed.
  AllocSite(const AllocSite<ctx> &) = delete;
  AllocSite(AllocSite<ctx> &&) = delete;
  AllocSite<ctx> &operator=(const AllocSite<ctx> &) = delete;
  AllocSite<ctx> &operator=(AllocSite<ctx> &&) = delete;

  // getters
  [[nodiscard]] inline const ctx *getContext() const { return this->context; }

  [[nodiscard]] inline const llvm::Value *getValue() const {
    return this->value;
  }

  [[nodiscard]] inline AllocKind getAllocType() const { return this->type; }

  [[nodiscard]] inline const llvm::Module *getLLVMModule() const {
    if (value == nullptr) {
      return nullptr;
    }

    if (auto I = llvm::dyn_cast<llvm::Instruction>(value)) {
      return I->getModule();
    } else if (auto G = llvm::dyn_cast<llvm::GlobalValue>(value)) {
      return G->getParent();
    }

    // what else can it be??
    return nullptr;
  }

  friend FIObject<ctx>;
  friend MemBlock<ctx>;
};

} // namespace xray

#endif