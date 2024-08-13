//
// Created by peiming on 11/5/19.
//
#ifndef ASER_PTA_CALLSITE_H
#define ASER_PTA_CALLSITE_H

#include <llvm/IR/Instructions.h>

#include "aser/Util/Util.h"

namespace aser {

// wrapper around llvm::CallSite,
// but resolve constant expression evaluated to a function
class CallSite {
private:
  const llvm::CallBase *CB;
  static const llvm::Function *resolveTargetFunction(const llvm::Value *);

public:
  explicit CallSite(const llvm::Instruction *I)
      : CB(llvm::dyn_cast<llvm::CallBase>(I)) {}

  [[nodiscard]] inline bool isCallOrInvoke() const { return CB != nullptr; }

  [[nodiscard]] inline bool isIndirectCall() const {
    auto V = CB->getCalledOperand();
    if (llvm::isa<llvm::Function>(V->stripPointerCasts())) {
      return false;
    }

    if (CB->isIndirectCall()) {
      return true;
    }

    if (auto C = llvm::dyn_cast<llvm::Constant>(V)) {
      if (C->isNullValue()) {
        return true;
      }
    }

    return false;
  }

  [[nodiscard]] inline const llvm::Value *getCalledValue() const {
    return CB->getCalledOperand();
  }

  [[nodiscard]] inline const llvm::Function *getCalledFunction() const {
    return this->getTargetFunction();
  }

  [[nodiscard]] inline const llvm::Function *getTargetFunction() const {
    if (this->isIndirectCall()) {
      return nullptr;
    }

    auto targetFunction = CB->getCalledFunction();
    if (targetFunction != nullptr) {
      return targetFunction;
    }

    return resolveTargetFunction(CB->getCalledOperand());
  }

  [[nodiscard]] inline const llvm::Value *getReturnedArgOperand() const {
    return CB->getReturnedArgOperand();
  }

  [[nodiscard]] inline const llvm::Instruction *getInstruction() const {
    return CB;
  }

  [[nodiscard]] unsigned int getNumArgOperands() const {
    return CB->arg_size();
  }

  const llvm::Value *getArgOperand(unsigned int i) const {
    return CB->getArgOperand(i);
  }

  inline auto args() const -> decltype(CB->args()) { return CB->args(); }

  [[nodiscard]] inline auto arg_begin() const -> decltype(CB->arg_begin()) {
    return CB->arg_begin();
  }

  [[nodiscard]] inline auto arg_end() const -> decltype(CB->arg_end()) {
    return CB->arg_end();
  }

  inline llvm::Type *getType() const { return CB->getType(); };
};

} // namespace aser

#endif
