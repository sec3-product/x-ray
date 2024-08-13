//
// Created by peiming on 1/9/20.
//
#ifndef ASER_PTA_CONSTEXPRVISITOR_H
#define ASER_PTA_CONSTEXPRVISITOR_H

#include <llvm/IR/Constant.h>
#include <llvm/IR/Instructions.h>

#include "aser/Util/Log.h"

namespace aser {

/// This class defines a simple visitor class that may be used for
/// various SCEV analysis purposes.
template <typename SC, typename RetVal = void> struct ConstExprVisitor {
  RetVal visit(const llvm::Constant *C) {
    if (auto CE = llvm::dyn_cast<llvm::ConstantExpr>(C)) {
      switch (CE->getOpcode()) {
      case llvm::Instruction::GetElementPtr:
        return ((SC *)this)->visitGEP(llvm::dyn_cast<llvm::GEPOperator>(CE));
      case llvm::Instruction::BitCast:
        return ((SC *)this)
            ->visitBitCast(llvm::dyn_cast<llvm::BitCastOperator>(CE));
      default:
        // TODO handle more!
        LOG_ERROR("unhandled constant expression. type={}", *C);
        llvm_unreachable("h SCEV type!");
      }
    } else {
      return ((SC *)this)->visitConstant(C);
    }
  }
};

} // namespace aser

#endif // ASER_PTA_CONSTEXPRVISITOR_H
