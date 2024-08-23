//
// Created by peiming on 11/19/19.
//
#ifndef ASER_PTA_NOCTX_H
#define ASER_PTA_NOCTX_H

#include "CtxTrait.h"
#include <string>

// forward declaration
namespace llvm {
class Instruction;
}

namespace xray {

// for context insensitive PTA
using NoCtx = std::nullptr_t;

template <> struct CtxTrait<NoCtx> {
  // No runtime overhead when
  constexpr static const NoCtx *contextEvolve(const NoCtx *,
                                              const llvm::Instruction *) {
    return nullptr;
  }
  constexpr static const NoCtx *getInitialCtx() { return nullptr; }
  constexpr static const NoCtx *getGlobalCtx() { return nullptr; }

  inline static std::string toString(const NoCtx *, bool detailed = false) {
    return "<Empty>";
  }
  inline static void release(){};
};

} // namespace xray

#endif // ASER_PTA_NOCTX_H
