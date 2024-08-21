//
// Created by peiming on 11/3/19.
//
#ifndef ASER_PTA_FICANONICALIZER_H
#define ASER_PTA_FICANONICALIZER_H

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Value.h>

namespace xray {

// Canonicalizer should not collapse alloca, load, store, phi .. instructions
class FICanonicalizer {
public:
  static const llvm::Value *canonicalize(const llvm::Value *V);
};

} // namespace xray

#endif