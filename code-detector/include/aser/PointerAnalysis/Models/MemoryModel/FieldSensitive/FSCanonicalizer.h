//
// Created by peiming on 1/2/20.
//

#ifndef ASER_PTA_FSCANONICALIZER_H
#define ASER_PTA_FSCANONICALIZER_H

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Value.h>

namespace aser {

// Canonicalizer should not collapse alloca, load, store, phi .. instructions
class FSCanonicalizer {
public:
    static const llvm::Value *canonicalize(const llvm::Value *V);
};

}  // namespace aser


#endif  // ASER_PTA_FSCANONICALIZER_H
