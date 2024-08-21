//
// Created by peiming on 1/6/20.
//

#ifndef ASER_PTA_CANONICALIZEGEPPASS_H
#define ASER_PTA_CANONICALIZEGEPPASS_H

#include <llvm/Pass.h>

// TO canoicalize GEP instruction (required only by Field-Sensitive Pointer
// Analysis)

// 1st, turn
// %v = getelementptr (getelementptr idx1 ...) idx0 ...
// into
// %tmp = getelementptr idx1 ...
// %v = getelementptr %tmp, idx0 ...

// 2nd, split getelementptr that uses variable to index
// e.g.,
// %gep = getelementptr %base, 0, 0, %v1, 2, 3, %v2
// will be splitted into
// %gep1 = getelementptr %base, 0, 0;
// %gep2 = getelementptr %gep1, 0, %v1;
// %gep3 = getelementptr %gep2, 0, 2, 3;
// %gep4 = getelementptr %gep3, 0, %v2;

// 3rd, for every uses of inline asm
// change it to undef value

namespace xray {

class CanonicalizeGEPPass : public llvm::FunctionPass {

public:
  static char ID;
  explicit CanonicalizeGEPPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
  bool doInitialization(llvm::Module &M) override;
};

} // namespace xray

#endif // ASER_PTA_CANONICALIZEGEPPASS_H
