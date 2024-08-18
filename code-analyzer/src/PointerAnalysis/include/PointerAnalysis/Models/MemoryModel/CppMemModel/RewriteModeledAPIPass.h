//
// Created by peiming on 7/23/20.
//

#ifndef ASER_PTA_REWRITEMODELEDAPIPASS_H
#define ASER_PTA_REWRITEMODELEDAPIPASS_H

#include <llvm/Pass.h>
namespace aser {
namespace cpp {

class RewriteModeledAPIPass : public llvm::FunctionPass {
public:
  static char ID;
  explicit RewriteModeledAPIPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};

} // namespace cpp
} // namespace aser

#endif // ASER_PTA_REWRITEMODELEDAPIPASS_H
