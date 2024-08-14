//
// Created by peiming on 8/13/20.
//

#ifndef ASER_PTA_TRANSFORMCALLINSTBITCASTPASS_H
#define ASER_PTA_TRANSFORMCALLINSTBITCASTPASS_H

#include <llvm/Pass.h>

namespace aser {

class TransformCallInstBitCastPass : public llvm::ModulePass {
public:
  static char ID;
  explicit TransformCallInstBitCastPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &F) override;
};

} // namespace aser

#endif // ASER_PTA_TRANSFORMCALLINSTBITCASTPASS_H
