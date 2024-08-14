//
// Created by peiming on 4/15/20.
//
#include "PreProcessing/Passes/RemoveASMInstPass.h"

//#include <llvm/IR/CallSite.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/NoFolder.h>

using namespace aser;
using namespace llvm;

static bool destroyASMInst(Function &F, IRBuilder<NoFolder> &builder) {
  std::vector<Instruction *> removeThese;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto callInst = dyn_cast<CallBase>(&I)) {
        auto V = callInst->getCalledOperand();
        if (V != nullptr && isa<InlineAsm>(V)) {
          removeThese.push_back(callInst);
        }
      }
    }
  }

  for (auto I : removeThese) {
    I->replaceAllUsesWith(llvm::UndefValue::get(I->getType()));
    I->eraseFromParent();
  }
  return !removeThese.empty();
}

bool RemoveASMInstPass::runOnFunction(llvm::Function &F) {
  IRBuilder<NoFolder> builder(F.getContext());

  bool changed = destroyASMInst(F, builder);
  return changed;
}

char RemoveASMInstPass::ID = 0;
static RegisterPass<RemoveASMInstPass> CIP("", "Remove ASM Instruction",
                                           true, /*CFG only*/
                                           false /*is analysis*/);
