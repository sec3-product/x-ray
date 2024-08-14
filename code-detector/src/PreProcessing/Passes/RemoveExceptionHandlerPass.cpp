//
// Created by peiming on 1/10/20.
//
#include "PreProcessing/Passes/RemoveExceptionHandlerPass.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Util/Log.h"

using namespace aser;
using namespace llvm;

static BasicBlock *createUnReachableBB(Function &F) {
  auto BB = BasicBlock::Create(F.getContext(), "aser.unreachable", &F);
  IRBuilder<> builder(BB);
  builder.CreateUnreachable();

  return BB;
}

bool RemoveExceptionHandlerPass::doInitialization(Module &M) {
  LOG_DEBUG("Processing Exception Handlers");
  return false;
}

bool RemoveExceptionHandlerPass::runOnFunction(Function &F) {
  bool changed = false;
  BasicBlock *unReachableBB = nullptr;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto invokeInst = dyn_cast<InvokeInst>(&I)) {
        if (unReachableBB == nullptr) {
          unReachableBB = createUnReachableBB(F);
        }

        changed = true;
        invokeInst->setUnwindDest(unReachableBB);
      }
    }
  }

  if (changed) {
    EliminateUnreachableBlocks(F);
  }

  return changed;
}

char RemoveExceptionHandlerPass::ID = 0;
static RegisterPass<RemoveExceptionHandlerPass>
    REH("", "Remove Exception Handling Code in IR", false, /*CFG only*/
        false /*is analysis*/);
