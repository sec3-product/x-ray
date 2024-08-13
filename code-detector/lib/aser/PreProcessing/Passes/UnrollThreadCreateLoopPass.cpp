//
// Created by Yanze on 7/14/20.
//
#include "aser/PreProcessing/Passes/UnrollThreadCreateLoopPass.h"

#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/UnrollLoop.h>

#include "aser/PointerAnalysis/Program/CallSite.h"
#include "aser/Util/Log.h"

extern bool isConfiguredIndirectAPI(const llvm::Function *func);

using namespace aser;
using namespace llvm;

std::set<StringRef> UnrollThreadCreateLoopPass::THREAD_CREATE_API = {
    "pthread_create"};

static bool containsAny(const StringRef src,
                        const std::set<llvm::StringRef> &list) {
  for (auto e : list) {
    if (src.contains(e)) {
      return true;
    }
  }
  return false;
}

bool UnrollThreadCreateLoopPass::runOnLoop(Loop *L, LPPassManager &LPM) {
  // NOTE: right now we do not handle nested loops
  if (L->getLoopDepth() != 1)
    return false;
  auto F = L->getHeader()->getParent();

  // do not unroll loop in indirectFunctions: TDEngine
  //     static int32_t dnodeInitComponents() {
  //   for (int32_t i = 0; i < sizeof(tsDnodeComponents) /
  //   sizeof(tsDnodeComponents[0]); i++) {
  //     if (tsDnodeComponents[i].init() != 0) {
  if (isConfiguredIndirectAPI(F))
    return false;

  // if (canPeel(L)) {
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  AssumptionCache *AC =
      &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*F);
  TargetTransformInfo *TTI =
      &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(*F);

  OptimizationRemarkEmitter ORE(F);

  for (auto BB : L->getBlocks()) {
    for (BasicBlock::const_iterator BI = BB->begin(), BE = BB->end(); BI != BE;
         ++BI) {
      const Instruction *inst = cast<Instruction>(BI);
      if (isa<CallBase>(inst)) {
        auto CS = CallSite(inst);
        if (CS.getCalledFunction() == nullptr)
          continue;

        std::string fname = demangle(CS.getCalledFunction()->getName().str());
        if (containsAny(fname, THREAD_CREATE_API) ||
            CS.getCalledFunction()->getName().startswith(
                ".coderrect.thread.create.")) {
          // llvm::outs() << "======================================="
          //              << "\n";
          // llvm::outs() << "function: " << inst->getFunction()->getName() <<
          // "\n"; llvm::outs() << "can peel? " << canPeel(L) << "\n";
          // llvm::outs() << "is loop in simplified form? " <<
          // L->isLoopSimplifyForm() << "\n"; llvm::outs() << "does loop have
          // single exit? "
          //              << (!L->getExitingBlock() || !L->getUniqueExitBlock())
          //              << "\n";
          // llvm::outs() << "is the latch also the exiting block?"
          //              << (L->getLoopLatch() != L->getExitingBlock()) <<
          //              "\n";

          // NOTE: the peelLoop API seems to have soundness issue
          // some of the loop will fail the assertion if we turn on the
          // assertions in LLVM peelLoop(L, 1, LI, SE, DT, AC, true);

          // NOTE: it seems only if we set ULO.count to 2, the loop unroll will
          // take effect
          LOG_DEBUG("unroll loop once at: {}", *inst);
          // NOTE: the options can be more carefully tuned, I don't understand
          // all of them -- yanze
          UnrollLoopOptions ULO = {2, true, false, false, false, false};
          auto result = UnrollLoop(L, ULO, LI, SE, DT, AC, TTI, &ORE, true);
          // auto result = UnrollLoop(L, ULO, LI, SE, DT, AC, &ORE, true);
          if (result == LoopUnrollResult::Unmodified) {
            LOG_WARN("loop unrolling failed. loop = {}, function = {}", *L, *F);
            return false;
          }
          if (result == LoopUnrollResult::FullyUnrolled) {
            LPM.markLoopAsDeleted(*L);
          }
          return true;
        }
      }
    }
  }
  return false;
}

char UnrollThreadCreateLoopPass::ID = 0;
static RegisterPass<UnrollThreadCreateLoopPass>
    UPCL("", "Unroll Thread Creation Loop in IR", false, /*CFG only*/
         false /*is analysis*/);
