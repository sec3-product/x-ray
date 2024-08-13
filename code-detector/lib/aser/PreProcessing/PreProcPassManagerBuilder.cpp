//
// Created by peiming on 2/26/20.
//
#include "aser/PreProcessing/PreProcPassManagerBuilder.h"

#include <aser/PreProcessing/Passes/LoweringMemCpyPass.h>

#include "aser/PreProcessing/Passes/StandardHeapAPIRewritePass.h"
#include "aser/PreProcessing/Passes/TransformCallInstBitCastPass.h"
#include "aser/PreProcessing/Passes/WrapperFunIdentifyPass.h"
#include "aser/Util/Util.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"

using namespace aser;
using namespace llvm;

PreProcPassManagerBuilder::PreProcPassManagerBuilder()
    : useCFL(UseCFLAA::None), runInstCombine(false), enableLoopUnswitch(false),
      enableSimpleLoopUnswitch(false), beforeInlineHook(Noop()) {}

PreProcPassManagerBuilder::PreProcPassManagerBuilder(
    std::function<void(llvm::legacy::PassManagerBase &)> &&hook)
    : useCFL(UseCFLAA::None), runInstCombine(false), enableLoopUnswitch(false),
      enableSimpleLoopUnswitch(false), beforeInlineHook(hook) {}

static inline void
addInstructionCombiningPass(llvm::legacy::PassManagerBase &PM) {
  PM.add(createInstructionCombiningPass(true));
}

static void addInitialAliasAnalysisPasses(legacy::PassManagerBase &PM,
                                          UseCFLAA useCFL) {
  // we might want it later
  switch (useCFL) {
  case UseCFLAA::Steensgaard:
    PM.add(createCFLSteensAAWrapperPass());
    break;
  case UseCFLAA::Andersen:
    PM.add(createCFLAndersAAWrapperPass());
    break;
  case UseCFLAA::Both:
    PM.add(createCFLSteensAAWrapperPass());
    PM.add(createCFLAndersAAWrapperPass());
    break;
  default:
    break;
  }

  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  PM.add(createTypeBasedAAWrapperPass());
  PM.add(createScopedNoAliasAAWrapperPass());
}

void PreProcPassManagerBuilder::addFunctionSimplificationPasses(
    legacy::PassManagerBase &MPM) {
  // Start of function pass.
  // Break up aggregate allocas, using SSAUpdater.
  MPM.add(createSROAPass());
  MPM.add(createEarlyCSEPass(
      true /* Enable mem-ssa. */)); // Catch trivial redundancies

  // Speculative execution if the target has divergent branches; otherwise nop.
  // MPM.add(createSpeculativeExecutionIfHasBranchDivergencePass());
  // MPM.add(createJumpThreadingPass());         // Thread jumps.

  MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
  MPM.add(createCFGSimplificationPass());          // Merge & remove BBs

  // Combine silly seq's
  // TODO: we still want instruction combine, but the instcombine should not
  // delete the bitcast after heap allocation
  // TODO: APIs to help us to perseve the type information
  // TODO: implement our own version of inst combine pass
  if (runInstCombine) {
    MPM.add(createAggressiveInstCombinerPass());
    addInstructionCombiningPass(MPM);
  }

  MPM.add(createTailCallEliminationPass()); // Eliminate tail calls
  MPM.add(createCFGSimplificationPass());   // Merge & remove BBs
  MPM.add(createReassociatePass());         // Reassociate expressions

  // Begin the loop pass pipeline.
  // TODO: probably does not matter for now
  if (enableSimpleLoopUnswitch) {
    // The simple loop unswitch pass relies on separate cleanup passes. Schedule
    // them first so when we re-process a loop they run before other loop
    // passes.
    MPM.add(createLoopInstSimplifyPass());
    MPM.add(createLoopSimplifyCFGPass());
  }

  // Rotate Loop - disable header duplication at -Oz
  MPM.add(createLoopRotatePass());
  MPM.add(createLICMPass(/*LicmMssaOptCap, LicmMssaNoAccForPromotionCap*/));

  if (enableSimpleLoopUnswitch)
    MPM.add(createSimpleLoopUnswitchLegacyPass());
  else if (enableLoopUnswitch)
    MPM.add(createLoopUnswitchPass(false, false));

  // FIXME: We break the loop pass pipeline here in order to do full
  // simplify-cfg. Eventually loop-simplifycfg should be enhanced to replace the
  // need for this.
  MPM.add(createCFGSimplificationPass());
  if (runInstCombine) {
    addInstructionCombiningPass(MPM);
  }
  // We resume loop passes creating a second loop pipeline here.
  MPM.add(createIndVarSimplifyPass()); // Canonicalize indvars
  MPM.add(createLoopIdiomPass());      // Recognize idioms like memset.

  // addExtensionsToPM(EP_LateLoopOptimizations, MPM);

  MPM.add(createLoopDeletionPass()); // Delete dead loops

  // if (EnableLoopInterchange)
  //     MPM.add(createLoopInterchangePass()); // Interchange loops

  // Unroll small loops
  // TODO: probably make the loop harder to analyze
  // MPM.add(createSimpleLoopUnrollPass(OptLevel, DisableUnrollLoops,
  //                                    ForgetAllSCEVInLoopUnroll));

  // This ends the loop pass pipelines.
  MPM.add(createSCCPPass()); // Constant prop with SCCP

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  MPM.add(createBitTrackingDCEPass()); // Delete dead bit computations

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  if (runInstCombine) {
    addInstructionCombiningPass(MPM);
  }

  MPM.add(createCorrelatedValuePropagationPass());
  MPM.add(createDeadStoreEliminationPass()); // Delete dead stores
  MPM.add(createLICMPass(/*LicmMssaOptCap, LicmMssaNoAccForPromotionCap*/));

  // TODO: this could simplify the loop but might also have bad effect on debug
  // info? MPM.add(createLoopRerollPass());

  MPM.add(createAggressiveDCEPass());     // Delete dead instructions
  MPM.add(createCFGSimplificationPass()); // Merge & remove BBs

  // Clean up after everything.
  if (runInstCombine) {
    addInstructionCombiningPass(MPM);
  }
}

void PreProcPassManagerBuilder::populateFunctionPassManager(
    legacy::FunctionPassManager &FPM) {
  FPM.add(createEntryExitInstrumenterPass());

  // Add LibraryInfo if we have some.
  addInitialAliasAnalysisPasses(FPM, useCFL);

  // FPM.add(createLowerInvokePass());
  FPM.add(createSROAPass());
  FPM.add(createCFGSimplificationPass());
  FPM.add(createSROAPass());
  FPM.add(createEarlyCSEPass());
  FPM.add(createLowerExpectIntrinsicPass());
}

void PreProcPassManagerBuilder::populateModulePassManager(
    legacy::PassManagerBase &MPM) {
  // Allow forcing function attributes as a debugging and tuning aid.
  MPM.add(createForceFunctionAttrsLegacyPass());
  MPM.add(new StandardHeapAPIRewritePass()); // TODO: make it a function pass
  addInitialAliasAnalysisPasses(MPM, useCFL);

  // Infer attributes about declarations if possible.
  MPM.add(createInferFunctionAttrsLegacyPass());

  // TODO: probably split the call site might help context-sensitive analysis
  MPM.add(createCallSiteSplittingPass());

  // IP SCCP
  MPM.add(createIPSCCPPass());
  MPM.add(createCalledValuePropagationPass());

  // Infer attributes on declarations, call sites, arguments, etc.
  MPM.add(createAttributorLegacyPass());

  MPM.add(createGlobalOptimizerPass()); // Optimize out global vars
  // Promote any localized global vars.
  MPM.add(createPromoteMemoryToRegisterPass());

  MPM.add(createDeadArgEliminationPass()); // Dead argument elimination

  if (runInstCombine) {
    // do we need instruction combine pass?
    // it might destroy the bitcast that used for inferring type for heap
    // allocation
    addInstructionCombiningPass(MPM);
  }

  MPM.add(createCFGSimplificationPass()); // Clean up after IPCP & DAE

  // HOOKS here, passes run before inlining
  beforeInlineHook(MPM);

  // We add a module alias analysis pass here. In part due to bugs in the
  // analysis infrastructure this "works" in that the analysis stays alive
  // for the entire SCC pass run below.
  MPM.add(createGlobalsAAWrapperPass());

  // Start of CallGraph SCC passes.
  MPM.add(createPruneEHPass()); // Remove dead EH info

  MPM.add(createAlwaysInlinerLegacyPass());

  MPM.add(createPostOrderFunctionAttrsLegacyPass());

  // seems like it is useful
  // This pass promotes "by reference" arguments to be "by value" arguments.  In
  // practice, this means looking for internal functions that have pointer
  // arguments.  If it can prove, through the use of alias analysis, that an
  // argument is *only* loaded, then it can pass the value into the function
  // instead of the address of the value.  This can cause recursive
  // simplification of code and lead to the elimination of allocas (especially
  // in C++ template code like the STL).
  MPM.add(createArgumentPromotionPass()); // Scalarize uninlined fn args

  addFunctionSimplificationPasses(MPM);

  // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
  // pass manager that we are specifically trying to avoid. To prevent this
  // we must insert a no-op module pass to reset the pass manager.

  // this marks heap allocation APIs as well as getter/setter to always inline
  // we do it after function simplification so that it is easier to understand
  // the wrapper function
  MPM.add(new WrapperFunIdentifyPass());
  MPM.add(createAlwaysInlinerLegacyPass());

  // run the optimization all over again
  MPM.add(createPostOrderFunctionAttrsLegacyPass());
  MPM.add(createArgumentPromotionPass()); // Scalarize uninlined fn args
  addFunctionSimplificationPasses(MPM);

  MPM.add(createBarrierNoopPass());
  // MPM.add(createPartialInliningPass());

  // Remove avail extern fns and globals definitions if we aren't
  // compiling an object file for later LTO. For LTO we want to preserve
  // these so they are eligible for inlining at link-time. Note if they
  // are unreferenced they will be removed by GlobalDCE later, so
  // this only impacts referenced available externally globals.
  // Eventually they will be suppressed during codegen, but eliminating
  // here enables more opportunity for GlobalDCE as it may make
  // globals referenced by available external functions dead
  // and saves running remaining passes on the eliminated functions.
  MPM.add(createEliminateAvailableExternallyPass());

  // TODO: probably useless but also does no harm
  MPM.add(createReversePostOrderFunctionAttrsPass());

  // The inliner performs some kind of dead code elimination as it goes,
  // but there are cases that are not really caught by it. We might
  // at some point consider teaching the inliner about them, but it
  // is OK for now to run GlobalOpt + GlobalDCE in tandem as their
  // benefits generally outweight the cost, making the whole pipeline
  // faster.
  MPM.add(createGlobalOptimizerPass());
  MPM.add(createGlobalDCEPass());

  // we do not run beyond loop vectorization
  // these ensure our loop is in the canonical form
  MPM.add(createLoopSimplifyPass());
  MPM.add(createIndVarSimplifyPass());
}

void PreProcPassManagerBuilder::populatePreProcessingModuleManager(
    llvm::legacy::PassManagerBase &MPM) {
  // lowering memcpy before any optimization so that the pattern is fixed
  MPM.add(new LoweringMemCpyPass());
  MPM.add(new TransformCallInstBitCastPass());
  MPM.add(createAlwaysInlinerLegacyPass());
}
