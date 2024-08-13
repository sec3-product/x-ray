#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>

#include "aser/PreProcessing/IRPreProcessor.h"
#include "aser/PreProcessing/PreProcPassManagerBuilder.h"

using namespace aser;
using namespace llvm;
using namespace llvm::codegen;

static TargetMachine *GetTargetMachine(Triple TheTriple, StringRef CPUStr,
                                       StringRef FeaturesStr,
                                       const TargetOptions &Options) {
  std::string Error;
  std::string MArch;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(MArch, TheTriple, Error);
  // Some modules don't specify a triple, and this is okay.
  if (!TheTarget) {
    return nullptr;
  }

  return TheTarget->createTargetMachine(TheTriple.getTriple(), CPUStr,
                                        FeaturesStr, Options, None);
}

// static codegen::RegisterCodeGenFlags CFG;

void IRPreProcessor::runOnModule(
    llvm::Module &M,
    std::function<void(llvm::legacy::PassManagerBase &)> &&hook) {
  // TODO: Do we really need to know the target machine information?
  Triple ModuleTriple(M.getTargetTriple());
  std::string CPUStr, FeaturesStr;
  TargetMachine *Machine = nullptr;
  // const TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  // For LLVM 12
  const TargetOptions Options = InitTargetOptionsFromCodeGenFlags(ModuleTriple);
  if (ModuleTriple.getArch()) {
    CPUStr = getCPUStr();
    FeaturesStr = getFeaturesStr();
    Machine = GetTargetMachine(ModuleTriple, CPUStr, FeaturesStr, Options);
  } else if (ModuleTriple.getArchName() != "unknown" &&
             ModuleTriple.getArchName() != "") {
    // err: do not know target machine type
    return;
  }
  std::unique_ptr<TargetMachine> TM(Machine);

  // Override function attributes based on CPUStr, FeaturesStr, and command line
  // flags.
  // setFunctionAttributes(CPUStr, FeaturesStr, M);

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build.
  legacy::PassManager Passes;
  legacy::PassManager PrePasses;
  legacy::FunctionPassManager FPasses(&M);
  PreProcPassManagerBuilder builder(std::move(hook));

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  // target-info and target transfromInfo
  TargetLibraryInfoImpl TLII(ModuleTriple);
  Passes.add(new TargetLibraryInfoWrapperPass(TLII));
  Passes.add(createTargetTransformInfoWrapperPass(TM ? TM->getTargetIRAnalysis()
                                                     : TargetIRAnalysis()));

  FPasses.add(new TargetLibraryInfoWrapperPass(TLII));
  FPasses.add(createTargetTransformInfoWrapperPass(
      TM ? TM->getTargetIRAnalysis() : TargetIRAnalysis()));
  builder.populateFunctionPassManager(FPasses);
  // builder.populateModulePassManager(Passes);

  builder.populatePreProcessingModuleManager(PrePasses);

  // pre processing
  PrePasses.run(M);

  // function opt
  FPasses.doInitialization();
  for (Function &F : M) {
    FPasses.run(F);
  }
  FPasses.doFinalization();

  // optimization
  // Passes.run(M);
}
