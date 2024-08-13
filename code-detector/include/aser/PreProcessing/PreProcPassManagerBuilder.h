//
// Created by peiming on 2/26/20.
//
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#ifndef ASER_PTA_PREPROCPASSMANAGERBUILDER_H
#define ASER_PTA_PREPROCPASSMANAGERBUILDER_H

namespace aser {

enum class UseCFLAA { None, Steensgaard, Andersen, Both };

// inspired by llvm::PassManagerBuilder
class PreProcPassManagerBuilder {
private:
  UseCFLAA useCFL;
  bool runInstCombine;
  bool enableLoopUnswitch;
  bool enableSimpleLoopUnswitch;

  std::function<void(llvm::legacy::PassManagerBase &)> beforeInlineHook;
  void addFunctionSimplificationPasses(llvm::legacy::PassManagerBase &MPM);

public:
  PreProcPassManagerBuilder();
  explicit PreProcPassManagerBuilder(
      std::function<void(llvm::legacy::PassManagerBase &)> &&);

  void populateModulePassManager(llvm::legacy::PassManagerBase &MPM);
  void populateFunctionPassManager(llvm::legacy::FunctionPassManager &FPM);
  void populatePreProcessingModuleManager(llvm::legacy::PassManagerBase &MPM);
};

} // namespace aser

#endif // ASER_PTA_PREPROCPASSMANAGERBUILDER_H
