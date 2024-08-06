//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Affine + SCF + Standard dialects to the LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                                  Standard --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Sequence.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "sol/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct SolToLLVMLoweringPass
    : public PassWrapper<SolToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
};
}  // end anonymous namespace

void SolToLLVMLoweringPass::runOnOperation() {
  // llvm::outs() << "in SolToLLVMLoweringPass\n";

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.

  // OwningRewritePatternList patterns;
  // MLIRContext *context = &getContext();
  RewritePatternSet patterns(&getContext());

  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  // patterns.insert<PrintOpLowering>(&getContext());
  // patterns.insert<GenericCallOpLowering>(&getContext());
  // patterns.insert<ObjectOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();

  // llvm::outs() << "in applyFullConversion\n";

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::sol::createLowerToLLVMPass() {
  return std::make_unique<SolToLLVMLoweringPass>();
}
