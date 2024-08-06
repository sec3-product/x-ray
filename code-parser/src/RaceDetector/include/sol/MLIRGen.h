#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

namespace mlir {
class MLIRContext;
}  // namespace mlir

namespace sol {
class ModuleAST;

/// Emit IR for the given moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGenFull(mlir::MLIRContext &context,
                                              sol::ModuleAST &moduleAST);

}  // namespace sol
