#ifndef MLIR_ST_MLIRGEN_H_
#define MLIR_ST_MLIRGEN_H_

#include <ast/AST.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <memory>

namespace mlir {
class MLIRContext;
}  // namespace mlir

namespace st {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);

}  // namespace st

namespace stx {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGenFull(mlir::MLIRContext &context,
                                              stx::ModuleAST &moduleAST);

}  // namespace stx

#endif  // MLIR_ST_MLIRGEN_H_