#ifndef MLIR_ST_MLIRGEN_H_
#define MLIR_ST_MLIRGEN_H_

#include <ast/AST.h>

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
}  // namespace mlir

namespace st {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);

}  // namespace st

namespace stx {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGenFull(mlir::MLIRContext &context,
                                  stx::ModuleAST &moduleAST);

}  // namespace stx

#endif  // MLIR_ST_MLIRGEN_H_
