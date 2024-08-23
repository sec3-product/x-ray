

## WIP: LLVM 17+ supports

The support of LLVM 17 and later version is still WIP. We appreciate any suggestions or contributions to help solving compatibility of latest LLVM on the open-sourced X-Ray tool.

### 1- Opaque pointers (solved)

Starting from LLVM 17, only opaque pointers are supported. Typed pointers are not supported anymore. 

About opaque pointer, LLVM's official documentation:  [](https://releases.llvm.org/17.0.1/docs/OpaquePointers.html)

[https://releases.llvm.org/17.0.1/docs/OpaquePointers.html](https://releases.llvm.org/17.0.1/docs/OpaquePointers.html)  

In the official doc, LLVM gives migration Instructions:

In order to support opaque pointers, two types of changes tend to be necessary. The first is the removal of all calls to PointerType::getElementType() and Type::getPointerElementType().

### 2- SCF and Standard expression refactored (to be solved)

In LLVM 17, all Dialects related to `StandardOps` have changed. The original `StandardOps` directory has been refactored and merged into the new directory structure. 

Most standard operations, such as constant operations and branching operations, have been dispersed into other related dialect directories, such as mlir/Dialect/ControlFlow, etc.

About SCF: Starting from LLVM 14, the functionality of `SCFToStandard` conversion has been integrated into the `ControlFlowToLLVM` conversion. The conversion is responsible for converting various control flow structures (including SCF and other control flows) in MLIR to LLVM DialectIR.

Our tools (smallrace and LLVMRace) use SCF and Standard expressions in many places. A lot of modification is needed.

### 3- MLIR IR-related code refactored

In LLVM 15+, some of MLIR IR-related code is re-organized, for the purpose of refactoring the MLIR codebase into a more modular structure. In particular, the mlir/lib directory now contains the core implementation of MLIR, while various dialects, conversions, and other features have been dispersed into other directories.

Some headers related to MLIR IR have undergone similar path changes.

  
### 4- Trivial changes (almost finished)

Multiple changes in the LLVM IR API, such as deprecated functions, changes in the number of function parameters, and data types.
Some of them are frequently used in X-Ray tool, such as `CreateInBoundsGEP()`, `getNumArgOperands()`, etc.

