#ifndef RACEDETECTOR_CUDAMODEL_H
#define RACEDETECTOR_CUDAMODEL_H

#include <set>

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace aser {

// forward declaration
class CallSite;

namespace CUDAModel {

bool isFork(const llvm::StringRef func_name);

bool isFork(const llvm::Function *func);

bool isFork(const llvm::Instruction *inst);
bool isGetThreadIdX(const llvm::Function *func);
bool isGetThreadIdX(const llvm::StringRef func_name);
bool isGetThreadIdY(const llvm::Function *func);
bool isGetThreadIdY(const llvm::StringRef func_name);
bool isGetThreadIdZ(const llvm::Function *func);
bool isGetThreadIdZ(const llvm::StringRef func_name);

bool isGetBlockIdX(const llvm::Function *func);
bool isGetBlockIdX(const llvm::StringRef func_name);
bool isGetBlockIdY(const llvm::Function *func);
bool isGetBlockIdY(const llvm::StringRef func_name);
bool isGetBlockIdZ(const llvm::Function *func);
bool isGetBlockIdZ(const llvm::StringRef func_name);

bool isGetBlockDimX(const llvm::Function *func);
bool isGetBlockDimX(const llvm::StringRef func_name);
bool isGetBlockDimY(const llvm::Function *func);
bool isGetBlockDimY(const llvm::StringRef func_name);
bool isGetBlockDimZ(const llvm::Function *func);
bool isGetBlockDimZ(const llvm::StringRef func_name);

bool isGetGridDimX(const llvm::Function *func);
bool isGetGridDimX(const llvm::StringRef func_name);
bool isGetGridDimY(const llvm::Function *func);
bool isGetGridDimY(const llvm::StringRef func_name);
bool isGetGridDimZ(const llvm::Function *func);
bool isGetGridDimZ(const llvm::StringRef func_name);

bool isAnyCUDACall(const llvm::Function *func);
bool isAnyCUDACall(const llvm::StringRef func_name);

}  // namespace CUDAModel
}  // namespace aser

#endif
