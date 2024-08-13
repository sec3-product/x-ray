//
// Created by peiming on 1/16/20.
//

#ifndef ASER_PTA_DEFAULTHEAPMODEL_H
#define ASER_PTA_DEFAULTHEAPMODEL_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include <set>

namespace aser {

class DefaultHeapModel {
private:
  // TODO: there should be more -> memalign, etc
  const llvm::SmallDenseSet<llvm::StringRef, 4> heapAllocAPIs{
      "malloc", "calloc", "_Znam", "_Znwm", "??2@YAPEAX_K@Z"};

protected:
  static llvm::Type *getNextBitCastDestType(const llvm::Instruction *allocSite);

  // infer the type for calloc-like memory allocation.
  // NOTE: this can be used for sub class as well as general routine
  [[nodiscard]] static llvm::Type *
  inferCallocType(const llvm::Function *fun, const llvm::Instruction *allocSite,
                  int numArgNo = 0, int sizeArgNo = 1);

  // infer the type for malloc-like memory allocation.
  // NOTE: this can be used for sub class as well as general routine
  // NOTE:
  // if sizeArgNo < 0:
  //    the type should be modelled as unlimited bound array.
  [[nodiscard]] static llvm::Type *
  inferMallocType(const llvm::Function *fun, const llvm::Instruction *allocSite,
                  int sizeArgNo = 0);

public:
  inline bool isCalloc(const llvm::Function *fun) const {
    if (fun->hasName()) {
      return fun->getName().equals("calloc");
    }
    return false;
  }

  inline bool isHeapAllocFun(const llvm::Function *fun) const {
    if (fun->hasName()) {
      return heapAllocAPIs.find(fun->getName()) != heapAllocAPIs.end();
    }
    return false;
  }

  inline llvm::Type *
  inferHeapAllocType(const llvm::Function *fun,
                     const llvm::Instruction *allocSite) const {
    if (isCalloc(fun)) {
      // infer the type for calloc like function
      return inferCallocType(fun, allocSite);
    }

    // infer the type for malloc like function
    return inferMallocType(fun, allocSite);
  }
};

} // namespace aser

#endif // ASER_PTA_DEFAULTHEAPMODEL_H
