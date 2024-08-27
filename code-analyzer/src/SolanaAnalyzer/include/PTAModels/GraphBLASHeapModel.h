#pragma once

#include <Logger/Logger.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PatternMatch.h>

#include "PointerAnalysis/Models/DefaultHeapModel.h"
#include "PointerAnalysis/Program/CallSite.h"
#include "Util/Util.h"

namespace xray {

class GraphBLASHeapModel : public DefaultHeapModel {
private:
  // TODO: we can get rid of it! and use conflib for this!!
  const std::set<llvm::StringRef> HEAP_ALLOCATIONS{//"GB_realloc_memory"
                                                   "GB_malloc_memory",
                                                   "GB_calloc_memory"};

  const std::set<llvm::StringRef> HEAP_INITS{"GrB_Matrix_new", "GB_new"};

  // the set of APIs specified by users
  static std::set<llvm::StringRef> USER_HEAP_API;

public:
  // TODO: ensure that the user specified APIs return pointers!!
  static inline void init(std::vector<std::string> &heapAPIs) {
    for (auto &api : heapAPIs) {
      USER_HEAP_API.insert(api);
    }
  }

  inline bool isHeapAllocFun(const llvm::Function *fun) const {
    if (fun->hasName()) {
      return DefaultHeapModel::isHeapAllocFun(fun) ||
             isHeapInitFun(fun) || // any function in interceptHeapAllocation is
                                   // heap alloc function
             HEAP_ALLOCATIONS.find(fun->getName()) != HEAP_ALLOCATIONS.end() ||
             USER_HEAP_API.find(fun->getName().split(".").first) !=
                 USER_HEAP_API.end();
    }

    return false;
  }

  inline bool isHeapInitFun(const llvm::Function *fun) const {
    if (fun->hasName()) {
      return HEAP_INITS.find(fun->getName()) != HEAP_INITS.end();
    }
    return false;
  }

  // we used the next bitcast instruction as the heap object's element type
  // if we can not find it, return null.
  llvm::Type *inferHeapAllocType(const llvm::Function *fun,
                                 const llvm::Instruction *allocSite) const {
    if (DefaultHeapModel::isHeapAllocFun(fun)) {
      // if already handled by default heap model
      return DefaultHeapModel::inferHeapAllocType(fun, allocSite);
    }
    LOG_WARN("can not infer type for heap. type={}", *allocSite);
    return nullptr;
  }
}; // namespace xray

} // namespace xray
