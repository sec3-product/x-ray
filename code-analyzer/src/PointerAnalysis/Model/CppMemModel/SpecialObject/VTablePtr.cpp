//
// Created by peiming on 8/27/20.
//

#include "PointerAnalysis/Models/MemoryModel/CppMemModel/SpecialObject/VTablePtr.h"

#include "llvm/IR/Type.h"

using namespace llvm;

namespace xray {

bool isVTablePtrType(const llvm::Type *type) {
  static Type *vtableType = nullptr;

  if (vtableType == nullptr) {
    // vtable type i32 (...)**
    auto &C = type->getContext();
    auto elemTy = FunctionType::get(IntegerType::get(C, 32), true);
    vtableType = PointerType::get(PointerType::get(elemTy, 0), 0);
  }

  return type == vtableType;
}

} // namespace xray