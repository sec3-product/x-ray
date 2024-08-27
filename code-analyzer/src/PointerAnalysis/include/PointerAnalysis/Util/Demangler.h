#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>

#include "PointerAnalysis/Util/Util.h"

namespace xray {

struct Demangler : public llvm::ItaniumPartialDemangler {
private:
  std::string curStr;

public:
  /// strip the number post fix before demangling to avoid potential errors
  /// \return true on error, false otherwise
  bool partialDemangle(llvm::StringRef MangledName) {
    llvm::StringRef stripedName = stripNumberPostFix(MangledName);
    // ensure the lifetime is as long as the demangler so that the pointer is
    // valid during the whole mangling process
    curStr = stripedName.str();
    return ItaniumPartialDemangler::partialDemangle(curStr.c_str());
  }

  bool isCtor() const {
    if (this->isCtorOrDtor()) {
      llvm::StringRef baseName = this->getFunctionBaseName(nullptr, nullptr);
      if (!baseName.startswith("~")) {
        return true;
      }
    }
    return false;
  }

  bool isDtor() const {
    if (this->isCtorOrDtor()) {
      llvm::StringRef baseName = this->getFunctionBaseName(nullptr, nullptr);
      if (baseName.startswith("~")) {
        return true;
      }
    }
    return false;
  }
};

} // namespace xray
