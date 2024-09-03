#pragma once

#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>

#include "Rules/Rule.h"

namespace xray {

class Event;
class StaticThread;

class UntrustfulAccountDetector {
public:
  using CollectUntrustfulAccountFunc =
      std::function<void(llvm::StringRef, const Event *, SVE::Type, int)>;

  UntrustfulAccountDetector(
      const FuncArgTypesMap &funcArgTypesMap,
      const std::set<const llvm::Function *> &threadStartFunctions,
      const std::set<llvm::StringRef> &potentialOwnerAccounts,
      const std::set<llvm::StringRef> &globalStateAccounts,
      const FunctionFieldsMap &anchorStructFunctionFieldsMap,
      const std::map<llvm::StringRef, const Event *> &accountsPDAMap,
      const std::map<llvm::StringRef, const Event *>
          &accountsSeedProgramAddressMap,
      std::function<bool(llvm::StringRef)> accountTypeContainsMoreThanOneMint,
      std::function<bool(llvm::StringRef)> isInitFunc,
      CollectUntrustfulAccountFunc collectUntrustfulAccountFunc)

      : funcArgTypesMap(funcArgTypesMap),
        threadStartFunctions(threadStartFunctions),
        potentialOwnerAccounts(potentialOwnerAccounts),
        globalStateAccounts(globalStateAccounts),
        anchorStructFunctionFieldsMap(anchorStructFunctionFieldsMap),
        accountsPDAMap(accountsPDAMap),
        accountsSeedProgramAddressMap(accountsSeedProgramAddressMap),
        accountTypeContainsMoreThanOneMint(accountTypeContainsMoreThanOneMint),
        isInitFunc(isInitFunc),
        collectUntrustfulAccountFunc(collectUntrustfulAccountFunc) {}

  void detect(StaticThread *curThread);

private:
  bool hasThreadStartInitFunction(std::string symbol) const;
  bool isAnchorDataAccount(llvm::StringRef accountName) const {
    for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
      for (auto pair : fieldTypes) {
        if (pair.first.equals(accountName) &&
            (pair.second.startswith("Box<Account<'info") ||
             pair.second.startswith("Account<'info")))
          return true;
      }
    }
    return false;
  }
  bool isAnchorValidatedAccount(llvm::StringRef accountName) const {
    for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
      for (auto pair : fieldTypes) {
        if (pair.first.equals(accountName) &&
            (pair.second.contains("Program<") ||
             pair.second.contains("Sysvar<")))
          return true;
      }
    }
    return false;
  }

  bool isAccountPDA(llvm::StringRef accountName) const {
    for (auto [accountPda, inst] : accountsPDAMap) {
      if (accountPda.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountPDA: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountPDA false: " << accountName << "\n";

    return false;
  }

  bool isAccountUsedInSeedsProgramAddress(llvm::StringRef accountName) const {
    if (accountsSeedProgramAddressMap.find(accountName) !=
        accountsSeedProgramAddressMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountUsedInSeedsProgramAddress: " << accountName
                     << "\n";
      return true;
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountUsedInSeedsProgramAddress false: "
                   << accountName << "\n";

    return false;
  }

  bool isGlobalStateAccount(llvm::StringRef accountName) const {
    if (globalStateAccounts.find(accountName) != globalStateAccounts.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isGlobalStateAccount: " << accountName << "\n";

      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isGlobalStateAccount false: " << accountName << "\n";
    }
    return false;
  }

  const FuncArgTypesMap &funcArgTypesMap;
  const std::set<const llvm::Function *> &threadStartFunctions;
  const std::set<llvm::StringRef> &potentialOwnerAccounts;
  const std::set<llvm::StringRef> &globalStateAccounts;
  const FunctionFieldsMap &anchorStructFunctionFieldsMap;
  const std::map<llvm::StringRef, const Event *> &accountsPDAMap;
  const std::map<llvm::StringRef, const Event *> &accountsSeedProgramAddressMap;
  std::function<bool(llvm::StringRef)> accountTypeContainsMoreThanOneMint;
  std::function<bool(llvm::StringRef)> isInitFunc;
  CollectUntrustfulAccountFunc collectUntrustfulAccountFunc;
};

} // namespace xray
