#include "Rules/ArbitraryCPI.h"

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include "DebugFlags.h"
#include "PTAModels/GraphBLASModel.h"
#include "Rules/Rule.h"
#include "SVE.h"

namespace xray {

static llvm::StringRef stripToAccountInfo(llvm::StringRef account_name) {
  auto found = account_name.find(".to_account_info()");
  if (found != std::string::npos) {
    account_name = account_name.substr(0, found);
  }
  return account_name;
}

static llvm::StringRef stripCtxAccountsName(llvm::StringRef account_name) {
  if (account_name.contains("ctx.accounts.")) {
    // for anchor if not matched..
    auto found = account_name.find("ctx.accounts.");
    account_name = account_name.substr(found + 13);
  } else if (account_name.contains("ctx.")) {
    auto found = account_name.find("ctx.");
    account_name = account_name.substr(found + 4);
  }
  return account_name;
}

static bool isTrustedBuilder(llvm::StringRef BuilderName) {
  if (BuilderName.contains("system_instruction::")) {
    return true;
  }
  if (BuilderName.contains("spl_associated_token_account::")) {
    return true;
  }
  if (BuilderName.contains("spl_token::instruction::transfer_checked") ||
      BuilderName.contains("spl_token::instruction::mint_to_checked") ||
      BuilderName.contains("spl_token::instruction::approve_checked")) {
    return true;
  }
  return false;
}

struct CPIInfo {
  std::string Account;
  const llvm::Instruction *BuilderCall;
  std::string InstName;
  std::string BuiltBy;
  bool IsTrusted;
};

// getProgramIdAccountName locates and returns the account name used as the
// program_id for a CPI by inspecting the invoke/invoke_signed call. It traces
// back to the CPI‐builder call (e.g., spl_token::instruction::transfer) and
// extracts the first argument, which represents the target program’s account
// field.
static CPIInfo getProgramIdAccountName(const llvm::Instruction *InvokeInst) {
  // 1. Confirm an invoke/invoke_signed call.
  CallSite InvokeCS(const_cast<Instruction *>(InvokeInst));
  if (auto *F = InvokeCS.getCalledFunction()) {
    auto Name = F->getName();
    if (!Name.contains("program::invoke") &&
        !Name.startswith("sol.invoke_signed.") &&
        !Name.startswith("sol.invoke.")) {
      return CPIInfo{"", InvokeInst, "", "", false};
    }
  } else {
    return CPIInfo{"", InvokeInst, "", "", false};
  }

  const CallBase *BuilderCall = nullptr;
  auto MaybeBuilderCall = InvokeCS.getArgOperand(0);
  if (auto *CB = dyn_cast<CallBase>(MaybeBuilderCall); CB != nullptr) {
    BuilderCall = CB;
  } else {
    // 2. Walk backwards to find the real CallBase that built the instruction.
    for (auto *Prev = InvokeInst->getPrevNonDebugInstruction(); Prev;
         Prev = Prev->getPrevNonDebugInstruction()) {
      if (auto *CB = dyn_cast<CallBase>(Prev)) {
        auto Callee = CB->getCalledFunction();
        if (Callee != nullptr) {
          auto CalleeName = Callee->getName();
          if (CalleeName.contains("instruction")) {
            BuilderCall = CB;
            break;
          }
        }
      }
    }
  }
  if (!BuilderCall) {
    return CPIInfo{"", InvokeInst, "", "", false};
  }

  // 3. Record the builder.
  std::string BuiltBy = "<unknown>";
  if (auto *BF = BuilderCall->getCalledFunction(); BF != nullptr) {
    BuiltBy = BF->getName().str();
    if (DEBUG_RUST_API) {
      llvm::outs() << "    BuiltBy: " << BuiltBy << "\n";
    }
    // Heuristcally check if the builder is trusted.
    if (isTrustedBuilder(BuiltBy)) {
      return CPIInfo{"", InvokeInst, "", "", true};
    }
  }

  // 4. Try to extract its first argument as the program_id account.
  std::string Account;
  if (auto *GV = dyn_cast<GlobalVariable>(
          BuilderCall->getArgOperand(0)->stripPointerCasts())) {
    if (auto *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer())) {
      StringRef S = CDA->getAsString();
      // strip trailing “.key”
      if (auto idx = S.rfind(".key"); idx != StringRef::npos)
        S = S.substr(0, idx);
      Account = stripCtxAccountsName(S).str();
    }
  }

  // 5. Look one instruction further to find the assignment inst to the account.
  // This provides additional information for user review, especially for the
  // case that the account cannot be precisely analyzed.
  std::string InstName;
  for (auto *N = BuilderCall->getNextNonDebugInstruction(); N != nullptr;
       N = N->getNextNonDebugInstruction()) {
    if (auto *CB3 = dyn_cast<CallBase>(N); CB3 != nullptr) {
      if (auto *CF3 = CB3->getCalledFunction(); CF3 != nullptr) {
        if (CF3->getName() == "sol.model.opaqueAssign") {
          // first arg is the GEP pointing at the local
          if (auto *GEP = dyn_cast<GEPOperator>(CB3->getArgOperand(0));
              GEP != nullptr) {
            if (auto *GV = dyn_cast<GlobalVariable>(
                    GEP->getPointerOperand()->stripPointerCasts());
                GV != nullptr) {
              InstName = GV->getName().str();
            }
          }
          break;
        }
      }
    }
  }

  return CPIInfo{Account, BuilderCall, InstName, BuiltBy, false};
}

bool handleInvoke(const RuleContext &RC, const CallSite &CS) {
  auto TargetFuncName = CS.getTargetFunction()->getName();
  if (!TargetFuncName.startswith("sol.invoke_signed.") &&
      !TargetFuncName.startswith("sol.invoke.") &&
      !TargetFuncName.contains("program::invoke")) {
    return false;
  }
  // 1. Special-case for `create_associated_token_account` to mark the ATA
  // account invoked.
  if (auto CallValue = dyn_cast<CallBase>(CS.getArgOperand(0))) {
    CallSite CS2(CallValue);
    auto CS2Name = CS2.getTargetFunction()->getName();
    if (CS2Name.startswith("sol.create_associated_token_account.")) {
      auto value1 = CS.getArgOperand(1);
      auto valueName = LangModel::findGlobalString(value1);
      // split
      llvm::SmallVector<StringRef, 8> accounts_vec;
      valueName.split(accounts_vec, ',', -1, false);
      if (accounts_vec.size() > 3) {
        auto associate_account = accounts_vec[2];
        associate_account = stripToAccountInfo(associate_account);
        auto foundClone = associate_account.find(".clone()");
        if (foundClone != std::string::npos) {
          associate_account = associate_account.substr(0, foundClone);
        }
        if (DEBUG_RUST_API) {
          llvm::outs() << "sol.create_associated_token_account: "
                       << associate_account << "\n";
        }
        auto e = RC.createReadEvent();
        RC.getThread()->accountsInvokedMap[associate_account] = e;
      }
    }
  }

  // 2. Now run the general ABITRARY_CPI check.
  auto [Account, ProgInst, InstName, BuiltBy, IsTrusted] =
      getProgramIdAccountName(RC.getInst());
  if (IsTrusted) {
    // TODO: It currently returns `false` so that later rules that rely on
    // `sol.invoke` are evaluated.
    return false;
  }
  if (!Account.empty()) {
    if (!RC.getThread()->isAccountKeyValidated(Account)) {
      if (DEBUG_RUST_API) {
        llvm::outs() << "  sol.invoke: found unvalidated account: " << Account
                     << "\n";
      }
      RC.collectUntrustfulAccount(Account, SVE::Type::ARBITRARY_CPI, 9, "");
    }
    // TODO: It currently returns `false` so that later rules that rely on
    // `sol.invoke` are evaluated.
    return false;
  }
  // Unable to identify the account name. Report with warning.
  if (DEBUG_RUST_API) {
    llvm::outs() << "  sol.invoke: unable to identify account:\n"
                 << "    Func: " << TargetFuncName << "\n"
                 << "    Inst: " << *RC.getInst() << "\n"
                 << "    Account: " << Account << "\n"
                 << "    ProgInst: " << ProgInst << "\n"
                 << "    InstName: " << InstName << "\n"
                 << "    BuiltBy: " << BuiltBy << "\n";
  }
  std::string desc;
  if (!InstName.empty()) {
    desc = "The CPI may be vulnerable because it invokes whatever program is "
           "stored in `" +
           InstName + "`, whose value is built by the call to `" + BuiltBy +
           "`.";
  }
  RC.collectUntrustfulAccount(Account, SVE::Type::ARBITRARY_CPI, 9, desc);
  return false;
}

} // namespace xray
