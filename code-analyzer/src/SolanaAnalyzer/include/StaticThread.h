#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>

#include "DebugFlags.h"
#include "PTAModels/GraphBLASModel.h"

namespace xray {

using TID = uint32_t;
class Event;
class ForkEvent;

class StaticThread {
public:
  explicit StaticThread(const CallGraphNodeTy *entry)
      : id(curID++), entryNode(entry), threadHandle(nullptr) {
    auto result = tidToThread.emplace(id, this);
    assert(result.second);
    if (entryNode)
      startFunc = entryNode->getTargetFun()->getFunction();
  }

  StaticThread(const CallGraphNodeTy *entry, const llvm::Value *handle,
               const ForkEvent *P)
      : id(curID++), entryNode(entry), threadHandle(handle), parent(P) {
    auto result = tidToThread.emplace(id, this);
    assert(result.second);
    if (entryNode) {
      startFunc = entryNode->getTargetFun()->getFunction();
      initIDLInstructionName();
    }
  }
  StaticThread(const CallGraphNodeTy *entry, const llvm::Value *handle,
               const ForkEvent *P, const llvm::Function *f)
      : id(curID++), entryNode(entry), threadHandle(handle), parent(P),
        startFunc(f) {
    auto result = tidToThread.emplace(id, this);
    assert(result.second);
  }

  ~StaticThread() { tidToThread.erase(this->id); }

  static int getThreadNum() { return tidToThread.size(); }

  static void setThreadArg(TID tid,
                           std::map<uint8_t, const llvm::Constant *> &argMap);

  static std::map<uint8_t, const llvm::Constant *> *getThreadArg(TID tid);

  const CallGraphNodeTy *getEntryNode() { return this->entryNode; }

  const llvm::Value *getThreadHandle() { return threadHandle; }

  const TID getTID() const { return this->id; }

  const ForkEvent *getParentEvent() { return this->parent; }

  const llvm::Function *getStartFunction() { return startFunc; }

  inline const std::vector<ForkEvent *> &getForkSites() const {
    return forkSites;
  }

  void addForkSite(ForkEvent *e) { forkSites.push_back(e); }

  static StaticThread *getThreadByTID(TID tid) {
    assert(tidToThread.find(tid) != tidToThread.end());
    return tidToThread.at(tid);
  }

  static bool isSignatureUsedInAccountConstraint(TID tid, llvm::StringRef sig) {
    auto thread = getThreadByTID(tid);
    for (auto [account, hasConstraints] : thread->assertAccountConstraintsMap) {
      for (auto constraint : hasConstraints) {
        if (constraint.contains(sig)) {
          return true;
        }
      }
    }
    return false;
  }

  // extensions for sol
  // e.g., wallet_info -> next_account_info(account_info_iter)?;
  llvm::Function *anchor_struct_function;
  std::vector<std::pair<llvm::StringRef, llvm::StringRef>>
      anchorStructFunctionFields;

  bool isOnceOnlyOwnerOnlyInstruction = false;
  bool hasInitializedCheck = false;

  std::map<llvm::StringRef, const Event *> accountsMap;
  std::map<llvm::StringRef, bool> accountsMutMap;
  std::map<llvm::StringRef, const Event *> accountsDataWrittenMap;
  std::map<llvm::StringRef, const Event *> structAccountsMap;
  std::map<llvm::StringRef, const Event *> accountsCloseMap;
  std::map<llvm::StringRef, const Event *> accountsAnchorCloseMap;
  std::map<llvm::StringRef, const Event *> accountsUnpackedDataMap;
  std::map<llvm::StringRef, const Event *> accountsInvokedMap;

  std::map<llvm::StringRef, std::set<llvm::StringRef>> accountAliasesMap;
  std::map<llvm::StringRef, std::set<llvm::StringRef>> accountsSeedsMap;
  std::map<llvm::StringRef, std::set<llvm::StringRef>> assertHasOneFieldMap;
  std::map<llvm::StringRef, const Event *> accountsSelfSeedsMap;
  std::map<llvm::StringRef, std::set<llvm::StringRef>>
      assertAccountConstraintsMap;

  std::map<llvm::StringRef, const Event *> accountsPDAInInstructionMap;
  std::map<llvm::StringRef, const Event *> accountsPDAInitInInstructionMap;
  std::map<llvm::StringRef, const Event *> accountsPDAMutMap;

  // e.g., wallet_info.data -> wallet_info.data).borrow_mut
  std::map<llvm::StringRef, const Event *> borrowDataMutMap;
  std::map<llvm::StringRef, const Event *> borrowDataMap;
  std::map<llvm::StringRef, const Event *> borrowLamportsMutMap;
  std::map<llvm::StringRef, const Event *> borrowLamportsMap;
  std::map<llvm::StringRef, const Event *> borrowOtherMutMap;
  std::map<llvm::StringRef, const Event *> borrowOtherMap;
  std::map<llvm::StringRef, const Event *> memsetDataMap;

  // e.g.,    assert!(authority_info.is_signer);
  std::map<llvm::StringRef, const Event *> asssertProgramAddressMap;
  std::map<llvm::StringRef, const Event *> asssertSignerMap;
  std::map<llvm::StringRef, const Event *> asssertDiscriminatorMap;
  std::map<llvm::StringRef, const Event *> asssertOtherMap;
  // e.g., assert_eq!(wallet.vault, *vault_info.key);
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      accountsBumpMap;
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      assertKeyEqualMap;
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      checkDuplicateAccountKeyEqualMap;
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      accountContainsVerifiedMap;

  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      assertMintEqualMap;
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      assertBumpEqualMap;
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      assertOwnerEqualMap;
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      assertOtherEqualMap;
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      tokenTransferFromToMap;

  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      accountsInvokeAuthorityMap;
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      accountsBurnAuthorityMap;

  std::map<llvm::Function *,
           std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>>
      structAccountTypeMaps;

  std::map<const llvm::Function *, llvm::StringRef> mostRecentFuncReturnMap;

  std::set<llvm::StringRef> userProvidedInputStrings;

  bool isDuplicateAccountChecked(llvm::StringRef accountName1,
                                 llvm::StringRef accountName2) const;

  bool isUserProvidedString(llvm::StringRef &cons_seeds) const;

  void updateMostRecentFuncReturn(const llvm::Function *func,
                                  llvm::StringRef valueName);

  llvm::StringRef findReturnAliasAccount(const llvm::Function *func) {
    if (mostRecentFuncReturnMap.find(func) != mostRecentFuncReturnMap.end()) {
      return mostRecentFuncReturnMap.at(func);
    } else {
      auto &bb = func->getEntryBlock();
      if (bb.hasName() && bb.getName() == "sol.call") {
        for (auto &ii : bb) {
          if (isa<CallBase>(&ii)) {
            CallSite CS(&ii);
            auto targetF = CS.getTargetFunction();
            if (mostRecentFuncReturnMap.find(targetF) !=
                mostRecentFuncReturnMap.end()) {
              return mostRecentFuncReturnMap.at(targetF);
            }
          }
        }
      }
      return "";
    }
  }
  void addStructAccountType(llvm::Function *func, llvm::StringRef accountName,
                            llvm::StringRef typeName, const Event *e) {
    auto accountTypeMap = structAccountTypeMaps[func];
    auto pair = std::make_pair(accountName, typeName);
    accountTypeMap[pair] = e;
    structAccountTypeMaps[func] = accountTypeMap;
  }
  llvm::StringRef getStructAccountType(llvm::StringRef accountName) {
    for (auto [func, accountTypeMap] : structAccountTypeMaps) {
      for (auto [pair, e] : accountTypeMap) {
        if (pair.first.equals(accountName)) {
          return pair.second;
        }
      }
    }
    return "";
  }

  // e.g., **vault_info.lamports.borrow_mut() -= amount;
  std::map<llvm::StringRef, const Event *> accountLamportsUpdateMap;

  bool isInStructAccountsMap(llvm::StringRef accountName) {
    if (structAccountsMap.find(accountName) != structAccountsMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "structAccountsMap: " << accountName << "\n";

      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "structAccountsMap false: " << accountName << "\n";
    }
    return false;
  }

  bool isInAccountsMap(llvm::StringRef accountName) {
    if (accountsMap.find(accountName) != accountsMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isInAccountsMap: " << accountName << "\n";

      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isInAccountsMap false: " << accountName << "\n";
    }
    return false;
  }

  bool isInAccountOrStructAccountMap(llvm::StringRef accountName) {
    return isInAccountsMap(accountName) || isInStructAccountsMap(accountName);
  }

  bool isTokenTransferDestination(llvm::StringRef accountName) {
    for (auto [pair, inst] : tokenTransferFromToMap) {
      if (pair.second.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isTokenTransferDestination: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isTokenTransferDestination false: " << accountName
                   << "\n";
    return false;
  }

  bool isTokenTransferFromToAccount(llvm::StringRef accountName) {
    for (auto [pair, inst] : tokenTransferFromToMap) {
      if (pair.first.equals(accountName) || pair.second.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isTokenTransferFromToAccount: " << accountName
                       << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isTokenTransferFromToAccount false: " << accountName
                   << "\n";
    return false;
  }

  const Event *getTokenTransferFromToAccountEvent(llvm::StringRef accountName) {
    for (auto [pair, e] : tokenTransferFromToMap) {
      if (pair.first.equals(accountName) || pair.second.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isTokenTransferFromToAccount: " << accountName
                       << "\n";
        return e;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isTokenTransferFromToAccount false: " << accountName
                   << "\n";
    return nullptr;
  }

  llvm::StringRef getTokenTransferSourceAccount(llvm::StringRef accountName) {
    for (auto [pair, inst] : tokenTransferFromToMap) {
      if (pair.second.equals(accountName)) {
        return pair.first;
      }
    }
    return "";
  }

  bool isAccountDataWritten(llvm::StringRef accountName) {
    if (accountsDataWrittenMap.find(accountName) !=
        accountsDataWrittenMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountDataWritten: " << accountName << "\n";
      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountDataWritten false: " << accountName << "\n";
    }
    return false;
  }

  bool isAccountBorrowLamportsMut(llvm::StringRef accountName) {
    if (borrowLamportsMutMap.find(accountName) != borrowLamportsMutMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountBorrowLamportsMut: " << accountName << "\n";
      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountBorrowLamportsMut false: " << accountName
                     << "\n";
    }
    return false;
  }

  bool isAccountBorrowDataMut(llvm::StringRef accountName) {
    if (borrowDataMutMap.find(accountName) != borrowDataMutMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountBorrowDataMut: " << accountName << "\n";
      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountBorrowDataMut false: " << accountName << "\n";
    }
    return false;
  }

  bool isAccountBorrowData(llvm::StringRef accountName) {
    if (borrowDataMap.find(accountName) != borrowDataMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountBorrowData: " << accountName << "\n";

      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountBorrowData false: " << accountName << "\n";
    }
    return false;
  }

  bool isAccountMemsetData(llvm::StringRef accountName) {
    if (memsetDataMap.find(accountName) != memsetDataMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountMemsetData: " << accountName << "\n";
      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountMemsetData false: " << accountName << "\n";
    }
    return false;
  }

  bool isAccountsUnpackedData(llvm::StringRef accountName) {
    if (accountsUnpackedDataMap.find(accountName) !=
        accountsUnpackedDataMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "accountsUnpackedDataMap: " << accountName << "\n";
      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "accountsUnpackedDataMap false: " << accountName
                     << "\n";
    }
    return false;
  }

  bool isAccountLamportsUpdated(llvm::StringRef accountName) {
    if (accountLamportsUpdateMap.find(accountName) !=
        accountLamportsUpdateMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountLamportsUpdated: " << accountName << "\n";
      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountLamportsUpdated false: " << accountName
                     << "\n";
    }
    return false;
  }

  bool isAccountInvoked(llvm::StringRef accountName) {
    if (accountsInvokedMap.find(accountName) != accountsInvokedMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountInvoked: " << accountName << "\n";
      return true;
    } else {
      if (accountAliasesMap.find(accountName) != accountAliasesMap.end()) {
        auto aliasAccounts = accountAliasesMap.at(accountName);
        for (auto aliasAccount : aliasAccounts) {
          // TODO: remove circular aliases A->B ... B->A
          if (aliasAccount == accountName) {
            // aliasAccounts.erase(aliasAccount);
            continue;
          }
          if (accountAliasesMap.find(aliasAccount) != accountAliasesMap.end()) {
            auto aliasAccounts_x = accountAliasesMap.at(aliasAccount);
            if (aliasAccounts_x.find(accountName) != aliasAccounts_x.end()) {
              // aliasAccounts.erase(aliasAccount);
              continue;
            }
          }
          if (isAccountInvoked(aliasAccount)) {
            if (DEBUG_RUST_API)
              llvm::outs() << "isAccountInvoked: " << accountName
                           << " aliasAccount: " << aliasAccount << "\n";
            return true;
          }
        }
      }
    }

    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountInvoked false: " << accountName << "\n";
    return false;
  }

  bool isAccountPDAInInstruction(llvm::StringRef accountName) {
    for (auto [accountPda, inst] : accountsPDAInInstructionMap) {
      if (accountPda.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountPDAInInstruction: " << accountName << "\n";
        return true;
      }
    }
    if (isAccountsPDAInit(accountName))
      return true;
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountPDAInInstruction false: " << accountName
                   << "\n";

    return false;
  }

  bool isAuthorizedAccountPDAAndHasSignedSeeds(llvm::StringRef accountName) {
    for (auto [pair, e] : accountsBurnAuthorityMap) {
      if (pair.first.equals(accountName)) {
        auto from_or_to_account = pair.second;
        if (!from_or_to_account.empty()) {
          for (auto [accountSigner, inst] : asssertSignerMap) {
            if (isAccountUsedInSeedsOfAccount0(from_or_to_account,
                                               accountSigner)) {
              if (DEBUG_RUST_API)
                llvm::outs() << "isAuthorizedAccountPDAAndHasSignedSeeds: "
                             << accountName << "\n";
              return true;
            }
          }
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAuthorizedAccountPDAAndHasSignedSeeds false: "
                   << accountName << "\n";
    return false;
  }

  bool isAccountInvokedAsAuthority(llvm::StringRef accountName) {
    for (auto [pair, e] : accountsInvokeAuthorityMap) {
      if (pair.first.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountInvokedAsAuthority: " << accountName
                       << "\n";

        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountInvokedAsAuthority false: " << accountName
                   << "\n";
    return false;
  }

  bool isAccountInvokedAsBurnAuthority(llvm::StringRef accountName) {
    for (auto [pair, e] : accountsBurnAuthorityMap) {
      if (pair.first.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountInvokedAsBurnAuthority: " << accountName
                       << "\n";

        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountInvokedAsBurnAuthority false: " << accountName
                   << "\n";
    return false;
  }

  bool isAccountUsesSelfSeeds(llvm::StringRef accountName) {
    for (auto [accountSeed, inst] : accountsSelfSeedsMap) {
      if (accountSeed.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountUsesSelfSeeds: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountUsesSelfSeeds false: " << accountName << "\n";

    return false;
  }

  bool isAccountUsedInSeed(llvm::StringRef accountName) {
    for (auto [accountSeed, inst] : accountsSeedsMap) {
      if (accountSeed.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountUsedInSeed: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountUsedInSeed false: " << accountName << "\n";

    return false;
  }

  bool isAccountUsesAnotherAccountInSeed(llvm::StringRef accountName) {
    if (accountsSeedsMap.find(accountName) != accountsSeedsMap.end()) {
      for (auto accountSeed : accountsSeedsMap.at(accountName)) {
        if (!accountSeed.equals(accountName)) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountUsesAnotherAccountInSeed: " << accountName
                         << "\n";
          return true;
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountUsesAnotherAccountInSeed false: " << accountName
                   << "\n";

    return false;
  }

  bool
  isAccountReferencedByAnyOtherAccountInSeeds(llvm::StringRef accountName) {
    for (auto [account, seeds] : accountsSeedsMap) {
      if (!account.equals(accountName))
        if (seeds.find(accountName) != seeds.end()) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountReferencedByAnyOtherAccountInSeeds: "
                         << accountName << "\n";

          return true;
        }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountReferencedByAnyOtherAccountInSeeds false: "
                   << accountName << "\n";
    return false;
  }

  bool isAccountHasOneContraint(llvm::StringRef accountName) {
    if (assertHasOneFieldMap.find(accountName) != assertHasOneFieldMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountHasOneContraint: " << accountName << "\n";

      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountHasOneContraint false: " << accountName
                     << "\n";

      return false;
    }
  }

  bool
  isAccountReferencedByAnyOtherAccountInHasOne(llvm::StringRef accountName) {
    // TODO: Or hasOneField
    for (auto [account, hasOnes] : assertHasOneFieldMap) {
      if (hasOnes.find(accountName) != hasOnes.end()) {
        // llvm::outs() << "assertHasOneFieldMap: " << accountName << "
        // checking: " << account << "\n"; if (isAccountKeyValidated(account))
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountReferencedByAnyOtherAccountInHasOne: "
                       << accountName << "\n";

        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountReferencedByAnyOtherAccountInHasOne false: "
                   << accountName << "\n";

    return false;
  }

  bool
  isAccountValidatedBySignerAccountInConstraint(llvm::StringRef accountName) {
    for (auto [pair, inst] : assertKeyEqualMap) {
      if (isAccountSigner(pair.first) && pair.second.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountValidatedBySignerAccountInConstraint: "
                       << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountValidatedBySignerAccountInConstraint false: "
                   << accountName << "\n";

    return false;
  }

  bool
  isAccountReferencedBySignerAccountInConstraint(llvm::StringRef accountName) {
    for (auto [account, hasConstraints] : assertAccountConstraintsMap) {
      if (isAccountSigner(account)) {
        for (auto constraint : hasConstraints) {
          if (constraint.contains(accountName)) {
            if (DEBUG_RUST_API)
              llvm::outs() << "isAccountReferencedBySignerAccountInConstraint: "
                           << accountName << "\n";
            return true;
          }
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountReferencedBySignerAccountInConstraint false: "
                   << accountName << "\n";

    return false;
  }

  bool isAccountReferencedByAnyOtherAccountInConstraint(
      llvm::StringRef accountName) {
    for (auto [account, hasConstraints] : assertAccountConstraintsMap) {
      if (!account.equals(accountName)) {
        for (auto constraint : hasConstraints) {
          if (constraint.contains(accountName)) {
            if (DEBUG_RUST_API)
              llvm::outs()
                  << "isAccountReferencedByAnyOtherAccountInConstraint: "
                  << accountName << "\n";
            return true;
          }
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountReferencedByAnyOtherAccountInConstraint false: "
                   << accountName << "\n";

    return false;
  }

  bool isSignerAccountUsedSeedsOfAccount0(llvm::StringRef account0) {
    if (accountsSeedsMap.find(account0) != accountsSeedsMap.end()) {
      for (auto accountSeed : accountsSeedsMap[account0]) {
        if (isAccountSigner(accountSeed)) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isSignerAccountUsedSeedsOfAccount0: account0: "
                         << account0 << " accountSeed signer: " << accountSeed
                         << "\n";
          return true;
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isSignerAccountUsedSeedsOfAccount0 false: account0: "
                   << account0 << "\n";

    return false;
  }

  bool isAccountUsedInSeedsOfAccount0(llvm::StringRef account0,
                                      llvm::StringRef accountName) {
    if (accountsSeedsMap.find(account0) != accountsSeedsMap.end()) {
      for (auto accountSeed : accountsSeedsMap[account0]) {
        if (accountSeed.contains(accountName)) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountUsedInSeedsOfAccount0: account0: "
                         << account0 << " account: " << accountName << "\n";
          return true;
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountUsedInSeedsOfAccount0 false: account0: "
                   << account0 << " account: " << accountName << "\n";

    return false;
  }

  void computeIsOwnerOnly(std::set<llvm::StringRef> &potentialOwnerAccounts,
                          bool isInit = false) {
    if (isOwnerOnlyComputed) {
      return;
    }

    if (isInit) {
      for (auto [accountName, e] : accountsMap) {
        if (LangModel::isPreviledgeAccount(accountName)) {
          potentialOwnerAccounts.insert(accountName);
          if (DEBUG_RUST_API)
            llvm::outs() << "potentialOwnerAccounts: " << accountName
                         << " isInit: " << isInit << "\n";
        }
      }
      isOwnerOnly_ = true;
    } else {
      for (auto [accountName, e] : accountsMap) {
        if (isAccountSigner(accountName)) {
          if (LangModel::isPreviledgeAccount(accountName) &&
                  potentialOwnerAccounts.empty() &&
                  !LangModel::isUserProvidedAccount(accountName) ||
              potentialOwnerAccounts.find(accountName) !=
                  potentialOwnerAccounts.end()) {
            if (DEBUG_RUST_API)
              llvm::outs() << "potentialOwnerAccounts: " << accountName
                           << " isInit: " << isInit << "\n";
            isOwnerOnly_ = true;
            break;
          } else if (LangModel::isAuthorityAccount(accountName)) {
            isOwnerOnly_ = true;
            break;
          } else if (assertAccountConstraintsMap.find(accountName) !=
                     assertAccountConstraintsMap.end()) {
            auto hasConstraints = assertAccountConstraintsMap[accountName];
            for (auto constraint : hasConstraints) {
              if (constraint.contains("authority")) {
                if (DEBUG_RUST_API)
                  llvm::outs()
                      << "isAuthorityReferencedBySignerAccountInConstraint: "
                      << accountName << "\n";
                isOwnerOnly_ = true;
                break;
              }
            }
          }
        }
      }
    }
    if (DEBUG_RUST_API) {
      llvm::outs() << "computeIsOwnerOnly: " << isOwnerOnly_ << "\n";
    }
    isOwnerOnlyComputed = true;
  }

  bool isOwnerOnly() const { return isOwnerOnly_; }

  bool isAccountSignerMultiSigPDA() {
    for (auto [accountSigner, inst] : asssertSignerMap) {
      if (accountSigner.contains("multisig_pda")) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountSignerMultiSigPDA: " << accountSigner
                       << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountSignerMultiSigPDA false: "
                   << "\n";
    return false;
  }

  bool isAccountSignerVerifiedByPDAContains() {
    for (auto [accountSigner, inst] : asssertSignerMap) {
      for (auto [pair, inst2] : accountContainsVerifiedMap) {
        if (pair.first.equals(accountSigner) &&
            isAccountPDAInInstruction(pair.second)) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountSignerVerifiedByPDAContains: "
                         << accountSigner << "\n";
          return true;
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountSignerVerifiedByPDAContains false: "
                   << "\n";
    return false;
  }

  bool isAccountSigner(llvm::StringRef accountName) {
    for (auto [accountSigner, inst] : asssertSignerMap) {
      if (accountSigner.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountSigner: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountSigner false: " << accountName << "\n";

    return false;
  }

  bool isAccountAnchorClosed(llvm::StringRef accountName) {
    for (auto [accountClose, inst] : accountsAnchorCloseMap) {
      if (accountClose.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountAnchorClosed: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountAnchorClosed false: " << accountName << "\n";
    return false;
  }

  bool isAccountClosed(llvm::StringRef accountName) {
    for (auto [accountClose, inst] : accountsCloseMap) {
      if (accountClose.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountClosed: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountClosed false: " << accountName << "\n";
    return false;
  }

  bool accountsPDAMutable(llvm::StringRef accountName) {
    for (auto [accountPda, inst] : accountsPDAMutMap) {
      if (accountPda.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "accountsPDAInit: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "accountsPDAMutMap false: " << accountName << "\n";
    return false;
  }

  bool isAccountsPDAInit(llvm::StringRef accountName) {
    for (auto [accountPda, inst] : accountsPDAInitInInstructionMap) {
      if (accountPda.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountsPDAInit: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountsPDAInit false: " << accountName << "\n";
    return false;
  }

  bool isAccountDiscriminator(llvm::StringRef accountName) {
    if (asssertDiscriminatorMap.find(accountName) !=
        asssertDiscriminatorMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountDiscriminator: " << accountName << "\n";

      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountDiscriminator false: " << accountName << "\n";
    }
    return false;
  }

  bool isAliasAccountDiscriminator(llvm::StringRef accountName) {
    if (accountAliasesMap.find(accountName) != accountAliasesMap.end()) {
      auto aliasAccounts = accountAliasesMap.at(accountName);
      for (auto aliasAccount : aliasAccounts) {
        if (isAccountDiscriminator(aliasAccount)) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAliasAccountDiscriminator: " << accountName
                         << "\n";
          return true;
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAliasAccountDiscriminator false: " << accountName
                   << "\n";
    return false;
  }

  bool isAccountBumpValidated(llvm::StringRef bumpName) {
    for (auto [pair, inst] : assertBumpEqualMap) {
      if (pair.first.contains(bumpName) || pair.second.contains(bumpName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountBumpValidated: " << bumpName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountBumpValidated false: " << bumpName << "\n";
    return false;
  }

  bool isMintAccountValidated(llvm::StringRef accountName) {
    for (auto [pair, inst] : assertMintEqualMap) {
      if (pair.second.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isMintAccountValidated: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isMintAccountValidated false: " << accountName << "\n";
    return false;
  }

  bool isAccountMintValidated(llvm::StringRef accountName) {
    for (auto [pair, inst] : assertMintEqualMap) {
      if (pair.first.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountMintValidated: " << accountName << "\n";
        return true;
      }
      if (pair.first.equals(accountName.str() + ".mint")) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountMintValidated: " << accountName << ".key"
                       << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountMintValidated false: " << accountName << "\n";
    return false;
  }

  bool isAccountProgramAddressKeyValidated(llvm::StringRef accountName) {
    for (auto [pair, inst] : assertKeyEqualMap) {
      if (pair.first.contains(accountName)) {
        if (asssertProgramAddressMap.find(pair.second) !=
            asssertProgramAddressMap.end()) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountProgramAddressKeyValidated: "
                         << accountName << "\n";
          return true;
        }
      }
      if (pair.second.contains(accountName)) {
        if (asssertProgramAddressMap.find(pair.first) !=
            asssertProgramAddressMap.end()) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountProgramAddressKeyValidated: "
                         << accountName << "\n";
          return true;
        }
      }
    }
    for (auto [pair, inst] : assertOtherEqualMap) {
      if (pair.first.contains(accountName) && pair.first.contains(".key")) {
        if (asssertProgramAddressMap.find(pair.second) !=
            asssertProgramAddressMap.end()) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountProgramAddressKeyValidated: "
                         << accountName << "\n";
          return true;
        }
      }
      if (pair.second.contains(accountName) && pair.second.contains(".key")) {
        if (asssertProgramAddressMap.find(pair.first) !=
            asssertProgramAddressMap.end()) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountProgramAddressKeyValidated: "
                         << accountName << "\n";
          return true;
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountProgramAddressKeyValidated false: "
                   << accountName << "\n";
    return false;
  }

  bool isAccountKeyValidated(llvm::StringRef accountName) {
    for (auto [pair, inst] : assertKeyEqualMap) {
      if (pair.first.equals(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountKeyValidated: " << accountName << "\n";
        return true;
      }
      if (pair.first.equals(accountName.str() + ".key")) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountKeyValidated: " << accountName << ".key"
                       << "\n";
        return true;
      }
      if (pair.first.equals(accountName.str() + ".key()")) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountKeyValidated: " << accountName << ".key()"
                       << "\n";
        return true;
      }

      if (accountAliasesMap.find(accountName) != accountAliasesMap.end()) {
        auto aliasAccounts = accountAliasesMap.at(accountName);
        for (auto aliasAccount : aliasAccounts) {
          // TODO: remove circular aliases A->B ... B->A
          if (aliasAccount == accountName) {
            // aliasAccounts.erase(aliasAccount);
            continue;
          }
          if (accountAliasesMap.find(aliasAccount) != accountAliasesMap.end()) {
            auto aliasAccounts_x = accountAliasesMap.at(aliasAccount);
            if (aliasAccounts_x.find(accountName) != aliasAccounts_x.end()) {
              // aliasAccounts.erase(aliasAccount);
              continue;
            }
          }
          if (isAccountKeyValidated(aliasAccount)) {
            if (DEBUG_RUST_API)
              llvm::outs() << "isAccountKeyValidated: " << accountName
                           << " aliasAccount: " << aliasAccount << "\n";
            return true;
          }
        }
      }
    }

    if (isAccountReferencedByAnyOtherAccountInHasOne(accountName))
      return true;

    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountKeyValidated false: " << accountName << "\n";
    return false;
  }

  bool isAccountKeyValidatedSafe(llvm::StringRef accountName) {
    for (auto [pair, inst] : assertKeyEqualMap) {
      if (pair.first.contains(accountName) && pair.second.contains("address") ||
          pair.second.contains(accountName) && pair.first.contains("address")) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountKeyValidatedSafe: " << accountName << "\n";

        return true;
      }
      if (pair.first.contains(accountName) &&
              pair.second.contains(".signer_authority") ||
          pair.second.contains(accountName) &&
              pair.first.contains(".signer_authority")) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountKeyValidatedSafe signer_authority: "
                       << accountName << "\n";

        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountKeyValidatedSafe false: " << accountName
                   << "\n";

    return false;
  }

  bool isAccountOwnedBySigner(llvm::StringRef accountName) {
    for (auto [pair, inst] : assertOwnerEqualMap) {
      if (pair.first.contains(accountName) && isAccountSigner(pair.second)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountOwnedBySigner: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountOwnedBySigner false: " << accountName << "\n";

    return false;
  }

  bool isAccountOwnerValidated(llvm::StringRef accountName) {
    for (auto [pair, inst] : assertOwnerEqualMap) {
      if (pair.first.contains(accountName) ||
          pair.second.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountOwnerValidated: " << accountName << "\n";

        return true;
      }
    }
    // find alias account
    if (accountAliasesMap.find(accountName) != accountAliasesMap.end()) {
      auto aliasAccounts = accountAliasesMap.at(accountName);
      for (auto aliasAccount : aliasAccounts) {
        // TODO: remove circular aliases A->B ... B->A
        if (aliasAccount == accountName) {
          // aliasAccounts.erase(aliasAccount);
          continue;
        }
        if (accountAliasesMap.find(aliasAccount) != accountAliasesMap.end()) {
          auto aliasAccounts_x = accountAliasesMap.at(aliasAccount);
          if (aliasAccounts_x.find(accountName) != aliasAccounts_x.end()) {
            // aliasAccounts.erase(aliasAccount);
            continue;
          }
        }
        if (isAccountOwnerValidated(aliasAccount)) {
          if (DEBUG_RUST_API)
            llvm::outs() << "isAccountOwnerValidated: " << accountName
                         << " aliasAccount: " << aliasAccount << "\n";
          return true;
        }
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountOwnerValidated false: " << accountName << "\n";

    return false;
  }

  const llvm::Function *startFunc = nullptr;
  std::string anchor_idl_instruction_name;
  std::string convertToAnchorString(StringRef idl_instruction_name) {
    auto res = idl_instruction_name.str();
    if (idl_instruction_name.contains("_")) {
      llvm::SmallVector<StringRef, 3> anchor_name_vec;
      idl_instruction_name.split(anchor_name_vec, '_', -1, false);
      for (uint i = 0; i < anchor_name_vec.size(); i++) {
        auto str = anchor_name_vec[i].str();
        if (i == 0)
          res = str;
        else {
          str[0] = std::toupper(str[0]);
          res = res + str;
        }
      }
    }
    return res;
  }

private:
  bool initIDLInstructionName();

  static TID curID;
  static std::map<TID, StaticThread *> tidToThread;
  static std::map<TID, std::map<uint8_t, const llvm::Constant *>> threadArgs;

  const CallGraphNodeTy *entryNode;
  const ForkEvent *parent = nullptr;
  const llvm::Value *threadHandle;
  const TID id;
  bool isOwnerOnly_ = false;
  bool isOwnerOnlyComputed = false;

  std::vector<ForkEvent *> forkSites;
};

} // namespace xray
