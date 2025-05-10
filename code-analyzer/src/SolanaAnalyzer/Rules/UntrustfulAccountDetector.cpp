#include "Rules/UntrustfulAccountDetector.h"

#include <string>

#include <llvm/Support/raw_ostream.h>

#include "DebugFlags.h"
#include "StaticThread.h"

namespace xray {

void UntrustfulAccountDetector::detect(StaticThread *curThread) {
  auto funcName = curThread->startFunc->getName();
  llvm::outs() << "\n**************** attack surface #" << curThread->getTID()
               << ": " << funcName.substr(0, funcName.find_last_of("."))
               << " **************** \n";

  // Dump debugging information for accounts if requested.
  if (DEBUG_RUST_API) {
    for (auto [accountName, e] : curThread->accountsMap) {
      llvm::outs() << "account_x: " << accountName << "\n";
    }
    for (auto [pair, e] : curThread->assertKeyEqualMap) {
      llvm::outs() << "assertKeyEqualMap: " << pair.first
                   << " == " << pair.second << "\n";
    }
    for (auto [account, aliasAccounts] : curThread->accountAliasesMap) {
      llvm::outs() << "account: " << account << " aliases: "
                   << "\n";
      for (auto aliasAccount : aliasAccounts) {
        llvm::outs() << "         " << aliasAccount << "\n";
      }
    }
    for (auto [pair, e] : curThread->assertOwnerEqualMap) {
      llvm::outs() << "assertOwnerEqualMap: " << pair.first
                   << " == " << pair.second << "\n";
    }
    for (auto [pair, e] : curThread->assertOtherEqualMap) {
      llvm::outs() << "assertOtherEqualMap: " << pair.first
                   << " == " << pair.second << "\n";
    }
    for (auto [account, e] : curThread->accountsInvokedMap) {
      llvm::outs() << "accountsInvokedMap: " << account << "\n";
    }
  }

  bool isInit = isInitFunc(funcName);
  bool isPotentiallyOwnerOnly =
      funcName.contains("create_") || funcName.contains("new_");
  bool isOwnerOnly = curThread->isOnceOnlyOwnerOnlyInstruction ||
                     curThread->isAccountSignerMultiSigPDA() ||
                     curThread->isAccountSignerVerifiedByPDAContains() ||
                     curThread->isOwnerOnly();
  if (DEBUG_RUST_API) {
    llvm::outs() << "isOwnerOnly: " << isOwnerOnly << "\n";
  }

  // Traverse all the accounts and check for untrustful accounts.
  for (auto [accountName, e] : curThread->accountsMap) {
    if (DEBUG_RUST_API) {
      llvm::outs() << "UntrustfulAccountDetector::detect account: " << accountName << "\n";
    }
    // Skip initialization function.
    if (isInit) {
      continue;
    }

    // skip initialization func
    bool isUnvalidate = false;
    // missing owner check
    // ctx.accounts.authority.key != &token.owner
    if (!isUnvalidate &&
        (curThread->isAccountBorrowData(accountName) &&
         !curThread->isAccountBorrowDataMut(accountName)) &&
        !curThread->isAccountBorrowLamportsMut(accountName) &&
        !isAnchorDataAccount(accountName) && !isAccountPDA(accountName) &&
        !curThread->isAccountKeyValidated(accountName) &&
        !curThread->isAccountInvoked(accountName) &&
        !curThread->isAccountOwnerValidated(accountName) &&
        !curThread->isAccountUsedInSeed(accountName)) {
      // llvm::errs() << "==============VULNERABLE:
      // MissingOwnerCheck!============\n";
      collectUntrustfulAccountFunc(accountName, e, SVE::Type::MISS_OWNER, 10, "");
      isUnvalidate = true;
    }
    if (!isUnvalidate && curThread->isAccountBorrowData(accountName) &&
        !curThread->isAccountBorrowDataMut(accountName)) {
      if (!curThread->isAccountLamportsUpdated(accountName) &&
          !curThread->isAccountKeyValidated(accountName) &&
          (!curThread->isAccountOwnerValidated(accountName))) {
        // llvm::errs() << "==============VULNERABLE: Account
        // Unvalidated!============\n";
        if (!isOwnerOnly) {
          auto e1 = curThread->borrowDataMap[accountName];
          collectUntrustfulAccountFunc(
              accountName, e1, SVE::Type::ACCOUNT_UNVALIDATED_BORROWED, 9, "");
          isUnvalidate = true;
        }
      }
    }

    // TODO: define pattern for missing check for p_asset_mint
    // there are some hidden semantc-level invariants between accounts

    if (LangModel::isAuthorityAccount(accountName)) {
      // checker for missing signer accounts
      if (!isUnvalidate && !curThread->isAccountSigner(accountName) &&
          !curThread->isAccountValidatedBySignerAccountInConstraint(
              accountName) &&
          !isAccountPDA(accountName) &&
          !curThread->isAccountInvokedAsAuthority(accountName) &&
          !curThread->isAccountInvoked(accountName) &&
          !curThread->isAccountBorrowDataMut(accountName)) {
        if (!curThread->isAccountReferencedByAnyOtherAccountInHasOne(
                accountName) &&
            !curThread->isAccountReferencedByAnyOtherAccountInConstraint(
                accountName) &&
            !curThread->isAccountKeyValidatedSafe(accountName) &&
            !isAnchorDataAccount(accountName)) {
          bool isValidatedBySigner = false;
          if (accountName.endswith("_info")) {
            auto accountName_x =
                accountName.substr(0, accountName.find("_info"));
            if (curThread->isAccountValidatedBySignerAccountInConstraint(
                    accountName_x)) {
              isValidatedBySigner = true;
            }
          }
          // heuristic: skip checking signer on "manager_fee"
          if (!isOwnerOnly && !isValidatedBySigner &&
              !accountName.contains("pda") && !accountName.contains("mint") &&
              !accountName.contains("new_") && !accountName.contains("from_") &&
              !accountName.contains("to_") && !accountName.contains("_fee")) {
            collectUntrustfulAccountFunc(accountName, e, SVE::Type::MISS_SIGNER,
                                         10, "");
            isUnvalidate = true;
          }

          // habitat_owner_token_account.owner is signer
        }
      }

    } else { // !LangModel::isAuthorityAccount(accountName)
      // TODO: define pattern for mut declared but not updated account
      //&& !isAccountPDA(accountName) - this condition removed, because PDA
      // can still be faked...
      if (!isUnvalidate && !isAnchorValidatedAccount(accountName) &&
          !curThread->isAccountLamportsUpdated(accountName) &&
          !curThread->isAccountKeyValidated(accountName) &&
          (!curThread->isAccountOwnerValidated(accountName) &&
           !curThread->isAccountSigner(accountName))) {
        // llvm::errs() << "==============VULNERABLE: Account
        // Unvalidated!============\n";
        if (!isOwnerOnly) {
          if (!curThread->isAccountPDAInInstruction(accountName) &&
              !isGlobalStateAccount(accountName) &&
              !curThread->isAccountInvoked(accountName) &&
              !curThread->isAccountBorrowDataMut(accountName) &&
              !curThread->isAccountDiscriminator(accountName) &&
              !curThread->isAccountDataWritten(accountName) &&
              !curThread->isAccountAnchorClosed(accountName)) {
            bool skipOrderAccount = false;
            bool skipCloseAccount = false;
            bool skipMintAccount = false;
            bool skipUserAccount = false;
            if (accountName.contains("order")) {
              // TODO: order_account, skip if contains two mints
              // get anchor account type
              auto type = curThread->getStructAccountType(accountName);
              if (accountTypeContainsMoreThanOneMint(type)) {
                skipOrderAccount = true;
              }
            } else if (accountName.contains("close")) {
              if (curThread->startFunc->getName().contains("close"))
                skipCloseAccount = true;
            } else {
              auto type = curThread->getStructAccountType(accountName);
              // && curThread->isMintAccountValidated(accountName)
              if (type.contains("Mint"))
                skipMintAccount = true;
              if (type.contains("TokenAccount") &&
                  accountName.contains("user_"))
                skipUserAccount = true;
              else if ((accountName.contains("user_") ||
                        accountName.contains("multisig_") ||
                        accountName.contains("market_")) &&
                       curThread->isAccountHasOneContraint(accountName)) {
                skipUserAccount = true;
              }

              // for non-Anchor account
              // habitat_mint
              if (accountName.endswith("_account_info")) {
                auto accountName_x =
                    accountName.substr(0, accountName.find("_account_info"));
                if (curThread->isAccountKeyValidated(accountName_x)) {
                  skipMintAccount = true;
                }
              }
            }
            if (!isPotentiallyOwnerOnly && !skipOrderAccount &&
                !skipCloseAccount && !skipMintAccount &&
                !skipUserAccount && //! isPotentiallyOwnerOnly &&
                !accountName.contains("new_") &&
                !accountName.contains("payer") &&
                !accountName.contains("delegat") &&
                !accountName.contains("dest") &&
                !accountName.contains("receiver") &&
                !accountName.equals("to")) {
              if (!curThread->isAccountReferencedBySignerAccountInConstraint(
                      accountName) &&
                  !curThread->isAccountValidatedBySignerAccountInConstraint(
                      accountName) &&
                  !curThread->isAccountReferencedByAnyOtherAccountInSeeds(
                      accountName) &&
                  !isAccountUsedInSeedsProgramAddress(accountName)) {

                llvm::outs() << "==============VULNERABLE: Account other ==============\n";
                llvm::outs() << "accountName: " << accountName << "\n"
                  << "  validated? " << curThread->isAccountKeyValidated(accountName) << "\n";
                collectUntrustfulAccountFunc(
                    accountName, e, SVE::Type::ACCOUNT_UNVALIDATED_OTHER, 8, "");
                isUnvalidate = true;
              }
            }
          } else if (curThread->isAccountInvoked(accountName)) {
            // TODO: check if account is isolated.
            // llvm::errs()
            //     << "==============VULNERABLE: Invoke Account Potentially
            //     Unvalidated!============\n";
            // collectUntrustfulAccountFunc(accountName, e,
            // SVE::Type::ACCOUNT_UNVALIDATED_OTHER, 5);
            // isUnvalidate = true;
          }
        }
      }
    }
  }

  // check bump_seed_canonicalization_insecure
  if (DEBUG_RUST_API) {
    llvm::outs() << "checking bump_seed_canonicalization_insecure\n";
  }
  for (auto [pair, e] : curThread->accountsBumpMap) {
    auto bumpName = pair.first;
    auto account = pair.second;
    // llvm::outs() << "bumpName: " << bumpName << "\n";
    if (!curThread->isAccountBumpValidated(bumpName) &&
        !curThread->isSignerAccountUsedSeedsOfAccount0(account)) {
      if (!isOwnerOnly) {
        collectUntrustfulAccountFunc(bumpName, e, SVE::Type::BUMP_SEED, 9, "");
      }
    }
  }
}

bool UntrustfulAccountDetector::hasThreadStartInitFunction(
    std::string symbol) const {
  for (auto func : threadStartFunctions) {
    if (func->getName().contains(symbol)) {
      if (DEBUG_RUST_API) {
        llvm::outs() << "hasThreadStartInitFunction: " << func->getName() << " "
                     << symbol << "\n";
      }
      return true;
    }
  }
  return false;
};

} // namespace xray
