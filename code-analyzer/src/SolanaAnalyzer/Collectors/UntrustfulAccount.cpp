#include "Collectors/UntrustfulAccount.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <llvm/IR/Value.h>
#include <nlohmann/json.hpp>

#include "LogColor.h" // info
#include "SVE.h"
#include "SourceInfo.h"

using namespace xray;
using namespace llvm;

using json = nlohmann::json;

extern bool PRINT_IMMEDIATELY;
extern bool TERMINATE_IMMEDIATELY;

// Not to limit the number of bugs we collected
// by default we only collect at most 25 cases for each type of bug
static bool nolimit = false;
constexpr unsigned int DEFAULT_BUDGET = 25;

/* --------------------------------

           UntrustfulAccount

----------------------------------- */

// static fields
uint xray::UntrustfulAccount::budget = DEFAULT_BUDGET;
std::vector<UntrustfulAccount> xray::UntrustfulAccount::untrustfulAccounts;
std::set<const llvm::Value *> xray::UntrustfulAccount::apiSigs;
std::set<std::string> xray::UntrustfulAccount::cpiSigs;

std::set<std::vector<std::string>> xray::UntrustfulAccount::callStackSigs;

void xray::UntrustfulAccount::init(int configReportLimit,
                                   bool configNoReportLimit) {
  if (configReportLimit != -1) {
    budget = configReportLimit;
  }
  nolimit = configNoReportLimit;
}

std::string xray::UntrustfulAccount::getErrorMsg(SVE::Type type) {
  std::string msg;
  switch (type) {
  case SVE::Type::ACCOUNT_UNVALIDATED_BORROWED:
    msg = "The account is not validated before parsing its data:";
    break;
  case SVE::Type::ACCOUNT_UNVALIDATED_OTHER:
    msg = "The account is not properly validated and may be untrustful:";
    break;
  case SVE::Type::ACCOUNT_UNVALIDATED_PDA:
    msg = "The PDA account is not properly validated and may be untrustful:";
    break;
  case SVE::Type::ACCOUNT_UNVALIDATED_DESTINATION:
    msg = "The account is used as destination in token transfer without "
          "validation and it could be the same as "
          "the transfer source account:";
    break;
  case SVE::Type::ACCOUNT_INCORRECT_AUTHORITY:
    msg = "The PDA account may be incorrectly used as a shared authority and "
          "may allow any account to transfer "
          "or burn tokens:";
    break;
  case SVE::Type::INSECURE_INIT_IF_NEEDED:
    msg = "The `init_if_needed` keyword in anchor-lang prior to v0.24.x has a "
          "critical security bug:";
    break;
  case SVE::Type::MISS_OWNER:
    msg = "The account info is missing owner check:";
    break;
  case SVE::Type::MISS_SIGNER:
    msg = "The account info is missing signer check:";
    break;
  case SVE::Type::INSECURE_SPL_TOKEN_CPI:
    msg = "The spl_token account may be arbitrary:";
    break;
  case SVE::Type::INSECURE_ACCOUNT_REALLOC:
    msg = "The account realloc in solana_program prior to v1.10.29 may cause "
          "programs to malfunction:";
    break;
  case SVE::Type::INSECURE_ASSOCIATED_TOKEN:
    msg = "The associated token account may be faked:";
    break;
  case SVE::Type::MALICIOUS_SIMULATION:
    msg = "The program may be malicious:";
    break;
  case SVE::Type::DIV_BY_ZERO:
    msg = "The arithmetic operation may result in a div-by-zero error:";
    break;
  case SVE::Type::BUMP_SEED:
    msg = "The account's bump seed is not validated and may be vulnerable to "
          "seed canonicalization attacks:";
    break;
  case SVE::Type::INSECURE_PDA_SHARING:
    msg = "The PDA sharing with these seeds is insecure:";
    break;
  case SVE::Type::ACCOUNT_CLOSE:
    msg = "The account closing is insecure:";
    break;
  default:
    assert(false && "unhandled untrustful account:");
    break;
  }
  return msg;
}

// we report at most 1 UntrustfulAccount bug for each function call
bool xray::UntrustfulAccount::filter(SVE::Type type, SourceInfo &srcInfo) {
  // for CPI
  if (SVE::Type::ACCOUNT_CLOSE == type) {
    auto sig = srcInfo.sig();
    if (cpiSigs.find(sig) != cpiSigs.end()) {
      // llvm::outs() << "filter true:" << srcInfo.sig() << "\n";
      return true;
    } else {
      cpiSigs.insert(sig);
      // llvm::outs() << "filter false:" << srcInfo.sig() << "\n";
      return false;
    }
  }
  const llvm::Value *v = srcInfo.getValue();
  if (apiSigs.find(v) != apiSigs.end()) {
    // llvm::outs() << "filter true:" << srcInfo.sig() << "\n";
    return true;
  } else {
    apiSigs.insert(v);
    // llvm::outs() << "filter false:" << srcInfo.sig() << "\n";
    return false;
  }
}

// we report at most 1 UntrustfulAccount bug for each call stack
bool xray::UntrustfulAccount::filterByCallStack(std::vector<std::string> &st0) {
  auto st = st0;
  // this one is special: keep only the last few entries
  if (st0.size() > 1) {
    std::vector<std::string> vect(st0.begin() + st0.size() - 1, st0.end());
    st = vect;
  }

  if (callStackSigs.find(st) != callStackSigs.end()) {
    return true;
  }
#pragma omp critical(callStackSigs)
  { callStackSigs.insert(st); }
  return false;
}

void xray::UntrustfulAccount::collect(
    llvm::StringRef accountName, const Event *e,
    std::map<TID, std::vector<CallEvent *>> &callEventTraces, SVE::Type type,
    int P) {
  SourceInfo srcInfo = getSourceLoc(e->getInst());
  if (filter(type, srcInfo))
    return;

  if (SVE::Type::MISS_SIGNER == type) {
    // SKIP PDA is_signer
    if (getSourceLinesForSoteria(srcInfo, 1).find(" PDA") !=
            std::string::npos ||
        getSourceLinesForSoteria(srcInfo, 2).find(" PDA") != std::string::npos)
      return;
  }

  // for Anchor accounts, _ is ignored by default
  bool isIgnored = accountName.startswith("_");
  bool isHidden = false;
  if (accountName.contains("_no_check"))
    isIgnored = true; // skip no_check
  if (SVE::isCheckerDisabled(type))
    isHidden = true;

  if (SVE::Type::ACCOUNT_UNVALIDATED_DESTINATION == type) {
    bool isDestinationIgnored = customizedFilterSoteriaIgnoreSymbol(e, "dest");
    if (isDestinationIgnored)
      isIgnored = true;
  } else if (SVE::Type::MISS_SIGNER == type) {
    bool isSignerIgnored = customizedFilterSoteriaIgnoreSymbol(e, "signer");
    if (isSignerIgnored)
      isIgnored = true;
  } else if (SVE::Type::ACCOUNT_UNVALIDATED_OTHER == type) {
    bool isUntrustfulIgnored =
        customizedFilterSoteriaIgnoreSymbol(e, "untrust");
    if (isUntrustfulIgnored)
      isIgnored = true;
  }
  std::vector<std::string> st;
  TID tid = e->getTID();
  EventID id = e->getID();

  auto &callTrace = callEventTraces[tid];
  std::string last_str = "";
  for (CallEvent *ce : callTrace) {
    if (ce->getID() > id)
      break;
    if (ce->getEndID() == 0 || ce->getEndID() > id) {
      auto call_str = ce->getCallSiteString(true);
      if (last_str != call_str)
        st.push_back(call_str);
      last_str = call_str;
    }
  }
  // st.erase(st.begin(), st.begin() + 2);
#pragma omp critical(budget)
  {
    if (nolimit || budget > 0) {
      srcInfo.setStackTrace(st);
      auto msg = getErrorMsg(type); // TODO: add source location
      untrustfulAccounts.emplace_back(accountName.str(), srcInfo, msg, type, P,
                                      isIgnored, isHidden);
      --budget;
      if (PRINT_IMMEDIATELY)
        untrustfulAccounts.back().print();
      // intentionally commented out since UntrustfulAccount needs improvement
      // if (TERMINATE_IMMEDIATELY) exit(1);
    }
  }
}

xray::UntrustfulAccount::UntrustfulAccount(std::string account,
                                           SourceInfo &srcInfo, std::string msg,
                                           SVE::Type t, int P, bool isIgnored,
                                           bool isHidden)
    : apiInst(srcInfo), errorMsg(msg), type(t), accountName(account), p(P),
      ignore(isIgnored), hide(isHidden) {
  id = SVE::getTypeID(t);
  name = SVE::database[id]["name"];
  description = SVE::database[id]["description"];
  url = SVE::database[id]["url"];
}

json xray::UntrustfulAccount::to_json() {
  json j({{"priority", p},
          {"account", accountName},
          {"inst", apiInst},
          {"errorMsg", errorMsg},
          {"id", id},
          {"hide", hide},
          {"ignore", ignore},
          {"description", description},
          {"url", url}});
  return j;
}

void xray::UntrustfulAccount::print() {
  outs() << "ignored: " << ignore << "\n";
  // llvm::outs() << "=============This account may be
  // UNTRUSTFUL!================\n";
  llvm::errs() << "==============VULNERABLE: " << name << "!============\n";
  outs() << "Found a potential vulnerability at line " << apiInst.getLine()
         << ", column " << apiInst.getCol() << " in " << apiInst.getFilename()
         << "\n";
  // outs() << errorMsg << "\n";
  outs() << description << ":\n";
  outs() << apiInst.getSnippet();
  outs() << ">>>Stack Trace:\n";
  printStackTrace(apiInst.getStackTrace());
  outs() << "\n";
  outs() << "For more info, see " << url << "\n\n\n";
}

void xray::UntrustfulAccount::printAll() {
  std::sort(untrustfulAccounts.begin(), untrustfulAccounts.end());
  for (auto r : untrustfulAccounts) {
    r.print();
  }
}

void xray::UntrustfulAccount::printSummary() {
  info("detected " + std::to_string(untrustfulAccounts.size()) +
       " untrustful accounts in total.");
}

