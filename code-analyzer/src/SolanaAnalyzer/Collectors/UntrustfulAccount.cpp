#include "Collectors/UntrustfulAccount.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <llvm/IR/Value.h>
#include <llvm/Support/ErrorHandling.h>
#include <nlohmann/json.hpp>

#include "DebugFlags.h"
#include "LogColor.h" // info
#include "SVE.h"
#include "SourceInfo.h"

using namespace xray;
using namespace llvm;

// Not to limit the number of bugs we collected
// by default we only collect at most 25 cases for each type of bug
unsigned int xray::UntrustfulAccount::budget = 25;
std::vector<UntrustfulAccount> xray::UntrustfulAccount::untrustfulAccounts;
static bool nolimit = false;

std::set<const llvm::Value *> xray::UntrustfulAccount::apiSigs;
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
  case SVE::Type::MISS_OWNER:
    msg = "The account info is missing owner check:";
    break;
  case SVE::Type::MISS_SIGNER:
    msg = "The account info is missing signer check:";
    break;
  case SVE::Type::MALICIOUS_SIMULATION:
    msg = "The program may be malicious:";
    break;
  case SVE::Type::BUMP_SEED:
    msg = "The account's bump seed is not validated and may be vulnerable to "
          "seed canonicalization attacks:";
    break;
  case SVE::Type::INSECURE_PDA_SHARING:
    msg = "The PDA sharing with these seeds is insecure:";
    break;
  case SVE::Type::ARBITRARY_CPI:
    msg = "The CPI may be vulnerable and invoke an arbitrary program: ";
    break;
  default:
    llvm::errs() << "Unhandled type: " << static_cast<int>(type) << "\n";
    llvm_unreachable("unhandled untrustful account");
    break;
  }
  return msg;
}

bool xray::UntrustfulAccount::filter(SVE::Type type, SourceInfo &srcInfo) {
  const llvm::Value *v = srcInfo.getValue();
  // we report at most 1 UntrustfulAccount bug for each function call
  if (apiSigs.find(v) != apiSigs.end()) {
    // llvm::outs() << "filter true:" << srcInfo.sig() << "\n";
    return true;
  }

  apiSigs.insert(v);
  // llvm::outs() << "filter false:" << srcInfo.sig() << "\n";
  return false;
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
    int P, std::string additionalInfo) {
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

  if (SVE::Type::MISS_SIGNER == type) {
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
      if (!additionalInfo.empty()) {
        msg = additionalInfo;
      }
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

nlohmann::json xray::UntrustfulAccount::to_json() const {
  nlohmann::json j({{"priority", p},
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

void xray::UntrustfulAccount::print() const {
  llvm::errs() << "==============VULNERABLE: " << name << "!============\n";
  outs() << "Found a potential vulnerability at line " << apiInst.getLine()
         << ", column " << apiInst.getCol() << " in " << apiInst.getFilename()
         << "\n";
  if (!errorMsg.empty()) {
    outs() << errorMsg << "\n\n";
  } else {
    outs() << description << ":\n\n";
  }
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
