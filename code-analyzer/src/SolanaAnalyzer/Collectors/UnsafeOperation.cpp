#include "Collectors/UnsafeOperation.h"

#include <algorithm>
#include <cassert>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "LogColor.h"

#define DEFAULT_BUDGET 25

using json = nlohmann::json;

extern bool PRINT_IMMEDIATELY;
extern bool TERMINATE_IMMEDIATELY;

// Not to limit the number of bugs we collected
// by default we only collect at most 25 cases for each type of bug
static bool nolimit = false;

/* --------------------------------

           UnsafeOperation

----------------------------------- */

// static fields
uint xray::UnsafeOperation::budget = DEFAULT_BUDGET;
std::vector<xray::UnsafeOperation> xray::UnsafeOperation::unsafeOperations;

// used for filtering
std::set<std::string> xray::UnsafeOperation::apiSigs;
std::set<std::vector<std::string>> xray::UnsafeOperation::callStackSigs;

void xray::UnsafeOperation::init(int configReportLimit,
                                 bool configNoReportLimit) {
  if (configReportLimit != -1) {
    budget = configReportLimit;
  }
  nolimit = configNoReportLimit;
}

std::string xray::UnsafeOperation::getErrorMsg(SVE::Type type) {
  std::string msg;
  switch (type) {
  case SVE::Type::OVERFLOW_ADD:
    msg = "The add operation may result in overflows:\n";
    break;
  case SVE::Type::OVERFLOW_SUB:
    msg = "The sub operation may result in underflows:\n";
    break;
  case SVE::Type::OVERFLOW_MUL:
    msg = "The mul operation may result in overflows:\n";
    break;
  case SVE::Type::OVERFLOW_DIV:
    msg =
        "The div operation may result in divide-by-zero errors or overflows:\n";
    break;
  case SVE::Type::INCORRECT_BREAK_LOGIC:
    msg = "Loop break instead of continue (jet-v1 exploit):\n";
    break;
  case SVE::Type::BIDIRECTIONAL_ROUNDING:
    msg =
        "The arithmetics here may suffer from inconsistent rounding errors:\n";
    break;
  case SVE::Type::CAST_TRUNCATE:
    msg = "The cast operation here may lose precision due to truncation:\n";
    break;
  case SVE::Type::CRITICAL_REDUNDANT_CODE:
    msg = "The code may be redundant or unused, but appears critical:";
    break;
  case SVE::Type::ORDER_RACE_CONDITION:
    msg = "The instruction may suffer from a race condition between order "
          "cancellation and order recreation by "
          "an attacker:";
    break;
  case SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_ADD:
    msg = "The account may break the ABI of the deployed on-chain program as "
          "it does not exist in the IDL "
          "available on Anchor:";
    break;
  case SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_MUT:
    msg = "The mutable account may break the ABI of the deployed on-chain "
          "program as it is immutable according "
          "to the IDL available on Anchor:";
    break;
  case SVE::Type::REENTRANCY_ETHER:
    msg = "The function may suffer from reentrancy attacks due to the use of "
          "call.value, which can invoke an "
          "external contract's fallback function:";
    break;
  case SVE::Type::ARBITRARY_SEND_ERC20:
    msg = "The function may allow an attacker to send from an arbitrary "
          "address, instead of from the msg.sender:";
    break;
  case SVE::Type::SUISIDE_SELFDESTRUCT:
    msg = "The function may allow an attacker to destruct the contract:";
    break;
  case SVE::Type::MISS_INIT_UNIQUE_ADMIN_CHECK:
    msg = "The init function misses checking admin uniqueness and may allow an "
          "attacker to call the init "
          "function more than once:";
    break;
  case SVE::Type::BIT_SHIFT_OVERFLOW:
    msg = "The bit shift operation may result in overflows:";
    break;
  case SVE::Type::DIV_PRECISION_LOSS:
    msg = "The division operation here may lose precision:\n";
    break;
  case SVE::Type::VULNERABLE_SIGNED_INTEGER_I128:
    msg = "The I128 signed integer implementation in Move may be vulnerable "
          "and is not recommended:\n";
    break;
  case SVE::Type::INCORRECT_TOKEN_CALCULATION:
    msg = "The token amount calculation may be incorrect. Consider using the "
          "reserves instead of the balances:\n";
    break;

  default:
    assert(false && "Unhandled UnsafeOperation");
    break;
  }
  return msg;
}

bool xray::UnsafeOperation::filter(SourceInfo &srcInfo) {
  if (apiSigs.find(srcInfo.sig()) != apiSigs.end()) {
    return true;
  }
#pragma omp critical(apiSigs)
  { apiSigs.insert(srcInfo.sig()); }
  return false;
}

bool xray::UnsafeOperation::filterByCallStack(std::vector<std::string> &st0) {
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

extern bool hasOverFlowChecks;
void xray::UnsafeOperation::collect(
    const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces,
    SVE::Type type, int P) {
  if (hasOverFlowChecks && type != SVE::Type::CAST_TRUNCATE)
    return;

  SourceInfo srcInfo = getSourceLoc(e->getInst());
  if (filter(srcInfo))
    return;

  bool isHidden = false;
  bool isIgnored = false;
  if (SVE::isCheckerDisabled(type))
    isHidden = true;

  if (type == SVE::Type::CRITICAL_REDUNDANT_CODE) {
    bool isRedundantIgnored =
        customizedFilterSoteriaIgnoreSymbol(e, "redundant");
    if (isRedundantIgnored)
      isIgnored = true;
  }

  // std::vector<std::string> st = getStackTrace(e, callEventTraces,
  // srcInfo.isCpp());
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
      unsafeOperations.emplace_back(srcInfo, msg, type, P, isIgnored, isHidden);
      --budget;
      if (PRINT_IMMEDIATELY)
        unsafeOperations.back().print();
      // intentionally commented out since unsafeOperations needs improvement
      // if (TERMINATE_IMMEDIATELY) exit(1);
    }
  }
}

xray::UnsafeOperation::UnsafeOperation(SourceInfo &srcInfo, std::string msg,
                                       SVE::Type t, int P, bool isIgnored,
                                       bool isHidden)
    : apiInst(srcInfo), errorMsg(msg), type(t), p(P), ignore(isIgnored),
      hide(isHidden) {
  id = SVE::getTypeID(t);
  name = SVE::SOLANA_SVE_DB[id]["name"];
  description = SVE::SOLANA_SVE_DB[id]["description"];
  url = SVE::SOLANA_SVE_DB[id]["url"];
}

json xray::UnsafeOperation::to_json() {
  json j({{"priority", p},
          {"inst", apiInst},
          {"errorMsg", errorMsg},
          {"id", id},
          {"hide", hide},
          {"ignore", ignore},
          {"description", description},
          {"url", url}});
  return j;
}

void xray::UnsafeOperation::print() {
  // llvm::outs() << "=============This arithmetic operation may be
  // UNSAFE!================\n"; outs() << "Found a potential vulnerability at
  // line " << apiInst.getLine() << ", column " << apiInst.getCol()
  //        << " in " << apiInst.getFilename() << "\n";
  // outs() << errorMsg << "\n";
  // outs() << apiInst.getSnippet();
  // outs() << ">>>Stack Trace:\n";
  // printStackTrace(apiInst.getStackTrace());
  // outs() << "\n";
  outs() << "ignored: " << ignore << "\n";
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

void xray::UnsafeOperation::printAll() {
  std::sort(unsafeOperations.begin(), unsafeOperations.end());
  for (auto r : unsafeOperations) {
    r.print();
  }
}

void xray::UnsafeOperation::printSummary() {
  info("detected " + std::to_string(unsafeOperations.size()) +
       " unsafe operations in total.");
}

