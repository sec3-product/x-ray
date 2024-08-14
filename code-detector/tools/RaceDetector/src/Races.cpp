#include "Races.h"

#include <fstream>

#include <llvm/Support/Regex.h>

#include "RDUtil.h"

#define DEFAULT_BUDGET 25

using namespace aser;
using namespace std;
using namespace llvm;

// for cost quote
extern unsigned int NUM_OF_IR_LINES;
extern unsigned int NUM_OF_ATTACK_VECTORS;
extern unsigned int NUM_OF_FUNCTIONS;
extern unsigned int TOTAL_SOL_COST;
extern unsigned int TOTAL_SOL_TIME;
extern set<StringRef> SMART_CONTRACT_ADDRESSES;
extern llvm::cl::opt<std::string> ConfigOutputPath;
extern cl::opt<std::string> TargetModulePath;
extern bool CONFIG_NO_FILTER;

extern bool PRINT_IMMEDIATELY;
extern bool TERMINATE_IMMEDIATELY;

// Not to limit the number of bugs we collected
// by default we only collect at most 25 cases for each type of bug
static bool nolimit = false;

/* --------------------------------

            Order Violation

----------------------------------- */

std::vector<aser::OrderViolation> aser::OrderViolation::ovs;
uint aser::OrderViolation::budget = DEFAULT_BUDGET;

void aser::OrderViolation::collect(
    const MemAccessEvent *e1, const MemAccessEvent *e2, const ObjTy *obj,
    std::map<TID, std::vector<CallEvent *>> &callEventTraces, int P) {
  SourceInfo srcInfo1 = getSourceLoc(e1->getInst());
  SourceInfo srcInfo2 = getSourceLoc(e2->getInst());
  SourceInfo sharedObjLoc = getSourceLoc(obj->getValue());

  // TODO: real ov filter
  static std::set<std::string> ovfilter;
  std::string sig1 = getRaceRawLineSig(srcInfo1);
  std::string sig2 = getRaceRawLineSig(srcInfo2);
  if (filterStrPattern(sig1) || filterStrPattern(sig2)) {
    return;
  }
  if (sig1.empty() || sig2.empty()) {
    return;
  }
  if (ovfilter.find(sig1) != ovfilter.end() ||
      ovfilter.find(sig2) != ovfilter.end()) {
    return;
  }

  auto st1 = getStackTrace(e1, callEventTraces, true);
  auto st2 = getStackTrace(e2, callEventTraces, true);

  P = customizedPriorityAdjust(P, sharedObjLoc.getName(), srcInfo1, srcInfo2,
                               st1, st2);
#pragma omp critical(ovfilter)
  {
    ovfilter.insert(sig1);
    ovfilter.insert(sig2);
  }
#pragma omp critical(budget)
  {
    if (nolimit || budget > 0) {
      srcInfo1.setStackTrace(st1);
      srcInfo2.setStackTrace(st2);
      sharedObjLoc.setAccessPath(obj->getFieldAccessPath());
      if (sharedObjLoc.isGlobalValue())
        P++; // P = Priority::LEVEL6;

      if (e1->getType() == EventType::Write ||
          e1->getType() == EventType::APIWrite)
        srcInfo1.setWrite();
      if (e2->getType() == EventType::Write ||
          e2->getType() == EventType::APIWrite)
        srcInfo2.setWrite();

      ovs.emplace_back(srcInfo1, srcInfo2, sharedObjLoc, P);
      --budget;
      if (PRINT_IMMEDIATELY)
        ovs.back().print();
      if (TERMINATE_IMMEDIATELY)
        exit(1);
    }
  }
}

void aser::OrderViolation::print() const {
  outs() << "\n==== Found an Order Violation between: \n"
         << "line " << access1.getLine() << ", column " << access1.getCol()
         << " in " << access1.getFilename() << " AND "
         << "line " << access2.getLine() << ", column " << access2.getCol()
         << " in " << access2.getFilename() << "\n";

  if (access1.isWrite())
    highlight("Thread 1 (write): ");
  else
    highlight("Thread 1 (read): ");
  outs() << access1.getSnippet();
  outs() << ">>>Stack Trace:\n";
  printStackTrace(access1.getStackTrace());
  if (access2.isWrite())
    highlight("Thread 2 (write): ");
  else
    highlight("Thread 2 (read): ");
  outs() << access2.getSnippet();
  outs() << ">>>Stack Trace:\n";
  printStackTrace(access2.getStackTrace());
}

json aser::OrderViolation::to_json() const {
  json ov;
  ov["priority"] = priority;
  ov["access1"] = access1;
  ov["access2"] = access2;
  return ov;
}

void aser::OrderViolation::printAll() {
  for (auto const &ov : OrderViolation::getOvs()) {
    ov.print();
  }
}

void aser::OrderViolation::printSummary() {
  info("detected " + to_string(OrderViolation::getNumOvs()) +
       " order violations in total.");
}

void aser::OrderViolation::init(int configReportLimit,
                                bool configNoReportLimit) {
  if (configReportLimit != -1) {
    budget = configReportLimit;
  }
  nolimit = configNoReportLimit;
}

/* --------------------------------

            Dead Lock

----------------------------------- */

// static fields
uint aser::DeadLock::budget = DEFAULT_BUDGET;
vector<DeadLock> aser::DeadLock::deadlocks;

void aser::DeadLock::init(int configReportLimit, bool configNoReportLimit) {
  if (configReportLimit != -1) {
    budget = configReportLimit;
  }
  nolimit = configNoReportLimit;
}

aser::DeadLock::DeadLock(std::vector<SourceInfo> &locks,
                         std::vector<std::vector<SourceInfo>> &traces, int P)
    : lockNum(locks.size()), locks(locks), dlTraces(traces), p(P) {}

// TODO: how to filter deadlock
// TODO: does deadlock need stacktrace?
void aser::DeadLock::collect(
    std::vector<const ObjTy *> locks,
    std::vector<std::vector<const LockEvent *>> dlTraces, int P) {
#pragma omp critical(budget)
  {
    if (nolimit || budget > 0) {
      auto size = dlTraces.size();
      std::vector<std::vector<SourceInfo>> dlInfo(size);
      // sort the acquiring events into their happening order
      // and get their source info
      for (int idx = 0; idx < size; idx++) {
        auto &trace = dlTraces[idx];
        std::sort(trace.begin(), trace.end(),
                  [](const LockEvent *e1, const LockEvent *e2) -> bool {
                    return (e1->getID() < e2->getID());
                  });
        for (auto e : trace) {
          dlInfo[idx].push_back(aser::getSourceLoc(e->getInst()));
        }
      }
      std::vector<SourceInfo> lockInfo;
      for (auto o : locks) {
        lockInfo.push_back(aser::getSourceLoc(o->getValue()));
      }
      // collect the serializable deadlock info info static field
      deadlocks.emplace_back(lockInfo, dlInfo, P);
      if (PRINT_IMMEDIATELY)
        deadlocks.back().print();
      if (TERMINATE_IMMEDIATELY)
        exit(1);
      --budget;
    }
  }
}

void aser::DeadLock::printAll() {
  for (auto dl : deadlocks) {
    dl.print();
  }
}

void aser::DeadLock::printSummary() {
  info("detected " + to_string(deadlocks.size()) + " deadlocks in total.");
}

void aser::DeadLock::print() {
  // TODO show full call stack
  outs() << "\nFound a potential deadlock:\n";
  // int lockId=1;
  // for(auto l: locks){
  //     outs()<<"  lock"<<lockId++<<": "<<l.str();
  // }
  int threadId = 1;
  for (auto trace : dlTraces) {
    outs() << "====thread" << threadId++ << " lock history====\n";
    for (auto e : trace) {
      outs() << e.str();
    }
  }
}

json &aser::DeadLock::to_json() {
  if (j.empty()) {
    j["acquires"] = dlTraces;
    j = json{{"priority", p},
             {"number", lockNum},
             {"locks", locks},
             {"acquires", dlTraces}};
  }
  return j;
}

/* --------------------------------

           CosplayAccounts

----------------------------------- */

// static fields
uint aser::CosplayAccounts::budget = DEFAULT_BUDGET;
vector<CosplayAccounts> aser::CosplayAccounts::cosplayAccounts;
// used for filtering
set<string> aser::CosplayAccounts::apiSigs;

void aser::CosplayAccounts::init(int configReportLimit,
                                 bool configNoReportLimit) {
  if (configReportLimit != -1) {
    budget = configReportLimit;
  }
  nolimit = configNoReportLimit;
}

string aser::CosplayAccounts::getErrorMsg(SVE::Type type) {
  string msg;
  switch (type) {
  case SVE::Type::COSPLAY_FULL:
    msg = "These two data types have the same layout:\n";
    break;
  case SVE::Type::COSPLAY_PARTIAL:
    msg = "These two data types are partially compatible:\n";
    break;
  case SVE::Type::ACCOUNT_DUPLICATE:
    msg = "The data type may contain duplicated mutable accounts:";
    break;
  case SVE::Type::PDA_SEEDS_COLLISIONS:
    msg = "These two PDA accounts may have the same seeds, which may lead to "
          "PDA collisions:";
    break;
  case SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_ORDER:
    msg = "These two accounts are reordered in the instruction and may break "
          "the ABI of the deployed on-chain "
          "program, according to the IDL available on Anchor:";
    break;
  default:
    assert(false && "unhandled CosplayAccounts");
    break;
  }
  return msg;
}

// we report at most 1 UntrustfulAccount bug for each function call
bool aser::CosplayAccounts::filter(SourceInfo &srcInfo) {
  if (apiSigs.find(srcInfo.sig()) != apiSigs.end()) {
    return true;
  }
#pragma omp critical(apiSigs)
  { apiSigs.insert(srcInfo.sig()); }
  return false;
}

void aser::CosplayAccounts::collect(
    const Event *e1, const Event *e2,
    map<TID, vector<CallEvent *>> &callEventTraces, SVE::Type type, int P) {
  SourceInfo srcInfo1 = getSourceLoc(e1->getInst());
  SourceInfo srcInfo2 = getSourceLoc(e2->getInst());

  std::string sig = getRaceSrcSig(srcInfo1, srcInfo2);
  if (apiSigs.find(sig) != apiSigs.end()) {
    return;
  } else {
    apiSigs.insert(sig);
  }

  bool isIgnored = false;
  bool isHidden = false;
  if (SVE::isCheckerDisabled(type))
    isHidden = true;
  if (customizedFilterIgnoreLocations(e1, e2)) {
    isIgnored = true;
  }

  std::vector<std::string> st1;
  std::vector<std::string> st2;

  TID tid1 = e1->getTID();
  TID tid2 = e1->getTID();
  EventID id1 = e1->getID();
  EventID id2 = e2->getID();

  auto &callTrace1 = callEventTraces[tid1];
  for (CallEvent *ce : callTrace1) {
    if (ce->getID() > id1)
      break;
    if (ce->getEndID() == 0 || ce->getEndID() > id1) {
      st1.push_back(ce->getCallSiteString(true));
    }
  }
  auto &callTrace2 = callEventTraces[tid2];
  for (CallEvent *ce : callTrace2) {
    if (ce->getID() > id2)
      break;
    if (ce->getEndID() == 0 || ce->getEndID() > id2) {
      st2.push_back(ce->getCallSiteString(true));
    }
  }
#pragma omp critical(budget)
  {
    if (nolimit || budget > 0) {
      srcInfo1.setStackTrace(st1);
      srcInfo2.setStackTrace(st1);
      auto msg = getErrorMsg(type); // TODO: add source location
      if (srcInfo1.getLine() < srcInfo2.getLine())
        cosplayAccounts.emplace_back(srcInfo1, srcInfo2, msg, type, P,
                                     isIgnored, isHidden);
      else
        cosplayAccounts.emplace_back(srcInfo2, srcInfo1, msg, type, P,
                                     isIgnored, isHidden);
      --budget;
      if (PRINT_IMMEDIATELY)
        cosplayAccounts.back().print();
      // intentionally commented out since UntrustfulAccount needs improvement
      // if (TERMINATE_IMMEDIATELY) exit(1);
    }
  }
}

aser::CosplayAccounts::CosplayAccounts(SourceInfo &srcInfo1,
                                       SourceInfo &srcInfo2, std::string msg,
                                       SVE::Type t, int P, bool isIgnored,
                                       bool isHidden)
    : apiInst1(srcInfo1), apiInst2(srcInfo2), errorMsg(msg), type(t), p(P),
      ignore(isIgnored), hide(isHidden) {
  id = SVE::getTypeID(t);
  name = SVE::SOLANA_SVE_DB[id]["name"];
  description = SVE::SOLANA_SVE_DB[id]["description"];
  url = SVE::SOLANA_SVE_DB[id]["url"];
}

json aser::CosplayAccounts::to_json() {
  json j({{"priority", p},
          {"inst1", apiInst1},
          {"inst2", apiInst2},
          {"errorMsg", errorMsg},
          {"id", id},
          {"hide", hide},
          {"ignore", ignore},
          {"description", description},
          {"url", url}});
  return j;
}

void aser::CosplayAccounts::print() {
  // llvm::errs() << "==============VULNERABLE: Possible Accounts Cosplay
  // Attacks!============\n"; outs() << errorMsg; outs() << " Data Type1 defined
  // at line " << apiInst1.getLine() << ", column " << apiInst1.getCol() << " in
  // "
  //        << apiInst1.getFilename() << "\n";
  // outs() << apiInst1.getSnippet();
  // outs() << ">>>Stack Trace:\n";
  // printStackTrace(apiInst1.getStackTrace());
  // outs() << " Data Type2 defined at line " << apiInst2.getLine() << ", column
  // " << apiInst2.getCol() << " in "
  //        << apiInst2.getFilename() << "\n";
  // outs() << apiInst2.getSnippet();
  // outs() << ">>>Stack Trace:\n";
  // printStackTrace(apiInst2.getStackTrace());
  // outs() << "\n";
  outs() << "ignored: " << ignore << "\n";
  llvm::errs() << "==============VULNERABLE: " << name << "!============\n";
  outs() << description << ":\n";
  auto desc_type = "Data type";
  if (type == SVE::Type::PDA_SEEDS_COLLISIONS)
    desc_type = "PDA account";
  else if (type == SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_ORDER)
    desc_type = "Account";
  outs() << " " << desc_type << "1 defined at line " << apiInst1.getLine()
         << ", column " << apiInst1.getCol() << " in " << apiInst1.getFilename()
         << "\n";
  outs() << apiInst1.getSnippet();
  outs() << ">>>Stack Trace:\n";
  printStackTrace(apiInst1.getStackTrace());
  outs() << " " << desc_type << "2 defined at line " << apiInst2.getLine()
         << ", column " << apiInst2.getCol() << " in " << apiInst2.getFilename()
         << "\n";
  outs() << apiInst2.getSnippet();
  outs() << ">>>Stack Trace:\n";
  printStackTrace(apiInst2.getStackTrace());
  outs() << "\n";
  outs() << "For more info, see " << url << "\n\n\n";
}

void aser::CosplayAccounts::printAll() {
  std::sort(cosplayAccounts.begin(), cosplayAccounts.end());
  for (auto r : cosplayAccounts) {
    r.print();
  }
}

void aser::CosplayAccounts::printSummary() {
  info("detected " + to_string(cosplayAccounts.size()) +
       " accounts cosplay issues in total.");
}

/* --------------------------------

           UntrustfulAccount

----------------------------------- */

// static fields
uint aser::UntrustfulAccount::budget = DEFAULT_BUDGET;
vector<UntrustfulAccount> aser::UntrustfulAccount::untrustfulAccounts;
// used for filtering
// std::map<SVE::Type, set<const llvm::Value *>>
// aser::UntrustfulAccount::apiSigsMap;
std::set<const llvm::Value *> aser::UntrustfulAccount::apiSigs;
std::set<std::string> aser::UntrustfulAccount::cpiSigs;

std::set<std::vector<std::string>> aser::UntrustfulAccount::callStackSigs;

void aser::UntrustfulAccount::init(int configReportLimit,
                                   bool configNoReportLimit) {
  if (configReportLimit != -1) {
    budget = configReportLimit;
  }
  nolimit = configNoReportLimit;
}

string aser::UntrustfulAccount::getErrorMsg(SVE::Type type) {
  string msg;
  switch (type) {
  case SVE::Type::ACCOUNT_UNVALIDATED_BORROWED:
    msg = "The account is not validated before parsing its data:";
    break;
  case SVE::Type::ACCOUNT_UNVALIDATED_OTHER:
    msg = "The account is not properly validated and may be untrustful:";
    // msg = "The account info is not trustful:\n";
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
  case SVE::Type::MISS_CPI_RELOAD:
    msg = "The token account is missing reload after CPI:";
    break;
  case SVE::Type::MISS_ACCESS_CONTROL_UNSTAKE:
    msg = "The unstake instruction may be missing an access_control account "
          "validation:";
    break;
  case SVE::Type::ARBITRARY_CPI:
    msg = "The CPI may invoke an arbitrary program:";
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
  case SVE::Type::UNSAFE_SYSVAR_API:
    msg = "The sysvar::instructions API is unsafe and deprecated:";
    break;
  case SVE::Type::DIV_BY_ZERO:
    msg = "The arithmetic operation may result in a div-by-zero error:";
    break;
  case SVE::Type::REINIT:
    msg = "The account is vulnerable to program re-initialization:";
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
bool aser::UntrustfulAccount::filter(SVE::Type type, SourceInfo &srcInfo) {
  // for CPI
  if (SVE::Type::ARBITRARY_CPI == type || SVE::Type::ACCOUNT_CLOSE == type) {
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
bool aser::UntrustfulAccount::filterByCallStack(std::vector<std::string> &st0) {
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

void aser::UntrustfulAccount::collect(
    llvm::StringRef accountName, const Event *e,
    map<TID, vector<CallEvent *>> &callEventTraces, SVE::Type type, int P) {
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

aser::UntrustfulAccount::UntrustfulAccount(std::string account,
                                           SourceInfo &srcInfo, std::string msg,
                                           SVE::Type t, int P, bool isIgnored,
                                           bool isHidden)
    : apiInst(srcInfo), errorMsg(msg), type(t), accountName(account), p(P),
      ignore(isIgnored), hide(isHidden) {
  id = SVE::getTypeID(t);
  name = SVE::SOLANA_SVE_DB[id]["name"];
  description = SVE::SOLANA_SVE_DB[id]["description"];
  url = SVE::SOLANA_SVE_DB[id]["url"];
}

json aser::UntrustfulAccount::to_json() {
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
void aser::UntrustfulAccount::print() {
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
void aser::UntrustfulAccount::printAll() {
  std::sort(untrustfulAccounts.begin(), untrustfulAccounts.end());
  for (auto r : untrustfulAccounts) {
    r.print();
  }
}

void aser::UntrustfulAccount::printSummary() {
  info("detected " + to_string(untrustfulAccounts.size()) +
       " untrustful accounts in total.");
}

/* --------------------------------

           UnSafeOperation

----------------------------------- */

// static fields
uint aser::UnSafeOperation::budget = DEFAULT_BUDGET;
vector<UnSafeOperation> aser::UnSafeOperation::unsafeOperations;
// used for filtering
set<string> aser::UnSafeOperation::apiSigs;
std::set<std::vector<std::string>> aser::UnSafeOperation::callStackSigs;

void aser::UnSafeOperation::init(int configReportLimit,
                                 bool configNoReportLimit) {
  if (configReportLimit != -1) {
    budget = configReportLimit;
  }
  nolimit = configNoReportLimit;
}

string aser::UnSafeOperation::getErrorMsg(SVE::Type type) {
  string msg;
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
    assert(false && "Unhandled UnSafeOperation");
    break;
  }
  return msg;
}

bool aser::UnSafeOperation::filter(SourceInfo &srcInfo) {
  if (apiSigs.find(srcInfo.sig()) != apiSigs.end()) {
    return true;
  }
#pragma omp critical(apiSigs)
  { apiSigs.insert(srcInfo.sig()); }
  return false;
}

bool aser::UnSafeOperation::filterByCallStack(std::vector<std::string> &st0) {
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
void aser::UnSafeOperation::collect(
    const Event *e, map<TID, vector<CallEvent *>> &callEventTraces,
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

aser::UnSafeOperation::UnSafeOperation(SourceInfo &srcInfo, std::string msg,
                                       SVE::Type t, int P, bool isIgnored,
                                       bool isHidden)
    : apiInst(srcInfo), errorMsg(msg), type(t), p(P), ignore(isIgnored),
      hide(isHidden) {
  id = SVE::getTypeID(t);
  name = SVE::SOLANA_SVE_DB[id]["name"];
  description = SVE::SOLANA_SVE_DB[id]["description"];
  url = SVE::SOLANA_SVE_DB[id]["url"];
}

json aser::UnSafeOperation::to_json() {
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
void aser::UnSafeOperation::print() {
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
void aser::UnSafeOperation::printAll() {
  std::sort(unsafeOperations.begin(), unsafeOperations.end());
  for (auto r : unsafeOperations) {
    r.print();
  }
}

void aser::UnSafeOperation::printSummary() {
  info("detected " + to_string(unsafeOperations.size()) +
       " unsafe operations in total.");
}

/* --------------------------------

                Utils

----------------------------------- */
extern llvm::cl::opt<std::string> ConfigOutputPath;

void aser::outputJSON() {
  std::string path;
  if (!ConfigOutputPath.empty()) {
    path = ConfigOutputPath;
  } else {
    info("writing detection results to ./races.json");
    path = "races.json";
  }

  std::vector<json> ovJsons;
  for (auto const &ov : OrderViolation::getOvs()) {
    ovJsons.emplace_back(ov.to_json());
  }

  std::vector<json> dlJsons;
  for (auto &r : DeadLock::deadlocks) {
    dlJsons.emplace_back(r.to_json());
  }

  std::vector<json> uaccountsJsons;
  std::sort(UntrustfulAccount::untrustfulAccounts.begin(),
            UntrustfulAccount::untrustfulAccounts.end());
  for (auto &r : UntrustfulAccount::untrustfulAccounts) {
    uaccountsJsons.emplace_back(r.to_json());
  }
  std::vector<json> usafeOperationsJsons;
  std::sort(UnSafeOperation::unsafeOperations.begin(),
            UnSafeOperation::unsafeOperations.end());
  for (auto &r : UnSafeOperation::unsafeOperations) {
    usafeOperationsJsons.emplace_back(r.to_json());
  }
  std::vector<json> cosplayAccountsJsons;
  std::sort(CosplayAccounts::cosplayAccounts.begin(),
            CosplayAccounts::cosplayAccounts.end());
  for (auto &r : CosplayAccounts::cosplayAccounts) {
    cosplayAccountsJsons.emplace_back(r.to_json());
  }

  json rs;
  rs["raceConditions"] = std::vector<json>();
  rs["orderViolations"] = ovJsons;
  rs["deadLocks"] = dlJsons;
  rs["untrustfulAccounts"] = uaccountsJsons;
  rs["unsafeOperations"] = usafeOperationsJsons;
  rs["cosplayAccounts"] = cosplayAccountsJsons;
  rs["version"] = 1;
  rs["generatedAt"] = getCurrentTimeStr();
  rs["bcfile"] = TargetModulePath.getValue();
  rs["numOfIRLines"] = NUM_OF_IR_LINES;
  rs["numOfAttackVectors"] = NUM_OF_ATTACK_VECTORS;
  rs["numOfFunctions"] = NUM_OF_FUNCTIONS;
  rs["addresses"] = SMART_CONTRACT_ADDRESSES;
  std::ofstream output(path, std::ofstream::out);
  output << rs;
  output.close();
}
