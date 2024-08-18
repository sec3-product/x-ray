#include "Collectors/Races.h"

#include <fstream>

#include "AccountIDL.h"
#include "Collectors/CosplayAccounts.h"
#include "Collectors/UnsafeOperation.h"
#include "Collectors/UntrustfulAccount.h"
#include "DebugFlags.h"
#include "LogColor.h"

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

extern std::set<llvm::StringRef> SMART_CONTRACT_ADDRESSES;
extern llvm::cl::opt<std::string> ConfigOutputPath;
extern cl::opt<std::string> TargetModulePath;

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
  std::sort(UnsafeOperation::unsafeOperations.begin(),
            UnsafeOperation::unsafeOperations.end());
  for (auto &r : UnsafeOperation::unsafeOperations) {
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
