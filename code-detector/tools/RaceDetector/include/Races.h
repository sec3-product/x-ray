#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "RDUtil.h"
#include "SourceInfo.h"

namespace aser {

class SVE {
public:
  enum Type {
    MISS_OWNER,
    MISS_SIGNER,
    OVERFLOW_ADD,
    OVERFLOW_SUB,
    OVERFLOW_MUL,
    OVERFLOW_DIV,
    ACCOUNT_UNVALIDATED_OTHER,
    ACCOUNT_UNVALIDATED_BORROWED,
    ACCOUNT_UNVALIDATED_PDA,
    ACCOUNT_UNVALIDATED_DESTINATION,
    ACCOUNT_INCORRECT_AUTHORITY,
    ACCOUNT_DUPLICATE,
    ACCOUNT_CLOSE,
    COSPLAY_FULL,
    COSPLAY_PARTIAL,
    PDA_SEEDS_COLLISIONS,
    DIV_BY_ZERO,
    REINIT,
    BUMP_SEED,
    INSECURE_PDA_SHARING,
    ARBITRARY_CPI,
    MALICIOUS_SIMULATION,
    OUTDATED_DEPENDENCY,
    UNSAFE_RUST,
    OVERPAY,
    STALE_PRICE_FEED,
    MISS_INIT_TOKEN_MINT,
    MISS_RENT_EXEMPT,
    MISS_FREEZE_AUTHORITY,
    MISS_CPI_RELOAD,
    MISS_ACCESS_CONTROL_UNSTAKE,
    CRITICAL_REDUNDANT_CODE,
    ORDER_RACE_CONDITION,
    FLASHLOAN_RISK,
    BIDIRECTIONAL_ROUNDING,
    CAST_TRUNCATE,
    UNSAFE_SYSVAR_API,
    INCORRECT_BREAK_LOGIC,
    INCORRECT_CONDITION_CHECK,
    EXPONENTIAL_CALCULATION,
    INCORRECT_DIVISION_LOGIC,
    INCORRECT_TOKEN_CALCULATION,
    INSECURE_INIT_IF_NEEDED,
    INSECURE_SPL_TOKEN_CPI,
    INSECURE_ACCOUNT_REALLOC,
    INSECURE_ASSOCIATED_TOKEN,
    ACCOUNT_IDL_INCOMPATIBLE_ADD,
    ACCOUNT_IDL_INCOMPATIBLE_MUT,
    ACCOUNT_IDL_INCOMPATIBLE_ORDER,
    REENTRANCY_ETHER,
    ARBITRARY_SEND_ERC20,
    SUISIDE_SELFDESTRUCT,
    MISS_INIT_UNIQUE_ADMIN_CHECK,
    BIT_SHIFT_OVERFLOW,
    DIV_PRECISION_LOSS,
    VULNERABLE_SIGNED_INTEGER_I128
  };
  static std::map<SVE::Type, std::string> sveTypeIdMap;
  static std::set<std::string> disabledCheckers;
  static void addTypeID(std::string ID, SVE::Type type);
  static std::string getTypeID(SVE::Type type);
  static void addDisabledChecker(std::string ID);
  static bool isCheckerDisabled(SVE::Type type);
};

class OrderViolation {
  // const MemAccessEvent *e1, *e2;
  SourceInfo access1, access2, sharedObj;
  int priority;

  // static std::map<const MemAccessEvent *, OrderViolation> ovs;
  static std::vector<OrderViolation> ovs;

  static unsigned int budget;

public:
  OrderViolation(SourceInfo access1, SourceInfo access2, SourceInfo sharedObj,
                 int P)
      : access1(access1), access2(access2), sharedObj(sharedObj), priority(P) {}

  void print() const;
  json to_json() const;

  static void collect(const MemAccessEvent *e1, const MemAccessEvent *e2,
                      const ObjTy *obj,
                      std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                      int P);

  static size_t getNumOvs() { return ovs.size(); }
  static const std::vector<OrderViolation> &getOvs() { return ovs; }

  static void printAll();
  static void printSummary();
  static void init(int configReportLimit, bool configNoReportLimit);
};

class DeadLock {
private:
  int p;
  // number of locks involved in the races
  // currently we only support 2 & 3
  const int lockNum;
  // the set of locks (abstract obj) being acquired circularly
  std::vector<SourceInfo> locks;
  // the ordered acquiring events for each thread
  // e.g.
  // a 2-thread-2-lock deadlock:
  // dlTraces:
  //      t1: event1 (acquire lock1) -> event2 (acquire lock2)
  //      t2: event3 (acquire lock2) -> event4 (acquire lock1)
  std::vector<std::vector<SourceInfo>> dlTraces;

  json j;

  static unsigned int budget;

public:
  // a collection of serializable race condition
  static std::vector<DeadLock> deadlocks;

  static void init(int configReportLimit, bool configNoReportLimit);

  static void collect(std::vector<const ObjTy *> locks,
                      std::vector<std::vector<const LockEvent *>> dlTraces,
                      int P);
  // print the text report for all the collected races
  static void printAll();
  // print summary for all the collect races
  // this should be the default terminal behavior for racedetect
  static void printSummary();

  DeadLock(std::vector<SourceInfo> &locks,
           std::vector<std::vector<SourceInfo>> &traces, int P);

  inline int getPriority() { return this->p; }

  void print();

  json &to_json();

  inline bool operator<(DeadLock &dl) const {
    // the race with higher priority should be placed at an earlier place
    if (this->p > dl.getPriority()) {
      return true;
    }
    return false;
  }
};

class UnSafeOperation {
private:
  int p;
  // The potentially buggy API call
  SourceInfo apiInst;
  std::string errorMsg;
  SVE::Type type;
  std::string id;
  std::string name;
  std::string description;
  std::string url;
  bool ignore;
  bool hide;
  static unsigned int budget;

  static std::set<std::string> apiSigs;

  static bool filter(SourceInfo &srcInfo);

  static std::set<std::vector<std::string>> callStackSigs;

  static bool filterByCallStack(std::vector<std::string> &st);

  static std::string getErrorMsg(SVE::Type type);

public:
  static std::vector<UnSafeOperation> unsafeOperations;

  static void init(int configReportLimit, bool configNoReportLimit);

  static void collect(const Event *e,
                      std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                      SVE::Type type, int P);

  // print the text report for all the collected races
  static void printAll();
  // print summary for all the collect races
  // this should be the default terminal behavior for racedetect
  static void printSummary();

  UnSafeOperation(SourceInfo &srcInfo, std::string msg, SVE::Type type, int P,
                  bool isIgnored, bool isHidden);

  inline int getPriority() { return this->p; }

  json to_json();
  void print();
  inline bool operator<(UnSafeOperation &mapi) const {
    // the race with higher priority should be placed at an earlier place
    if (this->p > mapi.getPriority()) {
      return true;
    }
    return false;
  }
};

class UntrustfulAccount {
  // public:
  //     enum Type { UNTRUST, OWNER, SIGNER, UNSAFEMATH, REINIT, DUPLICATE,
  //     BUMP, PDA, CLOSE };

private:
  int p;
  // The potentially buggy API call
  SourceInfo apiInst;
  std::string errorMsg;
  SVE::Type type;
  std::string id;
  std::string name;
  std::string accountName;
  std::string description;
  std::string url;
  bool ignore;
  bool hide;
  static unsigned int budget;
  static std::map<SVE::Type, std::set<const llvm::Value *>> apiSigsMap;
  static std::set<std::string> cpiSigs;
  static std::set<const llvm::Value *> apiSigs;
  static bool filter(SVE::Type type, SourceInfo &srcInfo);
  static std::set<std::vector<std::string>> callStackSigs;
  static bool filterByCallStack(std::vector<std::string> &st);
  static std::string getErrorMsg(SVE::Type type);

public:
  static std::vector<UntrustfulAccount> untrustfulAccounts;
  static void init(int configReportLimit, bool configNoReportLimit);
  static void collect(llvm::StringRef accountName, const Event *e,
                      std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                      SVE::Type type, int P);
  // print the text report for all the collected races
  static void printAll();
  // print summary for all the collect races
  // this should be the default terminal behavior for racedetect
  static void printSummary();
  UntrustfulAccount(std::string account, SourceInfo &srcInfo, std::string msg,
                    aser::SVE::Type t, int P, bool isIgnored, bool isHidden);
  inline int getPriority() { return this->p; }
  json to_json();
  void print();
  inline bool operator<(UntrustfulAccount &mapi) const {
    // the race with higher priority should be placed at an earlier place
    if (this->p > mapi.getPriority()) {
      return true;
    }
    return false;
  }
};

class CosplayAccounts {
  // public:
  //     enum Type { FULL, PARTIAL };

private:
  int p;
  SourceInfo apiInst1;
  SourceInfo apiInst2;
  std::string errorMsg;
  SVE::Type type;
  std::string id;
  std::string name;
  std::string description;
  std::string url;
  bool ignore;
  bool hide;
  static unsigned int budget;
  static std::set<std::string> apiSigs;
  static bool filter(SourceInfo &srcInfo);
  static std::set<std::vector<std::string>> callStackSigs;
  static std::string getErrorMsg(SVE::Type type);

public:
  static std::vector<CosplayAccounts> cosplayAccounts;
  static void init(int configReportLimit, bool configNoReportLimit);
  static void collect(const Event *e1, const Event *e2,
                      std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                      SVE::Type type, int P);
  // print the text report for all the collected races
  static void printAll();
  // print summary for all the collect races
  // this should be the default terminal behavior for racedetect
  static void printSummary();
  CosplayAccounts(SourceInfo &srcInfo1, SourceInfo &srcInfo2, std::string msg,
                  SVE::Type type, int P, bool isIgnored, bool isHidden);
  inline int getPriority() { return this->p; }
  json to_json();
  void print();
  inline bool operator<(CosplayAccounts &mapi) const {
    // the race with higher priority should be placed at an earlier place
    if (this->p > mapi.getPriority()) {
      return true;
    }
    return false;
  }
};

/* -----------------------------------

                Utils

-------------------------------------- */

// public API to output all types of bugs into a single JSON file
void outputJSON();
void ignoreRaceLocations(Event *e1, Event *e2);

/* -----------------------------------

    implicitly used by nlohmann/json

-------------------------------------- */
inline void to_json(json &j, const SourceInfo &si) {
  j = json{{"line", si.getLine()},
           {"col", si.getCol()},
           {"filename", si.getFilename()},
           {"dir", si.getDir()},
           {"sourceLine", si.getSourceLine()},
           {"snippet", si.getSnippet()},
           {"stacktrace", si.getStackTrace()}};
};

} // namespace aser
