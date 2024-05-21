// This file declares a set of serializable data structure
// of all kinds of
#ifndef RACEDETECTOR_RACES_H
#define RACEDETECTOR_RACES_H

#include "RDUtil.h"

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

// the format of JSON output
// https://coderrect.atlassian.net/wiki/spaces/COD/pages/9863332/Reporter
class DataRace {
private:
    int p;
    // memory access 1
    SourceInfo access1;

    // memory access 2
    SourceInfo access2;

    // shared object
    SourceInfo objInfo;
    unsigned objLine;
    std::string objName;
    std::string objTy;
    std::string objDir;
    std::string objFilename;
    std::string objSrcLine;
    std::string objField;

    // OpenMP region information
    bool isOmpRace;
    std::string ompFilename;
    std::string ompDir;
    std::string ompSnippet;
    std::vector<std::string> callingCtx;

    // The max number of race we report (even in the JSON report)
    static unsigned int budget;
    // The max number of openmp race
    static unsigned int omp_budget;
    // race signatures: based on source code information
    static std::set<std::string> raceSigs;
    static std::set<std::string> rawLineSigs;
    static std::map<std::string, bool> methodPairs;

public:
    static void init(int configReportLimit, bool configNoReportLimit);
    // a collection of serializable data races
    static std::vector<DataRace> races;
    // print the text report for all the collected races
    static void printAll();
    // print summary for all the collect races
    // this should be the default terminal behavior for racedetect
    static void printSummary();
    // write `races` into a JSON file
    static void outputJSON();

    // return true if the race is filtered
    // otherwise false
    static bool filter(Event *e1, Event *e2, const ObjTy *obj, SourceInfo &srcInfo1, SourceInfo &srcInfo2,
                       SourceInfo &sharedObj);

    // print & collect regular races (pthread races)
    static void collect(Event *e1, Event *e2, const ObjTy *obj,
                        std::map<TID, std::vector<CallEvent *>> &callEventTraces, int P);
    // print & collect OpenMP races
    // OpenMP will have some extra debugging information
    static void collectOMP(Event *e1, Event *e2, const ObjTy *obj,
                           std::map<TID, std::vector<CallEvent *>> &callEventTraces, CallingCtx &callingCtx,
                           const llvm::Instruction *ompRegion, int P);

    DataRace(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, int P);
    DataRace(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, bool isOmpRace, int P);
    DataRace(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, const std::string &ompFilename,
             const std::string &ompDir, const std::string &snippet, const std::vector<std::string> &callingCtx, int P);

    inline int getPriority() { return this->p; }
    inline SourceInfo &getAccess1() { return this->access1; }
    inline SourceInfo &getAccess2() { return this->access2; }

    void print();

    // generate JSON string
    json to_json();

    inline bool operator<(DataRace &dr) const {
        // the race with higher priority should be placed at an earlier place
        if (this->p > dr.getPriority()) {
            return true;
        }
        // TODO: do we need this for non-determinism?
        // else if (this->access1 < dr.getAccess1()) {
        //     return true;
        // } else if (this->access1 == dr.getAccess1()) {
        //     return this->access2 < dr.getAccess2();
        // }
        return false;
    }
};

class OrderViolation {
    // const MemAccessEvent *e1, *e2;
    SourceInfo access1, access2, sharedObj;
    int priority;

    // static std::map<const MemAccessEvent *, OrderViolation> ovs;
    static std::vector<OrderViolation> ovs;

    static unsigned int budget;

public:
    OrderViolation(SourceInfo access1, SourceInfo access2, SourceInfo sharedObj, int P)
        : access1(access1), access2(access2), sharedObj(sharedObj), priority(P) {}

    void print() const;
    json to_json() const;

    static void collect(const MemAccessEvent *e1, const MemAccessEvent *e2, const ObjTy *obj,
                        std::map<TID, std::vector<CallEvent *>> &callEventTraces, int P);

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

    static void collect(std::vector<const ObjTy *> locks, std::vector<std::vector<const LockEvent *>> dlTraces, int P);
    // print the text report for all the collected races
    static void printAll();
    // print summary for all the collect races
    // this should be the default terminal behavior for racedetect
    static void printSummary();

    DeadLock(std::vector<SourceInfo> &locks, std::vector<std::vector<SourceInfo>> &traces, int P);

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

class MismatchedAPI {
public:
    enum Type { REENTRANT_LOCK, MISS_LOCK, MISS_UNLOCK, MISS_SIGNAL, UNCHECKED_COND_WAIT };

private:
    int p;
    // The potentially buggy API call
    SourceInfo apiInst;
    SourceInfo apiInst2;
    // The error message for this API call
    std::string errorMsg;
    std::string errorMsg2;

    static unsigned int budget;

    static std::set<std::string> apiSigs;

    static bool filter(SourceInfo &srcInfo);

    static std::set<std::vector<std::string>> callStackSigs;

    static bool filterByCallStack(std::vector<std::string> &st);

    static std::string getErrorMsg(Type type);

public:
    static std::vector<MismatchedAPI> mismatchedAPIs;

    static void init(int configReportLimit, bool configNoReportLimit);

    static bool filterUncheckedCondWait(SourceInfo &srcInfo);

    static void collect(const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces, Type type, int P);
    static void collect(const Event *e, const Event *e_last, std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                        Type type, int P);
    // print the text report for all the collected races
    static void printAll();
    // print summary for all the collect races
    // this should be the default terminal behavior for racedetect
    static void printSummary();

    MismatchedAPI(SourceInfo &srcInfo, std::string msg, int P);
    MismatchedAPI(SourceInfo &srcInfo, SourceInfo &srcInfo2, std::string msg, std::string msg2, int P);

    inline int getPriority() { return this->p; }

    json to_json();
    void print();
    inline bool operator<(MismatchedAPI &mapi) const {
        // the race with higher priority should be placed at an earlier place
        if (this->p > mapi.getPriority()) {
            return true;
        }
        return false;
    }
};

class UnSafeOperation {
    // public:
    //     enum Type { ADD, SUB, MUL, DIV };

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

    static void collect(const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces, SVE::Type type,
                        int P);

    // print the text report for all the collected races
    static void printAll();
    // print summary for all the collect races
    // this should be the default terminal behavior for racedetect
    static void printSummary();

    UnSafeOperation(SourceInfo &srcInfo, std::string msg, SVE::Type type, int P, bool isIgnored, bool isHidden);

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
    //     enum Type { UNTRUST, OWNER, SIGNER, UNSAFEMATH, REINIT, DUPLICATE, BUMP, PDA, CLOSE };

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
                        std::map<TID, std::vector<CallEvent *>> &callEventTraces, SVE::Type type, int P);
    // print the text report for all the collected races
    static void printAll();
    // print summary for all the collect races
    // this should be the default terminal behavior for racedetect
    static void printSummary();
    UntrustfulAccount(std::string account, SourceInfo &srcInfo, std::string msg, aser::SVE::Type t, int P,
                      bool isIgnored, bool isHidden);
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
    static void collect(const Event *e1, const Event *e2, std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                        SVE::Type type, int P);
    // print the text report for all the collected races
    static void printAll();
    // print summary for all the collect races
    // this should be the default terminal behavior for racedetect
    static void printSummary();
    CosplayAccounts(SourceInfo &srcInfo1, SourceInfo &srcInfo2, std::string msg, SVE::Type type, int P, bool isIgnored,
                    bool isHidden);
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
// Time of Check Time of Use
class TOCTOU {
private:
    int p;
    // memory access 1
    SourceInfo access1;

    // memory access 2
    SourceInfo access2;

    // shared object
    SourceInfo objInfo;
    unsigned objLine;
    std::string objName;
    std::string objDir;
    std::string objFilename;
    std::string objSrcLine;
    std::string objField;

    // OpenMP region information
    bool isOmpRace;
    std::string ompFilename;
    std::string ompDir;
    std::string ompSnippet;
    std::vector<std::string> callingCtx;

    // The max number of race we report (even in the JSON report)
    static unsigned int budget;
    // The max number of openmp race
    static unsigned int omp_budget;
    // race signatures: based on source code information
    static std::set<std::string> raceSigs;
    static std::set<std::string> rawLineSigs;
    static std::set<std::string> methodPairs;

public:
    static void init(int configReportLimit, bool configNoReportLimit);
    // a collection of serializable data races
    static std::vector<TOCTOU> races;
    // print the text report for all the collected races
    static void printAll();
    // print summary for all the collect races
    // this should be the default terminal behavior for racedetect
    static void printSummary();
    // write `races` into a JSON file
    static void outputJSON();

    // return true if the race is filtered
    // otherwise false
    static bool filter(Event *e1, Event *e2, const ObjTy *obj, SourceInfo &srcInfo1, SourceInfo &srcInfo2,
                       SourceInfo &sharedObj);

    // print & collect regular races (pthread races)
    static void collect(Event *e1, Event *e2, const ObjTy *obj,
                        std::map<TID, std::vector<CallEvent *>> &callEventTraces, int P);

    TOCTOU(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, int P);
    TOCTOU(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, bool isOmpRace, int P);
    TOCTOU(SourceInfo &srcInfo1, SourceInfo &srcInfo2, SourceInfo &objInfo, const std::string &ompFilename,
           const std::string &ompDir, const std::string &snippet, const std::vector<std::string> &callingCtx, int P);

    inline int getPriority() { return this->p; }
    inline SourceInfo &getAccess1() { return this->access1; }
    inline SourceInfo &getAccess2() { return this->access2; }

    void print();

    // generate JSON string
    json to_json();

    inline bool operator<(TOCTOU &dr) const {
        // the race with higher priority should be placed at an earlier place
        if (this->p > dr.getPriority()) {
            return true;
        }
        // TODO: do we need this for non-determinism?
        // else if (this->access1 < dr.getAccess1()) {
        //     return true;
        // } else if (this->access1 == dr.getAccess1()) {
        //     return this->access2 < dr.getAccess2();
        // }
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

}  // namespace aser

#endif