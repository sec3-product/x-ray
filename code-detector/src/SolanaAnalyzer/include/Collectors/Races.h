#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Graph/Event.h"
#include "SVE.h"
#include "SourceInfo.h"

namespace aser {

using json = nlohmann::json;

class OrderViolation {
  SourceInfo access1, access2, sharedObj;
  int priority;

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

/* -----------------------------------

                Utils

-------------------------------------- */

// public API to output all types of bugs into a single JSON file
void outputJSON(std::string);
void ignoreRaceLocations(Event *e1, Event *e2);

} // namespace aser
