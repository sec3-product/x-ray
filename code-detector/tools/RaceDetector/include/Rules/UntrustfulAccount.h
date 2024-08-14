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

class UntrustfulAccount {
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
                    SVE::Type t, int P, bool isIgnored, bool isHidden);
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

} // namespace aser
