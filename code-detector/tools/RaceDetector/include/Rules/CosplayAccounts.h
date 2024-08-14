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

class CosplayAccounts {
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

} // namespace aser
