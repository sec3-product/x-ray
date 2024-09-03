#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Graph/Event.h"
#include "SVE.h"
#include "SourceInfo.h"

namespace xray {

using json = nlohmann::json;

class CosplayAccounts {
public:
  CosplayAccounts(SourceInfo &srcInfo1, SourceInfo &srcInfo2, std::string msg,
                  SVE::Type type, int P, bool isIgnored, bool isHidden);

  static void init(int configReportLimit, bool configNoReportLimit);
  static void collect(const Event *e1, const Event *e2,
                      std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                      SVE::Type type, int P);

  static void printAll();
  static void printSummary();
  int getPriority() const { return this->p; }
  json to_json() const;
  void print() const;
  bool operator<(CosplayAccounts &mapi) const {
    // the race with higher priority should be placed at an earlier place
    return (this->p > mapi.getPriority());
  }

  static std::vector<CosplayAccounts> cosplayAccounts;

  static int cosplayFullCount;
  static int cosplayPartialCount;
private:
  static bool filter(SourceInfo &srcInfo);
  static std::string getErrorMsg(SVE::Type type);

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
  static std::set<std::vector<std::string>> callStackSigs;
};

} // namespace xray
