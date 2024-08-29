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

class UnsafeOperation {
public:
  UnsafeOperation(SourceInfo &srcInfo, std::string msg, SVE::Type type, int P,
                  bool isIgnored, bool isHidden);

  static void init(int configReportLimit, bool configNoReportLimit);
  static void collect(const Event *e,
                      std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                      SVE::Type type, int P);

  static void printAll();
  static void printSummary();

  int getPriority() const { return this->p; }
  nlohmann::json to_json() const;
  void print() const;

  bool operator<(UnsafeOperation &mapi) const {
    // the race with higher priority should be placed at an earlier place
    if (this->p > mapi.getPriority()) {
      return true;
    }
    return false;
  }

  static std::vector<UnsafeOperation> unsafeOperations;

private:
  static bool filter(SourceInfo &srcInfo);
  static bool filterByCallStack(std::vector<std::string> &st);
  static std::string getErrorMsg(SVE::Type type);

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
  static std::set<std::vector<std::string>> callStackSigs;
};

} // namespace xray
