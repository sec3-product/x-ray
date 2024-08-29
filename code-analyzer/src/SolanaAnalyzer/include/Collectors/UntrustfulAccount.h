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

class UntrustfulAccount {
public:
  UntrustfulAccount(std::string account, SourceInfo &srcInfo, std::string msg,
                    SVE::Type t, int P, bool isIgnored, bool isHidden);

  static std::vector<UntrustfulAccount> untrustfulAccounts;
  static void init(int configReportLimit, bool configNoReportLimit);
  static void collect(llvm::StringRef accountName, const Event *e,
                      std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                      SVE::Type type, int P);

  static void printAll();
  static void printSummary();

  int getPriority() const { return this->p; }
  json to_json() const;
  void print() const;

  bool operator<(UntrustfulAccount &mapi) const {
    // the race with higher priority should be placed at an earlier place
    if (this->p > mapi.getPriority()) {
      return true;
    }
    return false;
  }

private:
  static bool filter(SVE::Type type, SourceInfo &srcInfo);
  static bool filterByCallStack(std::vector<std::string> &st);
  static std::string getErrorMsg(SVE::Type type);

  int p;
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
  static std::set<const llvm::Value *> apiSigs;
  static std::set<std::vector<std::string>> callStackSigs;
};

} // namespace xray
