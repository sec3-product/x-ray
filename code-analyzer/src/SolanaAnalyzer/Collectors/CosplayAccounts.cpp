#include "Collectors/CosplayAccounts.h"

#include <algorithm>
#include <cassert>
#include <set>
#include <string>
#include <vector>

#include <llvm/Support/ErrorHandling.h>
#include <nlohmann/json.hpp>

#include "DebugFlags.h"
#include "LogColor.h"
#include "SVE.h"
#include "SourceInfo.h"

// Not to limit the number of bugs we collected
// by default we only collect at most 25 cases for each type of bug
unsigned int xray::CosplayAccounts::budget = 25;
std::vector<xray::CosplayAccounts> xray::CosplayAccounts::cosplayAccounts;
static bool nolimit = false;
int xray::CosplayAccounts::cosplayFullCount = 0;
int xray::CosplayAccounts::cosplayPartialCount = 0;

// used for filtering
std::set<std::string> xray::CosplayAccounts::apiSigs;

void xray::CosplayAccounts::init(int configReportLimit,
                                 bool configNoReportLimit) {
  if (configReportLimit != -1) {
    budget = configReportLimit;
  }
  nolimit = configNoReportLimit;
}

std::string xray::CosplayAccounts::getErrorMsg(SVE::Type type) {
  std::string msg;
  switch (type) {
  case SVE::Type::COSPLAY_FULL:
    msg = "These two data types have the same layout:\n";
    break;
  case SVE::Type::COSPLAY_PARTIAL:
    msg = "These two data types are partially compatible:\n";
    break;

  default:
    llvm::errs() << "Unhandled type: " << static_cast<int>(type) << "\n";
    llvm_unreachable("unhandled CosplayAccounts");
    break;
  }
  return msg;
}

// we report at most 1 UntrustfulAccount bug for each function call
bool xray::CosplayAccounts::filter(SourceInfo &srcInfo) {
  if (apiSigs.find(srcInfo.sig()) != apiSigs.end()) {
    return true;
  }
#pragma omp critical(apiSigs)
  { apiSigs.insert(srcInfo.sig()); }
  return false;
}

void xray::CosplayAccounts::collect(
    const Event *e1, const Event *e2,
    std::map<TID, std::vector<CallEvent *>> &callEventTraces, SVE::Type type,
    int P) {
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
      if (type == SVE::Type::COSPLAY_FULL) 
            cosplayFullCount++;
      else if (type == SVE::Type::COSPLAY_PARTIAL) 
            cosplayPartialCount++;
      if (PRINT_IMMEDIATELY)
        cosplayAccounts.back().print();
      // intentionally commented out since UntrustfulAccount needs improvement
      // if (TERMINATE_IMMEDIATELY) exit(1);
    }
  }
}

xray::CosplayAccounts::CosplayAccounts(SourceInfo &srcInfo1,
                                       SourceInfo &srcInfo2, std::string msg,
                                       SVE::Type t, int P, bool isIgnored,
                                       bool isHidden)
    : apiInst1(srcInfo1), apiInst2(srcInfo2), errorMsg(msg), type(t), p(P),
      ignore(isIgnored), hide(isHidden) {
  id = SVE::getTypeID(t);
  name = SVE::database[id]["name"];
  description = SVE::database[id]["description"];
  url = SVE::database[id]["url"];
}

nlohmann::json xray::CosplayAccounts::to_json() const {
  nlohmann::json j({{"priority", p},
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

void xray::CosplayAccounts::print() const {
  llvm::errs() << "==============VULNERABLE: " << name << "!============\n";
  outs() << description << ":\n";
  outs() << " Data type 1 defined at line " << apiInst1.getLine() << ", column "
         << apiInst1.getCol() << " in " << apiInst1.getFilename() << "\n";
  outs() << apiInst1.getSnippet();
  outs() << ">>>Stack Trace:\n";
  printStackTrace(apiInst1.getStackTrace());
  outs() << " Data type 2 defined at line " << apiInst2.getLine() << ", column "
         << apiInst2.getCol() << " in " << apiInst2.getFilename() << "\n";
  outs() << apiInst2.getSnippet();
  outs() << ">>>Stack Trace:\n";
  printStackTrace(apiInst2.getStackTrace());
  outs() << "\n";
  outs() << "For more info, see " << url << "\n\n\n";
}

void xray::CosplayAccounts::printAll() {
  std::sort(cosplayAccounts.begin(), cosplayAccounts.end());
  for (auto r : cosplayAccounts) {
    r.print();
  }
}

void xray::CosplayAccounts::printSummary() {
  info("detected " + std::to_string(cosplayAccounts.size()) +
       " accounts cosplay issues in total.");
}
