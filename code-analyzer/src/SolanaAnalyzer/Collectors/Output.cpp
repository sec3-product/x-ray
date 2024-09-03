#include "Collectors/Output.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "AccountIDL.h"
#include "Collectors/CosplayAccount.h"
#include "Collectors/UnsafeOperation.h"
#include "Collectors/UntrustfulAccount.h"
#include "DebugFlags.h"
#include "LogColor.h"
#include "SolanaAnalysisPass.h"

using json = nlohmann::json;

namespace xray {

std::string getCurrentTimeStr() {
  auto t = std::time(nullptr);
  auto local = *std::localtime(&t);

  std::ostringstream tzoss;
  auto offset = std::localtime(&t)->tm_gmtoff / 3600;
  tzoss << "GMT" << (offset >= 0 ? "+" : "") << offset;

  std::ostringstream oss;
  oss << std::put_time(&local, "%a %d %b %Y %T %p") << " " << tzoss.str();
  return oss.str();
}

void outputJSON(std::string OutputPath) {
  std::string path;
  if (!OutputPath.empty()) {
    path = OutputPath;
  } else {
    info("writing detection results to ./races.json");
    path = "races.json";
  }

  std::vector<json> uaccountsJsons;
  std::sort(UntrustfulAccount::untrustfulAccounts.begin(),
            UntrustfulAccount::untrustfulAccounts.end());
  for (const auto &r : UntrustfulAccount::untrustfulAccounts) {
    uaccountsJsons.emplace_back(r.to_json());
  }

  std::vector<json> usafeOperationsJsons;
  std::sort(UnsafeOperation::unsafeOperations.begin(),
            UnsafeOperation::unsafeOperations.end());
  for (const auto &r : UnsafeOperation::unsafeOperations) {
    usafeOperationsJsons.emplace_back(r.to_json());
  }

  std::vector<json> cosplayAccountsJsons;
  std::sort(CosplayAccount::cosplayAccounts.begin(),
            CosplayAccount::cosplayAccounts.end());
  for (const auto &r : CosplayAccount::cosplayAccounts) {
    cosplayAccountsJsons.emplace_back(r.to_json());
  }

  json rs;
  rs["version"] = 1;
  rs["irFile"] = TARGET_MODULE_PATH;
  rs["numOfIRLines"] = NUM_OF_IR_LINES;
  rs["numOfAttackVectors"] = NUM_OF_ATTACK_VECTORS;
  rs["addresses"] = SMART_CONTRACT_ADDRESSES;
  rs["untrustfulAccounts"] = uaccountsJsons;
  rs["unsafeOperations"] = usafeOperationsJsons;
  rs["cosplayAccounts"] = cosplayAccountsJsons;
  rs["generatedAt"] = getCurrentTimeStr();
  std::ofstream output(path, std::ofstream::out);
  output << rs;
  output.close();
}

} // namespace xray
