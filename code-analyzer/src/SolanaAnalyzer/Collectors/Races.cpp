#include "Collectors/Races.h"

#include <fstream>

#include "AccountIDL.h"
#include "Collectors/CosplayAccounts.h"
#include "Collectors/UnsafeOperation.h"
#include "Collectors/UntrustfulAccount.h"
#include "DebugFlags.h"
#include "LogColor.h"
#include "SolanaAnalysisPass.h"

#define DEFAULT_BUDGET 25

using namespace aser;
using namespace std;
using namespace llvm;

// for cost quote
extern unsigned int NUM_OF_IR_LINES;
extern unsigned int NUM_OF_ATTACK_VECTORS;

extern std::set<llvm::StringRef> SMART_CONTRACT_ADDRESSES;

/* --------------------------------

                Utils

----------------------------------- */
void aser::outputJSON(std::string OutputPath) {
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
  for (auto &r : UntrustfulAccount::untrustfulAccounts) {
    uaccountsJsons.emplace_back(r.to_json());
  }
  std::vector<json> usafeOperationsJsons;
  std::sort(UnsafeOperation::unsafeOperations.begin(),
            UnsafeOperation::unsafeOperations.end());
  for (auto &r : UnsafeOperation::unsafeOperations) {
    usafeOperationsJsons.emplace_back(r.to_json());
  }
  std::vector<json> cosplayAccountsJsons;
  std::sort(CosplayAccounts::cosplayAccounts.begin(),
            CosplayAccounts::cosplayAccounts.end());
  for (auto &r : CosplayAccounts::cosplayAccounts) {
    cosplayAccountsJsons.emplace_back(r.to_json());
  }

  json rs;
  rs["raceConditions"] = std::vector<json>();
  rs["untrustfulAccounts"] = uaccountsJsons;
  rs["unsafeOperations"] = usafeOperationsJsons;
  rs["cosplayAccounts"] = cosplayAccountsJsons;
  rs["version"] = 1;
  rs["generatedAt"] = getCurrentTimeStr();
  rs["bcfile"] = TARGET_MODULE_PATH;
  rs["numOfIRLines"] = NUM_OF_IR_LINES;
  rs["numOfAttackVectors"] = NUM_OF_ATTACK_VECTORS;
  rs["addresses"] = SMART_CONTRACT_ADDRESSES;
  std::ofstream output(path, std::ofstream::out);
  output << rs;
  output.close();
}
