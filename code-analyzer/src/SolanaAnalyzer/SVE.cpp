#include "SVE.h"

#include <map>
#include <set>
#include <string>

namespace xray {

/* --------------------------------

           SVE Types

----------------------------------- */

std::map<SVE::Type, std::string> SVE::sveTypeIdMap;

void SVE::addTypeID(std::string ID, SVE::Type type) { sveTypeIdMap[type] = ID; }

std::string SVE::getTypeID(SVE::Type type) {
  if (sveTypeIdMap.find(type) != sveTypeIdMap.end())
    return sveTypeIdMap.at(type);
  else
    return "";
}

std::set<std::string> SVE::disabledCheckers;
void SVE::addDisabledChecker(std::string ID) { disabledCheckers.insert(ID); }
bool SVE::isCheckerDisabled(SVE::Type type) {
  auto id = SVE::getTypeID(type);
  return (disabledCheckers.find(id) != disabledCheckers.end());
}

SVE::Database SVE::database;

void SVE::init(Database sves) {
  addTypeID("1001", Type::MISS_SIGNER);
  addTypeID("1002", Type::MISS_OWNER);

  addTypeID("1003", Type::OVERFLOW_ADD);
  addTypeID("1004", Type::OVERFLOW_SUB);
  addTypeID("1005", Type::OVERFLOW_MUL);
  addTypeID("1006", Type::OVERFLOW_DIV);

  addTypeID("1007", Type::ACCOUNT_UNVALIDATED_BORROWED);
  addTypeID("1019", Type::ACCOUNT_UNVALIDATED_OTHER);

  addTypeID("1010", Type::COSPLAY_FULL);
  addTypeID("1011", Type::COSPLAY_PARTIAL);

  addTypeID("1014", Type::BUMP_SEED);
  addTypeID("1015", Type::INSECURE_PDA_SHARING);
  addTypeID("1016", Type::ARBITRARY_CPI);
  addTypeID("1017", Type::MALICIOUS_SIMULATION);

  addTypeID("2001", Type::INCORRECT_BREAK_LOGIC);
  addTypeID("2002", Type::INCORRECT_CONDITION_CHECK);
  addTypeID("2003", Type::EXPONENTIAL_CALCULATION);
  addTypeID("2004", Type::INCORRECT_DIVISION_LOGIC);
  addTypeID("2005", Type::INCORRECT_TOKEN_CALCULATION);

  // Set disabled checkers.
  for (auto &[key, valueMap] : sves) {
    auto on = valueMap["on"];
    if (on == "false") {
      addDisabledChecker(key);
    }
  }

  database = std::move(sves);
}

} // namespace xray
