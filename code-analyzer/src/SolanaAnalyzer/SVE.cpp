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
  addTypeID("1008", Type::ACCOUNT_DUPLICATE);
  addTypeID("1009", Type::ACCOUNT_CLOSE);
  addTypeID("1010", Type::COSPLAY_FULL);
  addTypeID("1011", Type::COSPLAY_PARTIAL);
  addTypeID("1012", Type::DIV_BY_ZERO);
  addTypeID("1013", Type::REINIT);
  addTypeID("1014", Type::BUMP_SEED);
  addTypeID("1015", Type::INSECURE_PDA_SHARING);
  addTypeID("1016", Type::ARBITRARY_CPI);
  addTypeID("1017", Type::MALICIOUS_SIMULATION);
  addTypeID("1018", Type::UNSAFE_SYSVAR_API);
  addTypeID("1019", Type::ACCOUNT_UNVALIDATED_OTHER);
  addTypeID("1020", Type::OUTDATED_DEPENDENCY);
  addTypeID("1021", Type::UNSAFE_RUST);
  addTypeID("1022", Type::OVERPAY);
  addTypeID("1023", Type::STALE_PRICE_FEED);
  addTypeID("1024", Type::MISS_INIT_TOKEN_MINT);
  addTypeID("1025", Type::MISS_RENT_EXEMPT);
  addTypeID("1026", Type::MISS_FREEZE_AUTHORITY);
  addTypeID("1027", Type::FLASHLOAN_RISK);
  addTypeID("1028", Type::BIDIRECTIONAL_ROUNDING);
  addTypeID("1029", Type::CAST_TRUNCATE);
  addTypeID("1030", Type::ACCOUNT_UNVALIDATED_PDA);
  addTypeID("1031", Type::ACCOUNT_UNVALIDATED_DESTINATION);
  addTypeID("1032", Type::ACCOUNT_INCORRECT_AUTHORITY);
  addTypeID("1033", Type::INSECURE_INIT_IF_NEEDED);
  addTypeID("1034", Type::INSECURE_SPL_TOKEN_CPI);
  addTypeID("1035", Type::INSECURE_ASSOCIATED_TOKEN);
  addTypeID("1036", Type::INSECURE_ACCOUNT_REALLOC);
  addTypeID("1037", Type::PDA_SEEDS_COLLISIONS);
  addTypeID("2001", Type::INCORRECT_BREAK_LOGIC);
  addTypeID("2002", Type::INCORRECT_CONDITION_CHECK);
  addTypeID("2003", Type::EXPONENTIAL_CALCULATION);
  addTypeID("2004", Type::INCORRECT_DIVISION_LOGIC);
  addTypeID("2005", Type::INCORRECT_TOKEN_CALCULATION);
  addTypeID("3002", Type::CRITICAL_REDUNDANT_CODE);
  addTypeID("3005", Type::MISS_CPI_RELOAD);
  addTypeID("3006", Type::MISS_ACCESS_CONTROL_UNSTAKE);
  addTypeID("3007", Type::ORDER_RACE_CONDITION);
  addTypeID("3008", Type::ACCOUNT_IDL_INCOMPATIBLE_ADD);
  addTypeID("3009", Type::ACCOUNT_IDL_INCOMPATIBLE_MUT);
  addTypeID("3010", Type::ACCOUNT_IDL_INCOMPATIBLE_ORDER);
  addTypeID("10001", Type::REENTRANCY_ETHER);
  addTypeID("10002", Type::ARBITRARY_SEND_ERC20);
  addTypeID("10003", Type::SUISIDE_SELFDESTRUCT);
  addTypeID("20001", Type::MISS_INIT_UNIQUE_ADMIN_CHECK);
  addTypeID("20002", Type::BIT_SHIFT_OVERFLOW);
  addTypeID("20003", Type::DIV_PRECISION_LOSS);
  addTypeID("20004", Type::VULNERABLE_SIGNED_INTEGER_I128);

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
