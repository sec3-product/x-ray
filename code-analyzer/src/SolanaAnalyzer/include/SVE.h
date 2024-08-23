#pragma once

#include <map>
#include <set>
#include <string>

namespace xray {

class SVE {
public:
  enum Type {
    MISS_OWNER,
    MISS_SIGNER,
    OVERFLOW_ADD,
    OVERFLOW_SUB,
    OVERFLOW_MUL,
    OVERFLOW_DIV,
    ACCOUNT_UNVALIDATED_BORROWED,
    ACCOUNT_UNVALIDATED_OTHER,
    COSPLAY_FULL,
    COSPLAY_PARTIAL,
    BUMP_SEED,
    INSECURE_PDA_SHARING,
    MALICIOUS_SIMULATION,
    INCORRECT_BREAK_LOGIC,
    INCORRECT_CONDITION_CHECK,
    EXPONENTIAL_CALCULATION,
    INCORRECT_DIVISION_LOGIC,
    INCORRECT_TOKEN_CALCULATION
  };

  using Database = std::map<std::string, std::map<std::string, std::string>>;

  static Database database;
  static void init(Database);

  static std::map<SVE::Type, std::string> sveTypeIdMap;
  static std::set<std::string> disabledCheckers;
  static void addTypeID(std::string ID, SVE::Type type);
  static std::string getTypeID(SVE::Type type);
  static void addDisabledChecker(std::string ID);
  static bool isCheckerDisabled(SVE::Type type);
};

} // namespace xray
