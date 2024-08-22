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
    ACCOUNT_UNVALIDATED_OTHER,
    ACCOUNT_UNVALIDATED_BORROWED,
    ACCOUNT_UNVALIDATED_PDA,
    ACCOUNT_UNVALIDATED_DESTINATION,
    ACCOUNT_INCORRECT_AUTHORITY,
    ACCOUNT_DUPLICATE,
    ACCOUNT_CLOSE,
    COSPLAY_FULL,
    COSPLAY_PARTIAL,
    PDA_SEEDS_COLLISIONS,
    DIV_BY_ZERO,
    REINIT,
    BUMP_SEED,
    INSECURE_PDA_SHARING,
    ARBITRARY_CPI,
    MALICIOUS_SIMULATION,
    // OUTDATED_DEPENDENCY,
    // UNSAFE_RUST,
    // OVERPAY,
    // STALE_PRICE_FEED,
    // MISS_INIT_TOKEN_MINT,
    // MISS_RENT_EXEMPT,
    // MISS_FREEZE_AUTHORITY,
    // MISS_CPI_RELOAD,
    // MISS_ACCESS_CONTROL_UNSTAKE,
    // CRITICAL_REDUNDANT_CODE,
    // ORDER_RACE_CONDITION,
    // FLASHLOAN_RISK,
    // BIDIRECTIONAL_ROUNDING,
    // CAST_TRUNCATE,
    // UNSAFE_SYSVAR_API,
    INCORRECT_BREAK_LOGIC,
    INCORRECT_CONDITION_CHECK,
    EXPONENTIAL_CALCULATION,
    INCORRECT_DIVISION_LOGIC,
    INCORRECT_TOKEN_CALCULATION,
    INSECURE_INIT_IF_NEEDED,
    INSECURE_SPL_TOKEN_CPI,
    INSECURE_ACCOUNT_REALLOC,
    INSECURE_ASSOCIATED_TOKEN,
    // ACCOUNT_IDL_INCOMPATIBLE_ADD,
    // ACCOUNT_IDL_INCOMPATIBLE_MUT,
    // ACCOUNT_IDL_INCOMPATIBLE_ORDER,
    // REENTRANCY_ETHER,
    // ARBITRARY_SEND_ERC20,
    // SUISIDE_SELFDESTRUCT,
    // MISS_INIT_UNIQUE_ADMIN_CHECK,
    // BIT_SHIFT_OVERFLOW,
    DIV_PRECISION_LOSS,
    // VULNERABLE_SIGNED_INTEGER_I128
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
