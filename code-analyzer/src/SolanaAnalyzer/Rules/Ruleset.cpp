#include "Rules/Ruleset.h"

#include "Rules/ArbitraryCPI.h"
#include "Rules/Break.h"
#include "Rules/InsecurePDA.h"
#include "Rules/MaliciousSimulation.h"
#include "Rules/OverflowAdd.h"
#include "Rules/OverflowDiv.h"
#include "Rules/OverflowMul.h"
#include "Rules/OverflowSub.h"

namespace xray {

bool Ruleset::evaluate(const RuleContext &RC, const xray::CallSite &CS) const {
  for (const auto &R : Rules) {
    if (R.handle(RC, CS)) {
      return true;
    }
  }
  return false;
}

Ruleset Ruleset::createNonRustModelRuleset() {
  Ruleset RS;

  // Overflow add rules.
  RS.addRule(Rule(handlePlus));
  RS.addRule(Rule(handlePlusEqual));

  // Overflow sub rules.
  RS.addRule(Rule(handleMinus));
  RS.addRule(Rule(handleMinusEqual));

  // Overflow mul rules.
  RS.addRule(Rule(handleMul));

  // Overflow div rules.
  RS.addRule(Rule(handleDiv));

  // Malicious simulation rules.
  RS.addRule(Rule(handleComparisonEqual));

  // Detect arbitrary CPI calls.
  RS.addRule(Rule(handleInvoke));

  return RS;
}

Ruleset Ruleset::createRustModelRuleset() {
  Ruleset RS;

  // Break logic.
  RS.addRule(Rule(handleBreak));

  // Insecure PDA sharing.
  RS.addRule(Rule(handleOpaqueAssign));

  return Ruleset();
}

} // namespace xray
