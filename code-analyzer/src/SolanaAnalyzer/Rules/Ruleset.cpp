#include "Rules/Ruleset.h"

#include "Rules/OverflowAdd.h"
#include "Rules/OverflowDiv.h"
#include "Rules/OverflowMul.h"
#include "Rules/OverflowSub.h"

namespace xray {

Ruleset Ruleset::createRustModelRuleset() {
  Ruleset RS;

  // Overflow add rules.
  RS.addRule(Rule(matchPlus, handlePlus));
  RS.addRule(Rule(matchPlusEqual, handlePlusEqual));

  // Overflow sub rules.
  RS.addRule(Rule(matchMinus, handleMinus));
  RS.addRule(Rule(matchMinusEqual, handleMinusEqual));

  // Overflow mul rules.
  RS.addRule(Rule(matchMul, handleMul));

  // Overflow div rules.
  RS.addRule(Rule(matchDiv, handleDiv));

  return RS;
}

Ruleset Ruleset::createNonRustModelRuleset() { return Ruleset(); }

} // namespace xray
