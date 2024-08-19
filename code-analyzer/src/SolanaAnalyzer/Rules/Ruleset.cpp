#include "Rules/Ruleset.h"

#include "Rules/OverflowAdd.h"
#include "Rules/OverflowSub.h"

namespace aser {

Ruleset Ruleset::createRustModelRuleset() {
  Ruleset RS;
  RS.addRule(Rule(matchPlusEqual, handlePlusEqual));
  RS.addRule(Rule(matchMinusEqual, handleMinusEqual));
  return RS;
}

Ruleset Ruleset::createNonRustModelRuleset() { return Ruleset(); }

} // namespace aser
