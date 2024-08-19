#pragma once

#include <memory>

#include "OverflowAdd.h"
#include "OverflowSub.h"
#include "Ruleset.h"

namespace aser {

class RulesetFactory {
public:
  static Ruleset createRustModelRuleset() {
    Ruleset RS;
    RS.addRule(Rule(matchPlusEqual, handlePlusEqual));
    RS.addRule(Rule(matchMinusEqual, handleMinusEqual));
    return RS;
  }

  static Ruleset createNonRustModelRuleset() { return Ruleset(); }
};

} // namespace aser
