#pragma once

#include <memory>

#include "OverflowAdd.h"
#include "Ruleset.h"

namespace aser {

class RulesetFactory {
public:
  static Ruleset createRustModelRuleset() {
    Ruleset RS;
    RS.addRule(Rule(matchPlusEqual, handlePlusEqual));
    return RS;
  }

  static Ruleset createNonRustModelRuleset() { return Ruleset(); }
};

} // namespace aser
