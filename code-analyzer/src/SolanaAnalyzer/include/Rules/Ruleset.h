#pragma once

#include <utility>
#include <vector>

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

class Ruleset {
public:
  void addRule(Rule R) { Rules.push_back(std::move(R)); }

  bool evaluate(const RuleContext &RC, const xray::CallSite &CS) const {
    for (const auto &R : Rules) {
      if (R.match(CS)) {
        R.handle(RC, CS);
        return true;
      }
    }
    return false;
  }

  static Ruleset createRustModelRuleset();
  static Ruleset createNonRustModelRuleset();

private:
  std::vector<Rule> Rules;
};

} // namespace xray
