#pragma once

#include <utility>
#include <vector>

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace aser {

class Ruleset {
public:
  void addRule(Rule R) { Rules.push_back(std::move(R)); }

  bool evaluate(const RuleContext &RC, const aser::CallSite &CS) const {
    for (const auto &R : Rules) {
      if (R.match(CS)) {
        R.handle(RC, CS);
        return true;
      }
    }
    return false;
  }

private:
  std::vector<Rule> Rules;
};

} // namespace aser
