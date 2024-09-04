#pragma once

#include <utility>
#include <vector>

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

class Ruleset {
public:
  void addRule(Rule R) { Rules.push_back(std::move(R)); }
  // Evaludates the ruleset against the given call site. It returns true if the
  // call site has been fully processed and no further rules should be
  // evaluated, false otherwise.
  bool evaluate(const RuleContext &RC, const xray::CallSite &CS) const;

  static Ruleset createRustModelRuleset();
  static Ruleset createNonRustModelRuleset();

private:
  std::vector<Rule> Rules;
};

} // namespace xray
