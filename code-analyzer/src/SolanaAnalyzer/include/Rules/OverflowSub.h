#pragma once

#include <map>
#include <utility>

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace aser {

bool matchMinusEqual(const CallSite &);
void handleMinusEqual(const RuleContext &, const CallSite &);

bool matchMinus(const CallSite &);
void handleMinus(const RuleContext &, const CallSite &);

}; // namespace aser
