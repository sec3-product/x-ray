
#pragma once

#include <map>
#include <utility>

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace aser {

bool matchMul(const CallSite &);
void handleMul(const RuleContext &, const CallSite &);

}; // namespace aser
