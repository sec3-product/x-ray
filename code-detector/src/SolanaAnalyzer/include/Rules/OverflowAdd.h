#pragma once

#include <map>
#include <utility>

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace aser {

bool matchPlusEqual(const CallSite &CS);
void handlePlusEqual(const RuleContext &RC, const CallSite &CS);

}; // namespace aser
