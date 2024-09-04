#pragma once

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

bool matchComparisonEqual(const CallSite &);
void handleComparisonEqual(const RuleContext &, const CallSite &);

}; // namespace xray
