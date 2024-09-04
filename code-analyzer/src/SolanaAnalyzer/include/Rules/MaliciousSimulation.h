#pragma once

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

bool handleComparisonEqual(const RuleContext &, const CallSite &);

}; // namespace xray
