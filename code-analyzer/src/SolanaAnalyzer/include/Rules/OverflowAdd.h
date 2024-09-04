#pragma once

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

bool handlePlusEqual(const RuleContext &, const CallSite &);
bool handlePlus(const RuleContext &, const CallSite &);

}; // namespace xray
