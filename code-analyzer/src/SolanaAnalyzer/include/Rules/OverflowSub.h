#pragma once

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

bool handleMinusEqual(const RuleContext &, const CallSite &);
bool handleMinus(const RuleContext &, const CallSite &);

}; // namespace xray
