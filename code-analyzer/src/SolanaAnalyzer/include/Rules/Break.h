#pragma once

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

bool matchBreak(const CallSite &);
void handleBreak(const RuleContext &, const CallSite &);

}; // namespace xray
