#pragma once

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

bool matchPlusEqual(const CallSite &);
void handlePlusEqual(const RuleContext &, const CallSite &);

bool matchPlus(const CallSite &);
void handlePlus(const RuleContext &, const CallSite &);

}; // namespace xray
