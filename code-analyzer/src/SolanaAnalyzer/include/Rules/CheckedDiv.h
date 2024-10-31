#pragma once

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

bool handleCheckedDiv(const RuleContext &RC, const CallSite &CS);

}; // namespace xray
