#pragma once

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace xray {

bool handleMul(const RuleContext &, const CallSite &);

}; // namespace xray
