#pragma once

#include <map>
#include <utility>

#include <PointerAnalysis/Program/CallSite.h>

#include "Rule.h"

namespace aser {

bool matchDiv(const CallSite &);
void handleDiv(const RuleContext &, const CallSite &);

}; // namespace aser

