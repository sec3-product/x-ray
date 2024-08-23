#include "DebugFlags.h"

namespace xray {

bool CONFIG_SHOW_SUMMARY;
bool CONFIG_SHOW_DETAIL;

bool CONFIG_CHECK_UncheckedAccount;
bool hasOverFlowChecks;
bool anchorVersionTooOld;
bool splVersionTooOld;
bool solanaVersionTooOld;

bool DEBUG_RUST_API;
bool PRINT_IMMEDIATELY;
bool TERMINATE_IMMEDIATELY;

bool DEBUG_HB;       // Referenced by Graph/ReachabilityEngine.h.
bool DEBUG_LOCK_STR; // Referenced by LocksetManager.cpp.

} // namespace xray
