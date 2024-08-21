#pragma once

namespace xray {

// Debug flags defined in SolanaAnalysisPass.cpp.
extern bool CONFIG_SHOW_SUMMARY;
extern bool CONFIG_SHOW_DETAIL;

extern bool CONFIG_CHECK_UncheckedAccount;
extern bool hasOverFlowChecks;
extern bool anchorVersionTooOld;
extern bool splVersionTooOld;
extern bool solanaVersionTooOld;

extern bool DEBUG_RUST_API;
extern bool PRINT_IMMEDIATELY;
extern bool TERMINATE_IMMEDIATELY;

extern bool DEBUG_HB;
extern bool DEBUG_LOCK_STR;

} // namespace xray
