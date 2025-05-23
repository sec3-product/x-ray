#include "Rules/MaliciousSimulation.h"

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include "DebugFlags.h"
#include "PTAModels/GraphBLASModel.h"
#include "Rules/Rule.h"
#include "SVE.h"

namespace xray {

bool handleComparisonEqual(const RuleContext &RC, const CallSite &CS) {
  auto targetFuncName = CS.getCalledFunction()->getName();
  if (!targetFuncName.startswith("sol.>=") &&
      !targetFuncName.startswith("sol.<=")) {
    return false;
  }

  if (DEBUG_RUST_API) {
    llvm::outs() << "sol.>=: " << *RC.getInst() << "\n";
  }
  auto value1 = CS.getArgOperand(0);
  auto value2 = CS.getArgOperand(1);
  auto valueName1 = LangModel::findGlobalString(value1);
  auto valueName2 = LangModel::findGlobalString(value2);
  if (valueName1.contains("slot") && valueName2.contains("slot") &&
      !valueName2.contains("price")) {
    RC.collectUntrustfulAccount(valueName1, SVE::Type::MALICIOUS_SIMULATION,
                                11, "");
  }
  return true;
}

} // namespace xray
