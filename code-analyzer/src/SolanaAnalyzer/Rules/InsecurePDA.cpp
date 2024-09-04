#include "Rules/InsecurePDA.h"

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include "DebugFlags.h"
#include "PTAModels/GraphBLASModel.h"
#include "Rules/Rule.h"
#include "SVE.h"

namespace xray {

bool handleOpaqueAssign(const RuleContext &RC, const CallSite &CS) {
  auto calledFuncName = CS.getCalledFunction()->getName();
  if (!calledFuncName.startswith("sol.model.opaqueAssign")) {
    return false;
  }
  auto value = CS.getArgOperand(0);
  auto valueName = LangModel::findGlobalString(value);
  if (llvm::isa<llvm::CallBase>(CS.getArgOperand(1))) {
    return false;
  }
  auto valueName1 = LangModel::findGlobalString(CS.getArgOperand(1));
  if (valueName.contains("seeds")) {
    if (valueName1.contains("pool.") && !valueName1.contains("fee_") &&
        !valueName1.contains("dest")) {
      // report pda_sharing_secure
      RC.collectUntrustfulAccount(valueName1, SVE::Type::INSECURE_PDA_SHARING,
                                  5);
      return true;
    }
  }
  return false;
}

} // namespace xray
