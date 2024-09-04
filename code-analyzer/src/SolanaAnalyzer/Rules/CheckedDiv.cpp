#include "Rules/CheckedDiv.h"

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>

#include "DebugFlags.h"
#include "PTAModels/GraphBLASModel.h"
#include "Rules/Rule.h"
#include "SVE.h"

namespace xray {

bool handleCheckedDiv(const RuleContext &RC, const CallSite &CS) {
  auto targetFuncName = CS.getCalledFunction()->getName();
  if (!targetFuncName.startswith("sol.checked_div.")) {
    return false;
  }

  if (DEBUG_RUST_API) {
    llvm::outs() << "sol.checked_div: " << *RC.getInst() << "\n";
  }

  if (RC.isInLoop() && CS.getNumArgOperands() > 1) {
    auto value = CS.getArgOperand(1);
    if (auto denominator_inst = llvm::dyn_cast<llvm::CallBase>(value)) {
      CallSite CS2(denominator_inst);
      if (CS2.getTargetFunction()->getName().startswith("sol.checked_")) {
        RC.collectUnsafeOperation(SVE::Type::INCORRECT_DIVISION_LOGIC, 8);
      }
    }
  }
  return true;
}

} // namespace xray

