#include "Rules/Break.h"

#include <llvm/Support/raw_ostream.h>

#include "DebugFlags.h"

namespace xray {

bool matchBreak(const CallSite &CS) {
  return CS.getTargetFunction()->getName().equals("sol.model.break");
}

void handleBreak(const RuleContext &RC, const CallSite &CS) {
  if (DEBUG_RUST_API) {
    llvm::outs() << "sol.model.break: " << *RC.getInst() << "\n";
  }
  if (!RC.isInLoop()) {
    return;
  }

  // check potential break => continue
  // caller instruction
  auto inst0 = RC.getLastInst();
  if (auto inst1 = inst0->getPrevNonDebugInstruction()) {
    if (auto callValue2 = llvm::dyn_cast<llvm::CallBase>(inst1)) {
      CallSite CS2(callValue2);
      if (CS2.getTargetFunction()->getName().equals("sol.if")) {
        if (auto inst2 = inst1->getPrevNonDebugInstruction()) {
          if (auto inst3 = inst2->getPrevNonDebugInstruction()) {
            if (auto callValue3 = dyn_cast<CallBase>(inst3)) {
              CallSite CS3(callValue3);
              if (CS3.getTargetFunction()->getName().startswith(
                      "sol.Pubkey::default.")) {
                // sol.Pubkey::default.0
                RC.collectUnsafeOperation(SVE::Type::INCORRECT_BREAK_LOGIC, 10);
              }
            }
          }
        }
      }
    }
  }
}

} // namespace xray
