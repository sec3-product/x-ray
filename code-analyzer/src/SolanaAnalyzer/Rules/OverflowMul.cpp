#include "Rules/OverflowMul.h"

#include <map>
#include <utility>

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include "PTAModels/GraphBLASModel.h"
#include "Rules/Rule.h"

namespace aser {

bool matchMul(const CallSite &callSite) {
  return callSite.getTargetFunction()->getName().equals("sol.*");
}

void handleMul(const RuleContext &ruleContext, const CallSite &callSite) {
  if (DEBUG_RUST_API) {
    llvm::outs() << "sol.*: " << *ruleContext.getInst() << "\n";
  }
  auto value1 = callSite.getArgOperand(0);
  auto value2 = callSite.getArgOperand(1);
  if (llvm::isa<llvm::Argument>(value1) || llvm::isa<llvm::Argument>(value2)) {
    if (!ruleContext.isSafeType(value1) && !ruleContext.isSafeType(value2)) {
      ruleContext.collectUnsafeOperation(SVE::Type::OVERFLOW_MUL, 6);
    }
  }
}

} // namespace aser
