#include "Rules/OverflowDiv.h"

#include <map>
#include <utility>

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include "PTAModels/GraphBLASModel.h"
#include "Rules/Rule.h"

namespace aser {

bool matchDiv(const CallSite &callSite) {
  return callSite.getTargetFunction()->getName().equals("sol./");
}

void handleDiv(const RuleContext &ruleContext, const CallSite &callSite) {
  if (DEBUG_RUST_API) {
    llvm::outs() << "sol./: " << *ruleContext.getInst() << "\n";
  }
  auto value1 = callSite.getArgOperand(0);
  auto value2 = callSite.getArgOperand(1);
  if (llvm::isa<llvm::Argument>(value2)) {
    if (!ruleContext.isSafeType(value2)) {
      ruleContext.collectUnsafeOperation(SVE::Type::OVERFLOW_DIV, 5);
    }
  }
  if (auto value = llvm::dyn_cast<llvm::CallBase>(callSite.getArgOperand(0))) {
    CallSite CS2(value);
    if (CS2.getTargetFunction()->getName().equals("sol.*")) {
      auto valueName2 = LangModel::findGlobalString(callSite.getArgOperand(1));
      if (valueName2.contains("total") && valueName2.contains("upply")) {
        auto liquidity_ = LangModel::findGlobalString(CS2.getArgOperand(0));
        auto balance_ = LangModel::findGlobalString(CS2.getArgOperand(1));
        if (liquidity_.contains("liquidity") && balance_.contains("balance")) {
          ruleContext.collectUnsafeOperation(
              SVE::Type::INCORRECT_TOKEN_CALCULATION, 9);
        }
      } else if (!isConstant(valueName2.str())) {
        SourceInfo srcInfo = getSourceLoc(ruleContext.getInst());
        if (srcInfo.getSourceLine().find(" as u") == std::string::npos) {
          ruleContext.collectUnsafeOperation(SVE::Type::DIV_PRECISION_LOSS, 8);
        }
      }
    }
  }
}

} // namespace aser

