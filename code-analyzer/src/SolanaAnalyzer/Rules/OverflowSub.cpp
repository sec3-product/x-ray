#include "Rules/OverflowSub.h"

#include <map>
#include <utility>

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include "PTAModels/GraphBLASModel.h"
#include "Rules/Rule.h"

namespace aser {

bool matchMinusEqual(const CallSite &CS) {
  return CS.getTargetFunction()->getName().equals("sol.-=");
}

void handleMinusEqual(const RuleContext &ruleContext,
                      const CallSite &callSite) {
  if (DEBUG_RUST_API) {
    llvm::outs() << "sol.-=: " << *ruleContext.getInst() << "\n";
  }
  auto value = callSite.getArgOperand(0);
  if (auto call_inst = dyn_cast<CallBase>(value)) {
    CallSite CS2(call_inst);
    if (CS2.getTargetFunction()->getName().startswith("sol.borrow_mut.")) {
      if (DEBUG_RUST_API) {
        llvm::outs() << "borrow_mut: " << *value << "\n";
      }
      auto value2 = CS2.getArgOperand(0);

      auto valueName = LangModel::findGlobalString(value2);
      if (DEBUG_RUST_API) {
        llvm::outs() << "data: " << valueName << "\n";
      }
      auto e = ruleContext.createReadEvent();
      ruleContext.getThread()->accountLamportsUpdateMap[valueName] = e;
    }
  }
  auto value2 = callSite.getArgOperand(1);
  if (llvm::isa<llvm::Argument>(value) || llvm::isa<llvm::Argument>(value2)) {
    if (!ruleContext.isSafeType(value) && !ruleContext.isSafeType(value2)) {
      ruleContext.collectUnsafeOperation(SVE::Type::OVERFLOW_SUB, 8);
    }
  } else {
    auto valueName2 = LangModel::findGlobalString(value2);
    if (valueName2.equals("1") && !ruleContext.isInLoop()) {
      if (!ruleContext.isSafeVariable(value) &&
          !ruleContext.hasValueLessMoreThan(value, false)) {
        ruleContext.collectUnsafeOperation(SVE::Type::OVERFLOW_SUB, 8);
      }
    } else if (valueName2.contains("fee")) {
      // tricky base_fee, last line < base_fee
      SourceInfo srcInfo = getSourceLoc(ruleContext.getInst());
      auto line_above = getSourceLinesForSoteria(srcInfo, 1);
      if (line_above.find("<") != std::string::npos &&
          line_above.find(valueName2.str()) != std::string::npos) {
        ruleContext.collectUnsafeOperation(SVE::Type::OVERFLOW_SUB, 8);
      }
    }
  }
}

bool matchMinus(const CallSite &callSite) {
  return callSite.getTargetFunction()->getName().equals("sol.-");
}

void handleMinus(const RuleContext &ruleContext, const CallSite &callSite) {
  if (DEBUG_RUST_API) {
    llvm::outs() << "sol.-: " << *ruleContext.getInst() << "\n";
  }
  auto value1 = callSite.getArgOperand(0);
  auto value2 = callSite.getArgOperand(1);
  if (llvm::isa<llvm::Argument>(value1) || llvm::isa<llvm::Argument>(value2)) {
    if (!ruleContext.isSafeType(value1) && !ruleContext.isSafeType(value2)) {
      if (!ruleContext.isSafeVariable(value1)) {
        ruleContext.collectUnsafeOperation(SVE::Type::OVERFLOW_SUB, 8);
      }
    }
  } else {
    auto valueName = LangModel::findGlobalString(value2);
    if (valueName.contains("rent")) {
      if (auto lamport_inst = llvm::dyn_cast<llvm::CallBase>(value1)) {
        CallSite CS2(lamport_inst);
        if (CS2.getTargetFunction()->getName().startswith("sol.lamports.")) {
          ruleContext.collectUnsafeOperation(SVE::Type::OVERFLOW_SUB, 8);
        }
      }
    }
  }
}

} // namespace aser
