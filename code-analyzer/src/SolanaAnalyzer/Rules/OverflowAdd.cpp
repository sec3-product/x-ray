#include "Rules/OverflowAdd.h"

#include <map>
#include <utility>

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include "PTAModels/GraphBLASModel.h"
#include "Rules/Rule.h"

namespace aser {

bool matchPlusEqual(const CallSite &CS) {
  return CS.getTargetFunction()->getName().equals("sol.+=");
}

void handlePlusEqual(const RuleContext &RC, const CallSite &CS) {
  if (DEBUG_RUST_API) {
    llvm::outs() << "PlusEqualHandler: "
                 << "sol.+=: " << *RC.getInst() << "\n";
  }

  auto value = CS.getArgOperand(0);
  if (auto call_inst = llvm::dyn_cast<llvm::CallBase>(value)) {
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
      auto e = RC.createReadEvent();
      RC.getThread()->accountLamportsUpdateMap[valueName] = e;
    }
  }

  auto value2 = CS.getArgOperand(1);
  auto valueName2 = LangModel::findGlobalString(value2);
  bool isArgument = false;
  if (valueName2.contains(".")) {
    // += bet_order_account.stake;
    valueName2 = valueName2.substr(0, valueName2.find("."));
    for (auto pair : RC.getFuncArgTypesMap()[RC.getFunc()]) {
      if (pair.first.contains(valueName2)) {
        isArgument = true;
        break;
      }
    }
  }

  if (llvm::isa<llvm::Argument>(value) || llvm::isa<llvm::Argument>(value2) ||
      isArgument || llvm::dyn_cast<llvm::CallBase>(value2)) {
    if (!RC.isSafeType(value) && !RC.isSafeType(value2)) {
      RC.collectUnsafeOperation(SVE::Type::OVERFLOW_ADD, 8);
    }
  } else if (valueName2.equals("1") && !RC.isInLoop()) {
    if (!RC.isSafeVariable(value) && !RC.hasValueLessMoreThan(value, true)) {
      RC.collectUnsafeOperation(SVE::Type::OVERFLOW_ADD, 8);
    }
  }
}

bool matchPlus(const CallSite &callSite) {
  return callSite.getTargetFunction()->getName().equals("sol.+");
}

void handlePlus(const RuleContext &ruleContext, const CallSite &callSite) {
  if (DEBUG_RUST_API) {
    llvm::outs() << "sol.+: " << *ruleContext.getInst() << "\n";
  }
  auto value1 = callSite.getArgOperand(0);
  auto value2 = callSite.getArgOperand(1);
  if (llvm::isa<llvm::Argument>(value1) || llvm::isa<llvm::Argument>(value2)) {
    if (!ruleContext.isSafeType(value1) && !ruleContext.isSafeType(value2)) {
      if (!ruleContext.isSafeVariable(value1)) {
        ruleContext.collectUnsafeOperation(SVE::Type::OVERFLOW_ADD, 8);
      }
    }
  }
}

} // namespace aser
