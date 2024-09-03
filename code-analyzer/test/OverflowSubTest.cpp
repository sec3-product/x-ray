#include <PointerAnalysis/Program/CallSite.h>
#include <gtest/gtest.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "Rules/OverflowSub.h"
#include "Rules/Rule.h"

namespace xray {

class TestableRuleContext : public RuleContext {
public:
  TestableRuleContext(bool safeType, bool inLoop, bool safeVariable)
      : RuleContext(nullptr, nullptr, dummyFuncArgTypesMap, nullptr,
                    createReadEvent, nullptr, nullptr),
        safeType(safeType), inLoop(inLoop), safeVariable(safeVariable),
        unsafeOps(0) {}

  static Event *createReadEvent(const llvm::Instruction *) { return nullptr; }

  bool isSafeType(const llvm::Value *value) const override { return safeType; }

  bool hasValueLessMoreThan(const llvm::Value *value,
                            bool flag) const override {
    return false;
  }

  bool isInLoop() const override { return inLoop; }

  bool isSafeVariable(const llvm::Value *value) const override {
    return safeVariable;
  }

  void collectUnsafeOperation(SVE::Type type, int size) const override {
    unsafeOps++;
  }

  size_t unsafeOperations() const { return unsafeOps; }

private:
  bool safeType;
  bool inLoop;
  bool safeVariable;
  xray::FuncArgTypesMap dummyFuncArgTypesMap;
  mutable size_t unsafeOps;
};

TEST(HandleMinusEqualTest, NoOverflowTriggered) {
  llvm::LLVMContext context;
  llvm::Module module("testModule", context);

  llvm::FunctionType *funcType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
  llvm::Function *function = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "testFunc", module);

  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(context, "entry", function);
  llvm::CallInst *callInst = llvm::CallInst::Create(function, "", block);

  xray::CallSite callSite(callInst);
  TestableRuleContext ruleContext(true,  // safeType
                                  false, // inLoop
                                  true); // safeVariable

  handleMinusEqual(ruleContext, callSite);

  // No unsafe operation was collected.
  EXPECT_EQ(0, ruleContext.unsafeOperations());
}

// Test Case 2: Overflow Triggered by "sol.-="
TEST(HandleMinusEqualTest, OverflowTriggeredUnsafeTypes) {
  llvm::LLVMContext context;
  llvm::Module module("testModule", context);

  // Create the types for the arguments and the function.
  llvm::FunctionType *funcType =
    llvm::FunctionType::get(llvm::Type::getInt64Ty(context),
                            {llvm::Type::getInt64Ty(context), llvm::Type::getInt64Ty(context)},
                            false);
  llvm::Function *solMinusEqualFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "sol.-=", module);

  // Create a simple function to simulate the use of `sol.-=`.
  llvm::FunctionType *testFuncType =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
  llvm::Function *function = llvm::Function::Create(
      testFuncType, llvm::Function::ExternalLinkage, "overflowFunc", module);

  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(context, "entry", function);


  llvm::Value *simpleLHS = llvm::Constant::getNullValue(llvm::Type::getInt8PtrTy(context));
  llvm::Argument *valueRHS = new llvm::Argument(llvm::Type::getInt64Ty(context), "value1", function);

  llvm::CallInst *callInst = llvm::CallInst::Create(
      solMinusEqualFunc, {simpleLHS, valueRHS}, "sol_minus_equal", block);

  xray::CallSite callSite(callInst);

  TestableRuleContext ruleContext(false,  // safeType
                                  false,  // inLoop
                                  false); // safeVariable
  handleMinusEqual(ruleContext, callSite);

  // Verify that an unsafe operation was collected.
  EXPECT_EQ(1, ruleContext.unsafeOperations());
}

} // namespace xray
