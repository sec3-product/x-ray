#include <PointerAnalysis/Program/CallSite.h>
#include <gtest/gtest.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "Rules/OverflowMul.h"
#include "Rules/Rule.h"

#define DEBUG_UNIT_TEST 0

namespace xray {

class TestableRuleContext : public RuleContext {
public:
  TestableRuleContext(bool safeType, bool inLoop, bool safeVariable)
      : RuleContext(nullptr, nullptr, dummyFuncArgTypesMap, nullptr,
                    createReadEvent, nullptr, nullptr, nullptr),
        safeType(safeType), inLoop(inLoop), safeVariable(safeVariable),
        unsafeOps(0) {}

  TestableRuleContext(::xray::FuncArgTypesMap &funcArgTypesMap, bool safeType,
                      bool inLoop, bool safeVariable)
      : RuleContext(nullptr, nullptr, funcArgTypesMap, nullptr, createReadEvent,
                    nullptr, nullptr, nullptr),
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

TEST(handleMulTest, NoOverflowTriggered) {
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
                                  true,  // inLoop
                                  true); // safeVariable

  handleMul(ruleContext, callSite);

  // No unsafe operation was collected.
  EXPECT_EQ(0, ruleContext.unsafeOperations());
}

// Test Case 2: Overflow Triggered (Unsafe Types)
TEST(HandleMultiplyTest, BothUnsafeTypes) {
  llvm::LLVMContext context;
  llvm::Module module("testModule", context);

  // Create a function that will be used to simulate the call to sol.*
  llvm::FunctionType *testFuncType =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
  llvm::Function *function = llvm::Function::Create(
      testFuncType, llvm::Function::ExternalLinkage, "multiplyFunc", module);

  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(context, "entry", function);

  // Create two arguments using i64 type
  llvm::Argument *value1 =
      new llvm::Argument(llvm::Type::getInt64Ty(context), "value1", function);
  llvm::Argument *value2 =
      new llvm::Argument(llvm::Type::getInt64Ty(context), "value2", function);

  // Create a map that simulates FuncArgTypesMap for RuleContext
  ::xray::FuncArgTypesMap funcArgTypesMap;
  //   funcArgTypesMap[function][0] = std::make_pair(value1->getName(),
  //   llvm::StringRef("i64")); funcArgTypesMap[function][1] =
  //   std::make_pair(value2->getName(), llvm::StringRef("i64"));

  TestableRuleContext ruleContext(funcArgTypesMap,
                                  false, // safeType
                                  true,  // inLoop
                                  true); // safeVariable

  // Simulate the multiplication
  llvm::FunctionType *funcType = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context),
      {llvm::Type::getInt64Ty(context), llvm::Type::getInt64Ty(context)},
      false);
  llvm::Function *solMultiplyFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "sol.*", module);

  llvm::CallInst *callInst = llvm::CallInst::Create(
      solMultiplyFunc, {value1, value2}, "sol_multiply", block);
  xray::CallSite callSite(callInst);

  if (DEBUG_UNIT_TEST) {
    auto op1 = callSite.getArgOperand(0);
    auto op2 = callSite.getArgOperand(1);
    EXPECT_TRUE(llvm::isa<llvm::Argument>(op1));
    EXPECT_TRUE(llvm::isa<llvm::Argument>(op2));
  }
  handleMul(ruleContext, callSite);

  // Since both types are unsafe, ensure the context records 1 unsafe operation
  EXPECT_EQ(1, ruleContext.unsafeOperations());
}

} // namespace xray
