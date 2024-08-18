#include <PointerAnalysis/Program/CallSite.h>
#include <gtest/gtest.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "Rules/OverflowAdd.h"
#include "Rules/Rule.h"

namespace aser {

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
  aser::FuncArgTypesMap dummyFuncArgTypesMap;
  mutable size_t unsafeOps;
};

TEST(HandlePlusEqualTest, NoOverflowTriggered) {
  llvm::LLVMContext context;
  llvm::Module module("testModule", context);

  llvm::FunctionType *funcType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
  llvm::Function *function = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "testFunc", module);

  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(context, "entry", function);
  llvm::CallInst *callInst = llvm::CallInst::Create(function, "", block);

  aser::CallSite callSite(callInst);
  TestableRuleContext ruleContext(true,  // safeType
                                  false, // inLoop
                                  true); // safeVariable

  handlePlusEqual(ruleContext, callSite);

  // No unsafe operation was collected.
  EXPECT_EQ(0, ruleContext.unsafeOperations());
}

// Test Case 2: Overflow Triggered (Unsafe Types)
TEST(HandlePlusEqualTest, OverflowTriggeredUnsafeTypes) {
  llvm::LLVMContext context;
  llvm::Module module("testModule", context);

  // To Generate:
  // %16 = call i8* @"sol.+="(i8* getelementptr inbounds ([24 x i8], [24 x i8]*
  // @greeting_account.counter, i64 0, i64 0), i8* getelementptr inbounds ([1 x
  // i8], [1 x i8]* @"1", i64 0, i64 0)), !dbg !40

  // Create the types for the arguments and the function.
  llvm::Type *i8PtrType = llvm::Type::getInt8PtrTy(context);
  llvm::FunctionType *funcType =
      llvm::FunctionType::get(i8PtrType, {i8PtrType, i8PtrType}, false);
  llvm::Function *solPlusEqualFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "sol.+=", module);

  // Create a simple function to simulate the use of `sol.+=`.
  llvm::FunctionType *testFuncType =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
  llvm::Function *function = llvm::Function::Create(
      testFuncType, llvm::Function::ExternalLinkage, "overflowFunc", module);

  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(context, "entry", function);

  // Create the GEP instruction for the first argument
  // (greeting_account.counter).
  llvm::Value *greetingAccountCounter = llvm::ConstantExpr::getGetElementPtr(
      llvm::ArrayType::get(llvm::Type::getInt8Ty(context), 24),
      llvm::Constant::getNullValue(
          llvm::ArrayType::get(llvm::Type::getInt8Ty(context), 24)
              ->getPointerTo()),
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0),
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0));

  // Create a global string constant for "1".
  llvm::Constant *globalStr =
      llvm::ConstantDataArray::getString(context, "1", false);
  llvm::GlobalVariable *globalVar = new llvm::GlobalVariable(
      module, globalStr->getType(),
      true, // constant
      llvm::GlobalValue::PrivateLinkage, globalStr, "global_str_1");

  // Create the GEP instruction to get the pointer to the global string.
  llvm::Value *oneValue = llvm::ConstantExpr::getInBoundsGetElementPtr(
      globalStr->getType(), globalVar,
      llvm::ArrayRef<llvm::Constant *>{
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0),
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0)});

  llvm::CallInst *callInst = llvm::CallInst::Create(
      solPlusEqualFunc, {greetingAccountCounter, oneValue}, "sol_plus_equal",
      block);
  aser::CallSite callSite(callInst);

  TestableRuleContext ruleContext(false,  // safeType
                                  false,  // inLoop
                                  false); // safeVariable
  handlePlusEqual(ruleContext, callSite);

  // Verify that an unsafe operation was collected.
  EXPECT_EQ(1, ruleContext.unsafeOperations());
}

} // namespace aser
