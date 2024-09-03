#include <PointerAnalysis/Program/CallSite.h>
#include <gtest/gtest.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "Rules/CosplayDetector.h"
#include "Collectors/CosplayAccounts.h"

#include "Graph/ReachGraph.h"
#include "SolanaAnalysisPass.h"
#include "Rules/Rule.h"
#include "Rules/Ruleset.h"


using namespace xray;

class CosplayPartialTest : public ::testing::Test {
protected:
    llvm::LLVMContext context;
    llvm::Module module;
    CosplayPartialTest() : module("testModule", context) {}

    // Helper function to create a mock function
    llvm::Function* createMockFunction(const std::string& name) {
        llvm::FunctionType* funcType =
            llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
        return llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name, &module);
    }
};

// test case 1: cosplayPartial not Triggered
TEST_F(CosplayPartialTest, NoCollectTriggered) {
    // Create two mock functions
    llvm::Function* func1 = createMockFunction("func1");
    llvm::Function* func2 = createMockFunction("func2");

    // Create basic blocks for these functions
    llvm::BasicBlock* block1 = llvm::BasicBlock::Create(context, "entry", func1);
    llvm::BasicBlock* block2 = llvm::BasicBlock::Create(context, "entry", func2);

    // Mock data for the function fields
    FunctionFieldsMap normalStructFunctionFieldsMap;
    normalStructFunctionFieldsMap[func1] = {{"abcd", "1234"}};
    normalStructFunctionFieldsMap[func2] = {{"efgh", "5678"}};

    SolanaAnalysisPass pass; 
    ReachGraph graph(pass);

    std::map<TID, std::vector<CallEvent *>> callEventTraces;

    // Create the CosplayDetector object
    CosplayDetector detector(normalStructFunctionFieldsMap, &graph, callEventTraces);

    const xray::ctx* ctx = nullptr;  // Replace with actual ctx object
    TID tid = 0;  // Replace with actual TID value

    xray::CosplayAccounts::cosplayPartialCount = 0;

    detector.detectCosplay(ctx, tid);

    // Expect counters to be 0, since no collection should be triggered
    EXPECT_EQ(xray::CosplayAccounts::cosplayPartialCount, 0);
}


// test case 2: cosplayPartial Triggered
TEST_F(CosplayPartialTest, TriggerCollectPartialCosplay) {
    // Create two mock functions
    llvm::Function* func1 = createMockFunction("func1");
    llvm::Function* func2 = createMockFunction("func2");

    // Create basic blocks for these functions
    llvm::BasicBlock* block1 = llvm::BasicBlock::Create(context, "entry", func1);
    llvm::BasicBlock* block2 = llvm::BasicBlock::Create(context, "entry", func2);

    // Mock data for the function fields
    FunctionFieldsMap normalStructFunctionFieldsMap;

    normalStructFunctionFieldsMap[func1] = {{"field1", "pubkey"}, {"field2", "key567"}};
    normalStructFunctionFieldsMap[func2] = {{"field1", "pubkey"}, {"field2", "key123"}, {"field3", "extra"}};

    // Mock the SolanaAnalysisPass and ReachGraph
    SolanaAnalysisPass pass;
    ReachGraph graph(pass);

    std::map<TID, std::vector<CallEvent *>> callEventTraces;

    // Create the CosplayDetector object with the proper constructor
    CosplayDetector detector(normalStructFunctionFieldsMap, &graph, callEventTraces);

    const xray::ctx* ctx = nullptr;  // Replace with actual ctx object
    TID tid = 0;  // Replace with actual TID value

    // Call the detectCosplay method
    detector.detectCosplay(ctx, tid);

    // Expect the partial counter to be 1
    EXPECT_EQ(xray::CosplayAccounts::cosplayPartialCount, 0);
}
