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

class CosplayDetectorTest : public ::testing::Test {
protected:
    llvm::LLVMContext context;
    llvm::Module module;
    CosplayDetectorTest() : module("testModule", context) {}

    // Helper function to create a mock function
    llvm::Function* createMockFunction(const std::string& name) {
        llvm::FunctionType* funcType =
            llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
        return llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name, &module);
    }
};

// test case 1: cosplayFull not Triggered
TEST_F(CosplayDetectorTest, NoCollectTriggered) {
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

    xray::CosplayAccounts::cosplayFullCount = 0;
    xray::CosplayAccounts::cosplayPartialCount = 0;

    detector.detectCosplay(ctx, tid);

    // Expect counters to be 0, since no collection should be triggered
    EXPECT_EQ(xray::CosplayAccounts::cosplayFullCount, 0);
}


// test case 2: cosplayFull Triggered
TEST_F(CosplayDetectorTest, TriggerCollectFullCosplay) {
    // Create two mock functions
    llvm::Function* func1 = createMockFunction("func1");
    llvm::Function* func2 = createMockFunction("func2");

    // Create basic blocks for these functions
    llvm::BasicBlock* block1 = llvm::BasicBlock::Create(context, "entry", func1);
    llvm::BasicBlock* block2 = llvm::BasicBlock::Create(context, "entry", func2);

    // Mock data for the function fields (matching field types with "pubkey")
    FunctionFieldsMap normalStructFunctionFieldsMap;
    normalStructFunctionFieldsMap[func1] = {{"field1", "pubkey"}};
    normalStructFunctionFieldsMap[func2] = {{"field1", "pubkey"}};

    SolanaAnalysisPass pass;
    ReachGraph graph(pass);

    std::map<TID, std::vector<CallEvent *>> callEventTraces;

    // Create the CosplayDetector object
    CosplayDetector detector(normalStructFunctionFieldsMap, &graph, callEventTraces);

    const xray::ctx* ctx = nullptr;  // Replace with actual ctx object
    TID tid = 0;  // Replace with actual TID value

    xray::CosplayAccounts::cosplayFullCount = 0;
    xray::CosplayAccounts::cosplayPartialCount = 0;

    detector.detectCosplay(ctx, tid);

    // expect the vector collected 1 cosplayFullCount
    EXPECT_EQ(xray::CosplayAccounts::cosplayFullCount, 1);   
}


