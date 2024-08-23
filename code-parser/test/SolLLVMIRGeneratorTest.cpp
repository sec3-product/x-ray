#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>

#include "ast/AST.h"
#include "sol/SolLLVMIRGenerator.h"

namespace sol {

class SolLLVMIRGeneratorTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::vector<std::string> args{"SolLLVMIRGeneratorTest"};
    std::vector<char *> argv;
    for (const auto &arg : args) {
      argv.push_back(const_cast<char *>(arg.data()));
    }
    generator =
        new SolLLVMIRGenerator(static_cast<int>(argv.size()), argv.data());
  }

  void TearDown() override { delete generator; }

  std::string writeTestData(const std::string &filename,
                            const std::string &code) const {
    auto tempDir = std::filesystem::temp_directory_path();
    auto tempFilePath = tempDir / filename;
    std::ofstream outFile(tempFilePath);
    if (!outFile.is_open()) {
      ADD_FAILURE() << "Unable to open file for writing: " << tempFilePath;
      return "";
    }

    outFile << code;
    outFile.close();
    return tempFilePath.string();
  }

  SolLLVMIRGenerator *generator;
};

TEST_F(SolLLVMIRGeneratorTest, TestGenerateAST) {
  const std::string code = R"(
  fn main() {
      println!("Hello, world!");
  })";

  std::string filename = writeTestData("TestGenerateAST1.rs", code);
  ASSERT_FALSE(filename.empty());

  sol::ModuleAST mod;
  auto res = generator->generateAST(&mod, filename);
  ASSERT_TRUE(res);

  auto funcs = mod.getFunctions();
  ASSERT_EQ(1, funcs.size());
  EXPECT_EQ("TestGenerateAST1::main.0", funcs[0]->getName());

  auto body = funcs[0]->getBody();
  ASSERT_TRUE(body != nullptr);
  ASSERT_EQ(1, body->size());

  auto expr = body->at(0);
  auto *callExpr = llvm::dyn_cast<FunctionCallAST>(expr);
  ASSERT_TRUE(callExpr != nullptr);
  EXPECT_EQ("println.!1", callExpr->getCallee());

  // Generate LLVM IR.
  llvm::LLVMContext llvmContext;
  auto llvmModule = generator->generateLLVMIR(&mod, llvmContext);
  ASSERT_TRUE(llvmModule != nullptr);

  std::string errMsg;
  llvm::raw_string_ostream errStream(errMsg);
  ASSERT_FALSE(llvm::verifyModule(*llvmModule, &errStream)) << errStream.str();
}

}; // namespace sol
