#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest_prod.h>
#include <llvm/ADT/StringRef.h>

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace sol {
// From ast/AST.h.
class ModuleAST;
class FunctionAST;

class SolLLVMIRGenerator {
public:
  SolLLVMIRGenerator(int argc, char **argv);
  void run();

private:
  bool generateAST(sol::ModuleAST *mod, llvm::StringRef path);
  std::unique_ptr<llvm::Module> generateLLVMIR(sol::ModuleAST *mod,
                                               llvm::LLVMContext &llvmContext);
  bool handleRustFile(const std::string &path);
  bool handleDirectory(sol::ModuleAST *mod,
                       const std::filesystem::path &dir_path);

  std::vector<sol::FunctionAST *> processResults;

  FRIEND_TEST(SolLLVMIRGeneratorTest, TestGenerateAST);
};

}; // namespace sol
