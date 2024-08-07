#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace sol {

// From ast/AST.h.
class ModuleAST;
class FunctionAST;

class SolLLVMIRGenerator {
public:
  SolLLVMIRGenerator(int argc, char **argv);
  void run();

private:
  bool generateAST(sol::ModuleAST *mod);
  void generateLLVMIR(sol::ModuleAST *mod);
  bool handleRustFile(const std::string &path);
  bool handleDirectory(sol::ModuleAST *mod,
                       const std::filesystem::path &dir_path);
  std::vector<sol::FunctionAST *> process(const std::string &filename,
                                          const std::string &method,
                                          size_t line, std::string contents);

  std::vector<sol::FunctionAST *> processResults;
};

}; // namespace sol
