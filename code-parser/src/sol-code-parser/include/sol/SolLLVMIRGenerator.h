#pragma once

#include <map>
#include <string>

namespace sol {

// From ast/AST.h.
class ModuleAST;

class SolLLVMIRGenerator {
public:
  SolLLVMIRGenerator(int argc, char **argv);
  void Run();

protected:
  bool GenerateAST(sol::ModuleAST *mod);
  void GenerateLLVMIR(sol::ModuleAST *mod);
};

}; // namespace sol
