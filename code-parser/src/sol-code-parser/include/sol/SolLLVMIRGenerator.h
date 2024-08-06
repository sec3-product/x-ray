#pragma once

#include <map>
#include <string>

namespace sol {

// From ast/AST.h.
class ModuleAST;

class SolLLVMIRGenerator {
 public:
  SolLLVMIRGenerator(int argc, char **argv);
  int Run(sol::ModuleAST *mod);

 protected:
  bool InitParser(sol::ModuleAST *mod);
  void InitLLVMIR(sol::ModuleAST *mod);
};

};  // namespace sol
