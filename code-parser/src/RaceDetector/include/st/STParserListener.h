#ifndef MLIR_ST_PARSER_LISTENER_H_
#define MLIR_ST_PARSER_LISTENER_H_

#include "RustParserBaseListener.h"
#include "st/ParserWrapper.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace antlrcpp;
using namespace antlr4;

namespace st {

class STParserListener : public RustParserBaseListener {
 public:
  STParserListener(std::string fileName, std::string funcName, size_t line)
      : fileName(fileName), fnName(funcName), line_base(line) {
    llvm::outs() << "----listener filename: " << fileName << "-------"
                 << "\n";
  }
  std::unique_ptr<FunctionAST> getData() {
    Location loc({fileName, line_base, 0});

    PrototypeAST protoAST(loc, fnName, args);
    return std::make_unique<FunctionAST>(protoAST, exprList);
  }

 private:
  std::vector<VarDeclExprAST> args;
  ExprASTList exprList;
  std::vector<int> blockExprIdStack;
  std::map<const std::string, BlockExprAST *> blockExprMap;

  std::string fileName;
  std::string fnName;
  size_t line_base;

  std::string SPACE = "";
  int level = 0;
  int indentMore() {
    level++;
    SPACE += "  ";
    return level;
  }
  int indentLess() {
    level--;
    SPACE.pop_back();
    SPACE.pop_back();
    return level;
  }

  void enterCrate(RustParser::CrateContext * /*ctx*/) override {
    std::cout << SPACE << "enterCrate\n";
    indentMore();
  }
  void exitCrate(RustParser::CrateContext * /*ctx*/) override {
    indentLess();
    std::cout << SPACE << "exitCrate\n";
  }

};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<std::unique_ptr<FunctionAST>> records;

 public:
  auto begin() -> decltype(records.begin()) { return records.begin(); }
  auto end() -> decltype(records.end()) { return records.end(); }

  void addModule(std::unique_ptr<FunctionAST> record) {
    records.push_back(std::move(record));
  }
};
void dump(ModuleAST &);

}  // namespace st

#endif  // MLIR_ST_PARSER_LISTENER_H_
