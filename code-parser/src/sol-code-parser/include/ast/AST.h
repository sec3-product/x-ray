//
// Created by jncsw on 3/11/21.
//

#pragma once

#include <any>
#include <set>
#include <string>
#include <vector>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace sol {

/// Structure definition a location in a file.
struct Location {
  std::string file; ///< filename.
  size_t line;      ///< line number.
  size_t col;       ///< column number.
};

class BaseAST {
public:
  BaseAST(std::string ASTType) { this->ASTType = ASTType; }

  BaseAST() {}

  std::string Filename;
  size_t line_number_start = 0, line_number_end = 0, column_start = 0,
         column_end = 0;
  std::string ASTType;
  int level = 0;
  Location loc;
};

class CommentAST : public BaseAST {
public:
  CommentAST() : BaseAST("CommentAST") {}
  std::string Comment = "";
  std::string to_string() { return this->ASTType + " " + this->Comment; }
};

class VarAST : public BaseAST {
  std::string varname;

public:
  VarAST() : BaseAST("VarAST") {}
};

class BinaryAST : public BaseAST {
public:
  BinaryAST() : BaseAST("BinaryAST") {}
  VarAST *LHS = NULL;
  VarAST *RHS = NULL;
  std::string Operator = "";
};

class VarDeclAST : public BaseAST {
public:
  VarDeclAST() : BaseAST("VarDeclAST") {}
  std::string varname = "";
  std::string var_type = "";
  std::string initial_value = "";
};

struct VarType {
  std::string name;
  std::vector<int64_t> shape;
};

/// Base class for all expression nodes.
class ExprAST : public BaseAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_String,
    Expr_Symbol,
    Expr_Reserved,
    Expr_StructLiteral,
    Expr_Var,
    Expr_Primitive,
    Expr_Error,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
    Expr_Assign,
    Expr_Block,
    Expr_UnaryMessage,
    Expr_BinaryMessage,
    Expr_KeywordPair,
    Expr_KeywordMessage,
    Expr_FuncCall,
    Expr_FuncCall_Bin,
    Expr_FuncCall_Key,
    Expr_FuncCall_Cascade,
    Expr_RuntimeLiteral,
    Expr_ParsetimeLiteral,
    Expr_DynamicDictionary,
    Expr_DynamicArray,
    Default
  };

private:
  const ExprASTKind kind;
  Location location;

public:
  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}
  // virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

  std::string object_type = "";
  std::string source = "";
  ExprASTKind ASTType = Default;
  //    FunctionCallAST * Expression = NULL;
  //    AssignAST * assignStmt = NULL;
  //    BlockAST * blockAst = NULL;
  std::string ParsetimeLiteral = "";
  std::string primitive = "";
  std::string reference;
  bool valid() { return ASTType != ExprASTKind::Default; }

  std::string getName() {
    // todo: implement this
    return ParsetimeLiteral;
  }
  llvm::StringRef getVarName() {
    // todo: implement this
    return ParsetimeLiteral;
  }
};
/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, loc), name(name) {}

  std::string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};
/// Ex
// A block-list of expressions.
using ExprASTList = std::vector<ExprAST *>;
class LiteralExprAST : public ExprAST {
  std::string name;

public:
  LiteralExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Literal, loc), name(name) {}

  std::string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
};

class StringExprAST : public ExprAST {
  std::string name;

public:
  StringExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_String, loc), name(name) {}

  std::string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_String; }
};
class SymbolExprAST : public ExprAST {
  std::string name;

public:
  SymbolExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Symbol, loc), name(name) {}

  std::string getSymbol() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Symbol; }
};
class ReservedKeywordExprAST : public ExprAST {
  std::string name;

public:
  ReservedKeywordExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Reserved, loc), name(name) {}

  std::string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_Reserved;
  }
};
/// Expression class for numeric literals like "10".
class NumberExprAST : public ExprAST {
  long Val;

public:
  NumberExprAST(Location loc, long val) : ExprAST(Expr_Num, loc), Val(val) {}

  long getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

class DynamicDictionaryExprAST : public ExprAST {
  std::vector<ExprAST *> dict;

public:
  DynamicDictionaryExprAST(Location loc, std::vector<ExprAST *> dict)
      : ExprAST(Expr_DynamicDictionary, loc), dict(dict) {}

  std::vector<ExprAST *> &getExpressions() { return dict; };
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_DynamicDictionary;
  }
};
class DynamicArrayExprAST : public ExprAST {
  std::vector<ExprAST *> arr;

public:
  DynamicArrayExprAST(Location loc, std::vector<ExprAST *> arr)
      : ExprAST(Expr_DynamicArray, loc), arr(arr) {}
  std::vector<ExprAST *> &getExpressions() { return arr; };

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_DynamicArray;
  }
};

class PrimitiveExprAST : public ExprAST {
  std::string name;

public:
  PrimitiveExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Primitive, loc), name(name) {}

  std::string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_Primitive;
  }
};

class ParserErrorAST : public ExprAST {
  std::string name;

public:
  ParserErrorAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Error, loc), name(name) {}

  std::string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Error; }
};

class ReturnExprAST : public ExprAST {
  llvm::Optional<ExprAST *> expr;

public:
  ReturnExprAST(Location loc, llvm::Optional<ExprAST *> &expr)
      : ExprAST(Expr_Return, loc), expr(expr) {}

  llvm::Optional<ExprAST *> getExpr() {
    if (expr.hasValue())
      return expr;
    return llvm::None;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
};
/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
  ExprAST *arg;

public:
  PrintExprAST(Location loc, ExprAST *arg)
      : ExprAST(Expr_Print, loc), arg(arg) {}

  ExprAST *getArg() { return arg; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
};
class AssignExprAST : public ExprAST {
  std::string name; // variable name 'teset'
  ExprAST *var;
  ExprAST *valueExpr;

public:
  AssignExprAST(Location loc, ExprAST *var, ExprAST *valueExpr)
      : ExprAST(Expr_Assign, loc), var(var), valueExpr(valueExpr) {}

  std::string getVarName() { return var->getName(); }
  ExprAST *getValueExpr() { return valueExpr; }
  ExprAST *getLHS() { return var; };
  ExprAST *getRHS() { return valueExpr; };
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Assign; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
  const std::string name;
  const VarType type;
  ExprAST *initVal;

public:
  VarDeclExprAST(Location loc, std::string name, const VarType &type,
                 ExprAST *initVal = nullptr)
      : ExprAST(Expr_VarDecl, loc), name(name), type(type), initVal(initVal) {
    // llvm::outs() << "-----VarDeclExprAST ------varName: " << name << "\n";
  }
  llvm::StringRef getNameStringRef() const { return name; }
  std::string getName() const { return name; }
  ExprAST *getInitVal() { return initVal; }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

class PrototypeAST : public BaseAST {
  Location location;
  const std::string name;
  std::vector<VarDeclExprAST> args;
  ExprAST *ret;

public:
  PrototypeAST(Location location, std::string name,
               std::vector<VarDeclExprAST> &args, ExprAST *ret = nullptr)
      : location(location), name(name), args(args), ret(ret) {}

  const Location &loc() { return location; }
  std::string getName() { return name; }
  ExprAST *getRet() { return ret; };
  std::vector<VarDeclExprAST> &getArgs() { return args; }
};

class FunctionAST : public BaseAST {
private:
  Location loc;

public:
  FunctionAST(Location loc) : loc(loc) {}
  std::string function_name;
  PrototypeAST *proto;
  std::vector<VarDeclExprAST> temp_vars;
  std::set<std::string> writeVars;
  std::set<std::string> usedVars;
  void addUsedVar(std::string &name) { usedVars.insert(name); }
  std::vector<ExprAST *> body;
  std::string source = "";
  ExprAST *return_val = NULL;
  PrototypeAST *getProto() { return proto; }
  std::vector<VarDeclExprAST> &getLocals() { return temp_vars; }
  VarDeclExprAST &getLocal(int k) { return temp_vars[k]; }
  ExprASTList *getBody() { return &body; }
  std::string getName() { return function_name; }
};

class ClassAST : public BaseAST {
public:
  ClassAST() : BaseAST("ClassAST") {}
  std::vector<FunctionAST *> functions;
  bool valid = false;
  int line;
  std::string fileName = "";
  std::string class_name = "", environment = "", super_class = "",
              privateinfo = "", indexed_type = "";
  std::set<std::string> inst_vars;
  std::set<std::string> class_inst_vars;
  std::string imports = "";
  std::string category = "";
  std::string getName() { return class_name; }
  std::string getSuperClassName() { return super_class; }
};

class ModuleAST : public BaseAST {
private:
  std::vector<FunctionAST *> functions;

public:
  ModuleAST() : BaseAST("ModuleAST") {}

  FunctionAST *entry_point = NULL;
  std::string path = "";
  std::string path_config = "";
  std::map<std::string, std::string> configMap;
  std::map<std::string, ClassAST *> classesMap;
  void addFunctionAST(FunctionAST *funcAst) { functions.push_back(funcAst); }
  // TODO: if same class name is added multiple times, merge
  void addClassAST(ClassAST *classAst) {
    if (classesMap.find(classAst->class_name) != classesMap.end()) {
      auto classAst0 = classesMap.at(classAst->class_name);
      classAst0->inst_vars.insert(classAst->inst_vars.begin(),
                                  classAst->inst_vars.end());
      classAst0->class_inst_vars.insert(classAst->class_inst_vars.begin(),
                                        classAst->class_inst_vars.end());
    } else {
      classesMap[classAst->class_name] = classAst;
    }
  }
  std::vector<FunctionAST *> getFunctions() { return functions; }
  std::map<std::string, ClassAST *> &getClassesMap() { return classesMap; }
  uint getClassesMapSize() { return classesMap.size(); }
};

class KeywordPairAST;
class FunctionCallAST : public ExprAST {
public:
  FunctionCallAST(Location loc) : ExprAST(Expr_FuncCall, loc) {}
  FunctionCallAST(ExprASTKind kind, Location loc) : ExprAST(kind, loc) {}

  std::string callee;
  std::vector<ExprAST *> args;

  std::string function_name = "";
  //    std::string parameters;
  std::string params = "";
  ExprAST *syn = NULL;

  std::string getCallee() { return callee; }
  void setCallee(std::string str) { callee = str; }
  llvm::ArrayRef<ExprAST *> getArgs() { return args; }
  void addArg(ExprAST *arg) { args.push_back(arg); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_FuncCall;
  }
  // FunctionCallAST * next = NULL;
};

/// Expression class for code blocks, like "[Transcript show: race]".
class BlockExprAST : public ExprAST {
  const std::string source;
  std::vector<VarDeclExprAST> params;
  std::vector<VarDeclExprAST> locals;

public:
  FunctionAST *parentScope = nullptr;
  FunctionAST *func = nullptr;
  std::set<std::string> outerVars;
  BlockExprAST(Location loc, const std::string &source)
      : ExprAST(Expr_Block, loc), source(source) {}

  const std::string getBlockText() { return source; }
  void addLocal(VarDeclExprAST local) {
    // cout << "adding local: " << local.getName() << "\n";
    locals.push_back(local);
  }
  void addParameter(VarDeclExprAST param) {
    // cout << "adding parameter: " << param.getName() << "\n";
    params.push_back(param);
  }
  bool isLocal(std::string name) {
    for (auto it = locals.begin(); it < locals.end(); ++it) {
      if (it->getName() == name) {
        // cout << "is local: " << name << "\n";
        return true;
      }
    }
    return false;
  }
  bool isParameter(std::string name) {
    for (auto it = params.begin(); it < params.end(); ++it) {
      if (it->getName() == name) {
        // cout << "is parameter: " << name << "\n";
        return true;
      }
    }
    return false;
  }
  bool isBuiltInClass(std::string name) {
    if (name.length() > 0 && isupper(name.front())) {
      // cout << "isBuiltInClass: " << name << "\n";
      return true;
    }
    return false;
  }
  std::vector<VarDeclExprAST> &getBlockParams() { return params; }
  // block has its only locals
  std::vector<VarDeclExprAST> &getLocals() { return locals; }

  std::string getName() {
    if (!func)
      return "emptyblock";
    return func->function_name;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Block; }
};

struct SEM {
  int anonFuncCount = 0;
  int blockNum = 0; // indicator for block nesting
  int indentLevel = 0;
  FunctionAST *curScope; // current function being processed

  std::vector<FunctionAST *> Scopes;
  std::vector<std::string> curScopeName;
  std::any lastUnary; // last unary message
};

} // namespace sol
