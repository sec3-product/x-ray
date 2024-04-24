//
// Created by jncsw on 3/11/21.
//

#ifndef SMALLTALK_AST_H
#define SMALLTALK_AST_H

#include "antlr4-runtime.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace std;
using namespace antlr4;

namespace stx {
/// Structure definition a location in a file.
struct Location {
  std::string file;  ///< filename.
  int line;          ///< line number.
  int col;           ///< column number.
};

class BaseAST {
 public:
  BaseAST(string ASTType) { this->ASTType = ASTType; }

  BaseAST() {}

  string Filename = "";
  int line_number_start = 0, line_number_end = 0, column_start = 0,
      column_end = 0;
  string ASTType = "";
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
  VarAST* LHS = NULL;
  VarAST* RHS = NULL;
  string Operator = "";
};
class VarDeclAST : public BaseAST {
 public:
  VarDeclAST() : BaseAST("VarDeclAST") {}
  string varname = "";
  string var_type = "";
  string initial_value = "";
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

  const Location& loc() { return location; }

  string object_type = "";
  string source = "";
  ExprASTKind ASTType = Default;
  //    FunctionCallAST * Expression = NULL;
  //    AssignAST * assignStmt = NULL;
  //    BlockAST * blockAst = NULL;
  string ParsetimeLiteral = "";
  string primitive = "";
  string reference;
  bool valid() { return ASTType != ExprASTKind::Default; }

  string getName() {
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

  string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Var; }
};
/// Ex
// A block-list of expressions.
using ExprASTList = std::vector<ExprAST*>;
class LiteralExprAST : public ExprAST {
  std::string name;

 public:
  LiteralExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Literal, loc), name(name) {}

  string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Literal; }
};
class StringExprAST : public ExprAST {
  std::string name;

 public:
  StringExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_String, loc), name(name) {}

  string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_String; }
};
class SymbolExprAST : public ExprAST {
  std::string name;

 public:
  SymbolExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Symbol, loc), name(name) {}

  string getSymbol() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Symbol; }
};
class ReservedKeywordExprAST : public ExprAST {
  std::string name;

 public:
  ReservedKeywordExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Reserved, loc), name(name) {}

  string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
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
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Num; }
};

class DynamicDictionaryExprAST : public ExprAST {
  std::vector<ExprAST*> dict;

 public:
  DynamicDictionaryExprAST(Location loc, std::vector<ExprAST*> dict)
      : ExprAST(Expr_DynamicDictionary, loc), dict(dict) {}

  std::vector<ExprAST*>& getExpressions() { return dict; };
  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_DynamicDictionary;
  }
};
class DynamicArrayExprAST : public ExprAST {
  std::vector<ExprAST*> arr;

 public:
  DynamicArrayExprAST(Location loc, std::vector<ExprAST*> arr)
      : ExprAST(Expr_DynamicArray, loc), arr(arr) {}
  std::vector<ExprAST*>& getExpressions() { return arr; };

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_DynamicArray;
  }
};

class PrimitiveExprAST : public ExprAST {
  std::string name;

 public:
  PrimitiveExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Primitive, loc), name(name) {}

  string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_Primitive;
  }
};

class ParserErrorAST : public ExprAST {
  std::string name;

 public:
  ParserErrorAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Error, loc), name(name) {}

  string getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Error; }
};

class ReturnExprAST : public ExprAST {
  llvm::Optional<ExprAST*> expr;

 public:
  ReturnExprAST(Location loc, llvm::Optional<ExprAST*>& expr)
      : ExprAST(Expr_Return, loc), expr(expr) {}

  llvm::Optional<ExprAST*> getExpr() {
    if (expr.hasValue()) return expr;
    return llvm::None;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Return; }
};
/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
  ExprAST* arg;

 public:
  PrintExprAST(Location loc, ExprAST* arg)
      : ExprAST(Expr_Print, loc), arg(arg) {}

  ExprAST* getArg() { return arg; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Print; }
};
class AssignExprAST : public ExprAST {
  std::string name;  // variable name 'teset'
  VariableExprAST* var;
  ExprAST* valueExpr;

 public:
  AssignExprAST(Location loc, VariableExprAST* var, ExprAST* valueExpr)
      : ExprAST(Expr_Assign, loc), var(var), valueExpr(valueExpr) {}

  string getVarName() { return var->getName(); }
  ExprAST* getValueExpr() { return valueExpr; }
  VariableExprAST* getLHS() { return var; };
  ExprAST* getRHS() { return valueExpr; };
  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Assign; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
  const std::string name;
  const VarType type;
  ExprAST* initVal;

 public:
  VarDeclExprAST(Location loc, std::string name, const VarType& type,
                 ExprAST* initVal = nullptr)
      : ExprAST(Expr_VarDecl, loc), name(name), type(type), initVal(initVal) {
    // llvm::outs() << "-----VarDeclExprAST ------varName: " << name << "\n";
  }
  llvm::StringRef getNameStringRef() const { return name; }
  string getName() const { return name; }
  ExprAST* getInitVal() { return initVal; }
  const VarType& getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_VarDecl; }
};

class PrototypeAST : public BaseAST {
  Location location;
  const std::string name;
  std::vector<VarDeclExprAST> args;
  ExprAST* ret;

 public:
  PrototypeAST(Location location, std::string name,
               std::vector<VarDeclExprAST>& args, ExprAST* ret = nullptr)
      : location(location), name(name), args(args), ret(ret) {}

  const Location& loc() { return location; }
  string getName() { return name; }
  ExprAST* getRet() { return ret; };
  std::vector<VarDeclExprAST>& getArgs() { return args; }
};

class FunctionAST : public BaseAST {
 private:
  Location loc;

 public:
  FunctionAST(Location loc) : loc(loc) {}
  string function_name;
  PrototypeAST* proto;
  vector<VarDeclExprAST> temp_vars;
  set<string> writeVars;
  set<string> usedVars;
  void addUsedVar(std::string& name) { usedVars.insert(name); }
  vector<ExprAST*> body;
  string source = "";
  ExprAST* return_val = NULL;
  PrototypeAST* getProto() { return proto; }
  vector<VarDeclExprAST>& getLocals() { return temp_vars; }
  VarDeclExprAST& getLocal(int k) { return temp_vars[k]; }
  ExprASTList* getBody() { return &body; }
  string getName() { return function_name; }
};

class ClassAST : public BaseAST {
 public:
  ClassAST() : BaseAST("ClassAST") {}
  vector<FunctionAST*> functions;
  bool valid = false;
  int line;
  string fileName = "";
  string class_name = "", environment = "", super_class = "", privateinfo = "",
         indexed_type = "";
  set<std::string> inst_vars;
  set<std::string> class_inst_vars;
  string imports = "";
  string category = "";
  string getName() { return class_name; }
  string getSuperClassName() { return super_class; }
};
class ModuleAST : public BaseAST {
 private:
  std::vector<FunctionAST*> functions;

 public:
  ModuleAST() : BaseAST("ModuleAST") {}

  FunctionAST* entry_point = NULL;
  string path = "";
  std::map<string, ClassAST*> classesMap;
  void addFunctionAST(FunctionAST* funcAst) { functions.push_back(funcAst); }
  // TODO: if same class name is added multiple times, merge
  void addClassAST(ClassAST* classAst) {
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
  std::vector<FunctionAST*> getFunctions() { return functions; }
  std::map<string, ClassAST*>& getClassesMap() { return classesMap; }
  uint getClassesMapSize() { return classesMap.size(); }
};

class KeywordPairAST;
class FunctionCallAST : public ExprAST {
 public:
  FunctionCallAST(Location loc) : ExprAST(Expr_FuncCall, loc) {}
  FunctionCallAST(ExprASTKind kind, Location loc) : ExprAST(kind, loc) {}

  std::string callee;
  std::vector<ExprAST*> args;

  string function_name = "";
  //    string parameters;
  string params = "";
  ExprAST* syn = NULL;

  string getCallee() { return callee; }
  void setCallee(std::string str) { callee = str; }
  llvm::ArrayRef<ExprAST*> getArgs() { return args; }
  void addArg(ExprAST* arg) { args.push_back(arg); }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Call; }
  // FunctionCallAST * next = NULL;
};

/// Expression class for code blocks, like "[Transcript show: race]".
class BlockExprAST : public ExprAST {
  const std::string source;
  vector<VarDeclExprAST> params;
  vector<VarDeclExprAST> locals;

 public:
  FunctionAST* parentScope = nullptr;
  FunctionAST* func = nullptr;
  set<string> outerVars;
  BlockExprAST(Location loc, const std::string& source)
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
  vector<VarDeclExprAST>& getBlockParams() { return params; }
  // block has its only locals
  vector<VarDeclExprAST>& getLocals() { return locals; }

  string getName() {
    if (!func) return "emptyblock";
    return func->function_name;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Block; }
};

class UnaryMessageAST : public ExprAST {
 public:
  UnaryMessageAST(Location loc) : ExprAST(Expr_UnaryMessage, loc) {}
  bool startFlag = false;
  string source = "";
  vector<string> unaryTails;
  FunctionCallAST* fc = NULL;
  ExprAST* expr;

  //    UnaryMessageAST* next = NULL;
  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_UnaryMessage;
  }
};
class BinaryMessageAST : public ExprAST {
 public:
  BinaryMessageAST(Location loc) : ExprAST(Expr_BinaryMessage, loc) {}
  bool startFlag = false;
  string opr = "";
  UnaryMessageAST* operand = NULL;

  BinaryMessageAST* next = NULL;
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_BinaryMessage;
  }
};
class BinaryFunctionCallAST : public FunctionCallAST {
 public:
  BinaryFunctionCallAST(Location loc, UnaryMessageAST* umessage)
      : FunctionCallAST(Expr_FuncCall_Bin, loc), umessage(umessage) {}
  UnaryMessageAST* umessage;
  vector<string> umessagesTails;
  BinaryMessageAST* bmessages = NULL;
  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_FuncCall_Bin;
  }
};
class KeywordPairAST : public ExprAST {
 public:
  KeywordPairAST(Location loc) : ExprAST(Expr_KeywordPair, loc) {}
  string method_name = "";
  //    string parameters;
  BinaryFunctionCallAST* params;
  //    string params = "";
  string getName() { return method_name; }
  BinaryFunctionCallAST* getParams() { return params; }
  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_KeywordPair;
  }
};
class KeywordFunctionCallAST : public FunctionCallAST {
 private:
  BinaryFunctionCallAST* binSend;
  vector<KeywordPairAST*> messages;
  string func_name;

 public:
  KeywordFunctionCallAST(Location loc, BinaryFunctionCallAST* binSend,
                         vector<KeywordPairAST*> kwMsgs, string func_name)
      : FunctionCallAST(Expr_FuncCall_Key, loc),
        binSend(binSend),
        messages(kwMsgs),
        func_name(func_name) {}
  BinaryFunctionCallAST* getBinarySend() { return binSend; }
  vector<KeywordPairAST*> getKeywordMessages() { return messages; }
  string getName() { return func_name; }
  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_FuncCall_Key;
  }
};

class KeywordMessageAST : public ExprAST {
 private:
  std::string func_name;
  vector<KeywordPairAST*> messages;

 public:
  KeywordMessageAST(Location loc, vector<KeywordPairAST*> kwMsgs,
                    std::string func_name)
      : ExprAST(Expr_KeywordMessage, loc),
        messages(kwMsgs),
        func_name(func_name) {}

  string getName() { return func_name; }
  vector<KeywordPairAST*> getKeywordMessages() { return messages; }

  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_KeywordMessage;
  }
};

class CascadeFunctionCallAST : public FunctionCallAST {
 private:
 public:
  CascadeFunctionCallAST(Location loc)
      : FunctionCallAST(Expr_FuncCall_Cascade, loc) {}

  vector<KeywordMessageAST*> kwmsgStack;  // keyword msg
  vector<std::string> umsgStack;          // unary msg
  // TODO: merge them
  string func_name = "";
  string callType = "";
  FunctionCallAST* subCall = NULL;
  KeywordPairAST* umessage = NULL;
  void addKeyWordMessage(KeywordMessageAST* kwmsg) {
    kwmsgStack.push_back(kwmsg);
  }
  void addUnaryMessage(std::string umsg) { umsgStack.push_back(umsg); }
  vector<KeywordMessageAST*>& getKeywordMessagesStack() { return kwmsgStack; }
  vector<std::string>& getUnaryMessagesStack() { return umsgStack; }
  string getName() { return func_name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST* c) {
    return c->getKind() == Expr_FuncCall_Cascade;
  }
};

typedef struct {
  int anonFuncCount = 0;
  int blockNum = 0;  // indicator for block nesting
  int indentLevel = 0;
  FunctionAST* curScope;  // current function being processed

  vector<FunctionAST*> Scopes;
  vector<string> curScopeName;
  antlrcpp::Any lastUnary;  // last unary message
} SEM;
}  // namespace stx

#endif  // SMALLTALK_AST_H