#ifndef MLIR_ST_PARSER_WRAPPER_H_
#define MLIR_ST_PARSER_WRAPPER_H_

#include "RustParser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace antlr4;

namespace st {

/// Structure definition a location in a file.
struct Location {
  std::string file;  ///< filename.
  int line;          ///< line number.
  int col;           ///< column number.
};
/// A variable type with either name or shape information.
struct VarType {
 public:
  std::string name;
  std::vector<int64_t> shape;
};
/// Base class for all expression nodes.
class ExprAST {
 public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_StructLiteral,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
    Expr_Function,
    Expr_Assign,
    Expr_Block,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}
  // virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

 private:
  const ExprASTKind kind;
  Location location;
};

// A block-list of expressions.
using ExprASTList = std::vector<ExprAST *>;

/// Expression class for numeric literals like "10".
class NumberExprAST : public ExprAST {
  long Val;

 public:
  NumberExprAST(Location loc, long val) : ExprAST(Expr_Num, loc), Val(val) {}

  long getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<ExprAST *> args;
  int level;

 public:
  CallExprAST(Location loc, const std::string &callee,
              std::vector<ExprAST *> &args, int level)
      : ExprAST(Expr_Call, loc), callee(callee), args(args), level(level) {}

  llvm::StringRef getCallee() { return callee; }
  void setCallee(std::string str) { callee = str; }
  llvm::ArrayRef<ExprAST *> getArgs() { return args; }
  void addArg(ExprAST *arg) { args.push_back(arg); }
  int getLevel() { return level; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
};

/// Expression class for assignment, like "test := LamLLVMPrototype new".
class AssignExprData {
  std::string name;  // variable name 'test'
  mlir::Value v;
  int level;

 public:
  AssignExprData(mlir::Value v, llvm::StringRef name, int level)
      : v(v), name(name), level(level) {}

  llvm::StringRef getVarName() { return name; }
  mlir::Value getValue() { return v; }
  int getLevel() { return level; }
};

/// Expression class for assignment, like "test := LamLLVMPrototype new".
class AssignExprAST : public ExprAST {
  std::string name;  // variable name 'teset'
  ExprAST *valueExpr;
  int level;

 public:
  AssignExprAST(Location loc, llvm::StringRef name, ExprAST *valueExpr,
                int level)
      : ExprAST(Expr_Assign, loc),
        name(name),
        valueExpr(valueExpr),
        level(level) {}

  llvm::StringRef getVarName() { return name; }
  ExprAST *getValueExpr() { return valueExpr; }

  int getLevel() { return level; }
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Assign; }
};

/// Expression class for code blocks, like "[Transcript show: race]".
class BlockExprAST : public ExprAST {
  int level;
  const std::string text;
  ExprASTList blockAST;

 public:
  BlockExprAST(Location loc, const std::string &text, int level)
      : ExprAST(Expr_Block, loc), text(text), level(level) {}

  int getLevel() { return level; }
  ExprASTList getBlockAST() { return blockAST; }

  const std::string getBlockText() { return text; }
  void addExpr(ExprAST *expr) { blockAST.insert(blockAST.begin(), expr); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Block; }
};
/// Expression class for string literals, like "hi".
class LiteralExprAST : public ExprAST {
  std::string name;

 public:
  LiteralExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Literal, loc), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
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

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string name;

 public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, loc), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  llvm::Optional<ExprAST *> expr;

 public:
  ReturnExprAST(Location loc, llvm::Optional<ExprAST *> &expr)
      : ExprAST(Expr_Return, loc), expr(expr) {}

  llvm::Optional<ExprAST *> getExpr() {
    if (expr.hasValue()) return expr;
    return llvm::None;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
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
    llvm::outs() << "-----VarDeclExprAST ------varName: " << name << "\n";
  }

  llvm::StringRef getName() const { return name; }
  ExprAST *getInitVal() { return initVal; }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  Location location;
  const std::string name;
  std::vector<VarDeclExprAST> args;

 public:
  PrototypeAST(Location location, std::string name,
               std::vector<VarDeclExprAST> &args)
      : location(location), name(name), args(args) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }
  std::vector<VarDeclExprAST> getArgs() { return args; }
};

/// This class represents a function definition itself.
class FunctionAST : public ExprAST {
  PrototypeAST proto;
  ExprASTList body;

 public:
  FunctionAST(PrototypeAST &proto, ExprASTList &body)
      : ExprAST(Expr_Function, proto.loc()), proto((proto)), body((body)) {}
  PrototypeAST *getProto() { return &proto; }
  ExprASTList *getBody() { return &body; }
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_Function;
  }
};

}  // namespace st

#endif  // MLIR_ST_PARSER_WRAPPER_H_
