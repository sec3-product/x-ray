#include "sol/MLIRGen.h"

#include <atomic>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Verifier.h>

#include "ast/AST.h"
#include "sol/ScopedHashTableX.h"

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using namespace mlir;

const std::string SOL_ANON_FUNC_NAME = "sol.anonfun.";
const std::string ANON_NAME = "anon";
const std::string SOL_NEW = "sol.new";

// int ANON_FUNC_INDEX = 0;
std::atomic<int> ANON_FUNC_INDEX{0};
const std::string MAIN_FUNC_NAME = "main";
const std::string GLOBAL_VAR_NAME = "global_";
const std::string LOCAL_VAR_NAME = ""; //"local_";
const std::string GLOBAL_OP_NAME = "global_op_";

const std::string SOL_BUILT_IN_NAME =
    "sol."; // for all the built-in st function
const std::string SOL_BUILT_IN_MODEL_NAME = "sol.model.";

const std::string SOL_BUILT_IN_MODEL_CARGO_TOML = "sol.model.cargo.toml";
const std::string SOL_BUILT_IN_MODEL_TOML = "sol.model.toml";
const std::string SOL_BUILT_IN_MODEL_DECLARE_ID = "sol.model.declare_id";
const std::string SOL_BUILT_IN_MODEL_DECLARE_ID_ADDRESS =
    "sol.model.declare_id.address";
const std::string SOL_BUILT_IN_MODEL_NEW_TEMP = "sol.model.newTemp";
const std::string SOL_BUILT_IN_MODEL_NEW_OBJECT = "sol.model.newObject";
const std::string SOL_BUILT_IN_MODEL_NEW_OBJECT2 = "sol.model.newObject2";
const std::string SOL_BUILT_IN_MODEL_FUNC_ARG = "sol.model.funcArg";
const std::string SOL_BUILT_IN_MODEL_INST_VAR = "sol.model.instVar";
const std::string SOL_BUILT_IN_MODEL_CLASS_VAR = "sol.model.classVar";
const std::string SOL_BUILT_IN_MODEL_PARENT_VAR = "sol.model.parentVar";
const std::string SOL_BUILT_IN_MODEL_PARENT_SCOPE = "sol.model.parentScope";
const std::string SOL_BUILT_IN_MODEL_OPAQUE_ASSIGN = "sol.model.opaqueAssign";
const std::string SOL_BUILT_IN_MODEL_BINARY_OP = "sol.model.binaryop";
const std::string SOL_BUILT_IN_MODEL_BLOCK_PARAM = "sol.model.blockParam";

const std::string SOL_BUILT_IN_CLASS_NAME = "sol.class.";
const std::string SOL_BUILT_IN_CLASS_METADATA = "sol.class.metadata";
const std::string SOL_BUILT_IN_CLASS_SUPER = "sol.class.super";

extern std::set<std::string> declareIdAddresses;

extern int LOWER_BOUND_ID;
extern bool DEBUG_SOL;
/// A "module" matches a Toy source file: containing a list of functions.
mlir::ModuleOp theModule;
LLVM::LLVMPointerType llvmI8PtrTy;
mlir::Type llvmVoidTy, llvmI32Ty, llvmI64Ty;
/// A mapping for the functions that have been code generated to MLIR.
llvm::StringMap<mlir::FuncOp> functionMap;

namespace sol {

class MLIRGenImpl {
private:
  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  std::map<std::string, ClassAST *> classesInfoMap;
  std::map<std::string, std::vector<std::string>> inst_varsClassMap;
  std::map<std::string, std::vector<std::string>> class_varsClassMap;

  std::vector<std::map<std::string, std::pair<llvm::StringRef, mlir::Value>> *>
      symbolTableStack;

  llvm::ScopedHashTableX<StringRef, std::pair<StringRef, mlir::Value>>
      symbolTable2;
  using SymbolTableScopeT2 =
      llvm::ScopedHashTableXScope<StringRef, std::pair<StringRef, mlir::Value>>;
  std::vector<StringRef> functionStack;
  void pushNewFunction(StringRef funcName) {
    functionStack.push_back(funcName);
    auto map =
        new std::map<std::string, std::pair<llvm::StringRef, mlir::Value>>();
    symbolTableStack.push_back(map);
  }
  void popLastFunction(StringRef funcName) {
    if (functionStack.size() > 0 && functionStack.back().equals(funcName)) {
      functionStack.pop_back();

      auto map = symbolTableStack.back();
      symbolTableStack.pop_back();
      delete map;

    } else {
      if (DEBUG_SOL)
        llvm::outs() << "-----popLastFunction error------func: " << funcName
                     << "\n";
      // if functionStack is empty now, then clear all symbolTable
      // TODO: free memory
      // symbolTable2.clear();
      symbolTableStack.clear();
    }
  }
  llvm::StringRef findSymbolInParentScopes(std::string varName) {
    for (auto rit = symbolTableStack.rbegin() + 1;
         rit != symbolTableStack.rend(); ++rit) {
      std::map<std::string, std::pair<llvm::StringRef, mlir::Value>> &map =
          **rit;
      if (map.find(varName) != map.end()) {
        return map.at(varName).first;
      }
    }
    return "";
  }
  StringRef getCurrentFunctionName() {
    if (functionStack.size() > 0)
      return functionStack.back();
    else {
      if (DEBUG_SOL)
        llvm::outs()
            << "-----getCurrentFunctionName error------empty functionStack!!\n";
      return "sol.error.stub";
    }
  }
  std::string getCurrentNonAnonFunctionName() {
    int size = functionStack.size();
    if (size > 0) {
      for (int i = size - 1; i >= 0; i--) {
        auto name = functionStack[i];
        if (!name.startswith(SOL_ANON_FUNC_NAME))
          return name.str();
      }
      if (DEBUG_SOL)
        llvm::outs() << "-----getCurrentNonAnonFunctionName error------all "
                        "functions are anonymous!!\n";
      return functionStack.back().str(); // if we get here, something is wrong
    } else {
      if (DEBUG_SOL)
        llvm::outs() << "-----getCurrentNonAnonFunctionName error------empty "
                        "functionStack!!\n";
      return "sol.error.stub";
    }
  }

  bool matchCurrentScopeFunction(StringRef funcName) {
    // make sure
    if (functionStack.size() == 0) {
      if (DEBUG_SOL)
        llvm::outs() << "-----matchCurrentScopeFunction error------\n";
      return false;
    }
    return functionStack.back().equals(funcName);
  }

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(Location loc) {
    auto fileName = loc.file;
    if (fileName.empty())
      if (DEBUG_SOL)
        llvm::outs() << "-----empty file name------"
                     << "\n";
    // else
    //   if (DEBUG_SOL) llvm::outs() << "-----loc file name: " << loc.file <<
    //   "------"
    //                << "\n";

    return mlir::FileLineColLoc::get(builder.getStringAttr(fileName), loc.line,
                                     loc.col);
  }
  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(mlir::OpBuilder &builder, Location loc) {
    if (loc.file.empty())
      if (DEBUG_SOL)
        llvm::outs() << "-----empty file name------"
                     << "\n";
    // else
    //   if (DEBUG_SOL) llvm::outs() << "-----loc file name: " << loc.file <<
    //   "------"
    //                << "\n";

    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.file), loc.line,
                                     loc.col);
  }
  Value getOrCreateGlobalStringX(mlir::Location loc, std::string name,
                                 StringRef value) {
    if (name == "local_self") {
      auto v = getValueFromSymbolTable("this");
      if (v)
        return v;
    }

    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = theModule.lookupSymbol<LLVM::GlobalOp>(name))) {
      if (DEBUG_SOL)
        llvm::outs() << "-----adding GlobalString ------" << name << "\n";

      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(theModule.getBody());
      auto type = LLVM::LLVMArrayType::get(llvmI8PtrTy, value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    // return globalPtr;

    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(loc, llvmI8PtrTy, globalPtr,
                                       ArrayRef<Value>({cst0, cst0}));
  }
  FlatSymbolRefAttr
  getOrInsertUserDefinedFunctionX(mlir::Location mlocation,
                                  llvm::StringRef name,
                                  SmallVector<mlir::Value, 4> operands) {
    // special call: st.new
    if (name.str() == SOL_NEW) {
      name = SOL_BUILT_IN_MODEL_NEW_OBJECT;
    } else if (name.str() == SOL_NEW) {
      name = SOL_BUILT_IN_MODEL_NEW_OBJECT2;
    }

    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    if (DEBUG_SOL)
      llvm::outs() << "-----adding UserDefinedSolFunction ------" << name
                   << "\n";

    // Create a function declaration for opaqueObject, the signature is:
    //   * `i32 (i8*, ...)`
    std::vector<mlir::Type> inputTypes;
    for (auto operand : operands)
      inputTypes.push_back(operand.getType());

    auto mlirFnType = mlir::FunctionType::get(context, inputTypes, llvmI8PtrTy);
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto func =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // declare only

    return SymbolRefAttr::get(context, name);
  }

  FlatSymbolRefAttr getOrInsertSuperClassFunctionX() {
    llvm::StringRef name = SOL_BUILT_IN_CLASS_SUPER;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    // Create a function declaration, the signature is:
    //   * `void (i8*,i8*)`
    auto mlirFnType = mlir::FunctionType::get(
        context, {llvmI8PtrTy, llvmI8PtrTy}, llvmVoidTy);
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto func =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }

  FlatSymbolRefAttr getOrInsertAssignFunctionX(llvm::StringRef name) {
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    // Create a function declaration, the signature is:
    //   * `void (i8*,i8*)`
    auto mlirFnType = mlir::FunctionType::get(
        context, {llvmI8PtrTy, llvmI8PtrTy}, llvmVoidTy);
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto func =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }

  FlatSymbolRefAttr getOrInsertForkAtFunctionX(llvm::StringRef name) {
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    // Create a function declaration for opaqueObject, the signature is:
    //   * `void (i8*, i32...)`
    auto mlirFnType =
        mlir::FunctionType::get(context, {llvmI8PtrTy, llvmI64Ty}, llvmI8PtrTy);
    if (name.endswith("named:"))
      mlirFnType = mlir::FunctionType::get(
          context, {llvmI8PtrTy, llvmI64Ty, llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto func =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }

  FlatSymbolRefAttr getOrInsertBuiltInParentScopeX() {
    auto name = SOL_BUILT_IN_MODEL_PARENT_SCOPE;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType = mlir::FunctionType::get(
        context, {llvmI8PtrTy, llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }

  FlatSymbolRefAttr getOrInsertBuiltInArgTypeFunctionX() {
    auto name = SOL_BUILT_IN_MODEL_FUNC_ARG;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType = mlir::FunctionType::get(
        context, {llvmI8PtrTy, llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr getOrInsertBuiltInNewFunctionX() {
    auto name = SOL_BUILT_IN_MODEL_NEW_TEMP;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType =
        mlir::FunctionType::get(context, {llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr getOrInsertBuiltInDeclareIdFunctionX() {
    auto name = SOL_BUILT_IN_MODEL_DECLARE_ID;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType =
        mlir::FunctionType::get(context, {llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr getOrInsertBuiltInTOMLFunctionX() {
    auto name = SOL_BUILT_IN_MODEL_TOML;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType = mlir::FunctionType::get(
        context, {llvmI8PtrTy, llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr getOrInsertBuiltInParentVarFunctionX() {
    auto name = SOL_BUILT_IN_MODEL_PARENT_VAR;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType = mlir::FunctionType::get(
        context, {llvmI8PtrTy, llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr getOrInsertBuiltInInstVarFunctionX() {
    auto name = SOL_BUILT_IN_MODEL_INST_VAR;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType = mlir::FunctionType::get(
        context, {llvmI8PtrTy, llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr
  getOrInsertBuiltInBlockParamFunctionX(std::vector<mlir::Value> &operands) {
    auto name =
        SOL_BUILT_IN_MODEL_BLOCK_PARAM + "." + std::to_string(ANON_FUNC_INDEX);

    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);
    std::vector<mlir::Type> inputTypes;
    for (auto op : operands)
      inputTypes.push_back(op.getType());
    auto mlirFnType = mlir::FunctionType::get(context, inputTypes, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr getOrInsertBuiltInClassVarFunctionX() {
    auto name = SOL_BUILT_IN_MODEL_CLASS_VAR;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType =
        mlir::FunctionType::get(context, {llvmI8PtrTy}, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr getOrInsertBuiltInClassMetaFunctionX() {
    auto name = SOL_BUILT_IN_CLASS_METADATA;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);

    auto mlirFnType =
        mlir::FunctionType::get(context, {llvmI8PtrTy}, llvmI64Ty);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }
  FlatSymbolRefAttr
  getOrInsertBuiltInBinaryOpFunctionX(SmallVector<mlir::Value, 3> &operands) {
    auto name = SOL_BUILT_IN_MODEL_BINARY_OP;
    auto *context = builder.getContext();
    if (theModule.lookupSymbol<FuncOp>(name))
      return SymbolRefAttr::get(context, name);
    std::vector<mlir::Type> inputTypes;
    for (auto operand : operands)
      inputTypes.push_back(operand.getType());
    auto mlirFnType = mlir::FunctionType::get(context, inputTypes, llvmI8PtrTy);

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(theModule.getBody());
    [[maybe_unused]] auto newFunc =
        builder.create<FuncOp>(theModule.getLoc(), name, mlirFnType);
    // store into functionMap
    // functionMap.insert({func.getName(), func});

    return SymbolRefAttr::get(context, name);
  }

public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.getContext()->loadDialect<LLVM::LLVMDialect>();
    llvmVoidTy = LLVM::LLVMVoidType::get(builder.getContext());
    llvmI8PtrTy =
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8));
    llvmI32Ty = builder.getI32Type();
    llvmI64Ty = builder.getI64Type();
  }

  mlir::ModuleOp mlirGen(sol::ModuleAST &moduleAST) {
    // all classes
    classesInfoMap = moduleAST.getClassesMap();
    for (auto &[className, classAST] : classesInfoMap) {
      classesInfoMap[className] = classAST;
      for (auto value : classAST->inst_vars) {
        inst_varsClassMap[value].push_back(className);
      }
      for (auto value : classAST->class_inst_vars) {
        class_varsClassMap[value].push_back(className);
      }

      // Location loc({classAST->fileName, classAST->line, 0});
      [[maybe_unused]] auto loc = mlir::FileLineColLoc::get(
          builder.getStringAttr(classAST->fileName), classAST->line, 0);
      // auto value = classAST->inst_vars;
      // auto type = LLVM::LLVMType::getArrayTy(llvmI8PtrTy, value.size());
      // builder.create<LLVM::GlobalOp>(
      //     loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
      //     classAST->getName(), builder.getStrArrayAttr(value));
    }

    // main entry point
    if (moduleAST.entry_point) {
      auto func = mlirGen(*moduleAST.entry_point);
      if (func && functionMap.find(func.getName()) == functionMap.end()) {
        theModule.push_back(func);
        functionMap.insert({func.getName(), func});
      }
    }

    // functions for saving class-level meta data
    // define i64 @st.class.metadata() !dbg !3 {
    // %3 = call i8* @st.class.super(i8* getelementptr inbounds ([15 x i8], [15
    // x i8]* @LamAlarmHandler, i64 0, i64 0),CTAlarmHandler), !dbg !10
    if (moduleAST.getClassesMapSize() > 0) {
      auto func = mlirGenClassesMetadata(moduleAST);
      if (func && functionMap.find(func.getName()) == functionMap.end()) {
        theModule.push_back(func);
        functionMap.insert({func.getName(), func});
      }
    }

    // Toml config func
    {
      auto func = mlirGenTomlConfiguration(moduleAST);
      if (func && functionMap.find(func.getName()) == functionMap.end()) {
        theModule.push_back(func);
        functionMap.insert({func.getName(), func});
      }
    }
    // declare_id addresses func
    {
      auto func = mlirGenDeclareIdAddresses(moduleAST);
      if (func && functionMap.find(func.getName()) == functionMap.end()) {
        theModule.push_back(func);
        functionMap.insert({func.getName(), func});
      }
    }
    // all other functions
    // classes? => functions?
    for (auto funcAST : moduleAST.getFunctions()) {
      // skip repeated function name

      auto func = mlirGen(*funcAST);
      if (func && functionMap.find(func.getName()) == functionMap.end()) {
        theModule.push_back(func);
        functionMap.insert({func.getName(), func});
      }
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    // if(false)
    // if (mlir::failed(mlir::verify(theModule))) {
    //   // theModule.emitError("module verification error");
    //   // return nullptr;
    // }

    // theModule.dump();
    if (DEBUG_SOL)
      llvm::outs() << "\n-----MLIR-GEN-DONE------\n\n";

    return theModule;
  }

  mlir::LogicalResult declareX2(std::string varName, mlir::Value value,
                                StringRef funcName) {
    if (symbolTableStack.size() == 0) {
      if (DEBUG_SOL)
        llvm::outs() << "-----empty symbolTableStack---\n";
      return mlir::failure();
    }
    if (symbolTableStack.back()->count(varName)) {
      if (DEBUG_SOL)
        llvm::outs() << "-----symbolTable------key exists already: " << varName
                     << " func: " << funcName << " value: " << value << "\n";
      // TODO: erase it
      return mlir::failure();
    }

    (*symbolTableStack.back())[varName] = std::make_pair(funcName, value);

    // if (DEBUG_SOL) llvm::outs() << "-----symbolTable------key: " << varName
    //              << " func: " << funcName << " value: " << value << "\n";

    return mlir::success();
  }
  mlir::LogicalResult declareX(StringRef varName, mlir::Value value,
                               StringRef funcName) {
    if (symbolTable2.count(varName)) {
      if (DEBUG_SOL)
        llvm::outs() << "-----symbolTable------key exists already: " << varName
                     << " func: " << funcName << " value: " << value << "\n";
      // TODO: erase it
      return mlir::failure();
    }
    symbolTable2.insert(varName, {funcName, value});
    if (DEBUG_SOL)
      llvm::outs() << "-----symbolTable------key: " << varName
                   << " func: " << funcName << " value: " << value << "\n";

    return mlir::success();
  }
  mlir::Value mlirGen(FunctionCallAST &call) {
    auto callee = call.getCallee();
    // Processor::<{TOKEN_COUNT}>::process.3
    if (callee.find("::<") != std::string::npos) {
      auto found1 = callee.find("::<");
      auto found2 = callee.find(">");
      callee = callee.substr(0, found1) + callee.substr(found2 + 1);
      call.setCallee(callee);
    }

    if (DEBUG_SOL)
      llvm::outs() << "-----creating FunctionCallAST------callee: " << callee
                   << "\n";
    SmallVector<mlir::Value, 4> operands;
    for (auto param : call.getArgs()) {
      if (param != nullptr) {
        auto arg = mlirGen(*param);
        operands.push_back(arg);
      }
    }
    auto mlocation = loc(call.loc());

    if (DEBUG_SOL)
      llvm::outs() << "-----call on expr------name: " << callee
                   << " args_size: " << operands.size() << "\n";

    auto calleeRef = getOrInsertUserDefinedFunctionX(
        mlocation, SOL_BUILT_IN_NAME + callee, operands);
    auto callOp = builder.create<LLVM::CallOp>(mlocation, llvmI8PtrTy,
                                               calleeRef, operands);
    return callOp.getResult(0);
  }

  mlir::Value getValueFromSymbolTable(std::string varName) {
    if (symbolTableStack.back()->count(varName)) {
      auto result = symbolTableStack.back()->at(varName);
      auto funcName = result.first;
      auto value = result.second;

      // TODO: if this is in anonymous func, create op for var

      if (DEBUG_SOL)
        llvm::outs() << "-----found " << varName
                     << " in symbolTable------func: " << funcName
                     << " value: " << value << "\n";

      // make sure the function name match
      if (matchCurrentScopeFunction(funcName))
        return value;
    } else
      return nullptr;
  }
  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &expr) {
    std::string varName = expr.getName();

    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen VariableExprAST------name: " << varName
                   << "\n";
    // TODO: maintain symbolTable per func
    // if (auto variable = symbolTable.lookup(expr.getName()).first) {
    if (symbolTableStack.size() == 0) {
      if (DEBUG_SOL)
        llvm::outs() << "-----error: empty symbolTable------\n";
    }
    if (symbolTableStack.back()->count(expr.getName())) {
      auto result = symbolTableStack.back()->at(expr.getName());
      auto funcName = result.first;
      auto value = result.second;

      // TODO: if this is in anonymous func, create op for var

      if (DEBUG_SOL)
        llvm::outs() << "-----found in symbolTable------func: " << funcName
                     << " value: " << value << "\n";

      // make sure the function name match
      if (matchCurrentScopeFunction(funcName))
        return value;
      else {
        if (DEBUG_SOL)
          llvm::outs() << "-----did not match function scope------\n";
        // call st.model.parentScope
        auto newScopeRef = getOrInsertBuiltInParentScopeX();
        std::string name = LOCAL_VAR_NAME + varName;
        auto varLoc = loc(expr.loc());
        Value v2 = getOrCreateGlobalStringX(varLoc, name, varName);

        Value v1 = getOrCreateGlobalStringX(
            varLoc, GLOBAL_VAR_NAME + funcName.str(), funcName.str());

        auto callOp = builder.create<LLVM::CallOp>(
            varLoc, llvmI8PtrTy, newScopeRef, ArrayRef<Value>({v1, v2}));
        auto value = callOp.getResult(0);
        declareX2(name, value, funcName);
        return value;
      }
    } else {
      // std::string name = GLOBAL_VAR_NAME + expr.getName().str();
      if (DEBUG_SOL)
        llvm::outs() << "-----did not find in symbolTable------name: "
                     << varName << "\n";
      auto varLoc = loc(expr.loc());

      auto input = varName;
      // check classesInfoMap if input is an instance var or class var
      if (inst_varsClassMap.find(input) != inst_varsClassMap.end()) {
        if (DEBUG_SOL)
          llvm::outs() << "found inst_var: " << input << "\n";
        for (auto className : inst_varsClassMap.at(input)) {
          if (DEBUG_SOL)
            llvm::outs() << "possible className: " << className << "\n";
        }
        auto instVarRef = getOrInsertBuiltInInstVarFunctionX();
        Value v = getOrCreateGlobalStringX(varLoc, varName, varName);
        auto v0 = getValueFromSymbolTable("this");
        if (v0) {
          auto callOp = builder.create<LLVM::CallOp>(
              varLoc, llvmI8PtrTy, instVarRef, ArrayRef<Value>({v0, v}));
          auto value = callOp.getResult(0);
          return value;
        }

      } else if (class_varsClassMap.find(input) != class_varsClassMap.end()) {
        if (DEBUG_SOL)
          llvm::outs() << "found class_var: " << input << "\n";
        for (auto className : class_varsClassMap.at(input)) {
          if (DEBUG_SOL)
            llvm::outs() << "possible className: " << className << "\n";
        }
        auto classVarRef = getOrInsertBuiltInClassVarFunctionX();
        Value v = getOrCreateGlobalStringX(varLoc, varName, varName);
        auto callOp =
            builder.create<LLVM::CallOp>(varLoc, llvmI8PtrTy, classVarRef, v);
        auto value = callOp.getResult(0);
        return value;
      } else if (classesInfoMap.find(input) != classesInfoMap.end()) {
        if (DEBUG_SOL)
          llvm::outs() << "found className: " << input << "\n";
      } else {
        // todo: can be anonymous scope, go to search parent scope
        auto functionName = findSymbolInParentScopes(input);
        if (!functionName.empty()) {
          if (DEBUG_SOL)
            llvm::outs() << "found symbol in parent function: " << functionName
                         << "\n";
          if (false) {
            // alternatively, create a new parameter for the current anon func
          } else {
            auto parentVarRef = getOrInsertBuiltInParentVarFunctionX();
            Value v = getOrCreateGlobalStringX(varLoc, varName, varName);
            auto v0 = getValueFromSymbolTable("this");
            if (v0) {
              auto callOp = builder.create<LLVM::CallOp>(
                  varLoc, llvmI8PtrTy, parentVarRef, ArrayRef<Value>({v0, v}));
              auto value = callOp.getResult(0);
              return value;
            }
          }
        } else {
          // if (DEBUG_SOL) llvm::outs() << "unknown class or library? " <<
          // input << "\n";
        }
      }
    }
    return getOrCreateGlobalStringX(loc(expr.loc()), varName, varName);
  }
  // JEFF: invariant - should never return NULL
  mlir::Value mlirGen(BlockExprAST &block) {
    std::string funcName = SOL_ANON_FUNC_NAME +
                           getCurrentNonAnonFunctionName() + "." +
                           std::to_string(++ANON_FUNC_INDEX);
    auto locInfo = loc(block.loc());
    auto retBlock = getOrCreateGlobalStringX(locInfo, funcName, funcName);
    {
      // call st.model.blockParam for out variables
      std::vector<mlir::Value> blockOperands;
      blockOperands.push_back(retBlock);
      std::vector<std::string> outerParameterVars;
      for (auto o : block.outerVars) {
        // cout << "finding var to outerscope: " << o << endl;
        if (symbolTableStack.back()->count(o)) {
          // cout << "found local var to outerscope: " << o << endl;
          outerParameterVars.push_back(o);
          auto def = (*symbolTableStack.back())[o].second;
          blockOperands.push_back(def);
        } else {
          // likely instance variables
          // cout << "found instance var to outerscope: " << o << endl;
        }
      }
      // cout << "blockparam size: " << blockOperands.size() << endl;
      auto blockRef = getOrInsertBuiltInBlockParamFunctionX(blockOperands);
      [[maybe_unused]] auto tmp = builder.create<LLVM::CallOp>(
          locInfo, llvmI8PtrTy, blockRef, blockOperands);

      // Create a scope in the symbol table to hold variable declarations.
      // SymbolTableScopeT2 var_scope(symbolTable2);
      // make sure block has func
      // assert(block.func != nullptr)
      std::string anonName = funcName;
      // ANON_NAME + std::to_string(ANON_FUNC_INDEX);

      pushNewFunction(funcName);
      if (DEBUG_SOL)
        llvm::outs() << "NEW BlockExprAST ------name: " << funcName
                     << " anonName: " << anonName << "\n";

      if (!block.func) {
        // should not get there in normal
        popLastFunction(funcName);
        return retBlock;
      }
      // TODO: block scope
      // mlir::FuncOp func = mlirGen(*block.func);  // scope symbolTable not
      // reentrant?
      std::vector<VarDeclExprAST> args;
      // always add "this" to block code
      VarType type;
      type.name = "*i8";
      VarDeclExprAST arg0(block.loc(), "this", type);
      args.push_back(arg0);
      // for (auto param : block.getBlockParams()) args.push_back(param);
      for (auto var : outerParameterVars) {
        VarType type;
        type.name = "*i8";
        VarDeclExprAST param(block.loc(), var, type);
        args.push_back(param);
      }

      PrototypeAST protoAST(block.loc(), funcName, args);
      // func->proto = protoAST;

      // Create an MLIR function for the given prototype.
      auto blockFunc = mlirGen(protoAST);
      if (blockFunc) {
        theModule.push_back(blockFunc);
        functionMap.insert({blockFunc.getName(), blockFunc});

      } else {
        if (DEBUG_SOL)
          llvm::outs() << "-----failed to create BlockExprAST ------ name: "
                       << block.getName() << "\n";
        return retBlock;
      }

      auto &entryBlock = *blockFunc.addEntryBlock();
      auto protoArgs = protoAST.getArgs();
      // add parent scope pointer as the first arg
      // for (auto arg : protoArgs) {
      //   if (DEBUG_SOL) llvm::outs() << "block arg: " << arg.getName() <<
      //   "\n";
      // }
      // Declare all the function arguments in the symbol table.
      for (const auto name_value :
           llvm::zip(protoArgs, entryBlock.getArguments())) {
        declareX2(std::get<0>(name_value).getName(), std::get<1>(name_value),
                  funcName);
      }

      // Set the insertion point in the builder to the beginning of the function
      // body, it will be used throughout the codegen to create operations in
      // this function.
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(&entryBlock);
      auto retLoc = locInfo;
      mlir::Value retValue = entryBlock.getArguments().front();
      // temp variables
      for (auto vardecl : block.getLocals()) {
        declareLocalVariable(vardecl, funcName);
      }
      auto astlist = block.func->getBody();
      if (astlist->size() > 0) {
        if (DEBUG_SOL)
          llvm::outs() << "-----mlirGen BlockExprAST adding body------size: "
                       << astlist->size() << "\n";
        mlirGen(*astlist);
      }

      builder.create<mlir::LLVM::ReturnOp>(retLoc,
                                           llvm::makeArrayRef(retValue));

      popLastFunction(funcName);
      return retBlock;
      // if (protoArgs.size() > 0) return retBlock;  // callbacks with
      // parameters
      // auto parentFuncName = getCurrentFunctionName();
      // if (parentFuncName.startswith("st.fork")) return retBlock;  // fork
    }

    // TODO: insert call to this anonymous func, and return its ret
    // parent->this
    auto blockRef = SymbolRefAttr::get(builder.getContext(), funcName);
    auto parentFuncName = getCurrentFunctionName();
    auto parent =
        getOrCreateGlobalStringX(locInfo, parentFuncName.str(), parentFuncName);
    auto callOp =
        builder.create<LLVM::CallOp>(locInfo, llvmI8PtrTy, blockRef, parent);
    auto value = callOp.getResult(0);

    return value;
  }

  mlir::Value mlirGen(AssignExprAST &assign) {
    // visit vhs => v1
    mlir::Value v1 = mlirGen(*assign.getRHS());
    // if (DEBUG_SOL) llvm::outs() << "v1: " << v1 << "\n";
    // visit lhs => v2
    mlir::Value v2 = mlirGen(*assign.getLHS());

    // if (DEBUG_SOL) llvm::outs() << "v2: " << v2 << "\n";

    // create assign expression v2 = v1
    auto assignCallRef =
        getOrInsertAssignFunctionX(SOL_BUILT_IN_MODEL_OPAQUE_ASSIGN);
    builder.create<LLVM::CallOp>(loc(assign.loc()), llvmVoidTy, assignCallRef,
                                 ArrayRef<Value>({v2, v1}));
    // auto value = callOp.getResult(0);
    return v2;
  }
  // Codegen a list of expression, return failure if one of them hit an error.
  // return the last value
  void mlirGen(ExprASTList &exprList) {
    for (const auto &expr : exprList) {
      if (auto *ret = dyn_cast<ReturnExprAST>(expr)) {
        if (DEBUG_SOL)
          llvm::outs() << "-----mlirGen ExprASTList expr------ReturnExprAST"
                       << "\n";
        continue;
      }
      if (auto *print = dyn_cast<PrintExprAST>(expr)) {
        if (DEBUG_SOL)
          llvm::outs() << "-----mlirGen ExprASTList expr------PrintExprAST"
                       << "\n";
        continue;
      }
      if (auto *assign = dyn_cast<AssignExprAST>(expr)) {
        if (DEBUG_SOL)
          llvm::outs() << "-----mlirGen ExprASTList expr------AssignExprAST"
                       << "\n";
        // to be processed later
        mlirGen(*assign);
        continue;
      }
      {
        // if (DEBUG_SOL) llvm::outs() << "-----mlirGen ExprASTList
        // expr------OTHER"
        //              << "\n";
        mlirGen(*expr);
      }
    }
  }

  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto input = lit.getName();
    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen LiteralExprAST expr------" << input << "\n";
    // constant string
    std::string name = LOCAL_VAR_NAME + input;
    Value str = getOrCreateGlobalStringX(loc(lit.loc()), name, input);
    return str; // important
  }
  mlir::Value mlirGen(PrimitiveExprAST &str) {
    auto input = "primitivex";
    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen PrimitiveExprAST expr------" << input
                   << "\n";
    // todo: cast to i*8 to follow type of predefined funcs
    auto inputS = input;
    std::string name = LOCAL_VAR_NAME + inputS;
    Value v = getOrCreateGlobalStringX(loc(str.loc()), name, inputS);
    return v;
  }
  mlir::Value mlirGen(ParserErrorAST &str) {
    auto input = "parser.error";
    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen ParserErrorAST expr------" << input << "\n";
    // todo: cast to i*8 to follow type of predefined funcs
    auto inputS = input;
    std::string name = LOCAL_VAR_NAME + inputS;
    Value v = getOrCreateGlobalStringX(loc(str.loc()), name, inputS);
    return v;
  }

  mlir::Value mlirGen(StringExprAST &str) {
    auto input = str.getName();
    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen StringExprAST expr------" << input << "\n";
    // todo: cast to i*8 to follow type of predefined funcs
    auto inputS = input;
    std::string name = LOCAL_VAR_NAME + inputS;
    Value v = getOrCreateGlobalStringX(loc(str.loc()), name, inputS);
    return v;
  }

  mlir::Value mlirGen(SymbolExprAST &sym) {
    auto input = sym.getSymbol();
    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen SymbolExprAST expr------" << input << "\n";
    // todo: cast to i*8 to follow type of predefined funcs
    auto inputS = input;
    std::string name = LOCAL_VAR_NAME + inputS;
    Value str = getOrCreateGlobalStringX(loc(sym.loc()), name, inputS);
    return str;
  }
  mlir::Value mlirGen(ReservedKeywordExprAST &reserved) {
    auto input = reserved.getName();
    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen ReservedKeywordExprAST expr------" << input
                   << "\n";
    // todo: cast to i*8 to follow type of predefined funcs
    // TODO: handle self

    auto inputS = input;
    std::string name = LOCAL_VAR_NAME + inputS;
    Value str = getOrCreateGlobalStringX(loc(reserved.loc()), name, inputS);
    return str;
  }

  mlir::Value mlirGen(NumberExprAST &num) {
    auto input = num.getValue();
    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen NumberExprAST expr------" << input << "\n";
    // auto v = builder.getIntegerAttr(builder.getIndexType(), input);
    // return builder.create<LLVM::ConstantOp>(
    //     loc(num.loc()), LLVM::LLVMType::getInt64Ty(llvmDialect), v);

    // todo: cast to i*8 to follow type of predefined funcs
    auto inputS = std::to_string(input);
    Value str = getOrCreateGlobalStringX(loc(num.loc()),
                                         LOCAL_VAR_NAME + inputS, inputS);
    return str;
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    if (DEBUG_SOL)
      llvm::outs() << "ExprAST: " << expr.getKind()
                   << " loc: " << loc(expr.loc()) << "\n";

    switch (expr.getKind()) {
    case ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    case ExprAST::Expr_String:
      return mlirGen(cast<StringExprAST>(expr));
    case ExprAST::Expr_Symbol:
      return mlirGen(cast<SymbolExprAST>(expr));
    case ExprAST::Expr_Reserved:
      return mlirGen(cast<ReservedKeywordExprAST>(expr));
    case ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case ExprAST::Expr_Block:
      return mlirGen(cast<BlockExprAST>(expr));
    case ExprAST::Expr_Assign:
      return mlirGen(cast<AssignExprAST>(expr));
    case ExprAST::Expr_FuncCall: {
      return mlirGen(cast<FunctionCallAST>(expr));
    }
    case ExprAST::Expr_DynamicDictionary: {
      auto dictExprAST = cast<DynamicDictionaryExprAST>(expr);
      mlir::Value v = nullptr;
      for (auto dictExpr : dictExprAST.getExpressions())
        v = mlirGen(*dictExpr);
      return v;
    }
    case ExprAST::Expr_DynamicArray: {
      auto arrExprAST = cast<DynamicArrayExprAST>(expr);
      mlir::Value v = nullptr;
      for (auto arrExpr : arrExprAST.getExpressions())
        v = mlirGen(*arrExpr);
      return v;
    }
    case ExprAST::Expr_Primitive: {
      return mlirGen(cast<PrimitiveExprAST>(expr));
    }
    case ExprAST::Expr_Error: {
      return mlirGen(cast<ParserErrorAST>(expr));
    }
      //     auto funcExpr = cast<FunctionAST>(expr);
      //     auto name = funcExpr.getProto()->getName();
      // return SymbolRefAttr::get(name, builder.getContext());
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  void declareFunctionArgType(mlir::Location &locProto, const std::string &name,
                              const std::string &type) {
    auto funcArgRef = getOrInsertBuiltInArgTypeFunctionX();
    Value v1 = getOrCreateGlobalStringX(locProto, name, name);
    Value v2 = getOrCreateGlobalStringX(locProto, type, type);
    [[maybe_unused]] auto callOp = builder.create<LLVM::CallOp>(
        locProto, llvmI8PtrTy, funcArgRef, ArrayRef<Value>({v1, v2}));
    // auto value = callOp.getResult(0);
  }
  void declareLocalVariable(VarDeclExprAST &vardecl, llvm::StringRef funcName) {
    auto varName = vardecl.getName();
    auto varLoc = loc(vardecl.loc());
    // Specific handling for variable declarations, return statement, and
    // print. These can only appear in block list and not in nested
    // expressions.
    if (DEBUG_SOL)
      llvm::outs() << "-----mlirGen FunctionAST ------VarDeclExprAST: "
                   << varName << "\n";
    // i8* st.newTemp("name")

    auto newTempRef = getOrInsertBuiltInNewFunctionX();
    std::string name = LOCAL_VAR_NAME + varName;
    Value v = getOrCreateGlobalStringX(varLoc, name, varName);
    auto callOp =
        builder.create<LLVM::CallOp>(varLoc, llvmI8PtrTy, newTempRef, v);
    auto value = callOp.getResult(0);
    declareX2(vardecl.getName(), value,
              funcName); // ok this assume value is unique?
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  // return type: for main, void
  // for block and others, i8ptr

  mlir::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    llvm::SmallVector<mlir::Type, 4> argTypes;
    // by default, add *i8 to func's arg to represent "this"
    //"this" is either class object or class instance object
    if (proto.getName() != MAIN_FUNC_NAME) {
      // should be already added in parser AST

      if (proto.getArgs().size() == 0) {
        // recover from parser error
        VarType type;
        type.name = "*i8";
        VarDeclExprAST err(proto.loc(), "parser.error", type);
        proto.getArgs().push_back(err);
      }
    } else {
      // main
    }

    for (auto &arg : proto.getArgs()) {
      argTypes.push_back(llvmI8PtrTy);
    }

    auto returnType = llvmVoidTy;
    if (proto.getName() == MAIN_FUNC_NAME) {
      returnType = llvmI64Ty;
    } else // if (!proto.getRet())
    {
      returnType = llvmI8PtrTy;
    }
    auto func_type = builder.getFunctionType(argTypes, returnType);
    return mlir::FuncOp::create(location, proto.getName(), func_type);
  }
  mlir::FuncOp mlirGenTomlConfiguration(sol::ModuleAST &moduleAST) {
    auto funcName = SOL_BUILT_IN_MODEL_CARGO_TOML;
    auto location = mlir::FileLineColLoc::get(
        builder.getStringAttr(moduleAST.path_config), 0, 0);
    auto func_type = builder.getFunctionType(llvmI8PtrTy, llvmI64Ty);
    auto function = mlir::FuncOp::create(location, funcName, func_type);

    auto &entryBlock = *function.addEntryBlock();
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(&entryBlock);

    for (auto &[key, value] : moduleAST.configMap) {
      auto mlocation =
          mlir::FileLineColLoc::get(builder.getStringAttr("Cargo.toml"), 0, 0);
      SmallVector<mlir::Value, 2> operands;
      Value v1 = getOrCreateGlobalStringX(mlocation, key, key);
      Value v2 = getOrCreateGlobalStringX(mlocation, value, value);
      operands.push_back(v1);
      operands.push_back(v2);
      auto calleeRef = getOrInsertBuiltInTOMLFunctionX();
      builder.create<LLVM::CallOp>(mlocation, llvmI8PtrTy, calleeRef, operands);
    }

    mlir::Value value = builder.create<LLVM::ConstantOp>(
        location, llvmI64Ty, builder.getIntegerAttr(builder.getIndexType(), 0));
    builder.create<mlir::LLVM::ReturnOp>(location, llvm::makeArrayRef(value));

    return function;
  }
  mlir::FuncOp mlirGenDeclareIdAddresses(sol::ModuleAST &moduleAST) {
    auto funcName = SOL_BUILT_IN_MODEL_DECLARE_ID_ADDRESS;
    auto location = mlir::FileLineColLoc::get(
        builder.getStringAttr(moduleAST.path_config), 0, 0);
    auto func_type = builder.getFunctionType(llvmI8PtrTy, llvmI64Ty);
    auto function = mlir::FuncOp::create(location, funcName, func_type);

    auto &entryBlock = *function.addEntryBlock();
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(&entryBlock);

    for (auto addr : declareIdAddresses) {
      auto mlocation =
          mlir::FileLineColLoc::get(builder.getStringAttr("lib.rs"), 0, 0);
      SmallVector<mlir::Value, 1> operands;
      Value v1 = getOrCreateGlobalStringX(mlocation, addr, addr);
      operands.push_back(v1);
      auto calleeRef = getOrInsertBuiltInDeclareIdFunctionX();
      builder.create<LLVM::CallOp>(mlocation, llvmI8PtrTy, calleeRef, operands);
    }

    mlir::Value value = builder.create<LLVM::ConstantOp>(
        location, llvmI64Ty, builder.getIntegerAttr(builder.getIndexType(), 0));
    builder.create<mlir::LLVM::ReturnOp>(location, llvm::makeArrayRef(value));

    return function;
  }

  mlir::FuncOp mlirGenClassesMetadata(sol::ModuleAST &moduleAST) {
    auto funcName = SOL_BUILT_IN_CLASS_METADATA + "$" + moduleAST.path;
    if (LOWER_BOUND_ID > 0)
      funcName = funcName + "_" + std::to_string(LOWER_BOUND_ID);

    auto location =
        mlir::FileLineColLoc::get(builder.getStringAttr(moduleAST.path), 0, 0);
    auto func_type = builder.getFunctionType(llvmI8PtrTy, llvmI64Ty);
    auto function = mlir::FuncOp::create(location, funcName, func_type);

    auto &entryBlock = *function.addEntryBlock();
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(&entryBlock);

    auto calleeRef0 = getOrInsertBuiltInClassMetaFunctionX();
    Value v0 = entryBlock.getArguments()[0];
    builder.create<LLVM::CallOp>(location, llvmI64Ty, calleeRef0, v0);

    // all classes
    for (auto &[className, classAST] : classesInfoMap) {
      auto mlocation = mlir::FileLineColLoc::get(
          builder.getStringAttr(classAST->fileName), classAST->line, 0);
      SmallVector<mlir::Value, 2> operands;
      // classAST->getName()
      // classAST->getSuperClassName()
      Value v1 = getOrCreateGlobalStringX(mlocation, classAST->getName(),
                                          classAST->getName());
      Value v2 =
          getOrCreateGlobalStringX(mlocation, classAST->getSuperClassName(),
                                   classAST->getSuperClassName());
      operands.push_back(v1);
      operands.push_back(v2);
      auto calleeRef = getOrInsertSuperClassFunctionX();
      builder.create<LLVM::CallOp>(mlocation, llvmVoidTy, calleeRef, operands);
    }

    mlir::Value value = builder.create<LLVM::ConstantOp>(
        location, llvmI64Ty, builder.getIntegerAttr(builder.getIndexType(), 0));
    builder.create<mlir::LLVM::ReturnOp>(location, llvm::makeArrayRef(value));

    return function;
  }
  /// Emit a new function and add it to the MLIR module.
  mlir::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.

    // SymbolTableScopeT2 var_scope(symbolTable2);
    auto funcName = funcAST.getName();
    pushNewFunction(funcName);
    if (DEBUG_SOL)
      llvm::outs() << "NEW FunctionAST------name: " << funcName << "\n";

    // Create an MLIR function for the given prototype.
    auto function = mlirGen(*funcAST.getProto());
    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the
    // same argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();
    auto protoArgs = funcAST.getProto()->getArgs();
    // Declare all the function arguments in the symbol table.
    for (const auto name_value :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      declareX2(std::get<0>(name_value).getName(), std::get<1>(name_value),
                funcName);
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in
    // this function.
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(&entryBlock);
    auto locProto = loc(funcAST.getProto()->loc()); // loc(return_val->loc()
    // create SOL_BUILT_IN_MODEL_FUNC_ARG sol.model.funcArg
    for (auto arg : protoArgs) {
      if (DEBUG_SOL)
        llvm::outs() << "func arg: " << arg.getName()
                     << " type: " << arg.getType().name << "\n";
      declareFunctionArgType(locProto, arg.getName(), arg.getType().name);
    }

    // temp variables
    for (auto vardecl : funcAST.getLocals()) {
      declareLocalVariable(vardecl, funcName);
    }
    // symbolic execution?? let's do type inference in CR!
    // or let's do type inferece in MLIRGen?
    // if (false)
    { mlirGen(*funcAST.getBody()); }
    // if (false)
    {
      if (funcName != MAIN_FUNC_NAME) {
        mlir::Value retValue = entryBlock.getArguments().front();
        if (funcAST.getProto()->getRet()) {
          // TODO: return the correct value
          // return_val can be complicated
          retValue = mlirGen(*funcAST.return_val);
        }
        builder.create<mlir::LLVM::ReturnOp>(locProto,
                                             llvm::makeArrayRef(retValue));
      } else {
        mlir::Value value = builder.create<LLVM::ConstantOp>(
            locProto, llvmI64Ty,
            builder.getIntegerAttr(builder.getIndexType(), 0));
        builder.create<mlir::LLVM::ReturnOp>(locProto,
                                             llvm::makeArrayRef(value));
      }
    }
    popLastFunction(funcName);
    return function;
  }
};

mlir::OwningOpRef<mlir::ModuleOp> mlirGenFull(mlir::MLIRContext &context,
                                              sol::ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
};

} // namespace sol
