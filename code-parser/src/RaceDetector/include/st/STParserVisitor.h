#ifndef MLIR_ST_PARSER_VISITOR_H_
#define MLIR_ST_PARSER_VISITOR_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ast/AST.h"
#include "RustParserBaseVisitor.h"

using namespace stx;

bool DEBUG_SOL = false;

const std::string ANON_NAME = ".anon.";
std::map<std::string, stx::FunctionAST *> allFunctionMap;
std::map<std::string, uint> functionNamesMap;
std::map<std::string, uint> functionAnonNamesMap;
std::set<std::string> declareIdAddresses;

class STParserVisitor : public RustParserBaseVisitor {
 private:
  std::string fileName;
  std::string fnBaseName;
  std::string curFuncName;
  size_t line_base;
  std::string entryName;

  std::vector<stx::FunctionAST *> functions;

  void addNewFunction(stx::Location &loc, std::string fname,
                      std::vector<stx::VarDeclExprAST> &args,
                      stx::FunctionAST *func) {
    // if a previous function with the same name exists, append $k to this name
    // if (DEBUG_SOL) std::cout << "addNewFunction ->fname: " << fname << std::endl;
    auto k = functionNamesMap[fname];
    if (k > 0) {
      func->function_name = func->function_name + "$" + std::to_string(k);
    }
    if (DEBUG_SOL) {
      std::cout << "addNewFunction ->function_name: " << func->function_name << std::endl;
    }
    functionNamesMap[fname] = k + 1;
    auto protoAST = new PrototypeAST(loc, func->function_name, args);
    func->proto = protoAST;
    functions.push_back(func);
  }

  stx::Location getLoc(antlr4::ParserRuleContext *ctx) {
    size_t line = 0;
    size_t pos = 0;
    if (ctx) {
      line = ctx->getStart()->getLine();
      pos = ctx->getStart()->getCharPositionInLine();
    }
    stx::Location loc({fileName, line + line_base, pos});
    return loc;
  }

  bool existInCurScope(stx::FunctionAST *scope, std::string name) {
    if (scope->temp_vars.size() == 0) return false;
    for (auto it = scope->temp_vars.begin(); it < scope->temp_vars.end();
         ++it) {
      if (it->getName() == name) return true;
    }
    return false;
  }

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

 public:
  STParserVisitor(std::string fileName, std::string funcName, size_t line)
      : fileName(fileName), fnBaseName(funcName), line_base(line) {
    // llvm::outs() << "----visitor filename: " << fileName << "-------"<< "\n";
  }
  stx::SEM sem;

  virtual antlrcpp::Any visitCrate(RustParser::CrateContext *ctx) override {
    if (DEBUG_SOL) std::cout << "visitCrate path: " << fileName << std::endl;
    // TODO: collect all functions and
    visitChildren(ctx);
    if (DEBUG_SOL) std::cout << "visitCrate: end" << std::endl;
    return functions;
  }

  virtual antlrcpp::Any visitItem(RustParser::ItemContext *ctx) override {
    antlrcpp::Any result = nullptr;
    if (DEBUG_SOL)
      std::cout << SPACE << "visitItem begin: " << ctx->getText() << std::endl;
    indentMore();
    result = visitItemX(fnBaseName, ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitItem end " << std::endl;

    return result;
  }

  antlrcpp::Any visitItemX(std::string fnBaseName,
                           RustParser::ItemContext *ctx) {
    antlrcpp::Any result = nullptr;
    if (DEBUG_SOL)
      std::cout << SPACE << "visitItemX begin: " << ctx->getText() << std::endl;
    indentMore();

    stx::FunctionAST *funcAST_single = nullptr;
    if (ctx->visItem()) {
      for (auto outerAttribute : ctx->outerAttribute()) {
        auto attr = outerAttribute->getText();
        if (attr.find("#[test") != std::string::npos) {
          // skip #[test
          indentLess();
          if (DEBUG_SOL) std::cout << SPACE << "visitItemX end test code" << std::endl;
          return result;
        }
      }

      if (ctx->visItem()->function_()) {
        funcAST_single =
            visitFunction_X(fnBaseName, ctx->visItem()->function_());
        result = funcAST_single;
      } else if (ctx->visItem()->implementation()) {
        // impl State
        result =
            visitImplementation_X(fnBaseName, ctx->visItem()->implementation());
      } else if (ctx->visItem()->trait_()) {
        // pub trait RepayContext<'info>
        result = visitTrait_X(fnBaseName, ctx->visItem()->trait_());
      }
    }

    bool itemExploredInAnchorProgram = false;
    if (ctx->outerAttribute().size() > 0) {
      for (auto outerAttribute : ctx->outerAttribute()) {
        auto attr = outerAttribute->getText();
        if (attr == "#[program]") {
          if (DEBUG_SOL) std::cout << SPACE << "Anchor: " << attr << std::endl;
          if (ctx->visItem()) {
            itemExploredInAnchorProgram = true;
            if (auto moduleCtx = ctx->visItem()->module()) {
              auto loc = getLoc(ctx);
              auto name = moduleCtx->identifier()->getText();
              stx::FunctionAST *func = new stx::FunctionAST(loc);
              auto fnName = "model.anchor.program." + name;
              func->function_name = "sol." + fnName;
              std::vector<stx::ExprAST *> callInsts;

              for (auto itemCtx : moduleCtx->item()) {
                if (auto visItemCtx = itemCtx->visItem()) {
                  if (visItemCtx->function_()) {
                    auto funcAST =
                        visitFunction_X(fnBaseName, visItemCtx->function_());

                    // TODO: #[access_control(ctx.accounts.validate())]
                    // #[access_control(admin(&ctx.accounts.state,
                    // &ctx.accounts.admin))]
                    // #[access_control(DepositVault::accounts(&ctx.accounts,
                    // &deposit_args, ctx.program_id))] insert
                    // sol.access_control() to funcAST->body
                    for (auto attrCtx : itemCtx->outerAttribute()) {
                      auto attrCtx_str = attrCtx->attr()->getText();
                      // std::cout << SPACE << "  attrCtx: " << attrCtx_str << std::endl;
                      if (attrCtx->attr()->simplePath()->getText() ==
                          "access_control") {
                        stx::FunctionCallAST *fcall =
                            new stx::FunctionCallAST(getLoc(attrCtx));
                        fcall->callee = "model.access_control";
                        // ctx.accounts.validate
                        auto calleeName = attrCtx_str.substr(
                            15, attrCtx_str.length() - 15 - 1);
                        if (DEBUG_SOL)
                          std::cout << SPACE
                               << "  access_control callee: " << calleeName
                               << std::endl;
                        std::vector<stx::ExprAST *> args;
                        auto arg = new LiteralExprAST(getLoc(attrCtx->attr()),
                                                      calleeName);
                        args.push_back(arg);
                        fcall->args = args;

                        funcAST->body.insert(funcAST->body.begin(), fcall);
                      }
                    }
                    {
                      FunctionCallAST *fcall =
                          new FunctionCallAST(getLoc(visItemCtx));
                      //@sol.lib.log_message => @sol.log_message
                      auto calleeName = funcAST->getName();
                      if (auto found = calleeName.find("::"))
                        calleeName = calleeName.substr(found + 2);
                      fcall->callee =
                          calleeName;  // callee already added .count
                      if (DEBUG_SOL)
                        std::cout << SPACE << "  fcall->callee: " << fcall->callee
                             << std::endl;
                      std::vector<stx::ExprAST *> args;
                      for (auto declareExpr : funcAST->getProto()->getArgs()) {
                        auto literal = declareExpr.getName() + ":" +
                                       declareExpr.getType().name;
                        auto arg =
                            new LiteralExprAST(getLoc(visItemCtx), literal);
                        args.push_back(arg);
                      }
                      fcall->args = args;
                      callInsts.push_back(fcall);
                    }
                  }
                }
              }

              func->body = callInsts;
              std::vector<stx::VarDeclExprAST> args;
              addNewFunction(loc, func->function_name, args, func);
              // create main to call func
              auto main = allFunctionMap["main"];
              if (!main) {
                stx::FunctionAST *main = new FunctionAST(loc);
                main->function_name = "main";
                std::vector<stx::ExprAST *> res;
                auto program_id = new LiteralExprAST(getLoc(ctx), "program_id");
                // auto accounts = new LiteralExprAST(getLoc(ctx), "accounts");
                // auto instruction_data =
                //     new LiteralExprAST(getLoc(ctx), "instruction_data");
                res.push_back(program_id);
                // if (false)
                {
                  FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
                  fcall->callee = fnName;
                  if (DEBUG_SOL)
                    std::cout << SPACE << "  fcall->callee: " << fcall->callee
                         << std::endl;
                  std::vector<stx::ExprAST *> args;
                  args.push_back(program_id);
                  fcall->args = args;
                  res.push_back(fcall);
                }
                main->body = res;
                std::vector<stx::VarDeclExprAST> args;
                auto protoAST =
                    new PrototypeAST(loc, main->function_name, args);
                main->proto = protoAST;
                functions.push_back(main);
                allFunctionMap["main"] = main;
              } else {
                FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
                fcall->callee = fnName;
                if (DEBUG_SOL)
                  std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;
                std::vector<stx::ExprAST *> args;
                auto program_id = new LiteralExprAST(getLoc(ctx), "program_id");
                args.push_back(program_id);
                fcall->args = args;
                main->body.push_back(fcall);
              }
            }
          }

        } else if (attr == "#[account]" ||
                   attr.find("#[derive(Accounts") != std::string::npos) {
          //         if (s.rfind("titi", 0) == 0) { // pos=0 limits the search
          //         to the prefix
          //   // s starts with prefix
          // }
          if (DEBUG_SOL) std::cout << SPACE << "Anchor: " << attr << std::endl;
          if (ctx->visItem()) {
            if (ctx->visItem()->struct_() &&
                ctx->visItem()->struct_()->structStruct()) {
              result = visitStructStructX(
                  ctx->visItem()->struct_()->structStruct(), true);
            }
          }
        } else if (attr.find("#[derive(") != std::string::npos) {
          //#[derive(BorshSerialize, BorshDeserialize)]
          if (DEBUG_SOL) std::cout << SPACE << "non-Anchor struct: " << attr << std::endl;
          if (ctx->visItem()) {
            if (ctx->visItem()->struct_() &&
                ctx->visItem()->struct_()->structStruct()) {
              result = visitStructStructX(
                  ctx->visItem()->struct_()->structStruct(), false);
            }
          }
        } else if (attr.find("#[inline(") != std::string::npos ||
                   attr.find("#[allow(") != std::string::npos ||
                   attr.find("#[cfg") != std::string::npos ||
                   attr.find("#[proc_macro_attribute]") != std::string::npos) {
          if (DEBUG_SOL) std::cout << SPACE << "more attr: " << attr << std::endl;
        } else if (attr.find("#[access_control(") != std::string::npos) {
          //#[access_control(is_not_emergency(&ctx.accounts.nirv_center,
          //&ctx.accounts.signer))]
          if (DEBUG_SOL) std::cout << SPACE << "access_control: " << attr << std::endl;

          if (funcAST_single) {
            {
              FunctionCallAST *fcall =
                  new FunctionCallAST(getLoc(outerAttribute));
              fcall->callee = "model.access_control";
              // ctx.accounts.validate
              auto calleeName = attr.substr(17, attr.length() - 17 - 2);
              if (DEBUG_SOL)
                std::cout << SPACE << "  access_control callee: " << calleeName
                     << std::endl;
              std::vector<stx::ExprAST *> args;
              auto arg = new LiteralExprAST(getLoc(outerAttribute->attr()),
                                            calleeName);
              args.push_back(arg);
              fcall->args = args;

              funcAST_single->body.insert(funcAST_single->body.begin(), fcall);
            }
          }
        } else {
          if (DEBUG_SOL) std::cout << SPACE << "other attr: " << attr << std::endl;
          // for test
          // visitChildren(ctx);
        }
      }
    } else {
      // macroItem
      // if (DEBUG_SOL) std::cout << SPACE << "others: " << std::endl;
    }

    if (ctx->macroItem()) {
      if (auto macroInvocationSemi = ctx->macroItem()->macroInvocationSemi()) {
        // entrypoint!(process_instruction);
        if (auto simplePath = macroInvocationSemi->simplePath()) {
          auto macroName = simplePath->getText();
          if (DEBUG_SOL)
            std::cout << SPACE << "macroInvocationSemi simplePath: " << macroName
                 << std::endl;

          if (macroName == "entrypoint") {
            for (auto tokenTreeCtx : macroInvocationSemi->tokenTree()) {
              entryName = tokenTreeCtx->getText();
              if (DEBUG_SOL)
                std::cout << "entrypoint function: " << entryName << std::endl;
              // create a main function that calls "entryName"
              {
                auto loc = getLoc(ctx);
                stx::FunctionAST *func = new stx::FunctionAST(loc);
                std::string fnName = "main";
                func->function_name = fnName;

                std::vector<stx::ExprAST *> res;
                auto program_id = new LiteralExprAST(getLoc(ctx), "program_id");
                auto accounts = new LiteralExprAST(getLoc(ctx), "accounts");
                auto instruction_data =
                    new LiteralExprAST(getLoc(ctx), "instruction_data");
                res.push_back(program_id);
                res.push_back(accounts);
                res.push_back(instruction_data);
                {
                  FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
                  fcall->callee = entryName;
                  if (DEBUG_SOL)
                    std::cout << SPACE << "  fcall->callee: " << fcall->callee
                         << std::endl;
                  std::vector<stx::ExprAST *> args;
                  args.push_back(program_id);
                  args.push_back(accounts);
                  args.push_back(instruction_data);
                  fcall->args = args;
                  // ok - function names can collide - we need to append
                  // parameter count
                  fcall->callee =
                      fcall->callee + "." + std::to_string(args.size());
                  res.push_back(fcall);
                }

                func->body = res;
                std::vector<stx::VarDeclExprAST> args;
                auto protoAST =
                    new PrototypeAST(loc, func->function_name, args);
                func->proto = protoAST;
                functions.push_back(func);
              }
            }
          } else if (macroName == "solitaire") {
            // solitaire! {
            //     Initialize(InitializeData)                  => initialize,
            // }
            // TODO: create call sol.initialize()
            // pub fn initialize(
            //     ctx: &ExecutionContext,
            //     accs: &mut Initialize,
            //     data: InitializeData,
            // special treatment:

            auto loc = getLoc(ctx);
            stx::FunctionAST *func = new stx::FunctionAST(loc);
            std::string fnName = "main";
            func->function_name = fnName;
            std::vector<stx::ExprAST *> res, res2;
            std::string para0 = "ExecutionContext";
            std::string para1, para2, fname;
            for (auto tokenTreeCtx : macroInvocationSemi->tokenTree()) {
              auto tokenTree = tokenTreeCtx->getText();
              // if (DEBUG_SOL) std::cout << "solitaire tokenTree: " << tokenTree <<
              // std::endl;

              if (tokenTree.find("=>") != std::string::npos) {
                auto found = tokenTree.find(",");
                fname = tokenTree.substr(2, found - 2);
                // if (DEBUG_SOL) std::cout << "fname: " << fname << " para1: " <<
                // para1
                //      << " para2: " << para2 << std::endl;

                auto arg0 = new LiteralExprAST(getLoc(tokenTreeCtx), para0);
                auto arg1 = new LiteralExprAST(getLoc(tokenTreeCtx), para1);
                auto arg2 = new LiteralExprAST(getLoc(tokenTreeCtx), para2);
                {
                  FunctionCallAST *fcall =
                      new FunctionCallAST(getLoc(tokenTreeCtx));
                  fcall->callee = fname;
                  // if (DEBUG_SOL) std::cout << SPACE << "  fcall->callee: " <<
                  // fcall->callee << std::endl;
                  std::vector<stx::ExprAST *> args;
                  args.push_back(arg0);
                  args.push_back(arg1);
                  args.push_back(arg2);
                  fcall->args = args;
                  fcall->callee =
                      fcall->callee + "." + std::to_string(args.size());
                  res.push_back(fcall);
                }
                para1 = tokenTree.substr(found + 1);
              } else if (tokenTree.front() == '(') {
                para2 = tokenTree.substr(1, tokenTree.length() - 2);
              } else {
                para1 = tokenTree;
              }
            }
            // for analyzer to find threads, add sol.match

            FunctionCallAST *fmatch = new FunctionCallAST(loc);
            fmatch->callee = "match";
            // if (DEBUG_SOL) std::cout << SPACE << "  fmatch->callee: " <<
            // fmatch->callee << std::endl;
            fmatch->args = res;
            fmatch->callee = fmatch->callee + "." + std::to_string(res.size());
            res2.push_back(fmatch);

            func->body = res2;
            std::vector<stx::VarDeclExprAST> args;
            auto protoAST = new PrototypeAST(loc, func->function_name, args);
            func->proto = protoAST;
            functions.push_back(func);
          } else if (macroName.find("declare_id") != std::string::npos) {
            // solana_program::declare_id
            bool isProduction = true;
            for (auto outerAttribute : ctx->outerAttribute()) {
              auto attr = outerAttribute->getText();
              if ((attr.find("pro") != std::string::npos ||
                   attr.find("stable") != std::string::npos) &&
                  attr.find("not") == std::string::npos) {
                break;
              }
              if (attr.find("dev") != std::string::npos ||
                  attr.find("test") != std::string::npos ||
                  attr.find("local") != std::string::npos) {
                isProduction = false;
                break;
              }
            }
            // for (auto tokenTreeCtx : macroInvocationSemi->tokenTree()) {
            //   auto tokenTree = tokenTreeCtx->getText();
            //   if (DEBUG_SOL)
            //     std::cout << "solana_program::declare_id tokenTree: " << tokenTree
            //          << std::endl;
            // }
            if (isProduction) {
              if (macroInvocationSemi->tokenTree().size() == 1) {
                auto address = macroInvocationSemi->tokenTree()[0]->getText();
                if (address.length() > 16 &&
                    address.find(".") == std::string::npos) {
                  if (DEBUG_SOL)
                    std::cout << "solana_program::declare_id : " << address << std::endl;
                  declareIdAddresses.insert(
                      address.substr(1, address.length() - 2));
                }
              }
            }
          }
        }
      }
    }
    if (!itemExploredInAnchorProgram) {
      if (ctx->visItem()) {
        auto module = ctx->visItem()->module();
        if (module) {
          auto identifier = module->identifier()->getText();
          // special use identifier + "::" + fnBaseName
          for (auto item : module->item())
            result = visitItemX(identifier, item);
        }
      }
    }

    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitItemX end " << std::endl;

    return result;
  }

  virtual antlrcpp::Any visitStruct_(RustParser::Struct_Context *ctx) override {
    // if (DEBUG_SOL) std::cout << "visitStruct_ begin: " << ctx->getText() << std::endl;
    auto result = visitChildren(ctx);
    // if (DEBUG_SOL) std::cout << "visitStruct_ end " << std::endl;
    return result;
  }

  stx::FunctionAST *visitStructStructX(RustParser::StructStructContext *ctx,
                                       bool isAnchor) {
    // TODO: create a model function for each struct
    auto name = ctx->identifier()->getText();
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStructStructX begin: identifier: " << name << std::endl;

    indentMore();
    auto loc = getLoc(ctx);
    stx::FunctionAST *func = new stx::FunctionAST(loc);
    auto fnName = "sol.model.struct." + name;
    if (isAnchor) fnName = "sol.model.struct.anchor." + name;

    func->function_name = fnName;
    auto fieldCallFunc = "model.struct.field";
    auto constraintCallFunc = "model.struct.constraint";
    std::vector<stx::ExprAST *> callInsts;
    if (ctx->structFields())
      for (auto fieldCtx : ctx->structFields()->structField()) {
        if (fieldCtx->outerAttribute().size() == 1) {
          auto outerAttribute = fieldCtx->outerAttribute().front();
          auto attr = outerAttribute->attr();
          auto attrText = attr->getText();
          if (DEBUG_SOL)
            std::cout << SPACE << "structField attr: " << attrText << std::endl;
          // account(has_one = authority)
          auto found_account = attrText.find("account");
          if (found_account != std::string::npos) {
            FunctionCallAST *fcall = new FunctionCallAST(getLoc(attr));
            fcall->callee = constraintCallFunc;
            if (DEBUG_SOL)
              std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;
            auto cons = attrText.substr(8);
            cons.pop_back();
            auto arg0 = new LiteralExprAST(getLoc(attr), cons);
            std::vector<stx::ExprAST *> args;
            args.push_back(arg0);
            fcall->args = args;
            callInsts.push_back(fcall);
          }
        }
        auto field = fieldCtx->identifier()->getText();
        auto type = fieldCtx->type_()->getText();
        if (DEBUG_SOL)
          std::cout << SPACE << "structField identifier: " << field
               << " type: " << type << std::endl;

        FunctionCallAST *fcall =
            new FunctionCallAST(getLoc(fieldCtx->identifier()));
        fcall->callee = fieldCallFunc;
        if (DEBUG_SOL)
          std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;
        auto arg0 = new LiteralExprAST(getLoc(fieldCtx), field);
        auto arg1 = new LiteralExprAST(getLoc(fieldCtx), type);
        std::vector<stx::ExprAST *> args;
        args.push_back(arg0);
        args.push_back(arg1);
        fcall->args = args;
        callInsts.push_back(fcall);
      }

    func->body = callInsts;
    std::vector<stx::VarDeclExprAST> args;
    addNewFunction(loc, func->function_name, args, func);

    // auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStructStructX end " << std::endl;
    return func;
  }

  virtual antlrcpp::Any visitTupleStruct(
      RustParser::TupleStructContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTupleStruct begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTupleStruct end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitModule(RustParser::ModuleContext *ctx) override {
    //(module mod (identifier ddca_operating_account)

    if (DEBUG_SOL)
      std::cout << SPACE << "visitModule begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitModule end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitVisItem(RustParser::VisItemContext *ctx) override {
    // if (DEBUG_SOL) std::cout << "visitVisItem begin: " << ctx->getText() << std::endl;
    auto result = visitChildren(ctx->module());
    // if (DEBUG_SOL) std::cout << "visitVisItem end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitUseDeclaration(
      RustParser::UseDeclarationContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitUseDeclaration begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitUseDeclaration end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitMacroInvocationSemi(
      RustParser::MacroInvocationSemiContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitMacroInvocationSemi begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitMacroInvocationSemi end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitConstantItem(
      RustParser::ConstantItemContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitConstantItem begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitConstantItem end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitOuterAttribute(
      RustParser::OuterAttributeContext *ctx) override {
    //(outerAttribute # [ (attr (simplePath (simplePathSegment (identifier
    // program)))) ])
    if (DEBUG_SOL)
      std::cout << SPACE << "visitOuterAttribute begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitOuterAttribute end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitStaticItem(
      RustParser::StaticItemContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStaticItem begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStaticItem end " << std::endl;
    return result;
  }

  stx::FunctionAST *visitTrait_X(std::string fnBaseName_tmp,
                                 RustParser::Trait_Context *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTrait_X begin: " << ctx->getText() << std::endl;
    indentMore();
    stx::FunctionAST *result = nullptr;
    std::string typeName = ctx->identifier()->getText();
    std::vector<RustParser::AssociatedItemContext *> associatedItems =
        ctx->associatedItem();
    ;

    auto found = typeName.find("<");
    if (found != std::string::npos) typeName = typeName.substr(0, found);
    fnBaseName = fnBaseName_tmp + "::" + typeName;
    for (auto associateItemCtx : associatedItems) {
      if (associateItemCtx->function_()) {
        result = visitFunction_X(fnBaseName, associateItemCtx->function_());
      }
    }
    fnBaseName = fnBaseName_tmp;

    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTrait_X end " << std::endl;
    return result;
  }

  stx::FunctionAST *visitImplementation_X(std::string fnBaseName_tmp,
                                          RustParser::ImplementationContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitImplementation_X begin: " << ctx->getText()
           << std::endl;
    indentMore();
    stx::FunctionAST *result = nullptr;
    std::string typeName;
    std::vector<RustParser::AssociatedItemContext *> associatedItems;
    if (ctx->inherentImpl()) {
      // impl type name
      typeName = ctx->inherentImpl()->type_()->getText();
      associatedItems = ctx->inherentImpl()->associatedItem();
    } else if (ctx->traitImpl()) {
      typeName = ctx->traitImpl()->type_()->getText();
      associatedItems = ctx->traitImpl()->associatedItem();
    }
    if (!typeName.empty()) {
      auto found = typeName.find("<");
      if (found != std::string::npos) typeName = typeName.substr(0, found);
      if (typeName.rfind("crate::", 0) == 0) {
        typeName = typeName.substr(7);
      }

      fnBaseName = fnBaseName_tmp + "::" + typeName;
      for (auto associateItemCtx : associatedItems) {
        if (associateItemCtx->function_()) {
          result = visitFunction_X(fnBaseName, associateItemCtx->function_());
        }
      }
      fnBaseName = fnBaseName_tmp;
    }
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitImplementation_X end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitInherentImpl(
      RustParser::InherentImplContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitInherentImpl begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitInherentImpl end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitTraitImpl(
      RustParser::TraitImplContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTraitImpl begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTraitImpl end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitGenericParams(
      RustParser::GenericParamsContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitGenericParams begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitGenericParams end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitAssociatedItem(
      RustParser::AssociatedItemContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitAssociatedItem begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitAssociatedItem end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitFunction_(
      RustParser::Function_Context *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitFunction_ begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitFunction_ end " << std::endl;
    return result;
  }
  stx::FunctionAST *visitFunction_X(std::string fnBaseName,
                               RustParser::Function_Context *ctx) {
    curFuncName = ctx->identifier()->getText();
    auto fnName = fnBaseName + "::" + curFuncName;
    auto loc = getLoc(ctx);
    stx::FunctionAST *func = new stx::FunctionAST(loc);
    func->function_name = fnName;

    std::vector<stx::VarDeclExprAST> args;

    if (DEBUG_SOL) std::cout << SPACE << "visitFunction_X: " << fnName << std::endl;
    indentMore();

    if (ctx->functionParameters())
      args = visitFunctionParametersX(ctx->functionParameters());
    if (ctx->functionReturnType())
      visitFunctionReturnType(ctx->functionReturnType());

    // result = visitStatements(ctx->blockExpression()->statements());
    std::vector<stx::ExprAST *> res;
    if (ctx->blockExpression())
      res = visitStatementsX(ctx->blockExpression()->statements());
    func->body = res;
    // ok - function names can collide - we need to append parameter count
    func->function_name =
        func->function_name + "." + std::to_string(args.size());
    addNewFunction(loc, func->function_name, args, func);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitFunction_X end " << std::endl;
    return func;
  }

  virtual antlrcpp::Any visitFunctionReturnType(
      RustParser::FunctionReturnTypeContext *ctx) override {
    antlrcpp::Any result = nullptr;
    if (DEBUG_SOL)
      std::cout << SPACE << "visitFunctionReturnType begin: " << ctx->getText()
           << std::endl;
    // indentMore();
    // auto result = visitChildren(ctx);
    // indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitFunctionReturnType end " << std::endl;
    return result;
  }
  std::vector<stx::VarDeclExprAST> visitFunctionParametersX(
      RustParser::FunctionParametersContext *ctx) {
    std::vector<stx::VarDeclExprAST> args;
    auto loc = getLoc(ctx);

    antlrcpp::Any result = nullptr;
    if (DEBUG_SOL)
      std::cout << SPACE << "visitFunctionParameters begin: " << ctx->getText()
           << std::endl;
    indentMore();
    if (ctx->selfParam()) {
      // for &self
      auto paramName = ctx->selfParam()->getText();
      VarType type;
      type.name = paramName;
      auto arg = new stx::VarDeclExprAST(loc, paramName, type);
      if (DEBUG_SOL) std::cout << SPACE << "param: " << paramName << std::endl;
      args.push_back(*arg);
    }
    for (auto param : ctx->functionParam()) {
      if (DEBUG_SOL) std::cout << SPACE << "param: " << param->getText() << std::endl;
      if (param->functionParamPattern()) {
        auto paramName = param->functionParamPattern()->pattern()->getText();
        auto patternCtx = param->functionParamPattern()->pattern();
        if (auto patternCtx1 =
                patternCtx->patternNoTopAlt()[0]->patternWithoutRange()) {
          if (patternCtx1->identifierPattern())
            paramName =
                patternCtx1->identifierPattern()->identifier()->getText();
        }
        auto typeName = param->functionParamPattern()->type_()->getText();
        VarType type;
        type.name = typeName;
        auto arg = new stx::VarDeclExprAST(loc, paramName, type);
        // if (DEBUG_SOL) std::cout << SPACE << "arg paramName: " << paramName << "
        // type: " << typeName << std::endl;
        args.push_back(*arg);
      }
    }
    // auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitFunctionParameters end " << std::endl;
    return args;
  }
  virtual antlrcpp::Any visitFunctionQualifiers(
      RustParser::FunctionQualifiersContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitFunctionQualifiers begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitFunctionQualifiers end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitIdentifier(
      RustParser::IdentifierContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitIdentifier begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitIdentifier end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitBlockExpression(
      RustParser::BlockExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitBlockExpression begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitBlockExpression end " << std::endl;
    return result;
  }

  stx::ExprAST *visitBlockExpressionX(RustParser::BlockExpressionContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitBlockExpressionX begin: " << ctx->getText()
           << std::endl;
    indentMore();
    stx::ExprAST *res = nullptr;

    if (ctx->statements()) {
      if (auto lastExpr = ctx->statements()->expression()) {
        res = visitExpressionX(lastExpr);
      } else {
        auto blockExprStatements = ctx->statements()->statement();
        if (blockExprStatements.size() > 0) {
          auto lastStmt =
              blockExprStatements.back();  // blockExprStatements.size()-1
          res = visitStatementX(lastStmt);
        }
      }
      if (res) res = visitCustomizedIfBlockExpressionX(ctx, res, true);
    }
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitBlockExpressionX end " << std::endl;
    return res;
  }

  std::vector<stx::ExprAST *> visitStatementsX(RustParser::StatementsContext *ctx) {
    antlrcpp::Any result = nullptr;
    std::vector<stx::ExprAST *> ret;
    if (!ctx) return ret;  // empty block
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStatementsX begin: " << ctx->getText() << std::endl;
    indentMore();

    for (auto stmt : ctx->statement()) {
      auto stmtExpr = visitStatementX(stmt);
      if (stmtExpr) ret.push_back(stmtExpr);
    }
    if (ctx->expression()) {
      if (DEBUG_SOL)
        std::cout << SPACE << "expression: " << ctx->expression()->getText() << std::endl;
      // if (ctx->statement().empty())
      {
        auto expr = visitExpressionX(ctx->expression());
        if (expr) ret.push_back(expr);
      }
      // result = visitChildren(ctx->expression());
    }
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStatementsX end " << std::endl;
    return ret;
    // return result;
  }

  stx::ExprAST *visitStatementX(RustParser::StatementContext *ctx) {
    antlrcpp::Any result = nullptr;
    stx::ExprAST *expr = nullptr;

    if (DEBUG_SOL)
      std::cout << SPACE << "visitStatementX begin: " << ctx->getText() << std::endl;
    indentMore();
    // TODO handle CompoundAssignmentExpression
    // ifamount>0{letamountw=1000/amount;}**wallet_info.lamports.borrow_mut()-=amount

    if (ctx->expressionStatement()) {
      auto exprStmt = ctx->expressionStatement();
      if (auto blockExpr = exprStmt->expressionWithBlock()) {
        // TODO: ifExpression ifLetExpression matchExpression
        auto result = visitChildren(blockExpr);
        expr = visitExpressionWithBlockX(blockExpr);
        // if (blockExpr->ifExpression()) {
        //   expr = visitIfExpressionX(blockExpr->ifExpression());
        // } else if (blockExpr->matchExpression()) {
        //   expr = visitMatchExpressionX(blockExpr->matchExpression());
        // }
      } else if (auto expression = exprStmt->expression()) {
        auto result = visitChildren(exprStmt);

        expr = visitExpressionX(expression);
      }
    } else if (ctx->item()) {
      // assert!(authority_info.is_signer);
      // assert_eq!(wallet.authority,*authority_info.key);

      if (auto macroCtx = ctx->item()->macroItem())
        if (auto macroInvokeCtx = macroCtx->macroInvocationSemi()) {
          expr = visitMacroInvocationSemiX(macroInvokeCtx);
        }

    } else if (ctx->letStatement()) {
      // TODO: let account = next_account_info(accounts_iter)?;
      expr = visitLetStatementX(ctx->letStatement());

    } else if (ctx->macroInvocationSemi()) {
      // TODO: macro, e.g.,assert_eq!(g.floor(),3.0);
      auto macroName = ctx->macroInvocationSemi()->simplePath()->getText();
      if (DEBUG_SOL) std::cout << SPACE << "macroName: " << macroName << std::endl;
    }
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStatementX end " << std::endl;
    return expr;
    // return result;
  }

  stx::ExprAST *visitMacroInvocationSemiX(
      RustParser::MacroInvocationSemiContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitMacroInvocationSemiX: " << ctx->getText() << std::endl;

    stx::ExprAST *expr = nullptr;
    {
      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
      fcall->callee = ctx->simplePath()->getText();
      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;
      std::vector<stx::ExprAST *> args;
      std::vector<std::string> paraNames;
      std::string paraName = "";
      // require!(
      //       merkle_proof::verify(proof, distributor.root, node.0),
      //       InvalidProof
      //   );
      for (auto tokenCtx : ctx->tokenTree()) {
        // for (auto paramCtx : tokenCtx->tokenTreeToken())
        {  // concatenate until startswith ','
          auto varName = tokenCtx->getText();
          if (DEBUG_SOL) std::cout << SPACE << "  varName: " << varName << std::endl;
          if (varName.front() == ',') {
            // add paraName
            paraNames.push_back(paraName);
            paraName = varName.substr(1);
          } else if (varName.front() == '(') {
            // add func call
            paraName = paraName + varName;
          } else {
            auto found = varName.find(',');
            if (found != std::string::npos) {
              paraName = paraName + varName.substr(0, found);
              paraNames.push_back(paraName);
              paraName = varName.substr(found + 1);
            } else
              paraName = paraName + varName;
          }
        }
      }
      if (!paraName.empty()) paraNames.push_back(paraName);
      for (auto varName : paraNames) {
        auto *var = new VariableExprAST(getLoc(ctx), varName);
        args.push_back(var);
        if (DEBUG_SOL) std::cout << SPACE << "  fcall->args: " << varName << std::endl;
      }

      // ok - function names can collide - we need to append parameter count
      // macro !
      fcall->callee = fcall->callee + ".!" + std::to_string(args.size());

      // for assert, drop the second one
      // if (fcall->callee == "assert" || fcall->callee == "msg" ||
      //     fcall->callee == "info") {
      //   while (args.size() > 1) args.pop_back();
      // }

      fcall->args = args;
      return fcall;
    }
    if (DEBUG_SOL) std::cout << SPACE << "visitMacroInvocationSemiX end " << std::endl;
    return expr;
  }

  stx::ExprAST *visitLoopExpressionX(RustParser::LoopExpressionContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitLoopExpression begin: " << ctx->getText() << std::endl;
    indentMore();
    stx::ExprAST *res = nullptr;
    // auto result = visitChildren(ctx);

    FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
    std::vector<stx::ExprAST *> args;
    std::string loopCallFunc = "model.loop";
    RustParser::BlockExpressionContext *blockExpr = nullptr;
    if (ctx->iteratorLoopExpression()) {
      loopCallFunc = "model.loop.for";
      blockExpr = ctx->iteratorLoopExpression()->blockExpression();
    } else if (ctx->infiniteLoopExpression()) {
      loopCallFunc = "model.loop.infinite";
      blockExpr = ctx->infiniteLoopExpression()->blockExpression();
    } else if (ctx->predicateLoopExpression()) {
      loopCallFunc = "model.loop.while";
      blockExpr = ctx->predicateLoopExpression()->blockExpression();
    } else if (ctx->predicatePatternLoopExpression()) {
      loopCallFunc = "model.loop.while";
      blockExpr = ctx->predicatePatternLoopExpression()->blockExpression();
    }

    if (blockExpr) {
      auto index1 = functions.size();
      [[maybe_unused]] auto arg = visitBlockExpressionX(blockExpr);
      auto index2 = functions.size();
      if (index2 > index1) {
        auto funcName = functions[index2 - 1]->getName();
        // std::cout << SPACE << "model.loop funcName: " << funcName << std::endl;

        auto literal = new LiteralExprAST(getLoc(blockExpr), funcName);
        args.push_back(literal);
      }
    }
    fcall->args = args;
    fcall->callee = loopCallFunc + "." + std::to_string(args.size());
    res = fcall;
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitLoopExpression end " << std::endl;
    return res;
  }

  stx::ExprAST *visitExpressionWithBlockX(
      RustParser::ExpressionWithBlockContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitExpressionWithBlockX: " << ctx->getText() << std::endl;
    indentMore();
    stx::ExprAST *res = nullptr;
    if (ctx->ifExpression()) {
      res = visitIfExpressionX(ctx->ifExpression());
    } else if (ctx->ifLetExpression()) {
      // TODO: handle visitIfLetExpressionX
      res = visitIfLetExpressionX(ctx->ifLetExpression());
    } else if (ctx->matchExpression()) {
      res = visitMatchExpressionX(ctx->matchExpression());
    } else if (ctx->loopExpression()) {
      res = visitLoopExpressionX(ctx->loopExpression());
    } else if (ctx->blockExpression()) {
      res = visitBlockExpressionX(ctx->blockExpression());
    }
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitExpressionWithBlockX end" << std::endl;
    return res;
  }

  stx::ExprAST *visitExpressionX(RustParser::ExpressionContext *ctx) {
    stx::ExprAST *expr = nullptr;
    if (DEBUG_SOL)
      std::cout << SPACE << "visitExpressionX: " << ctx->getText() << std::endl;

    while (auto expressionCtx =
               dynamic_cast<RustParser::ErrorPropagationExpressionContext *>(
                   ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  stripping ErrorPropagationExpressionContext: "
             << ctx->getText() << std::endl;
      ctx = expressionCtx->expression();
    }

    if (auto callExpressionCtx =
            dynamic_cast<RustParser::CallExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  callExpressionCtx: " << ctx->getText() << std::endl;
      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
      // expr
      // callParams
      fcall->callee = callExpressionCtx->expression()->getText();
      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

      std::vector<stx::ExprAST *> args;
      if (callExpressionCtx->callParams()) {
        for (auto paramCtx : callExpressionCtx->callParams()->expression()) {
          auto varName = paramCtx->getText();
          // auto *var = new VariableExprAST(getLoc(ctx), varName);
          auto var = visitExpressionX(paramCtx);
          if (var != nullptr) args.push_back(var);
          if (DEBUG_SOL) std::cout << SPACE << "  fcall->args: " << varName << std::endl;
        }
      }
      // ok - function names can collide - we need to append parameter count
      fcall->callee = fcall->callee + "." + std::to_string(args.size());
      fcall->args = args;
      return fcall;
    } else if (auto methodCallExpressionCtx =
                   dynamic_cast<RustParser::MethodCallExpressionContext *>(
                       ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  MethodCallExpressionContext: " << ctx->getText()
             << std::endl;

      FunctionCallAST *fcall = new FunctionCallAST(
          getLoc(methodCallExpressionCtx->pathExprSegment()));
      // auto base = methodCallExpressionCtx->expression()->getText();
      auto pathExprSegment =
          methodCallExpressionCtx->pathExprSegment()->getText();
      // fcall->callee = base + "." + pathExprSegment;
      fcall->callee = pathExprSegment;
      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

      std::vector<stx::ExprAST *> args;
      auto baseExpr = visitExpressionX(methodCallExpressionCtx->expression());
      if (baseExpr != nullptr) args.push_back(baseExpr);
      if (methodCallExpressionCtx->callParams()) {
        for (auto paramCtx :
             methodCallExpressionCtx->callParams()->expression()) {
          auto varName = paramCtx->getText();
          // auto *var = new VariableExprAST(getLoc(ctx), varName);
          auto var = visitExpressionX(paramCtx);
          if (var != nullptr) args.push_back(var);
          if (DEBUG_SOL) std::cout << SPACE << "  fcall->args: " << varName << std::endl;
        }
      }
      // ok - function names can collide - we need to append parameter count
      fcall->callee = fcall->callee + "." + std::to_string(args.size());
      fcall->args = args;
      return fcall;

    } else if (auto closureExpressionCtx =
                   dynamic_cast<RustParser::ClosureExpression_Context *>(ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  ClosureExpression_Context: " << ctx->getText()
             << std::endl;

      if (auto closureExpr =
              closureExpressionCtx->closureExpression()->expression()) {
        return visitExpressionX(closureExpr);
      }

      if (DEBUG_SOL) std::cout << SPACE << "  ClosureExpression_Context end" << std::endl;
      indentLess();
    } else if (auto compareExpressionCtx =
                   dynamic_cast<RustParser::ComparisonExpressionContext *>(
                       ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  ComparisonExpressionContext: " << ctx->getText()
             << std::endl;

      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
      // expr
      // callParams
      fcall->callee = compareExpressionCtx->comparisonOperator()->getText();
      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

      std::vector<stx::ExprAST *> args;
      if (compareExpressionCtx->expression().size() > 1) {
        for (auto paramCtx : compareExpressionCtx->expression()) {
          auto varName = paramCtx->getText();
          // auto *var = new VariableExprAST(getLoc(ctx), varName);
          auto var = visitExpressionX(paramCtx);
          if (var != nullptr) args.push_back(var);
          if (DEBUG_SOL) std::cout << SPACE << "  fcall->args: " << varName << std::endl;
        }
      } else {
        auto var = visitExpressionX(compareExpressionCtx->expression(0));
        if (var != nullptr) args.push_back(var);
      }

      fcall->args = args;
      return fcall;

    } else if (auto compoundAssignmentExpressionCtx = dynamic_cast<
                   RustParser::CompoundAssignmentExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE
             << "  CompoundAssignmentExpressionContext: " << ctx->getText()
             << std::endl;
      auto compoundAssignCtx =
          compoundAssignmentExpressionCtx->compoundAssignOperator();
      FunctionCallAST *fcall = new FunctionCallAST(getLoc(compoundAssignCtx));
      fcall->callee = compoundAssignCtx->getText();
      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

      std::vector<stx::ExprAST *> args;
      for (auto paramCtx : compoundAssignmentExpressionCtx->expression()) {
        auto var = visitExpressionX(paramCtx);
        if (var != nullptr) args.push_back(var);
        if (DEBUG_SOL)
          std::cout << SPACE << "  fcall->args: " << paramCtx->getText() << std::endl;
      }

      fcall->args = args;
      return fcall;

    } else if (auto assignmentExpressionCtx =
                   dynamic_cast<RustParser::AssignmentExpressionContext *>(
                       ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  AssignmentExpressionContext: " << ctx->getText()
             << std::endl;
      auto leftVar = assignmentExpressionCtx->expression(0)->getText();
      auto var = new VariableExprAST(getLoc(assignmentExpressionCtx), leftVar);

      [[maybe_unused]] auto leftExpr = visitExpressionX(assignmentExpressionCtx->expression(0));
      auto rightExpr = visitExpressionX(assignmentExpressionCtx->expression(1));
      // make sure leftExpr and leftExpr rightExpr are not null
      if (rightExpr != nullptr) {
        auto assign_expr =
            new AssignExprAST(getLoc(assignmentExpressionCtx), var, rightExpr);
        return assign_expr;
      } else {
        if (DEBUG_SOL)
          std::cout << SPACE << "  AssignmentExpression is wrong!!!"
               << ctx->getText() << std::endl;
      }

    } else if (auto borrowExpressionCtx =
                   dynamic_cast<RustParser::BorrowExpressionContext *>(ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  BorrowExpressionContext: " << ctx->getText()
             << std::endl;

      auto res = visitExpressionX(borrowExpressionCtx->expression());
      if (DEBUG_SOL) std::cout << SPACE << "  BorrowExpressionContext end" << std::endl;
      indentLess();
      return res;

    } else if (auto derefExpressionCtx =
                   dynamic_cast<RustParser::DereferenceExpressionContext *>(
                       ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  DereferenceExpressionContext: " << ctx->getText()
             << std::endl;

      auto res = visitExpressionX(derefExpressionCtx->expression());
      if (DEBUG_SOL)
        std::cout << SPACE << "  DereferenceExpressionContext end" << std::endl;
      indentLess();
      return res;

    } else if (auto negationExpressionCtx =
                   dynamic_cast<RustParser::NegationExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  NegationExpressionContext: " << ctx->getText()
             << std::endl;
      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
      if (negationExpressionCtx->MINUS()) {
        // fcall->callee = negationExpressionCtx->MINUS()->getText();
        // distinguish minus and negation
        fcall->callee = "-!";
      } else
        fcall->callee = negationExpressionCtx->NOT()->getText();

      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

      std::vector<stx::ExprAST *> args;
      auto var = visitExpressionX(negationExpressionCtx->expression());
      if (var != nullptr) args.push_back(var);
      fcall->args = args;
      return fcall;

    } else if (auto structExpressionCtx_ =
                   dynamic_cast<RustParser::StructExpression_Context *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  StructExpression_Context: " << ctx->getText()
             << std::endl;

      if (structExpressionCtx_->structExpression()->structExprStruct()) {
        auto structExprStructCtx =
            structExpressionCtx_->structExpression()->structExprStruct();

        FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
        auto structName = "model.struct.new." +
                          structExprStructCtx->pathInExpression()->getText();
        fcall->callee = structName;
        if (DEBUG_SOL)
          std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

        std::vector<stx::ExprAST *> args;
        // auto structExpr = new LiteralExprAST(getLoc(ctx), structName);
        // args.push_back(structExpr);
        if (structExprStructCtx->structExprFields()) {
          for (auto exprFieldCtx :
               structExprStructCtx->structExprFields()->structExprField()) {
            std::string exprFieldName = "";
            if (exprFieldCtx->identifier()) {
              exprFieldName = exprFieldCtx->identifier()->getText();
            } else {
              exprFieldName = exprFieldCtx->tupleIndex()->getText();
            }
            auto var = new VariableExprAST(getLoc(exprFieldCtx->identifier()),
                                           exprFieldName);

            if (exprFieldCtx->expression()) {
              auto valueExpr = visitExpressionX(exprFieldCtx->expression());
              if (valueExpr != nullptr) {
                // assign
                auto assign_expr = new AssignExprAST(
                    getLoc(exprFieldCtx->identifier()), var, valueExpr);
                args.push_back(assign_expr);
              } else {
                std::cout << SPACE
                     << "NULL structExprFields: " << exprFieldCtx->getText()
                     << std::endl;
                args.push_back(var);
              }
            } else {
              args.push_back(var);
            }
          }
        }
        fcall->callee = fcall->callee + "." + std::to_string(args.size());
        fcall->args = args;
        return fcall;
      } else {
        auto literal = structExpressionCtx_->getText();
        return new LiteralExprAST(getLoc(ctx), literal);
      }

    } else if (auto literalExpressionCtx =
                   dynamic_cast<RustParser::LiteralExpression_Context *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  LiteralExpression_Context: " << ctx->getText()
             << std::endl;
      auto literal = literalExpressionCtx->literalExpression()->getText();
      return new LiteralExprAST(getLoc(ctx), literal);

    } else if (auto tupleExpressionCtx =
                   dynamic_cast<RustParser::TupleExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  TupleExpressionContext: " << ctx->getText() << std::endl;
      if (tupleExpressionCtx->tupleElements()) {
        auto literal = tupleExpressionCtx->tupleElements()->getText();
        return new LiteralExprAST(getLoc(ctx), literal);
      } else {
        auto literal = tupleExpressionCtx->getText();
        return new LiteralExprAST(getLoc(ctx), literal);
      }

    } else if (auto macroInvokeExpressionCtx = dynamic_cast<
                   RustParser::MacroInvocationAsExpressionContext *>(ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE
             << "  MacroInvocationAsExpressionContext: " << ctx->getText()
             << std::endl;
      // call func
      auto name =
          macroInvokeExpressionCtx->macroInvocation()->simplePath()->getText();
      auto delimTokenTree =
          macroInvokeExpressionCtx->macroInvocation()->delimTokenTree();

      // quote! {}
      if (delimTokenTree->LCURLYBRACE()) {
      }
      auto paraName = delimTokenTree->getText();
      // for (auto tokenTree : delimTokenTree->tokenTree()) {
      //   std::cout << SPACE << "  tokenTree: " << tokenTree->getText() << std::endl;
      // }
      auto literal = new LiteralExprAST(getLoc(delimTokenTree), paraName);

      FunctionCallAST *fcall =
          new FunctionCallAST(getLoc(macroInvokeExpressionCtx));

      std::vector<stx::ExprAST *> args;
      args.push_back(literal);
      fcall->args = args;
      auto macroCallFunc = "model.macro.";

      fcall->callee = macroCallFunc + name + ".!" + std::to_string(args.size());

      if (DEBUG_SOL)
        std::cout << SPACE << "  MacroInvocationAsExpressionContext end" << std::endl;
      indentLess();
      return fcall;

    } else if (auto arrayExpressionCtx =
                   dynamic_cast<RustParser::ArrayExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  ArrayExpressionContext: " << ctx->getText() << std::endl;
      auto literal = arrayExpressionCtx->getText();
      // auto literal = arrayExpressionCtx->innerAttribute()->getText();
      return new LiteralExprAST(getLoc(ctx), literal);

    } else if (auto fieldExpressionCtx =
                   dynamic_cast<RustParser::FieldExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  FieldExpressionContext: " << ctx->getText() << std::endl;
      auto varName = fieldExpressionCtx->getText();
      return new VariableExprAST(getLoc(ctx), varName);

    } else if (auto pathExpressionCtx =
                   dynamic_cast<RustParser::PathExpression_Context *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  PathExpression_Context: " << ctx->getText() << std::endl;
      auto varName = pathExpressionCtx->getText();
      return new VariableExprAST(getLoc(ctx), varName);

    } else if (auto indexExpressionCtx =
                   dynamic_cast<RustParser::IndexExpressionContext *>(ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  IndexExpressionContext: " << ctx->getText() << std::endl;
      // only the first one
      auto res = visitExpressionX(indexExpressionCtx->expression(0));
      if (DEBUG_SOL) std::cout << SPACE << "  IndexExpressionContext end" << std::endl;
      indentLess();
      return res;

    } else if (auto awaitExpressionCtx =
                   dynamic_cast<RustParser::AwaitExpressionContext *>(ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  AwaitExpressionContext: " << ctx->getText() << std::endl;
      auto res = visitExpressionX(awaitExpressionCtx->expression());
      if (DEBUG_SOL) std::cout << SPACE << "  AwaitExpressionContext end" << std::endl;
      indentLess();
      return res;
    } else if (auto tupleIndexExpressionCtx =
                   dynamic_cast<RustParser::TupleIndexingExpressionContext *>(
                       ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  TupleIndexingExpressionContext: " << ctx->getText()
             << std::endl;
      //    | expression '.' tupleIndex
      auto res = visitExpressionX(tupleIndexExpressionCtx->expression());
      if (DEBUG_SOL)
        std::cout << SPACE << "  TupleIndexingExpressionContext end" << std::endl;
      indentLess();
      return res;

    } else if (auto groupExpressionCtx =
                   dynamic_cast<RustParser::GroupedExpressionContext *>(ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  GroupedExpressionContext: " << ctx->getText()
             << std::endl;
      //    '(' innerAttribute* expression ')'
      auto res = visitExpressionX(groupExpressionCtx->expression());
      if (DEBUG_SOL) std::cout << SPACE << "  GroupedExpressionContext end" << std::endl;
      indentLess();
      return res;

    } else if (auto returnExpressionCtx =
                   dynamic_cast<RustParser::ReturnExpressionContext *>(ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  ReturnExpressionContext: " << ctx->getText()
             << std::endl;
      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
      fcall->callee = "return";  // special
      std::vector<stx::ExprAST *> args;
      //    'return' expression?
      if (returnExpressionCtx->expression()) {
        auto res = visitExpressionX(returnExpressionCtx->expression());
        if (res)
          args.push_back(res);
        else {
          if (DEBUG_SOL)
            std::cout << SPACE << "NULL ReturnExpressionContext: "
                 << returnExpressionCtx->expression()->getText() << std::endl;
        }
      }
      if (DEBUG_SOL) std::cout << SPACE << "  ReturnExpressionContext end" << std::endl;
      indentLess();
      fcall->args = args;
      fcall->callee = fcall->callee + "." + std::to_string(args.size());
      return fcall;
    } else if (auto typeCastExpressionCtx =
                   dynamic_cast<RustParser::TypeCastExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  TypeCastExpressionContext: " << ctx->getText()
             << std::endl;
      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
      fcall->callee = "typecast";  // special

      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

      std::vector<stx::ExprAST *> args;
      auto from = visitExpressionX(typeCastExpressionCtx->expression());
      if (from != nullptr) {
        auto toVar = typeCastExpressionCtx->typeNoBounds()->getText();
        auto to = new LiteralExprAST(getLoc(typeCastExpressionCtx), toVar);
        args.push_back(from);
        args.push_back(to);
      } else {
        if (DEBUG_SOL)
          std::cout << SPACE << "  TypeCastExpressionContext is wrong!!!"
               << ctx->getText() << std::endl;
      }
      fcall->args = args;
      return fcall;
    } else if (auto expressionWithBlock_Ctx =
                   dynamic_cast<RustParser::ExpressionWithBlock_Context *>(
                       ctx)) {
      indentMore();
      if (DEBUG_SOL)
        std::cout << SPACE << "  ExpressionWithBlock_Context: " << ctx->getText()
             << std::endl;
      auto expressionWithBlockCtx =
          expressionWithBlock_Ctx->expressionWithBlock();
      auto res = visitExpressionWithBlockX(expressionWithBlockCtx);
      if (DEBUG_SOL)
        std::cout << SPACE << "  ExpressionWithBlock_Context end" << std::endl;
      indentLess();
      return res;
    } else if (auto arithLogicExpressionCtx = dynamic_cast<
                   RustParser::ArithmeticOrLogicalExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE
             << "  ArithmeticOrLogicalExpressionContext: " << ctx->getText()
             << std::endl;

      // if (arithLogicExpressionCtx->STAR()) {
      //   return visitExpressionX(arithLogicExpressionCtx->expression(1));
      // }

      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
      // antlr4::tree::TerminalNode *STAR();
      // antlr4::tree::TerminalNode *SLASH();
      // antlr4::tree::TerminalNode *PERCENT();
      // antlr4::tree::TerminalNode *PLUS();
      // antlr4::tree::TerminalNode *MINUS();
      // ShlContext *shl();
      // ShrContext *shr();
      // antlr4::tree::TerminalNode *AND();
      // antlr4::tree::TerminalNode *CARET();
      // antlr4::tree::TerminalNode *OR();
      auto opName = arithLogicExpressionCtx->getText();
      if (arithLogicExpressionCtx->STAR()) {
        opName = arithLogicExpressionCtx->STAR()->getText();
      } else if (arithLogicExpressionCtx->SLASH()) {
        opName = arithLogicExpressionCtx->SLASH()->getText();
      } else if (arithLogicExpressionCtx->PERCENT()) {
        opName = arithLogicExpressionCtx->PERCENT()->getText();
      } else if (arithLogicExpressionCtx->PLUS()) {
        opName = arithLogicExpressionCtx->PLUS()->getText();
      } else if (arithLogicExpressionCtx->MINUS()) {
        opName = arithLogicExpressionCtx->MINUS()->getText();
      } else if (arithLogicExpressionCtx->shl()) {
        opName = arithLogicExpressionCtx->shl()->getText();
      } else if (arithLogicExpressionCtx->shr()) {
        opName = arithLogicExpressionCtx->shr()->getText();
      } else if (arithLogicExpressionCtx->AND()) {
        opName = arithLogicExpressionCtx->AND()->getText();
      } else if (arithLogicExpressionCtx->CARET()) {
        opName = arithLogicExpressionCtx->CARET()->getText();
      } else if (arithLogicExpressionCtx->OR()) {
        opName = arithLogicExpressionCtx->OR()->getText();
      } else {
      }
      fcall->callee = opName;  // special

      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

      std::vector<stx::ExprAST *> args;
      auto expr0 = visitExpressionX(arithLogicExpressionCtx->expression(0));
      auto expr1 = visitExpressionX(arithLogicExpressionCtx->expression(1));

      if (expr0 != nullptr && expr1 != nullptr) {
        args.push_back(expr0);
        args.push_back(expr1);
      } else {
        if (DEBUG_SOL)
          std::cout << SPACE
               << "TODO  ArithmeticOrLogicalExpressionContext is wrong!!!"
               << ctx->getText() << std::endl;
        return nullptr;
      }
      fcall->args = args;
      return fcall;

    } else if (auto lazyBoolExpressionCtx =
                   dynamic_cast<RustParser::LazyBooleanExpressionContext *>(
                       ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  LazyBooleanExpressionContext: " << ctx->getText()
             << std::endl;
      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));

      auto opName = lazyBoolExpressionCtx->getText();
      if (lazyBoolExpressionCtx->ANDAND()) {
        opName = lazyBoolExpressionCtx->ANDAND()->getText();
      } else if (lazyBoolExpressionCtx->OROR()) {
        opName = lazyBoolExpressionCtx->OROR()->getText();
      }

      fcall->callee = opName;  // || or &&

      if (DEBUG_SOL)
        std::cout << SPACE << "  fcall->callee: " << fcall->callee << std::endl;

      std::vector<stx::ExprAST *> args;
      auto expr0 = visitExpressionX(lazyBoolExpressionCtx->expression(0));
      auto expr1 = visitExpressionX(lazyBoolExpressionCtx->expression(1));

      if (expr0 != nullptr && expr1 != nullptr) {
        args.push_back(expr0);
        args.push_back(expr1);
      } else {
        if (DEBUG_SOL)
          std::cout << SPACE << "  ArithmeticOrLogicalExpressionContext is wrong!!!"
               << ctx->getText() << std::endl;
      }
      fcall->args = args;
      return fcall;

    } else if (auto callExpressionCtx =
                   dynamic_cast<RustParser::RangeExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  RangeExpressionContext: " << ctx->getText() << std::endl;
    } else if (auto callExpressionCtx = dynamic_cast<
                   RustParser::ErrorPropagationExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE
             << "  ErrorPropagationExpressionContext: " << ctx->getText()
             << std::endl;
    } else if (auto callExpressionCtx =
                   dynamic_cast<RustParser::CallExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  CallExpressionContext: " << ctx->getText() << std::endl;
    } else if (auto breakExpressionCtx =
                   dynamic_cast<RustParser::BreakExpressionContext *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  BreakExpressionContext: " << ctx->getText() << std::endl;
      FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
      fcall->callee = "model.break";
      std::vector<stx::ExprAST *> args;
      fcall->args = args;
      return fcall;
    } else if (auto callExpressionCtx = dynamic_cast<
                   RustParser::EnumerationVariantExpression_Context *>(ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE
             << "  EnumerationVariantExpression_Context: " << ctx->getText()
             << std::endl;
    } else if (auto callExpressionCtx =
                   dynamic_cast<RustParser::AttributedExpressionContext *>(
                       ctx)) {
      if (DEBUG_SOL)
        std::cout << SPACE << "  AttributedExpressionContext: " << ctx->getText()
             << std::endl;
    } else {
      if (DEBUG_SOL)
        std::cout << SPACE << "  Other Unknown Context: " << ctx->getText() << std::endl;
    }
    if (DEBUG_SOL) std::cout << SPACE << "visitExpressionX end " << std::endl;
    return expr;
  }

  stx::ExprAST *visitLetStatementX(RustParser::LetStatementContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitLetStatement begin: " << ctx->getText() << std::endl;
    indentMore();
    stx::ExprAST *expr = nullptr;
    auto varName = ctx->patternNoTopAlt()->getText();
    if (auto patternCtx = ctx->patternNoTopAlt()->patternWithoutRange()) {
      if (patternCtx->slicePattern()) {
        auto patternItemsCtx = patternCtx->slicePattern()->slicePatternItems();
        if (patternItemsCtx && patternItemsCtx->pattern().size() > 0) {
          FunctionCallAST *fcall = new FunctionCallAST(getLoc(patternItemsCtx));
          std::vector<stx::ExprAST *> args;
          for (auto item : patternItemsCtx->pattern()) {
            // TODO: handle mango [...]
            FunctionCallAST *fcall2 = new FunctionCallAST(getLoc(item));
            {
              auto literal = new LiteralExprAST(getLoc(item), item->getText());
              std::vector<stx::ExprAST *> args2;
              args2.push_back(literal);
              fcall2->args = args2;
              auto sliceItemCallFunc = "model.slice.item.";
              fcall2->callee = sliceItemCallFunc + std::to_string(args2.size());
            }
            args.push_back(fcall2);
          }
          fcall->args = args;
          auto sliceCallFunc = "model.slice.";
          fcall->callee = sliceCallFunc + std::to_string(args.size());
          expr = fcall;
        }
      } else {
        if (auto patternCtx1 = patternCtx->identifierPattern()) {
          varName = patternCtx1->identifier()->getText();
        }
        expr = new VariableExprAST(getLoc(ctx), varName);
      }
    }
    auto result = visitChildren(ctx);  // debug
    if (ctx->expression()) {
        stx::ExprAST *valueExpr = visitExpressionX(ctx->expression());
      // make sure valueExpr is not null
      if (valueExpr == nullptr) return expr;

      auto assign_expr = new AssignExprAST(getLoc(ctx), expr, valueExpr);
      expr = assign_expr;
    }
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitLetStatement end " << std::endl;
    return expr;
  }

  stx::ExprAST *visitIfExpressionX(RustParser::IfExpressionContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitIfExpressionX begin: " << ctx->getText() << std::endl;
    indentMore();
    stx::ExprAST *expr = nullptr;
    // if(left,right)
    FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
    // expr
    // callParams
    fcall->callee = "if";
    std::vector<stx::ExprAST *> args;
    auto condExpr = visitExpressionX(ctx->expression());
    if (condExpr != nullptr)
      args.push_back(condExpr);
    else
      std::cout << SPACE << "condExpr is null: " << ctx->getText() << std::endl;
    fcall->args = args;
    expr = fcall;

    // TODO: blockExpr as annoymous function
    {
      auto ifBlockExpr = visitCustomizedIfBlockExpressionX(
          ctx->blockExpression().front(), expr, true);
      if (ctx->blockExpression().size() > 1) {
        auto elseBlockExpr = visitCustomizedIfBlockExpressionX(
            ctx->blockExpression().back(), ifBlockExpr, false);
        return elseBlockExpr;
      } else if (ctx->ifExpression()) {
        auto elseIfExpr = visitIfExpressionX(ctx->ifExpression());
        if (ifBlockExpr && elseIfExpr) {
          // create call ifTrueFalse
          FunctionCallAST *fcall2 =
              new FunctionCallAST(getLoc(ctx->ifExpression()));
          fcall2->callee = "ifTrueFalse" + ANON_NAME;
          std::vector<stx::ExprAST *> args2;
          args2.push_back(ifBlockExpr);
          args2.push_back(elseIfExpr);
          fcall2->args = args2;
          return fcall2;
        } else if (elseIfExpr) {
          return elseIfExpr;
        }

      } else if (ctx->ifLetExpression()) {
        // auto ifLetExpr = visitIfLetExpressionX(ctx->ifLetExpression());
      }

      return ifBlockExpr;
    }

    // auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitIfExpressionX end " << std::endl;
    return expr;
  }

  virtual antlrcpp::Any visitIfExpression(
      RustParser::IfExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitIfExpression begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitIfExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitIfLetExpression(
      RustParser::IfLetExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitIfLetExpression begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitIfLetExpression end " << std::endl;
    return result;
  }

  // customized
  stx::ExprAST *visitCustomizedIfBlockExpressionX(
      RustParser::BlockExpressionContext *ctx, stx::ExprAST *expr, bool isTrue) {
    auto stmts = ctx->statements();
    auto loc = getLoc(ctx);
    stx::FunctionAST *func = new stx::FunctionAST(loc);
    auto fname = fnBaseName + "::" + curFuncName + ANON_NAME;
    auto k = functionAnonNamesMap[fname];
    k++;
    func->function_name = fname + std::to_string(k);
    functionAnonNamesMap[fname] = k;

    std::vector<stx::VarDeclExprAST> args;
    std::vector<stx::ExprAST *> res = visitStatementsX(stmts);
    func->body = res;
    // ok - function names can collide - we need to append parameter count
    // func->function_name =
    //     func->function_name + "." + std::to_string(args.size());
    addNewFunction(loc, func->function_name, args, func);

    // create call anon
    FunctionCallAST *fcall = new FunctionCallAST(loc);
    fcall->callee = func->function_name;
    std::vector<stx::ExprAST *> args1;
    args1.push_back(expr);
    fcall->args = args1;

    // create call ifTrue
    FunctionCallAST *fcall2 = new FunctionCallAST(loc);
    if (isTrue)
      fcall2->callee = "ifTrue" + ANON_NAME;
    else
      fcall2->callee = "ifFalse" + ANON_NAME;
    std::vector<stx::ExprAST *> args2;
    // args2.push_back(expr);
    args2.push_back(fcall);
    fcall2->args = args2;
    return fcall2;
  }

  stx::ExprAST *visitIfLetExpressionX(RustParser::IfLetExpressionContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitIfLetExpressionX begin: " << ctx->getText()
           << std::endl;
    indentMore();
    stx::ExprAST *expr = nullptr;

    if (ctx->expression()) {
      // TODO: let Some(x) = y
      //  if let Some(collection_authority_record) =
      //  delegate_collection_authority_record {
      //     let data = collection_authority_record
      //         .try_borrow_data()?;
      // }
      expr = visitExpressionX(ctx->expression());
      if (expr != nullptr) {
        auto varName = ctx->pattern()->getText();
        varName = ctx->expression()->getText();
        auto var = new VariableExprAST(getLoc(ctx->pattern()), varName);
        expr = new AssignExprAST(getLoc(ctx), var, expr);
      } else {
        std::cout << "NULL visitIfLetExpressionX: " << ctx->getText() << std::endl;
      }

      // create func for the block statements
      auto ifBlockExpr = visitCustomizedIfBlockExpressionX(
          ctx->blockExpression().front(), expr, true);
      if (ctx->blockExpression().size() > 1) {
        auto elseBlockExpr = visitCustomizedIfBlockExpressionX(
            ctx->blockExpression().back(), ifBlockExpr, false);
        return elseBlockExpr;
      } else if (ctx->ifExpression()) {
        // auto ifExpr = visitIfExpressionX(ctx->ifExpression());
      } else if (ctx->ifLetExpression()) {
        // auto ifLetExpr = visitIfLetExpressionX(ctx->ifLetExpression());
      }

      return ifBlockExpr;
    }

    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitIfLetExpressionX end " << std::endl;
    return expr;
  }

  stx::ExprAST *visitMatchExpressionX(RustParser::MatchExpressionContext *ctx) {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitMatchExpressionX begin: " << ctx->getText()
           << std::endl;
    indentMore();

    stx::ExprAST *expr = nullptr;
    // if(left,right)
    FunctionCallAST *fcall = new FunctionCallAST(getLoc(ctx));
    // expr
    // callParams
    fcall->callee = "match";
    std::vector<stx::ExprAST *> args;
    auto size = ctx->matchArms()->matchArmExpression().size();
    for (size_t i = 0; i < size; i++) {
      auto matchArm = ctx->matchArms()->matchArm()[i];
      auto matchArmExpression = ctx->matchArms()->matchArmExpression()[i];

      stx::ExprAST *matchExpr = nullptr;
      std::string debugExprText = "";
      if (matchArmExpression->expression()) {
        debugExprText = matchArmExpression->expression()->getText();
        matchExpr = visitExpressionX(matchArmExpression->expression());
      } else if (auto blockExpr = matchArmExpression->expressionWithBlock()) {
        debugExprText = blockExpr->getText();

        if (blockExpr->ifExpression()) {
          matchExpr = visitIfExpressionX(blockExpr->ifExpression());
        } else if (blockExpr->matchExpression()) {
          matchExpr = visitMatchExpressionX(blockExpr->matchExpression());
        } else if (blockExpr->blockExpression() &&
                   blockExpr->blockExpression()->statements()) {
          if (auto lastExpr =
                  blockExpr->blockExpression()->statements()->expression()) {
            matchExpr = visitExpressionX(lastExpr);
          } else {
            // TODO: a block of multiple statements
            auto literal = matchArm->pattern()->getText();
            expr = new LiteralExprAST(getLoc(matchArm), literal);
            matchExpr = visitCustomizedIfBlockExpressionX(
                blockExpr->blockExpression(), expr, true);

            // auto blockExprStatements =
            //     blockExpr->blockExpression()->statements()->statement();
            // auto lastStmt =
            //     blockExprStatements.back();  // blockExprStatements.size()-1
            // matchExpr = visitStatementX(lastStmt);
          }
        }
      }

      if (matchExpr) {
        args.push_back(matchExpr);
        if (DEBUG_SOL) std::cout << SPACE << "matched: " << debugExprText << std::endl;
      } else {
        if (DEBUG_SOL)
          std::cout << SPACE << "not matched: " << debugExprText << std::endl;
      }
    }
    //   : (matchArm '=>' matchArmExpression)* matchArm '=>' expression ','?
    // do the last one
    auto matchExpr = visitExpressionX(ctx->matchArms()->expression());
    if (matchExpr) {
      args.push_back(matchExpr);
      if (DEBUG_SOL)
        std::cout << SPACE
             << "matched: " << ctx->matchArms()->expression()->getText()
             << std::endl;
    } else {
      if (DEBUG_SOL)
        std::cout << SPACE
             << "not matched: " << ctx->matchArms()->expression()->getText()
             << std::endl;
    }
    fcall->args = args;
    // ok - function names can collide - we need to append parameter count
    fcall->callee = fcall->callee + "." + std::to_string(args.size());
    expr = fcall;

    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitMatchExpressionX end " << std::endl;
    return expr;
  }

  virtual antlrcpp::Any visitMatchExpression(
      RustParser::MatchExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitMatchExpression begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitMatchExpression end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitContinueExpression(
      RustParser::ContinueExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitContinueExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitContinueExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitAssignmentExpression(
      RustParser::AssignmentExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitAssignmentExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitAssignmentExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitMethodCallExpression(
      RustParser::MethodCallExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitMethodCallExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitMethodCallExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitLiteralExpression_(
      RustParser::LiteralExpression_Context *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitLiteralExpression_ begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitLiteralExpression_ end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitReturnExpression(
      RustParser::ReturnExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitReturnExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitReturnExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitExpressionStatement(
      RustParser::ExpressionStatementContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitExpressionStatement begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitExpressionStatement end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitTypeCastExpression(
      RustParser::TypeCastExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTypeCastExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTypeCastExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitPathExpression_(
      RustParser::PathExpression_Context *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitPathExpression_ begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitPathExpression_ end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitTupleExpression(
      RustParser::TupleExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTupleExpression begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTupleExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitIndexExpression(
      RustParser::IndexExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitIndexExpression begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitIndexExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitRangeExpression(
      RustParser::RangeExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitRangeExpression begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitRangeExpression end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitStructExpression_(
      RustParser::StructExpression_Context *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStructExpression_ begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStructExpression_ end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitTupleIndexingExpression(
      RustParser::TupleIndexingExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTupleIndexingExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTupleIndexingExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitNegationExpression(
      RustParser::NegationExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitNegationExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitNegationExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitCallExpression(
      RustParser::CallExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitCallExpression begin: " << ctx->getText() << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitCallExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitLazyBooleanExpression(
      RustParser::LazyBooleanExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitLazyBooleanExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    if (DEBUG_SOL) std::cout << SPACE << "visitLazyBooleanExpression end " << std::endl;
    indentLess();
    return result;
  }

  virtual antlrcpp::Any visitDereferenceExpression(
      RustParser::DereferenceExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitDereferenceExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();

    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitDereferenceExpression end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitExpressionWithBlock_(
      RustParser::ExpressionWithBlock_Context *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitExpressionWithBlock_ begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitExpressionWithBlock_ end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitGroupedExpression(
      RustParser::GroupedExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitGroupedExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitGroupedExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitBreakExpression(
      RustParser::BreakExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitBreakExpression begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitBreakExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitArithmeticOrLogicalExpression(
      RustParser::ArithmeticOrLogicalExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE
           << "visitArithmeticOrLogicalExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL)
      std::cout << SPACE << "visitArithmeticOrLogicalExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitFieldExpression(
      RustParser::FieldExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitFieldExpression begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitFieldExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitEnumerationVariantExpression_(
      RustParser::EnumerationVariantExpression_Context *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE
           << "visitEnumerationVariantExpression_ begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL)
      std::cout << SPACE << "visitEnumerationVariantExpression_ end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitComparisonExpression(
      RustParser::ComparisonExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitComparisonExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitComparisonExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitAttributedExpression(
      RustParser::AttributedExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitAttributedExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitAttributedExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitBorrowExpression(
      RustParser::BorrowExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitBorrowExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitBorrowExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitCompoundAssignmentExpression(
      RustParser::CompoundAssignmentExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE
           << "visitCompoundAssignmentExpression begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL)
      std::cout << SPACE << "visitCompoundAssignmentExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitClosureExpression_(
      RustParser::ClosureExpression_Context *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitClosureExpression_ begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitClosureExpression_ end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitArrayExpression(
      RustParser::ArrayExpressionContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitArrayExpression begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitArrayExpression end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitComparisonOperator(
      RustParser::ComparisonOperatorContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitComparisonOperator begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitComparisonOperator end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitCompoundAssignOperator(
      RustParser::CompoundAssignOperatorContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitCompoundAssignOperator begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitCompoundAssignOperator end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitMatchArmGuard(
      RustParser::MatchArmGuardContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << "visitMatchArmGuard begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << "visitMatchArmGuard end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitPattern(RustParser::PatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitPattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitPattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitPatternWithoutRange(
      RustParser::PatternWithoutRangeContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitPatternWithoutRange begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitPatternWithoutRange end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitLiteralPattern(
      RustParser::LiteralPatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitLiteralPattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitLiteralPattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitIdentifierPattern(
      RustParser::IdentifierPatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitIdentifierPattern begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitIdentifierPattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitWildcardPattern(
      RustParser::WildcardPatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitWildcardPattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitWildcardPattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitRestPattern(
      RustParser::RestPatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitRestPattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitRestPattern end " << std::endl;
    return result;
  }
  virtual antlrcpp::Any visitObsoleteRangePattern(
      RustParser::ObsoleteRangePatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitObsoleteRangePattern begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitObsoleteRangePattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitRangePatternBound(
      RustParser::RangePatternBoundContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitRangePatternBound begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitRangePatternBound end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitReferencePattern(
      RustParser::ReferencePatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitReferencePattern begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitReferencePattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitStructPattern(
      RustParser::StructPatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStructPattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStructPattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitStructPatternElements(
      RustParser::StructPatternElementsContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStructPatternElements begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStructPatternElements end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitStructPatternFields(
      RustParser::StructPatternFieldsContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStructPatternFields begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStructPatternFields end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitStructPatternField(
      RustParser::StructPatternFieldContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStructPatternField begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStructPatternField end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitStructPatternEtCetera(
      RustParser::StructPatternEtCeteraContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitStructPatternEtCetera begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitStructPatternEtCetera end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitTupleStructPattern(
      RustParser::TupleStructPatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTupleStructPattern begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTupleStructPattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitTupleStructItems(
      RustParser::TupleStructItemsContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << "visitTupleStructItems begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << "visitTupleStructItems end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitTuplePattern(
      RustParser::TuplePatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTuplePattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTuplePattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitTuplePatternItems(
      RustParser::TuplePatternItemsContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitTuplePatternItems begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitTuplePatternItems end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitGroupedPattern(
      RustParser::GroupedPatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitGroupedPattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitGroupedPattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitSlicePattern(
      RustParser::SlicePatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitSlicePattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitSlicePattern end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitSlicePatternItems(
      RustParser::SlicePatternItemsContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitSlicePatternItems begin: " << ctx->getText()
           << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitSlicePatternItems end " << std::endl;
    return result;
  }

  virtual antlrcpp::Any visitPathPattern(
      RustParser::PathPatternContext *ctx) override {
    if (DEBUG_SOL)
      std::cout << SPACE << "visitPathPattern begin: " << ctx->getText() << std::endl;
    indentMore();
    auto result = visitChildren(ctx);
    indentLess();
    if (DEBUG_SOL) std::cout << SPACE << "visitPathPattern end " << std::endl;
    return result;
  }
};

#endif  // MLIR_ST_PARSER_VISITOR_H_
