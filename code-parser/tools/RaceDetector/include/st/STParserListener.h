#ifndef MLIR_ST_PARSER_LISTENER_H_
#define MLIR_ST_PARSER_LISTENER_H_

#include "RustParserBaseListener.h"
#include "st/ParserWrapper.h"

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

  void enterCrate(RustParser::CrateContext * /*ctx*/) {
    std::cout << SPACE << "enterCrate\n";
    indentMore();
  }
  void exitCrate(RustParser::CrateContext * /*ctx*/) {
    indentLess();
    std::cout << SPACE << "exitCrate\n";
  }

  // void enterModule(SmalltalkParser::ModuleContext *ctx) {
  //   std::cout << SPACE << "enterModule\n";
  //   indentMore();
  //   if (ctx->function()) {
  //     auto id = ctx->function()->funcdecl()->IDENTIFIER();
  //     Location loc({fileName, ctx->getStart()->getLine() + line_base,
  //                   ctx->getStart()->getCharPositionInLine()});
  //     VarType type;
  //     type.name = "*i8";
  //     if (!id) {
  //       // TODO: add arg0 as this
  //       auto keywords = ctx->function()->funcdecl()->declPairs()->declPair();
  //       int count = 1;
  //       for (auto kw : keywords) {
  //         // fnName += kw->KEYWORD()->getText();
  //         VarDeclExprAST arg(loc, "arg" + count++, type);
  //         args.push_back(arg);
  //       }
  //     }
  //   }
  // }
  // void exitModule(SmalltalkParser::ModuleContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitModule\n";
  // }
  // void enterFunction(SmalltalkParser::FunctionContext *ctx) {
  //   std::cout << SPACE << "enterFunction: " << fnName << "\n";
  //   indentMore();
  // }
  // void exitFunction(SmalltalkParser::FunctionContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitFunction\n";
  // }
  // void enterScript(SmalltalkParser::ScriptContext *ctx) {
  //   std::cout << SPACE << "enterScript\n";
  //   indentMore();
  // }
  // void exitScript(SmalltalkParser::ScriptContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitScript\n";
  // }
  // void enterSequence(SmalltalkParser::SequenceContext *ctx) {
  //   std::cout << SPACE << "enterSequence\n";
  //   indentMore();
  // }
  // void exitSequence(SmalltalkParser::SequenceContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitSequence\n";
  // }
  // void enterWs(SmalltalkParser::WsContext *ctx) {
  //   std::cout << SPACE << "enterWs\n";
  //   indentMore();
  // }
  // void exitWs(SmalltalkParser::WsContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitWs\n";
  // }
  // void enterTemps2(SmalltalkParser::TempsContext *ctx) {
  //   std::cout << SPACE << "enterTemps2:";
  //   indentMore();
  //   for (auto id : ctx->IDENTIFIER()) {
  //     std::cout << " " << id->getSymbol()->toString();

  //     Location loc({fileName, id->getSymbol()->getLine() + line_base,
  //                   id->getSymbol()->getCharPositionInLine()});
  //     VarType type;
  //     // VarDeclExprAST var(loc, id->getSymbol()->getText(), type);
  //     // exprList.push_back(var);

  //     auto *var = new VarDeclExprAST(loc, id->getSymbol()->getText(), type);
  //     exprList.push_back(var);
  //   }
  //   std::cout << "\n";
  // }
  // void exitTemps2(SmalltalkParser::TempsContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitTemps2\n";
  // }
  // void enterTemps(SmalltalkParser::TempsContext *ctx) {
  //   std::cout << SPACE << "enterTemps\n";
  //   indentMore();
  // }
  // void exitTemps(SmalltalkParser::TempsContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitTemps\n";
  // }
  // void enterTemps1(SmalltalkParser::TempsContext *ctx) {
  //   std::cout << SPACE << "enterTemps1\n";
  //   indentMore();
  // }
  // void exitTemps1(SmalltalkParser::TempsContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitTemps1\n";
  // }
  // // void enterStatementAnswer(SmalltalkParser::StatementAnswerContext *ctx)
  // {
  // //   std::cout << SPACE << "enterStatementAnswer\n";
  // //   indentMore();
  // // }
  // // void exitStatementAnswer(SmalltalkParser::StatementAnswerContext *ctx) {
  // //   indentLess();
  // //   std::cout << SPACE << "exitStatementAnswer\n";
  // // }
  // // void enterStatementExpressionsAnswer(
  // //     SmalltalkParser::StatementExpressionsAnswerContext *ctx) {
  // //   std::cout << SPACE << "enterStatementExpressionsAnswer\n";
  // //   indentMore();
  // // }
  // // void exitStatementExpressionsAnswer(
  // //     SmalltalkParser::StatementExpressionsAnswerContext *ctx) {
  // //   indentLess();
  // //   std::cout << SPACE << "exitStatementExpressionsAnswer\n";
  // // }
  // // void enterStatementExpressions(
  // //     SmalltalkParser::StatementExpressionsContext *ctx) {
  // //   std::cout << SPACE << "enterStatementExpressions\n";
  // //   indentMore();
  // // }
  // // void exitStatementExpressions(
  // //     SmalltalkParser::StatementExpressionsContext *ctx) {
  // //   indentLess();
  // //   std::cout << SPACE << "exitStatementExpressions\n";
  // // }
  // void enterAnswer(SmalltalkParser::AnswerContext *ctx) {
  //   std::cout << SPACE << "enterAnswer\n";
  //   indentMore();
  // }
  // void exitAnswer(SmalltalkParser::AnswerContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitAnswer\n";
  // }

  // void handleTranscript(SmalltalkParser::KeywordMessageContext *ctx) {
  //   if (!ctx) std::cout << "handleTranscript empty ctx\n";
  //   // PrintExprAST
  //   auto pairs = ctx->keywordPair();
  //   {
  //     // std::cout <<"keyword msg text:
  //     // "<<ctx->keywordSend()->keywordMessage()->getText()<<"\n";

  //     for (auto kw : pairs) {
  //       // std::cout <<"keyword text: "<<kw->KEYWORD()->getText()<<"\n";

  //       if (kw->KEYWORD()->getText() == "show:") {
  //         Location loc({fileName, ctx->getStart()->getLine() + line_base,
  //                       ctx->getStart()->getCharPositionInLine()});

  //         if (auto op = kw->binarySend()->unarySend()->operand()) {
  //           if (op->reference()) {
  //             // case 2: reference
  //             std::cout << "todo handle reference\n";
  //             auto *arg = new VariableExprAST(loc, op->getText());
  //             auto *expr = new PrintExprAST(loc, arg);
  //             exprList.push_back(expr);
  //           } else {
  //             // case 1: literal string
  //             auto *arg = new LiteralExprAST(loc, op->getText());
  //             auto *expr = new PrintExprAST(loc, arg);
  //             exprList.push_back(expr);
  //           }
  //         } else {
  //           std::cout << "no operand\n";
  //         }
  //       } else {
  //         std::cout << "not transcript show\n";
  //       }
  //     }
  //   }
  // }
  // void enterExpression(SmalltalkParser::ExpressionContext *ctx) {
  //   std::cout << SPACE << "enterExpression: "
  //             << " @level " << level << "\n";  //<< ctx->getText()
  //   indentMore();
  //   if (ctx->assignment()) {
  //     Location loc(
  //         {fileName,
  //          ctx->assignment()->variable()->getStart()->getLine() + line_base,
  //          ctx->assignment()->variable()->getStart()->getCharPositionInLine()});

  //     ExprAST *valueExpr = nullptr;
  //     // todo handle assignment
  //     if (auto bsend = ctx->assignment()->expression()->binarySend()) {
  //       if (auto op = bsend->unarySend()->operand()) {
  //         if (op->literal() && op->literal()->parsetimeLiteral()) {
  //           if (op->literal()->parsetimeLiteral()->number()) {
  //             valueExpr = new NumberExprAST(loc, std::stol(op->getText()));
  //           } else {
  //             valueExpr = new LiteralExprAST(loc, op->getText());
  //           }
  //         }
  //       }
  //     }

  //     auto varName = ctx->assignment()->variable()->getText();
  //     std::cout << "adding assignment var: " << varName << "\n";
  //     auto *expr = new AssignExprAST(loc, varName, valueExpr, level + 2);
  //     exprList.push_back(expr);

  //   } else if (ctx->cascade()) {
  //     // Transcript cr; show: 'Size ',	self sharedState size
  //     printString. std::cout << "todo handle cascade\n";
  //   } else if (ctx->keywordSend()) {
  //     // std::cout << "todo handle keywordSend\n";

  //   } else if (ctx->binarySend()) {
  //     auto op = ctx->binarySend()->unarySend()->operand();
  //     if (op->reference() && ctx->binarySend()->unarySend()->unaryTail()) {
  //       auto id1 = ctx->binarySend()
  //                      ->unarySend()
  //                      ->operand()
  //                      ->reference()
  //                      ->variable()
  //                      ->IDENTIFIER();
  //       Location loc({fileName, id1->getSymbol()->getLine() + line_base,
  //                     id1->getSymbol()->getCharPositionInLine()});

  //       std::vector<ExprAST *> args;
  //       auto *arg = new VariableExprAST(loc, id1->getSymbol()->getText());
  //       args.push_back(arg);

  //       auto id2 = ctx->binarySend()
  //                      ->unarySend()
  //                      ->unaryTail()
  //                      ->unaryMessage()
  //                      ->unarySelector()
  //                      ->IDENTIFIER();
  //       auto name = id2->getSymbol()->getText();
  //       auto *expr = new CallExprAST(loc, std::string(name), args, level);
  //       exprList.push_back(expr);
  //     } else if (op->literal()) {
  //       if (op->literal()->parsetimeLiteral()) {
  //         std::cout << "todo add parsetimeLiteral: " << op->getText() <<
  //         "\n";

  //         if (ctx->binarySend()->unarySend()->unaryTail()) {
  //           std::vector<ExprAST *> args;

  //           Location loc1({fileName, op->getStart()->getLine() + line_base,
  //                          op->getStart()->getCharPositionInLine()});
  //           auto *arg = new LiteralExprAST(loc1, op->literal()->getText());
  //           args.push_back(arg);

  //           auto usend = ctx->binarySend()->unarySend();

  //           Location loc2({fileName, usend->getStart()->getLine() +
  //           line_base,
  //                          usend->getStart()->getCharPositionInLine()});
  //           auto name = usend->unaryTail()->unaryMessage()->getText();
  //           auto *expr = new CallExprAST(loc2, std::string(name), args,
  //           level); exprList.push_back(expr); std::cout << "adding
  //           LiteralExprAST CallExprAST: " << op->getText()
  //                     << " " << name << "\n";
  //         }
  //       } else if (op->literal()->runtimeLiteral()) {
  //         if (op->literal()->runtimeLiteral()->block()) {
  //           std::cout << "adding BlockExprAST: " << op->getText() << "\n";
  //           std::vector<ExprAST *> args;
  //           Location loc1({fileName, op->getStart()->getLine() + line_base,
  //                          op->getStart()->getCharPositionInLine()});
  //           auto *arg = new BlockExprAST(loc1, op->getText(), level);
  //           args.push_back(arg);
  //           // TODO: [[:inst | nil]]
  //           if (auto usend = ctx->binarySend()->unarySend()) {
  //             Location loc2({fileName, usend->getStart()->getLine() +
  //             line_base,
  //                            usend->getStart()->getCharPositionInLine()});

  //             if (usend->unaryTail()) {
  //               auto name = usend->unaryTail()->unaryMessage()->getText();
  //               auto *expr =
  //                   new CallExprAST(loc2, std::string(name), args, level);
  //               exprList.push_back(expr);
  //             } else {
  //               auto name = "anonymousBlock";
  //               auto *expr =
  //                   new CallExprAST(loc2, std::string(name), args, level);
  //               exprList.push_back(expr);
  //             }
  //           }

  //           blockExprMap[op->getText()] = arg;

  //         } else {
  //           std::cout << "todo add other runtimeLiteral: " << op->getText()
  //                     << "\n";
  //         }
  //       }
  //     }
  //   }
  // }
  // void exitExpression(SmalltalkParser::ExpressionContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitExpression\n";
  // }
  // void enterExpressions(SmalltalkParser::ExpressionsContext *ctx) {
  //   std::cout << SPACE << "enterExpressions\n";
  //   indentMore();
  // }
  // void exitExpressions(SmalltalkParser::ExpressionsContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitExpressions\n";
  // }
  // void enterExpressionList(SmalltalkParser::ExpressionListContext *ctx) {
  //   std::cout << SPACE << "enterExpressionList\n";
  //   indentMore();
  // }
  // void exitExpressionList(SmalltalkParser::ExpressionListContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitExpressionList\n";
  // }
  // void enterCascade(SmalltalkParser::CascadeContext *ctx) {
  //   std::cout << SPACE << "enterCascade\n";
  //   indentMore();
  // }
  // void exitCascade(SmalltalkParser::CascadeContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitCascade\n";
  // }
  // void enterMessage(SmalltalkParser::MessageContext *ctx) {
  //   std::cout << SPACE << "enterMessage\n";
  //   indentMore();
  // }
  // void exitMessage(SmalltalkParser::MessageContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitMessage\n";
  // }
  // void enterAssignment(SmalltalkParser::AssignmentContext *ctx) {
  //   std::cout << SPACE << "enterAssignment\n";
  //   indentMore();
  // }
  // void exitAssignment(SmalltalkParser::AssignmentContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitAssignment\n";
  // }
  // void enterVariable(SmalltalkParser::VariableContext *ctx) {
  //   std::cout << SPACE << "enterVariable: " << ctx->getText() << "\n";
  //   indentMore();
  // }
  // void exitVariable(SmalltalkParser::VariableContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitVariable\n";
  // }
  // void enterBinarySend(SmalltalkParser::BinarySendContext *ctx) {
  //   // std::cout << SPACE << "enterBinarySend\n";
  //   std::cout << SPACE << "enterBinarySend: "
  //             << " @level " << level << "\n";  //<< ctx->getText()
  //   indentMore();
  // }
  // void exitBinarySend(SmalltalkParser::BinarySendContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitBinarySend\n";
  // }
  // void enterUnarySend(SmalltalkParser::UnarySendContext *ctx) {
  //   std::cout << SPACE << "enterUnarySend\n";
  //   indentMore();
  // }
  // void exitUnarySend(SmalltalkParser::UnarySendContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitUnarySend\n";
  // }
  // void enterKeywordSend(SmalltalkParser::KeywordSendContext *ctx) {
  //   std::cout << SPACE << "enterKeywordSend: " << ctx->getStart()->getText()
  //             << "\n";
  //   indentMore();

  //   Location loc({fileName, ctx->getStart()->getLine() + line_base,
  //                 ctx->getStart()->getCharPositionInLine()});
  //   auto pairs = ctx->keywordMessage()->keywordPair();
  //   // Transcript
  //   if (ctx->binarySend()->getStart()->getText() == "Transcript") {
  //     // PrintExprAST
  //     std::cout << "handling Transcript calls\n";

  //     for (auto kw : pairs) {
  //       // std::cout <<"keyword text: "<<kw->KEYWORD()->getText()<<"\n";

  //       if (kw->KEYWORD()->getText() == "show:") {
  //         if (auto op = kw->binarySend()->unarySend()->operand()) {
  //           if (op->reference()) {
  //             // case 2: reference
  //             auto *arg = new VariableExprAST(loc, op->getText());
  //             auto *expr = new PrintExprAST(loc, arg);
  //             exprList.push_back(expr);
  //           } else {
  //             // case 1: literal string
  //             auto *arg = new LiteralExprAST((loc), op->getText());
  //             auto *expr = new PrintExprAST(loc, arg);
  //             exprList.push_back(expr);
  //           }
  //         } else {
  //           std::cout << "no operand\n";
  //         }
  //       } else {
  //         std::cout << "not transcript show\n";
  //       }
  //     }

  //   } else {
  //     std::cout << "handling other calls\n";
  //     // other calls  e.g.
  //     std::vector<ExprAST *> args;

  //     if (auto op = ctx->binarySend()->unarySend()->operand()) {
  //       // literal | reference | subexpression

  //       // self sharedState
  //       // OrderedCollection new
  //       if (auto tail = ctx->binarySend()->unarySend()->unaryTail()) {
  //         auto name = tail->unaryMessage()->unarySelector()->getText();
  //         std::cout << "adding unaryTail CallExprAST: " << name << "\n";
  //         std::vector<ExprAST *> args2;

  //         if (op->literal() && op->literal()->parsetimeLiteral()) {
  //           auto *arg = new LiteralExprAST(loc, op->getText());
  //           args2.push_back(arg);
  //         } else if (op->reference()) {
  //           auto *arg = new VariableExprAST(loc, op->getText());
  //           args2.push_back(arg);
  //         } else {
  //           std::cout << "op->reference  literal not true: " << op->getText()
  //                     << "\n";
  //         }
  //         auto *expr = new CallExprAST(loc, std::string(name), args2, level);
  //         args.push_back(expr);
  //       } else {
  //         //[] forkAt:30
  //         if (op->literal() && op->literal()->runtimeLiteral() &&
  //             op->literal()->runtimeLiteral()->block()) {
  //           std::cout << "adding BlockExprAST: " << op->getText() << "\n";
  //           auto *arg = new BlockExprAST(loc, op->getText(), level);
  //           args.push_back(arg);

  //           blockExprMap[op->getText()] = arg;
  //         } else if (op->literal() && op->literal()->parsetimeLiteral()) {
  //           std::cout << "adding LiteralExprAST: " << op->getText() << "\n";
  //           auto *arg = new LiteralExprAST(loc, op->getText());
  //           args.push_back(arg);
  //         } else if (op->reference()) {
  //           // shared at: index put: 1.
  //           std::cout << "adding VariableExprAST: " << op->getText() << "\n";
  //           auto *arg = new VariableExprAST(loc, op->getText());
  //           args.push_back(arg);
  //         } else if (op->subexpression()) {
  //           // ( Object errorSignal )
  //           auto expression = op->subexpression()->expression();
  //           std::vector<ExprAST *> args2;
  //           if (auto keyword = expression->keywordSend()) {
  //             std::cout << "todo: handle subexpression keyword\n";
  //             auto *expr =
  //                 new CallExprAST(loc, std::string("stub"), args2, level);
  //             args.push_back(expr);
  //           } else if (expression->binarySend()) {
  //             if (auto tail =
  //                     expression->binarySend()->unarySend()->unaryTail()) {
  //               auto name = tail->getText();
  //               std::cout << "adding subexpression CallExprAST: " << name
  //                         << "\n";
  //               if (expression->binarySend()
  //                       ->unarySend()
  //                       ->operand()
  //                       ->reference() ||
  //                   expression->binarySend()
  //                       ->unarySend()
  //                       ->operand()
  //                       ->literal()) {
  //                 auto objectname = expression->binarySend()
  //                                       ->unarySend()
  //                                       ->operand()
  //                                       ->getText();
  //                 auto *arg = new LiteralExprAST(loc, objectname);
  //                 args2.push_back(arg);
  //               }

  //               auto *expr =
  //                   new CallExprAST(loc, std::string(name), args2, level);
  //               args.push_back(expr);
  //             }
  //           }
  //         }
  //       }
  //     }

  //     // shared removeKey: index
  //     // shared at: index put: 1
  //     std::string callFuncName = "";
  //     // call func "x:y"
  //     for (auto kw : pairs) {
  //       callFuncName += kw->KEYWORD()->getText();

  //       if (auto op = kw->binarySend()->unarySend()->operand()) {
  //         if (op->reference()) {
  //           std::cout << "adding VariableExprAST: " << op->getText() << "\n";
  //           auto *arg = new VariableExprAST(loc, op->getText());
  //           args.push_back(arg);
  //         } else if (op->literal()) {
  //           if (op->literal()->parsetimeLiteral()) {
  //             if (op->literal()->parsetimeLiteral()->number()) {
  //               std::cout << "adding NumberExprAST: " << op->getText() <<
  //               "\n"; auto *arg = new NumberExprAST(loc,
  //               std::stol(op->getText())); args.push_back(arg);
  //             } else {
  //               std::cout << "adding other parsetimeLiteral: " <<
  //               op->getText()
  //                         << "\n";
  //               auto *arg = new LiteralExprAST(loc, op->getText());
  //               args.push_back(arg);
  //             }
  //           } else if (op->literal()->runtimeLiteral()) {
  //             if (op->literal()->runtimeLiteral()->block()) {
  //               std::cout << "adding BlockExprAST: " << op->getText() <<
  //               "\n"; auto *arg = new BlockExprAST(loc, op->getText(),
  //               level); args.push_back(arg);

  //               blockExprMap[op->getText()] = arg;

  //             } else {
  //               std::cout << "todo adding other runtimeLiteral: "
  //                         << op->getText() << "\n";
  //             }
  //           }
  //         } else if (op->subexpression()) {
  //           // put: (Object errorSignal handle: [:ex | ex returnWith: #error]
  //           // do: [Process generateRecursionLockOwners: processList]);
  //           // JEFF: we need a better parser tree to AST algorithm
  //           // traveral needs probably more recursion??

  //           std::vector<ExprAST *> args2;
  //           std::cout << "todo: handle keyword pair subexpression stub\n";
  //           auto *expr =
  //               new CallExprAST(loc, std::string("stub"), args2, level);
  //           args.push_back(expr);

  //         } else
  //           std::cout << "todo other operand\n";
  //       } else {
  //         std::cout << "other calls no operand\n";
  //       }
  //     }

  //     std::cout << "adding KEYWORD PAIR CallExprAST: " << callFuncName
  //               << " argsize: " << args.size() << "\n";
  //     auto *expr = new CallExprAST(loc, callFuncName, args, level);
  //     exprList.push_back(expr);
  //   }
  // }

  // void exitKeywordSend(SmalltalkParser::KeywordSendContext *ctx) {
  //   std::cout << SPACE << "exitKeywordSend\n";
  //   indentLess();
  // }
  // void enterKeywordMessage(SmalltalkParser::KeywordMessageContext *ctx) {
  //   std::cout << SPACE << "enterKeywordMessage: " <<
  //   ctx->getStart()->getText()
  //             << "\n";
  //   indentMore();
  // }
  // void exitKeywordMessage(SmalltalkParser::KeywordMessageContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitKeywordMessage\n";
  // }
  // void enterKeywordPair(SmalltalkParser::KeywordPairContext *ctx) {
  //   std::cout << SPACE
  //             << "enterKeywordPair: " <<
  //             ctx->KEYWORD()->getSymbol()->toString()
  //             << "\n";
  //   indentMore();
  // }
  // void exitKeywordPair(SmalltalkParser::KeywordPairContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitKeywordPair\n";
  // }
  // void enterOperand(SmalltalkParser::OperandContext *ctx) {
  //   std::cout << SPACE << "enterOperand\n";
  //   indentMore();
  // }
  // void exitOperand(SmalltalkParser::OperandContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitOperand\n";
  // }
  // void enterSubexpression(SmalltalkParser::SubexpressionContext *ctx) {
  //   std::cout << SPACE << "enterSubexpression\n";
  //   indentMore();
  // }
  // void exitSubexpression(SmalltalkParser::SubexpressionContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitSubexpression\n";
  // }
  // void enterLiteral(SmalltalkParser::LiteralContext *ctx) {
  //   std::cout << SPACE << "enterLiteral\n";
  //   indentMore();
  // }
  // void exitLiteral(SmalltalkParser::LiteralContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitLiteral\n";
  // }
  // void enterRuntimeLiteral(SmalltalkParser::RuntimeLiteralContext *ctx) {
  //   std::cout << SPACE << "enterRuntimeLiteral\n";
  //   indentMore();
  // }
  // void exitRuntimeLiteral(SmalltalkParser::RuntimeLiteralContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitRuntimeLiteral\n";
  // }
  // void enterBlock(SmalltalkParser::BlockContext *ctx) {
  //   // std::cout << SPACE << "enterBlock\n";
  //   std::cout << SPACE << "enterBlock: "
  //             << " @level " << level << "\n";  //<< ctx->getText()
  //   indentMore();

  //   Location loc({fileName, ctx->getStart()->getLine() + line_base,
  //                 ctx->getStart()->getCharPositionInLine()});
  //   auto *expr = new BlockExprAST(loc, ctx->getText(), level);
  //   exprList.push_back(expr);

  //   blockExprIdStack.push_back(exprList.size() - 1);
  // }
  // void exitBlock(SmalltalkParser::BlockContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitBlock: "
  //             << " @level " << level << "\n";  //<< ctx->getText()

  //   int id = blockExprIdStack.back();
  //   blockExprIdStack.pop_back();
  //   auto bexpr = exprList.at(id);
  //   if (auto *bexpr_ = llvm::dyn_cast<BlockExprAST>(bexpr)) {
  //     if (blockExprMap.find(ctx->getText()) == blockExprMap.end()) {
  //       std::cout << "TODO: block expr text: " << ctx->getText() << "\n";
  //       return;
  //     }
  //     std::cout << "block expr text: " << ctx->getText() << "\n";

  //     auto *block_expr = blockExprMap.at(ctx->getText());

  //     // let's do a quick preprocessing for blockexpr here

  //     int idx = id + 1;
  //     while (idx < exprList.size()) {
  //       std::cout << "idx: " << idx << " size: " << exprList.size() << "\n";
  //       auto expr = exprList.back();
  //       std::cout << "add block expr: " << exprList.size() - 1 << "\n";
  //       block_expr->addExpr(expr);

  //       exprList.pop_back();
  //     }
  //   }
  // }
  // void enterBlockParamList(SmalltalkParser::BlockParamListContext *ctx) {
  //   // std::cout << SPACE << "enterBlockParamList\n";
  //   std::cout << SPACE << "enterBlockParamList: " << ctx->getText()
  //             << " @level " << level << "\n";
  //   indentMore();
  // }
  // void exitBlockParamList(SmalltalkParser::BlockParamListContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitBlockParamList\n";
  // }
  // void enterDynamicDictionary(SmalltalkParser::DynamicDictionaryContext *ctx)
  // {
  //   std::cout << SPACE << "enterDynamicDictionary\n";
  //   indentMore();
  // }
  // void exitDynamicDictionary(SmalltalkParser::DynamicDictionaryContext *ctx)
  // {
  //   indentLess();
  //   std::cout << SPACE << "exitDynamicDictionary\n";
  // }
  // void enterDynamicArray(SmalltalkParser::DynamicArrayContext *ctx) {
  //   std::cout << SPACE << "enterDynamicArray\n";
  //   indentMore();
  // }
  // void exitDynamicArray(SmalltalkParser::DynamicArrayContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitDynamicArray\n";
  // }
  // void enterParsetimeLiteral(SmalltalkParser::ParsetimeLiteralContext *ctx) {
  //   std::cout << SPACE << "enterParsetimeLiteral\n";
  //   indentMore();
  // }
  // void exitParsetimeLiteral(SmalltalkParser::ParsetimeLiteralContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitParsetimeLiteral\n";
  // }
  // void enterNumber(SmalltalkParser::NumberContext *ctx) {
  //   // std::string num = "";
  //   // if(ctx->hex()){

  //   // }else if(ctx->stFloat()){

  //   // }else if(ctx->stInteger()){

  //   //     for(auto id: ctx->stInteger()->DIGIT()){
  //   //         num+=id->getSymbol()->getText();
  //   //     }
  //   //     if(ctx->stInteger()->MINUS())
  //   //     num = "-"+num;
  //   // }

  //   std::cout << SPACE << "enterNumber: " << ctx->getText() << "\n";
  //   indentMore();
  //   // std::cout << SPACE <<"enterNumberExp:
  //   // "<<ctx->RESERVED_WORD()->getSymbol()->toString()<<"\n"; indentMore();
  // }
  // void exitNumber(SmalltalkParser::NumberContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitNumber\n";
  // }
  // void enterNumberExp(SmalltalkParser::NumberExpContext *ctx) {
  //   std::cout << SPACE << "enterNumberExp\n";
  //   indentMore();
  // }
  // void exitNumberExp(SmalltalkParser::NumberExpContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitNumberExp\n";
  // }
  // void enterCharConstant(SmalltalkParser::CharConstantContext *ctx) {
  //   std::cout << SPACE << "enterCharConstant\n";
  //   indentMore();
  // }
  // void exitCharConstant(SmalltalkParser::CharConstantContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitCharConstant\n";
  // }
  // void enterHex_(SmalltalkParser::Hex_Context *ctx) {
  //   std::cout << SPACE << "enterHex\n";
  //   indentMore();
  // }
  // void exitHex_(SmalltalkParser::Hex_Context *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitHex\n";
  // }
  // void enterStInteger(SmalltalkParser::StIntegerContext *ctx) {
  //   std::cout << SPACE << "enterStInteger\n";
  //   indentMore();
  // }
  // void exitStInteger(SmalltalkParser::StIntegerContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitStInteger\n";
  // }
  // void enterStFloat(SmalltalkParser::StFloatContext *ctx) {
  //   std::cout << SPACE << "enterStFloat\n";
  //   indentMore();
  // }
  // void exitStFloat(SmalltalkParser::StFloatContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitStFloat\n";
  // }
  // void enterPseudoVariable(SmalltalkParser::PseudoVariableContext *ctx) {
  //   std::cout << SPACE << "enterPseudoVariable: "
  //             << ctx->RESERVED_WORD()->getSymbol()->toString() << "\n";
  //   indentMore();
  // }
  // void exitPseudoVariable(SmalltalkParser::PseudoVariableContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitPseudoVariable\n";
  // }
  // void enterString(SmalltalkParser::StringContext *ctx) {
  //   std::cout << SPACE << "enterString: " << ctx->getText() << "\n";
  //   indentMore();
  // }
  // void exitString(SmalltalkParser::StringContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitString\n";
  // }
  // void enterSymbol(SmalltalkParser::SymbolContext *ctx) {
  //   std::cout << SPACE << "enterSymbol\n";
  //   indentMore();
  // }
  // void exitSymbol(SmalltalkParser::SymbolContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitSymbol\n";
  // }
  // void enterPrimitive(SmalltalkParser::PrimitiveContext *ctx) {
  //   std::cout << SPACE << "enterPrimitive\n";
  //   indentMore();
  // }
  // void exitPrimitive(SmalltalkParser::PrimitiveContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitPrimitive\n";
  // }
  // void enterBareSymbol(SmalltalkParser::BareSymbolContext *ctx) {
  //   std::cout << SPACE << "enterBareSymbol\n";
  //   indentMore();
  // }
  // void exitBareSymbol(SmalltalkParser::BareSymbolContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitBareSymbol\n";
  // }
  // void enterLiteralArray(SmalltalkParser::LiteralArrayContext *ctx) {
  //   std::cout << SPACE << "enterLiteralArray\n";
  //   indentMore();
  // }
  // void exitLiteralArray(SmalltalkParser::LiteralArrayContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitLiteralArray\n";
  // }
  // void enterLiteralArrayRest(SmalltalkParser::LiteralArrayRestContext *ctx) {
  //   std::cout << SPACE << "enterLiteralArrayRest\n";
  //   indentMore();
  // }
  // void exitLiteralArrayRest(SmalltalkParser::LiteralArrayRestContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitLiteralArrayRest\n";
  // }
  // void enterBareLiteralArray(SmalltalkParser::BareLiteralArrayContext *ctx) {
  //   std::cout << SPACE << "enterBareLiteralArray\n";
  //   indentMore();
  // }
  // void exitBareLiteralArray(SmalltalkParser::BareLiteralArrayContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitBareLiteralArray\n";
  // }
  // void enterUnaryTail(SmalltalkParser::UnaryTailContext *ctx) {
  //   std::cout << SPACE << "enterUnaryTail\n";
  //   indentMore();
  // }
  // void exitUnaryTail(SmalltalkParser::UnaryTailContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitUnaryTail\n";
  // }
  // void enterUnaryMessage(SmalltalkParser::UnaryMessageContext *ctx) {
  //   std::cout << SPACE << "enterUnaryMessage\n";
  //   indentMore();
  // }
  // void exitUnaryMessage(SmalltalkParser::UnaryMessageContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitUnaryMessage\n";
  // }
  // void enterUnarySelector(SmalltalkParser::UnarySelectorContext *ctx) {
  //   // std::cout << SPACE <<"enterUnarySelector\n";
  //   std::cout << SPACE << "enterUnarySelector: "
  //             << ctx->IDENTIFIER()->getSymbol()->toString() << "\n";
  //   indentMore();
  // }
  // void exitUnarySelector(SmalltalkParser::UnarySelectorContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitUnarySelector\n";
  // }
  // void enterKeywords(SmalltalkParser::KeywordsContext *ctx) {
  //   std::cout << SPACE << "enterKeywords\n";
  //   indentMore();
  // }
  // void exitKeywords(SmalltalkParser::KeywordsContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitKeywords\n";
  // }
  // void enterReference(SmalltalkParser::ReferenceContext *ctx) {
  //   std::cout << SPACE << "enterReference\n";
  //   indentMore();
  // }
  // void exitReference(SmalltalkParser::ReferenceContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitReference\n";
  // }
  // void enterBinaryTail(SmalltalkParser::BinaryTailContext *ctx) {
  //   std::cout << SPACE << "enterBinaryTail\n";
  //   indentMore();
  // }
  // void exitBinaryTail(SmalltalkParser::BinaryTailContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitBinaryTail\n";
  // }
  // void enterBinaryMessage(SmalltalkParser::BinaryMessageContext *ctx) {
  //   // std::cout << SPACE << "enterBinaryMessage\n";
  //   std::cout << SPACE << "enterBinaryMessage: " << ctx->getText() << "
  //   @level "
  //             << level << "\n";
  //   indentMore();
  // }
  // void exitBinaryMessage(SmalltalkParser::BinaryMessageContext *ctx) {
  //   indentLess();
  //   std::cout << SPACE << "exitBinaryMessage\n";
  // }
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