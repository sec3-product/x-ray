
// Generated from /home/aaron/smallrace/utils/parser/Smalltalk.g4 by ANTLR 4.9

#pragma once


#include "antlr4-runtime.h"
#include "SmalltalkVisitor.h"


/**
 * This class provides an empty implementation of SmalltalkVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  SmalltalkBaseVisitor : public SmalltalkVisitor {
public:

  virtual antlrcpp::Any visitModule(SmalltalkParser::ModuleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFunction(SmalltalkParser::FunctionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFuncdecl(SmalltalkParser::FuncdeclContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclPairs(SmalltalkParser::DeclPairsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclPair(SmalltalkParser::DeclPairContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitScript(SmalltalkParser::ScriptContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSequence(SmalltalkParser::SequenceContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWs(SmalltalkParser::WsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitTemps(SmalltalkParser::TempsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStatements(SmalltalkParser::StatementsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAnswer(SmalltalkParser::AnswerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpression(SmalltalkParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpressions(SmalltalkParser::ExpressionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpressionList(SmalltalkParser::ExpressionListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCascade(SmalltalkParser::CascadeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMessage(SmalltalkParser::MessageContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAssignment(SmalltalkParser::AssignmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVariable(SmalltalkParser::VariableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBinarySend(SmalltalkParser::BinarySendContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnarySend(SmalltalkParser::UnarySendContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitKeywordSend(SmalltalkParser::KeywordSendContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitKeywordMessage(SmalltalkParser::KeywordMessageContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitKeywordPair(SmalltalkParser::KeywordPairContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitOperand(SmalltalkParser::OperandContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSubexpression(SmalltalkParser::SubexpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLiteral(SmalltalkParser::LiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitRuntimeLiteral(SmalltalkParser::RuntimeLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBlock(SmalltalkParser::BlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBlockParamList(SmalltalkParser::BlockParamListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDynamicDictionary(SmalltalkParser::DynamicDictionaryContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDynamicArray(SmalltalkParser::DynamicArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitParsetimeLiteral(SmalltalkParser::ParsetimeLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNumber(SmalltalkParser::NumberContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNumberExp(SmalltalkParser::NumberExpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCharConstant(SmalltalkParser::CharConstantContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitHex_(SmalltalkParser::Hex_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStInteger(SmalltalkParser::StIntegerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStFloat(SmalltalkParser::StFloatContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPseudoVariable(SmalltalkParser::PseudoVariableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitString(SmalltalkParser::StringContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPrimitive(SmalltalkParser::PrimitiveContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSymbol(SmalltalkParser::SymbolContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBareSymbol(SmalltalkParser::BareSymbolContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLiteralArray(SmalltalkParser::LiteralArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLiteralArrayRest(SmalltalkParser::LiteralArrayRestContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitByteLiteralArray(SmalltalkParser::ByteLiteralArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitByteLiteralArrayBody(SmalltalkParser::ByteLiteralArrayBodyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBareLiteralArray(SmalltalkParser::BareLiteralArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnaryTail(SmalltalkParser::UnaryTailContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnaryMessage(SmalltalkParser::UnaryMessageContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnarySelector(SmalltalkParser::UnarySelectorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitKeywords(SmalltalkParser::KeywordsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitReference(SmalltalkParser::ReferenceContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBinaryTail(SmalltalkParser::BinaryTailContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBinaryMessage(SmalltalkParser::BinaryMessageContext *ctx) override {
    return visitChildren(ctx);
  }


};

