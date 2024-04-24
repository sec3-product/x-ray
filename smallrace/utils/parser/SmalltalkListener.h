
// Generated from /home/aaron/smallrace/utils/parser/Smalltalk.g4 by ANTLR 4.9

#pragma once


#include "antlr4-runtime.h"
#include "SmalltalkParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by SmalltalkParser.
 */
class  SmalltalkListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterModule(SmalltalkParser::ModuleContext *ctx) = 0;
  virtual void exitModule(SmalltalkParser::ModuleContext *ctx) = 0;

  virtual void enterFunction(SmalltalkParser::FunctionContext *ctx) = 0;
  virtual void exitFunction(SmalltalkParser::FunctionContext *ctx) = 0;

  virtual void enterFuncdecl(SmalltalkParser::FuncdeclContext *ctx) = 0;
  virtual void exitFuncdecl(SmalltalkParser::FuncdeclContext *ctx) = 0;

  virtual void enterDeclPairs(SmalltalkParser::DeclPairsContext *ctx) = 0;
  virtual void exitDeclPairs(SmalltalkParser::DeclPairsContext *ctx) = 0;

  virtual void enterDeclPair(SmalltalkParser::DeclPairContext *ctx) = 0;
  virtual void exitDeclPair(SmalltalkParser::DeclPairContext *ctx) = 0;

  virtual void enterScript(SmalltalkParser::ScriptContext *ctx) = 0;
  virtual void exitScript(SmalltalkParser::ScriptContext *ctx) = 0;

  virtual void enterSequence(SmalltalkParser::SequenceContext *ctx) = 0;
  virtual void exitSequence(SmalltalkParser::SequenceContext *ctx) = 0;

  virtual void enterWs(SmalltalkParser::WsContext *ctx) = 0;
  virtual void exitWs(SmalltalkParser::WsContext *ctx) = 0;

  virtual void enterTemps(SmalltalkParser::TempsContext *ctx) = 0;
  virtual void exitTemps(SmalltalkParser::TempsContext *ctx) = 0;

  virtual void enterStatements(SmalltalkParser::StatementsContext *ctx) = 0;
  virtual void exitStatements(SmalltalkParser::StatementsContext *ctx) = 0;

  virtual void enterAnswer(SmalltalkParser::AnswerContext *ctx) = 0;
  virtual void exitAnswer(SmalltalkParser::AnswerContext *ctx) = 0;

  virtual void enterExpression(SmalltalkParser::ExpressionContext *ctx) = 0;
  virtual void exitExpression(SmalltalkParser::ExpressionContext *ctx) = 0;

  virtual void enterExpressions(SmalltalkParser::ExpressionsContext *ctx) = 0;
  virtual void exitExpressions(SmalltalkParser::ExpressionsContext *ctx) = 0;

  virtual void enterExpressionList(SmalltalkParser::ExpressionListContext *ctx) = 0;
  virtual void exitExpressionList(SmalltalkParser::ExpressionListContext *ctx) = 0;

  virtual void enterCascade(SmalltalkParser::CascadeContext *ctx) = 0;
  virtual void exitCascade(SmalltalkParser::CascadeContext *ctx) = 0;

  virtual void enterMessage(SmalltalkParser::MessageContext *ctx) = 0;
  virtual void exitMessage(SmalltalkParser::MessageContext *ctx) = 0;

  virtual void enterAssignment(SmalltalkParser::AssignmentContext *ctx) = 0;
  virtual void exitAssignment(SmalltalkParser::AssignmentContext *ctx) = 0;

  virtual void enterVariable(SmalltalkParser::VariableContext *ctx) = 0;
  virtual void exitVariable(SmalltalkParser::VariableContext *ctx) = 0;

  virtual void enterBinarySend(SmalltalkParser::BinarySendContext *ctx) = 0;
  virtual void exitBinarySend(SmalltalkParser::BinarySendContext *ctx) = 0;

  virtual void enterUnarySend(SmalltalkParser::UnarySendContext *ctx) = 0;
  virtual void exitUnarySend(SmalltalkParser::UnarySendContext *ctx) = 0;

  virtual void enterKeywordSend(SmalltalkParser::KeywordSendContext *ctx) = 0;
  virtual void exitKeywordSend(SmalltalkParser::KeywordSendContext *ctx) = 0;

  virtual void enterKeywordMessage(SmalltalkParser::KeywordMessageContext *ctx) = 0;
  virtual void exitKeywordMessage(SmalltalkParser::KeywordMessageContext *ctx) = 0;

  virtual void enterKeywordPair(SmalltalkParser::KeywordPairContext *ctx) = 0;
  virtual void exitKeywordPair(SmalltalkParser::KeywordPairContext *ctx) = 0;

  virtual void enterOperand(SmalltalkParser::OperandContext *ctx) = 0;
  virtual void exitOperand(SmalltalkParser::OperandContext *ctx) = 0;

  virtual void enterSubexpression(SmalltalkParser::SubexpressionContext *ctx) = 0;
  virtual void exitSubexpression(SmalltalkParser::SubexpressionContext *ctx) = 0;

  virtual void enterLiteral(SmalltalkParser::LiteralContext *ctx) = 0;
  virtual void exitLiteral(SmalltalkParser::LiteralContext *ctx) = 0;

  virtual void enterRuntimeLiteral(SmalltalkParser::RuntimeLiteralContext *ctx) = 0;
  virtual void exitRuntimeLiteral(SmalltalkParser::RuntimeLiteralContext *ctx) = 0;

  virtual void enterBlock(SmalltalkParser::BlockContext *ctx) = 0;
  virtual void exitBlock(SmalltalkParser::BlockContext *ctx) = 0;

  virtual void enterBlockParamList(SmalltalkParser::BlockParamListContext *ctx) = 0;
  virtual void exitBlockParamList(SmalltalkParser::BlockParamListContext *ctx) = 0;

  virtual void enterDynamicDictionary(SmalltalkParser::DynamicDictionaryContext *ctx) = 0;
  virtual void exitDynamicDictionary(SmalltalkParser::DynamicDictionaryContext *ctx) = 0;

  virtual void enterDynamicArray(SmalltalkParser::DynamicArrayContext *ctx) = 0;
  virtual void exitDynamicArray(SmalltalkParser::DynamicArrayContext *ctx) = 0;

  virtual void enterParsetimeLiteral(SmalltalkParser::ParsetimeLiteralContext *ctx) = 0;
  virtual void exitParsetimeLiteral(SmalltalkParser::ParsetimeLiteralContext *ctx) = 0;

  virtual void enterNumber(SmalltalkParser::NumberContext *ctx) = 0;
  virtual void exitNumber(SmalltalkParser::NumberContext *ctx) = 0;

  virtual void enterNumberExp(SmalltalkParser::NumberExpContext *ctx) = 0;
  virtual void exitNumberExp(SmalltalkParser::NumberExpContext *ctx) = 0;

  virtual void enterCharConstant(SmalltalkParser::CharConstantContext *ctx) = 0;
  virtual void exitCharConstant(SmalltalkParser::CharConstantContext *ctx) = 0;

  virtual void enterHex_(SmalltalkParser::Hex_Context *ctx) = 0;
  virtual void exitHex_(SmalltalkParser::Hex_Context *ctx) = 0;

  virtual void enterStInteger(SmalltalkParser::StIntegerContext *ctx) = 0;
  virtual void exitStInteger(SmalltalkParser::StIntegerContext *ctx) = 0;

  virtual void enterStFloat(SmalltalkParser::StFloatContext *ctx) = 0;
  virtual void exitStFloat(SmalltalkParser::StFloatContext *ctx) = 0;

  virtual void enterPseudoVariable(SmalltalkParser::PseudoVariableContext *ctx) = 0;
  virtual void exitPseudoVariable(SmalltalkParser::PseudoVariableContext *ctx) = 0;

  virtual void enterString(SmalltalkParser::StringContext *ctx) = 0;
  virtual void exitString(SmalltalkParser::StringContext *ctx) = 0;

  virtual void enterPrimitive(SmalltalkParser::PrimitiveContext *ctx) = 0;
  virtual void exitPrimitive(SmalltalkParser::PrimitiveContext *ctx) = 0;

  virtual void enterSymbol(SmalltalkParser::SymbolContext *ctx) = 0;
  virtual void exitSymbol(SmalltalkParser::SymbolContext *ctx) = 0;

  virtual void enterBareSymbol(SmalltalkParser::BareSymbolContext *ctx) = 0;
  virtual void exitBareSymbol(SmalltalkParser::BareSymbolContext *ctx) = 0;

  virtual void enterLiteralArray(SmalltalkParser::LiteralArrayContext *ctx) = 0;
  virtual void exitLiteralArray(SmalltalkParser::LiteralArrayContext *ctx) = 0;

  virtual void enterLiteralArrayRest(SmalltalkParser::LiteralArrayRestContext *ctx) = 0;
  virtual void exitLiteralArrayRest(SmalltalkParser::LiteralArrayRestContext *ctx) = 0;

  virtual void enterByteLiteralArray(SmalltalkParser::ByteLiteralArrayContext *ctx) = 0;
  virtual void exitByteLiteralArray(SmalltalkParser::ByteLiteralArrayContext *ctx) = 0;

  virtual void enterByteLiteralArrayBody(SmalltalkParser::ByteLiteralArrayBodyContext *ctx) = 0;
  virtual void exitByteLiteralArrayBody(SmalltalkParser::ByteLiteralArrayBodyContext *ctx) = 0;

  virtual void enterBareLiteralArray(SmalltalkParser::BareLiteralArrayContext *ctx) = 0;
  virtual void exitBareLiteralArray(SmalltalkParser::BareLiteralArrayContext *ctx) = 0;

  virtual void enterUnaryTail(SmalltalkParser::UnaryTailContext *ctx) = 0;
  virtual void exitUnaryTail(SmalltalkParser::UnaryTailContext *ctx) = 0;

  virtual void enterUnaryMessage(SmalltalkParser::UnaryMessageContext *ctx) = 0;
  virtual void exitUnaryMessage(SmalltalkParser::UnaryMessageContext *ctx) = 0;

  virtual void enterUnarySelector(SmalltalkParser::UnarySelectorContext *ctx) = 0;
  virtual void exitUnarySelector(SmalltalkParser::UnarySelectorContext *ctx) = 0;

  virtual void enterKeywords(SmalltalkParser::KeywordsContext *ctx) = 0;
  virtual void exitKeywords(SmalltalkParser::KeywordsContext *ctx) = 0;

  virtual void enterReference(SmalltalkParser::ReferenceContext *ctx) = 0;
  virtual void exitReference(SmalltalkParser::ReferenceContext *ctx) = 0;

  virtual void enterBinaryTail(SmalltalkParser::BinaryTailContext *ctx) = 0;
  virtual void exitBinaryTail(SmalltalkParser::BinaryTailContext *ctx) = 0;

  virtual void enterBinaryMessage(SmalltalkParser::BinaryMessageContext *ctx) = 0;
  virtual void exitBinaryMessage(SmalltalkParser::BinaryMessageContext *ctx) = 0;


};

