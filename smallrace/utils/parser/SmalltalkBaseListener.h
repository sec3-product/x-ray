
// Generated from /home/aaron/smallrace/utils/parser/Smalltalk.g4 by ANTLR 4.9

#pragma once


#include "antlr4-runtime.h"
#include "SmalltalkListener.h"


/**
 * This class provides an empty implementation of SmalltalkListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  SmalltalkBaseListener : public SmalltalkListener {
public:

  virtual void enterModule(SmalltalkParser::ModuleContext * /*ctx*/) override { }
  virtual void exitModule(SmalltalkParser::ModuleContext * /*ctx*/) override { }

  virtual void enterFunction(SmalltalkParser::FunctionContext * /*ctx*/) override { }
  virtual void exitFunction(SmalltalkParser::FunctionContext * /*ctx*/) override { }

  virtual void enterFuncdecl(SmalltalkParser::FuncdeclContext * /*ctx*/) override { }
  virtual void exitFuncdecl(SmalltalkParser::FuncdeclContext * /*ctx*/) override { }

  virtual void enterDeclPairs(SmalltalkParser::DeclPairsContext * /*ctx*/) override { }
  virtual void exitDeclPairs(SmalltalkParser::DeclPairsContext * /*ctx*/) override { }

  virtual void enterDeclPair(SmalltalkParser::DeclPairContext * /*ctx*/) override { }
  virtual void exitDeclPair(SmalltalkParser::DeclPairContext * /*ctx*/) override { }

  virtual void enterScript(SmalltalkParser::ScriptContext * /*ctx*/) override { }
  virtual void exitScript(SmalltalkParser::ScriptContext * /*ctx*/) override { }

  virtual void enterSequence(SmalltalkParser::SequenceContext * /*ctx*/) override { }
  virtual void exitSequence(SmalltalkParser::SequenceContext * /*ctx*/) override { }

  virtual void enterWs(SmalltalkParser::WsContext * /*ctx*/) override { }
  virtual void exitWs(SmalltalkParser::WsContext * /*ctx*/) override { }

  virtual void enterTemps(SmalltalkParser::TempsContext * /*ctx*/) override { }
  virtual void exitTemps(SmalltalkParser::TempsContext * /*ctx*/) override { }

  virtual void enterStatements(SmalltalkParser::StatementsContext * /*ctx*/) override { }
  virtual void exitStatements(SmalltalkParser::StatementsContext * /*ctx*/) override { }

  virtual void enterAnswer(SmalltalkParser::AnswerContext * /*ctx*/) override { }
  virtual void exitAnswer(SmalltalkParser::AnswerContext * /*ctx*/) override { }

  virtual void enterExpression(SmalltalkParser::ExpressionContext * /*ctx*/) override { }
  virtual void exitExpression(SmalltalkParser::ExpressionContext * /*ctx*/) override { }

  virtual void enterExpressions(SmalltalkParser::ExpressionsContext * /*ctx*/) override { }
  virtual void exitExpressions(SmalltalkParser::ExpressionsContext * /*ctx*/) override { }

  virtual void enterExpressionList(SmalltalkParser::ExpressionListContext * /*ctx*/) override { }
  virtual void exitExpressionList(SmalltalkParser::ExpressionListContext * /*ctx*/) override { }

  virtual void enterCascade(SmalltalkParser::CascadeContext * /*ctx*/) override { }
  virtual void exitCascade(SmalltalkParser::CascadeContext * /*ctx*/) override { }

  virtual void enterMessage(SmalltalkParser::MessageContext * /*ctx*/) override { }
  virtual void exitMessage(SmalltalkParser::MessageContext * /*ctx*/) override { }

  virtual void enterAssignment(SmalltalkParser::AssignmentContext * /*ctx*/) override { }
  virtual void exitAssignment(SmalltalkParser::AssignmentContext * /*ctx*/) override { }

  virtual void enterVariable(SmalltalkParser::VariableContext * /*ctx*/) override { }
  virtual void exitVariable(SmalltalkParser::VariableContext * /*ctx*/) override { }

  virtual void enterBinarySend(SmalltalkParser::BinarySendContext * /*ctx*/) override { }
  virtual void exitBinarySend(SmalltalkParser::BinarySendContext * /*ctx*/) override { }

  virtual void enterUnarySend(SmalltalkParser::UnarySendContext * /*ctx*/) override { }
  virtual void exitUnarySend(SmalltalkParser::UnarySendContext * /*ctx*/) override { }

  virtual void enterKeywordSend(SmalltalkParser::KeywordSendContext * /*ctx*/) override { }
  virtual void exitKeywordSend(SmalltalkParser::KeywordSendContext * /*ctx*/) override { }

  virtual void enterKeywordMessage(SmalltalkParser::KeywordMessageContext * /*ctx*/) override { }
  virtual void exitKeywordMessage(SmalltalkParser::KeywordMessageContext * /*ctx*/) override { }

  virtual void enterKeywordPair(SmalltalkParser::KeywordPairContext * /*ctx*/) override { }
  virtual void exitKeywordPair(SmalltalkParser::KeywordPairContext * /*ctx*/) override { }

  virtual void enterOperand(SmalltalkParser::OperandContext * /*ctx*/) override { }
  virtual void exitOperand(SmalltalkParser::OperandContext * /*ctx*/) override { }

  virtual void enterSubexpression(SmalltalkParser::SubexpressionContext * /*ctx*/) override { }
  virtual void exitSubexpression(SmalltalkParser::SubexpressionContext * /*ctx*/) override { }

  virtual void enterLiteral(SmalltalkParser::LiteralContext * /*ctx*/) override { }
  virtual void exitLiteral(SmalltalkParser::LiteralContext * /*ctx*/) override { }

  virtual void enterRuntimeLiteral(SmalltalkParser::RuntimeLiteralContext * /*ctx*/) override { }
  virtual void exitRuntimeLiteral(SmalltalkParser::RuntimeLiteralContext * /*ctx*/) override { }

  virtual void enterBlock(SmalltalkParser::BlockContext * /*ctx*/) override { }
  virtual void exitBlock(SmalltalkParser::BlockContext * /*ctx*/) override { }

  virtual void enterBlockParamList(SmalltalkParser::BlockParamListContext * /*ctx*/) override { }
  virtual void exitBlockParamList(SmalltalkParser::BlockParamListContext * /*ctx*/) override { }

  virtual void enterDynamicDictionary(SmalltalkParser::DynamicDictionaryContext * /*ctx*/) override { }
  virtual void exitDynamicDictionary(SmalltalkParser::DynamicDictionaryContext * /*ctx*/) override { }

  virtual void enterDynamicArray(SmalltalkParser::DynamicArrayContext * /*ctx*/) override { }
  virtual void exitDynamicArray(SmalltalkParser::DynamicArrayContext * /*ctx*/) override { }

  virtual void enterParsetimeLiteral(SmalltalkParser::ParsetimeLiteralContext * /*ctx*/) override { }
  virtual void exitParsetimeLiteral(SmalltalkParser::ParsetimeLiteralContext * /*ctx*/) override { }

  virtual void enterNumber(SmalltalkParser::NumberContext * /*ctx*/) override { }
  virtual void exitNumber(SmalltalkParser::NumberContext * /*ctx*/) override { }

  virtual void enterNumberExp(SmalltalkParser::NumberExpContext * /*ctx*/) override { }
  virtual void exitNumberExp(SmalltalkParser::NumberExpContext * /*ctx*/) override { }

  virtual void enterCharConstant(SmalltalkParser::CharConstantContext * /*ctx*/) override { }
  virtual void exitCharConstant(SmalltalkParser::CharConstantContext * /*ctx*/) override { }

  virtual void enterHex_(SmalltalkParser::Hex_Context * /*ctx*/) override { }
  virtual void exitHex_(SmalltalkParser::Hex_Context * /*ctx*/) override { }

  virtual void enterStInteger(SmalltalkParser::StIntegerContext * /*ctx*/) override { }
  virtual void exitStInteger(SmalltalkParser::StIntegerContext * /*ctx*/) override { }

  virtual void enterStFloat(SmalltalkParser::StFloatContext * /*ctx*/) override { }
  virtual void exitStFloat(SmalltalkParser::StFloatContext * /*ctx*/) override { }

  virtual void enterPseudoVariable(SmalltalkParser::PseudoVariableContext * /*ctx*/) override { }
  virtual void exitPseudoVariable(SmalltalkParser::PseudoVariableContext * /*ctx*/) override { }

  virtual void enterString(SmalltalkParser::StringContext * /*ctx*/) override { }
  virtual void exitString(SmalltalkParser::StringContext * /*ctx*/) override { }

  virtual void enterPrimitive(SmalltalkParser::PrimitiveContext * /*ctx*/) override { }
  virtual void exitPrimitive(SmalltalkParser::PrimitiveContext * /*ctx*/) override { }

  virtual void enterSymbol(SmalltalkParser::SymbolContext * /*ctx*/) override { }
  virtual void exitSymbol(SmalltalkParser::SymbolContext * /*ctx*/) override { }

  virtual void enterBareSymbol(SmalltalkParser::BareSymbolContext * /*ctx*/) override { }
  virtual void exitBareSymbol(SmalltalkParser::BareSymbolContext * /*ctx*/) override { }

  virtual void enterLiteralArray(SmalltalkParser::LiteralArrayContext * /*ctx*/) override { }
  virtual void exitLiteralArray(SmalltalkParser::LiteralArrayContext * /*ctx*/) override { }

  virtual void enterLiteralArrayRest(SmalltalkParser::LiteralArrayRestContext * /*ctx*/) override { }
  virtual void exitLiteralArrayRest(SmalltalkParser::LiteralArrayRestContext * /*ctx*/) override { }

  virtual void enterByteLiteralArray(SmalltalkParser::ByteLiteralArrayContext * /*ctx*/) override { }
  virtual void exitByteLiteralArray(SmalltalkParser::ByteLiteralArrayContext * /*ctx*/) override { }

  virtual void enterByteLiteralArrayBody(SmalltalkParser::ByteLiteralArrayBodyContext * /*ctx*/) override { }
  virtual void exitByteLiteralArrayBody(SmalltalkParser::ByteLiteralArrayBodyContext * /*ctx*/) override { }

  virtual void enterBareLiteralArray(SmalltalkParser::BareLiteralArrayContext * /*ctx*/) override { }
  virtual void exitBareLiteralArray(SmalltalkParser::BareLiteralArrayContext * /*ctx*/) override { }

  virtual void enterUnaryTail(SmalltalkParser::UnaryTailContext * /*ctx*/) override { }
  virtual void exitUnaryTail(SmalltalkParser::UnaryTailContext * /*ctx*/) override { }

  virtual void enterUnaryMessage(SmalltalkParser::UnaryMessageContext * /*ctx*/) override { }
  virtual void exitUnaryMessage(SmalltalkParser::UnaryMessageContext * /*ctx*/) override { }

  virtual void enterUnarySelector(SmalltalkParser::UnarySelectorContext * /*ctx*/) override { }
  virtual void exitUnarySelector(SmalltalkParser::UnarySelectorContext * /*ctx*/) override { }

  virtual void enterKeywords(SmalltalkParser::KeywordsContext * /*ctx*/) override { }
  virtual void exitKeywords(SmalltalkParser::KeywordsContext * /*ctx*/) override { }

  virtual void enterReference(SmalltalkParser::ReferenceContext * /*ctx*/) override { }
  virtual void exitReference(SmalltalkParser::ReferenceContext * /*ctx*/) override { }

  virtual void enterBinaryTail(SmalltalkParser::BinaryTailContext * /*ctx*/) override { }
  virtual void exitBinaryTail(SmalltalkParser::BinaryTailContext * /*ctx*/) override { }

  virtual void enterBinaryMessage(SmalltalkParser::BinaryMessageContext * /*ctx*/) override { }
  virtual void exitBinaryMessage(SmalltalkParser::BinaryMessageContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

