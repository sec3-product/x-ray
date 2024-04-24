
// Generated from /home/aaron/smallrace/utils/parser/Smalltalk.g4 by ANTLR 4.9

#pragma once


#include "antlr4-runtime.h"
#include "SmalltalkParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by SmalltalkParser.
 */
class  SmalltalkVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by SmalltalkParser.
   */
    virtual antlrcpp::Any visitModule(SmalltalkParser::ModuleContext *context) = 0;

    virtual antlrcpp::Any visitFunction(SmalltalkParser::FunctionContext *context) = 0;

    virtual antlrcpp::Any visitFuncdecl(SmalltalkParser::FuncdeclContext *context) = 0;

    virtual antlrcpp::Any visitDeclPairs(SmalltalkParser::DeclPairsContext *context) = 0;

    virtual antlrcpp::Any visitDeclPair(SmalltalkParser::DeclPairContext *context) = 0;

    virtual antlrcpp::Any visitScript(SmalltalkParser::ScriptContext *context) = 0;

    virtual antlrcpp::Any visitSequence(SmalltalkParser::SequenceContext *context) = 0;

    virtual antlrcpp::Any visitWs(SmalltalkParser::WsContext *context) = 0;

    virtual antlrcpp::Any visitTemps(SmalltalkParser::TempsContext *context) = 0;

    virtual antlrcpp::Any visitStatements(SmalltalkParser::StatementsContext *context) = 0;

    virtual antlrcpp::Any visitAnswer(SmalltalkParser::AnswerContext *context) = 0;

    virtual antlrcpp::Any visitExpression(SmalltalkParser::ExpressionContext *context) = 0;

    virtual antlrcpp::Any visitExpressions(SmalltalkParser::ExpressionsContext *context) = 0;

    virtual antlrcpp::Any visitExpressionList(SmalltalkParser::ExpressionListContext *context) = 0;

    virtual antlrcpp::Any visitCascade(SmalltalkParser::CascadeContext *context) = 0;

    virtual antlrcpp::Any visitMessage(SmalltalkParser::MessageContext *context) = 0;

    virtual antlrcpp::Any visitAssignment(SmalltalkParser::AssignmentContext *context) = 0;

    virtual antlrcpp::Any visitVariable(SmalltalkParser::VariableContext *context) = 0;

    virtual antlrcpp::Any visitBinarySend(SmalltalkParser::BinarySendContext *context) = 0;

    virtual antlrcpp::Any visitUnarySend(SmalltalkParser::UnarySendContext *context) = 0;

    virtual antlrcpp::Any visitKeywordSend(SmalltalkParser::KeywordSendContext *context) = 0;

    virtual antlrcpp::Any visitKeywordMessage(SmalltalkParser::KeywordMessageContext *context) = 0;

    virtual antlrcpp::Any visitKeywordPair(SmalltalkParser::KeywordPairContext *context) = 0;

    virtual antlrcpp::Any visitOperand(SmalltalkParser::OperandContext *context) = 0;

    virtual antlrcpp::Any visitSubexpression(SmalltalkParser::SubexpressionContext *context) = 0;

    virtual antlrcpp::Any visitLiteral(SmalltalkParser::LiteralContext *context) = 0;

    virtual antlrcpp::Any visitRuntimeLiteral(SmalltalkParser::RuntimeLiteralContext *context) = 0;

    virtual antlrcpp::Any visitBlock(SmalltalkParser::BlockContext *context) = 0;

    virtual antlrcpp::Any visitBlockParamList(SmalltalkParser::BlockParamListContext *context) = 0;

    virtual antlrcpp::Any visitDynamicDictionary(SmalltalkParser::DynamicDictionaryContext *context) = 0;

    virtual antlrcpp::Any visitDynamicArray(SmalltalkParser::DynamicArrayContext *context) = 0;

    virtual antlrcpp::Any visitParsetimeLiteral(SmalltalkParser::ParsetimeLiteralContext *context) = 0;

    virtual antlrcpp::Any visitNumber(SmalltalkParser::NumberContext *context) = 0;

    virtual antlrcpp::Any visitNumberExp(SmalltalkParser::NumberExpContext *context) = 0;

    virtual antlrcpp::Any visitCharConstant(SmalltalkParser::CharConstantContext *context) = 0;

    virtual antlrcpp::Any visitHex_(SmalltalkParser::Hex_Context *context) = 0;

    virtual antlrcpp::Any visitStInteger(SmalltalkParser::StIntegerContext *context) = 0;

    virtual antlrcpp::Any visitStFloat(SmalltalkParser::StFloatContext *context) = 0;

    virtual antlrcpp::Any visitPseudoVariable(SmalltalkParser::PseudoVariableContext *context) = 0;

    virtual antlrcpp::Any visitString(SmalltalkParser::StringContext *context) = 0;

    virtual antlrcpp::Any visitPrimitive(SmalltalkParser::PrimitiveContext *context) = 0;

    virtual antlrcpp::Any visitSymbol(SmalltalkParser::SymbolContext *context) = 0;

    virtual antlrcpp::Any visitBareSymbol(SmalltalkParser::BareSymbolContext *context) = 0;

    virtual antlrcpp::Any visitLiteralArray(SmalltalkParser::LiteralArrayContext *context) = 0;

    virtual antlrcpp::Any visitLiteralArrayRest(SmalltalkParser::LiteralArrayRestContext *context) = 0;

    virtual antlrcpp::Any visitByteLiteralArray(SmalltalkParser::ByteLiteralArrayContext *context) = 0;

    virtual antlrcpp::Any visitByteLiteralArrayBody(SmalltalkParser::ByteLiteralArrayBodyContext *context) = 0;

    virtual antlrcpp::Any visitBareLiteralArray(SmalltalkParser::BareLiteralArrayContext *context) = 0;

    virtual antlrcpp::Any visitUnaryTail(SmalltalkParser::UnaryTailContext *context) = 0;

    virtual antlrcpp::Any visitUnaryMessage(SmalltalkParser::UnaryMessageContext *context) = 0;

    virtual antlrcpp::Any visitUnarySelector(SmalltalkParser::UnarySelectorContext *context) = 0;

    virtual antlrcpp::Any visitKeywords(SmalltalkParser::KeywordsContext *context) = 0;

    virtual antlrcpp::Any visitReference(SmalltalkParser::ReferenceContext *context) = 0;

    virtual antlrcpp::Any visitBinaryTail(SmalltalkParser::BinaryTailContext *context) = 0;

    virtual antlrcpp::Any visitBinaryMessage(SmalltalkParser::BinaryMessageContext *context) = 0;


};

