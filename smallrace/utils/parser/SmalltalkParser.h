
// Generated from /home/aaron/smallrace/utils/parser/Smalltalk.g4 by ANTLR 4.9

#pragma once


#include "antlr4-runtime.h"




class  SmalltalkParser : public antlr4::Parser {
public:
  enum {
    SEPARATOR = 1, STRING = 2, PRIMITIVE = 3, COMMENT = 4, ST_COMMENT = 5, 
    BLOCK_START = 6, BLOCK_END = 7, CLOSE_PAREN = 8, OPEN_PAREN = 9, PIPE3 = 10, 
    PIPE2 = 11, PIPE = 12, PERIOD = 13, SEMI_COLON = 14, LLT = 15, GGT = 16, 
    LTE = 17, GTE = 18, NEQ = 19, LT = 20, GT = 21, AMP = 22, MINUS = 23, 
    BINARY_SELECTOR = 24, RESERVED_WORD = 25, IDENTIFIER = 26, IDENTIFIER2 = 27, 
    SPECIAL_UNDERLINE_IDENTIFIER = 28, CARROT = 29, UNDERSCORE = 30, UNDERSCORE2 = 31, 
    ASSIGNMENT = 32, COLON = 33, HASH = 34, DOLLAR = 35, EXP = 36, HEXNUM = 37, 
    INTNUM = 38, HEX = 39, DYNDICT_START = 40, DYNARR_END = 41, DYNARR_START = 42, 
    HEXDIGIT = 43, KEYWORD = 44, BLOCK_PARAM = 45, CHARACTER_CONSTANT = 46
  };

  enum {
    RuleModule = 0, RuleFunction = 1, RuleFuncdecl = 2, RuleDeclPairs = 3, 
    RuleDeclPair = 4, RuleScript = 5, RuleSequence = 6, RuleWs = 7, RuleTemps = 8, 
    RuleStatements = 9, RuleAnswer = 10, RuleExpression = 11, RuleExpressions = 12, 
    RuleExpressionList = 13, RuleCascade = 14, RuleMessage = 15, RuleAssignment = 16, 
    RuleVariable = 17, RuleBinarySend = 18, RuleUnarySend = 19, RuleKeywordSend = 20, 
    RuleKeywordMessage = 21, RuleKeywordPair = 22, RuleOperand = 23, RuleSubexpression = 24, 
    RuleLiteral = 25, RuleRuntimeLiteral = 26, RuleBlock = 27, RuleBlockParamList = 28, 
    RuleDynamicDictionary = 29, RuleDynamicArray = 30, RuleParsetimeLiteral = 31, 
    RuleNumber = 32, RuleNumberExp = 33, RuleCharConstant = 34, RuleHex_ = 35, 
    RuleStInteger = 36, RuleStFloat = 37, RulePseudoVariable = 38, RuleString = 39, 
    RulePrimitive = 40, RuleSymbol = 41, RuleBareSymbol = 42, RuleLiteralArray = 43, 
    RuleLiteralArrayRest = 44, RuleByteLiteralArray = 45, RuleByteLiteralArrayBody = 46, 
    RuleBareLiteralArray = 47, RuleUnaryTail = 48, RuleUnaryMessage = 49, 
    RuleUnarySelector = 50, RuleKeywords = 51, RuleReference = 52, RuleBinaryTail = 53, 
    RuleBinaryMessage = 54
  };

  explicit SmalltalkParser(antlr4::TokenStream *input);
  ~SmalltalkParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  class ModuleContext;
  class FunctionContext;
  class FuncdeclContext;
  class DeclPairsContext;
  class DeclPairContext;
  class ScriptContext;
  class SequenceContext;
  class WsContext;
  class TempsContext;
  class StatementsContext;
  class AnswerContext;
  class ExpressionContext;
  class ExpressionsContext;
  class ExpressionListContext;
  class CascadeContext;
  class MessageContext;
  class AssignmentContext;
  class VariableContext;
  class BinarySendContext;
  class UnarySendContext;
  class KeywordSendContext;
  class KeywordMessageContext;
  class KeywordPairContext;
  class OperandContext;
  class SubexpressionContext;
  class LiteralContext;
  class RuntimeLiteralContext;
  class BlockContext;
  class BlockParamListContext;
  class DynamicDictionaryContext;
  class DynamicArrayContext;
  class ParsetimeLiteralContext;
  class NumberContext;
  class NumberExpContext;
  class CharConstantContext;
  class Hex_Context;
  class StIntegerContext;
  class StFloatContext;
  class PseudoVariableContext;
  class StringContext;
  class PrimitiveContext;
  class SymbolContext;
  class BareSymbolContext;
  class LiteralArrayContext;
  class LiteralArrayRestContext;
  class ByteLiteralArrayContext;
  class ByteLiteralArrayBodyContext;
  class BareLiteralArrayContext;
  class UnaryTailContext;
  class UnaryMessageContext;
  class UnarySelectorContext;
  class KeywordsContext;
  class ReferenceContext;
  class BinaryTailContext;
  class BinaryMessageContext; 

  class  ModuleContext : public antlr4::ParserRuleContext {
  public:
    ModuleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FunctionContext *function();
    ScriptContext *script();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ModuleContext* module();

  class  FunctionContext : public antlr4::ParserRuleContext {
  public:
    FunctionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FuncdeclContext *funcdecl();
    antlr4::tree::TerminalNode *EOF();
    ScriptContext *script();
    WsContext *ws();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  FunctionContext* function();

  class  FuncdeclContext : public antlr4::ParserRuleContext {
  public:
    FuncdeclContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();
    WsContext *ws();
    DeclPairsContext *declPairs();
    VariableContext *variable();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *PIPE();
    antlr4::tree::TerminalNode *PIPE2();
    antlr4::tree::TerminalNode *PIPE3();
    antlr4::tree::TerminalNode *AMP();
    antlr4::tree::TerminalNode *BINARY_SELECTOR();
    antlr4::tree::TerminalNode *NEQ();
    antlr4::tree::TerminalNode *LLT();
    antlr4::tree::TerminalNode *GGT();
    antlr4::tree::TerminalNode *LTE();
    antlr4::tree::TerminalNode *GTE();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  FuncdeclContext* funcdecl();

  class  DeclPairsContext : public antlr4::ParserRuleContext {
  public:
    DeclPairsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    std::vector<DeclPairContext *> declPair();
    DeclPairContext* declPair(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DeclPairsContext* declPairs();

  class  DeclPairContext : public antlr4::ParserRuleContext {
  public:
    DeclPairContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KEYWORD();
    VariableContext *variable();
    WsContext *ws();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DeclPairContext* declPair();

  class  ScriptContext : public antlr4::ParserRuleContext {
  public:
    ScriptContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SequenceContext *sequence();
    antlr4::tree::TerminalNode *EOF();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    std::vector<PrimitiveContext *> primitive();
    PrimitiveContext* primitive(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ScriptContext* script();

  class  SequenceContext : public antlr4::ParserRuleContext {
  public:
    SequenceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TempsContext *temps();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    PrimitiveContext *primitive();
    StatementsContext *statements();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SequenceContext* sequence();

  class  WsContext : public antlr4::ParserRuleContext {
  public:
    WsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> SEPARATOR();
    antlr4::tree::TerminalNode* SEPARATOR(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMENT();
    antlr4::tree::TerminalNode* COMMENT(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  WsContext* ws();

  class  TempsContext : public antlr4::ParserRuleContext {
  public:
    TempsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> PIPE();
    antlr4::tree::TerminalNode* PIPE(size_t i);
    std::vector<antlr4::tree::TerminalNode *> PIPE2();
    antlr4::tree::TerminalNode* PIPE2(size_t i);
    std::vector<antlr4::tree::TerminalNode *> IDENTIFIER();
    antlr4::tree::TerminalNode* IDENTIFIER(size_t i);
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TempsContext* temps();

  class  StatementsContext : public antlr4::ParserRuleContext {
  public:
    StatementsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    AnswerContext *answer();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    ExpressionsContext *expressions();
    antlr4::tree::TerminalNode *PERIOD();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StatementsContext* statements();

  class  AnswerContext : public antlr4::ParserRuleContext {
  public:
    AnswerContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CARROT();
    ExpressionContext *expression();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    antlr4::tree::TerminalNode *PERIOD();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AnswerContext* answer();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    AssignmentContext *assignment();
    CascadeContext *cascade();
    KeywordSendContext *keywordSend();
    BinarySendContext *binarySend();
    PrimitiveContext *primitive();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    antlr4::tree::TerminalNode *PERIOD();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionContext* expression();

  class  ExpressionsContext : public antlr4::ParserRuleContext {
  public:
    ExpressionsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();
    std::vector<ExpressionListContext *> expressionList();
    ExpressionListContext* expressionList(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionsContext* expressions();

  class  ExpressionListContext : public antlr4::ParserRuleContext {
  public:
    ExpressionListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *PERIOD();
    ExpressionContext *expression();
    WsContext *ws();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionListContext* expressionList();

  class  CascadeContext : public antlr4::ParserRuleContext {
  public:
    CascadeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    KeywordSendContext *keywordSend();
    BinarySendContext *binarySend();
    std::vector<antlr4::tree::TerminalNode *> SEMI_COLON();
    antlr4::tree::TerminalNode* SEMI_COLON(size_t i);
    std::vector<MessageContext *> message();
    MessageContext* message(size_t i);
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CascadeContext* cascade();

  class  MessageContext : public antlr4::ParserRuleContext {
  public:
    MessageContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BinaryMessageContext *binaryMessage();
    UnaryMessageContext *unaryMessage();
    KeywordMessageContext *keywordMessage();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  MessageContext* message();

  class  AssignmentContext : public antlr4::ParserRuleContext {
  public:
    AssignmentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VariableContext *variable();
    antlr4::tree::TerminalNode *ASSIGNMENT();
    ExpressionContext *expression();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AssignmentContext* assignment();

  class  VariableContext : public antlr4::ParserRuleContext {
  public:
    VariableContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();
    antlr4::tree::TerminalNode *IDENTIFIER2();
    antlr4::tree::TerminalNode *UNDERSCORE();
    antlr4::tree::TerminalNode *UNDERSCORE2();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  VariableContext* variable();

  class  BinarySendContext : public antlr4::ParserRuleContext {
  public:
    BinarySendContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    UnarySendContext *unarySend();
    BinaryTailContext *binaryTail();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BinarySendContext* binarySend();

  class  UnarySendContext : public antlr4::ParserRuleContext {
  public:
    UnarySendContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    OperandContext *operand();
    WsContext *ws();
    UnaryTailContext *unaryTail();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  UnarySendContext* unarySend();

  class  KeywordSendContext : public antlr4::ParserRuleContext {
  public:
    KeywordSendContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BinarySendContext *binarySend();
    KeywordMessageContext *keywordMessage();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  KeywordSendContext* keywordSend();

  class  KeywordMessageContext : public antlr4::ParserRuleContext {
  public:
    KeywordMessageContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    std::vector<KeywordPairContext *> keywordPair();
    KeywordPairContext* keywordPair(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  KeywordMessageContext* keywordMessage();

  class  KeywordPairContext : public antlr4::ParserRuleContext {
  public:
    KeywordPairContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KEYWORD();
    BinarySendContext *binarySend();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  KeywordPairContext* keywordPair();

  class  OperandContext : public antlr4::ParserRuleContext {
  public:
    OperandContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LiteralContext *literal();
    ReferenceContext *reference();
    SubexpressionContext *subexpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OperandContext* operand();

  class  SubexpressionContext : public antlr4::ParserRuleContext {
  public:
    SubexpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *OPEN_PAREN();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *CLOSE_PAREN();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SubexpressionContext* subexpression();

  class  LiteralContext : public antlr4::ParserRuleContext {
  public:
    LiteralContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    RuntimeLiteralContext *runtimeLiteral();
    ParsetimeLiteralContext *parsetimeLiteral();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LiteralContext* literal();

  class  RuntimeLiteralContext : public antlr4::ParserRuleContext {
  public:
    RuntimeLiteralContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DynamicDictionaryContext *dynamicDictionary();
    DynamicArrayContext *dynamicArray();
    BlockContext *block();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RuntimeLiteralContext* runtimeLiteral();

  class  BlockContext : public antlr4::ParserRuleContext {
  public:
    BlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *BLOCK_START();
    antlr4::tree::TerminalNode *BLOCK_END();
    antlr4::tree::TerminalNode *PIPE3();
    BlockParamListContext *blockParamList();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    SequenceContext *sequence();
    antlr4::tree::TerminalNode *PIPE();
    TempsContext *temps();
    antlr4::tree::TerminalNode *PIPE2();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BlockContext* block();

  class  BlockParamListContext : public antlr4::ParserRuleContext {
  public:
    BlockParamListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> BLOCK_PARAM();
    antlr4::tree::TerminalNode* BLOCK_PARAM(size_t i);
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BlockParamListContext* blockParamList();

  class  DynamicDictionaryContext : public antlr4::ParserRuleContext {
  public:
    DynamicDictionaryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DYNDICT_START();
    antlr4::tree::TerminalNode *DYNARR_END();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    ExpressionsContext *expressions();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DynamicDictionaryContext* dynamicDictionary();

  class  DynamicArrayContext : public antlr4::ParserRuleContext {
  public:
    DynamicArrayContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DYNARR_START();
    antlr4::tree::TerminalNode *DYNARR_END();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    ExpressionsContext *expressions();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DynamicArrayContext* dynamicArray();

  class  ParsetimeLiteralContext : public antlr4::ParserRuleContext {
  public:
    ParsetimeLiteralContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PseudoVariableContext *pseudoVariable();
    NumberContext *number();
    CharConstantContext *charConstant();
    LiteralArrayContext *literalArray();
    StringContext *string();
    SymbolContext *symbol();
    ByteLiteralArrayContext *byteLiteralArray();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ParsetimeLiteralContext* parsetimeLiteral();

  class  NumberContext : public antlr4::ParserRuleContext {
  public:
    NumberContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    NumberExpContext *numberExp();
    Hex_Context *hex_();
    StFloatContext *stFloat();
    StIntegerContext *stInteger();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  NumberContext* number();

  class  NumberExpContext : public antlr4::ParserRuleContext {
  public:
    NumberExpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EXP();
    std::vector<StIntegerContext *> stInteger();
    StIntegerContext* stInteger(size_t i);
    StFloatContext *stFloat();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  NumberExpContext* numberExp();

  class  CharConstantContext : public antlr4::ParserRuleContext {
  public:
    CharConstantContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CHARACTER_CONSTANT();
    antlr4::tree::TerminalNode *DOLLAR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CharConstantContext* charConstant();

  class  Hex_Context : public antlr4::ParserRuleContext {
  public:
    Hex_Context(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *HEXNUM();
    antlr4::tree::TerminalNode *MINUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Hex_Context* hex_();

  class  StIntegerContext : public antlr4::ParserRuleContext {
  public:
    StIntegerContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INTNUM();
    antlr4::tree::TerminalNode *MINUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StIntegerContext* stInteger();

  class  StFloatContext : public antlr4::ParserRuleContext {
  public:
    StFloatContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> INTNUM();
    antlr4::tree::TerminalNode* INTNUM(size_t i);
    antlr4::tree::TerminalNode *PERIOD();
    antlr4::tree::TerminalNode *MINUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StFloatContext* stFloat();

  class  PseudoVariableContext : public antlr4::ParserRuleContext {
  public:
    PseudoVariableContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RESERVED_WORD();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PseudoVariableContext* pseudoVariable();

  class  StringContext : public antlr4::ParserRuleContext {
  public:
    StringContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STRING();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StringContext* string();

  class  PrimitiveContext : public antlr4::ParserRuleContext {
  public:
    PrimitiveContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *PRIMITIVE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PrimitiveContext* primitive();

  class  SymbolContext : public antlr4::ParserRuleContext {
  public:
    SymbolContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *HASH();
    BareSymbolContext *bareSymbol();
    WsContext *ws();
    antlr4::tree::TerminalNode *SPECIAL_UNDERLINE_IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SymbolContext* symbol();

  class  BareSymbolContext : public antlr4::ParserRuleContext {
  public:
    BareSymbolContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();
    antlr4::tree::TerminalNode *UNDERSCORE();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *PIPE();
    antlr4::tree::TerminalNode *PIPE2();
    antlr4::tree::TerminalNode *PIPE3();
    antlr4::tree::TerminalNode *AMP();
    antlr4::tree::TerminalNode *BINARY_SELECTOR();
    antlr4::tree::TerminalNode *NEQ();
    antlr4::tree::TerminalNode *LTE();
    antlr4::tree::TerminalNode *GTE();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *GGT();
    antlr4::tree::TerminalNode *LLT();
    antlr4::tree::TerminalNode *OPEN_PAREN();
    antlr4::tree::TerminalNode *CLOSE_PAREN();
    std::vector<antlr4::tree::TerminalNode *> KEYWORD();
    antlr4::tree::TerminalNode* KEYWORD(size_t i);
    StringContext *string();
    antlr4::tree::TerminalNode *RESERVED_WORD();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BareSymbolContext* bareSymbol();

  class  LiteralArrayContext : public antlr4::ParserRuleContext {
  public:
    LiteralArrayContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LiteralArrayRestContext *literalArrayRest();
    antlr4::tree::TerminalNode *HASH();
    antlr4::tree::TerminalNode *OPEN_PAREN();
    antlr4::tree::TerminalNode *BLOCK_START();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    DynamicDictionaryContext *dynamicDictionary();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LiteralArrayContext* literalArray();

  class  LiteralArrayRestContext : public antlr4::ParserRuleContext {
  public:
    LiteralArrayRestContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CLOSE_PAREN();
    antlr4::tree::TerminalNode *BLOCK_END();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    std::vector<LiteralContext *> literal();
    LiteralContext* literal(size_t i);
    std::vector<BareLiteralArrayContext *> bareLiteralArray();
    BareLiteralArrayContext* bareLiteralArray(size_t i);
    std::vector<BareSymbolContext *> bareSymbol();
    BareSymbolContext* bareSymbol(size_t i);
    std::vector<ByteLiteralArrayBodyContext *> byteLiteralArrayBody();
    ByteLiteralArrayBodyContext* byteLiteralArrayBody(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LiteralArrayRestContext* literalArrayRest();

  class  ByteLiteralArrayContext : public antlr4::ParserRuleContext {
  public:
    ByteLiteralArrayContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *HASH();
    ByteLiteralArrayBodyContext *byteLiteralArrayBody();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ByteLiteralArrayContext* byteLiteralArray();

  class  ByteLiteralArrayBodyContext : public antlr4::ParserRuleContext {
  public:
    ByteLiteralArrayBodyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *BLOCK_START();
    antlr4::tree::TerminalNode *BLOCK_END();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    std::vector<antlr4::tree::TerminalNode *> INTNUM();
    antlr4::tree::TerminalNode* INTNUM(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ByteLiteralArrayBodyContext* byteLiteralArrayBody();

  class  BareLiteralArrayContext : public antlr4::ParserRuleContext {
  public:
    BareLiteralArrayContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *OPEN_PAREN();
    LiteralArrayRestContext *literalArrayRest();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BareLiteralArrayContext* bareLiteralArray();

  class  UnaryTailContext : public antlr4::ParserRuleContext {
  public:
    UnaryTailContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    UnaryMessageContext *unaryMessage();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);
    UnaryTailContext *unaryTail();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  UnaryTailContext* unaryTail();

  class  UnaryMessageContext : public antlr4::ParserRuleContext {
  public:
    UnaryMessageContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    UnarySelectorContext *unarySelector();
    WsContext *ws();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  UnaryMessageContext* unaryMessage();

  class  UnarySelectorContext : public antlr4::ParserRuleContext {
  public:
    UnarySelectorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  UnarySelectorContext* unarySelector();

  class  KeywordsContext : public antlr4::ParserRuleContext {
  public:
    KeywordsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> KEYWORD();
    antlr4::tree::TerminalNode* KEYWORD(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  KeywordsContext* keywords();

  class  ReferenceContext : public antlr4::ParserRuleContext {
  public:
    ReferenceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VariableContext *variable();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ReferenceContext* reference();

  class  BinaryTailContext : public antlr4::ParserRuleContext {
  public:
    BinaryTailContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BinaryMessageContext *binaryMessage();
    BinaryTailContext *binaryTail();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BinaryTailContext* binaryTail();

  class  BinaryMessageContext : public antlr4::ParserRuleContext {
  public:
    BinaryMessageContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *PIPE();
    antlr4::tree::TerminalNode *PIPE2();
    antlr4::tree::TerminalNode *PIPE3();
    antlr4::tree::TerminalNode *AMP();
    antlr4::tree::TerminalNode *BINARY_SELECTOR();
    antlr4::tree::TerminalNode *NEQ();
    antlr4::tree::TerminalNode *LLT();
    antlr4::tree::TerminalNode *GGT();
    antlr4::tree::TerminalNode *LTE();
    antlr4::tree::TerminalNode *GTE();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GT();
    UnarySendContext *unarySend();
    OperandContext *operand();
    std::vector<WsContext *> ws();
    WsContext* ws(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BinaryMessageContext* binaryMessage();


private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

