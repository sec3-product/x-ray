
// Generated from /home/aaron/smallrace/utils/parser/Smalltalk.g4 by ANTLR 4.9

#pragma once


#include "antlr4-runtime.h"




class  SmalltalkLexer : public antlr4::Lexer {
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

  explicit SmalltalkLexer(antlr4::CharStream *input);
  ~SmalltalkLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames() const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

  virtual bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.
  bool COLONSempred(antlr4::RuleContext *_localctx, size_t predicateIndex);

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

