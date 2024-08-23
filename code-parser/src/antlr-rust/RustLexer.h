
// Generated from RustLexer.g4 by ANTLR 4.13.1

#pragma once

#include "antlr4-runtime.h"

class RustLexer : public antlr4::Lexer {
public:
  enum {
    KW_AS = 1,
    KW_BREAK = 2,
    KW_CONST = 3,
    KW_CONTINUE = 4,
    KW_CRATE = 5,
    KW_ELSE = 6,
    KW_ENUM = 7,
    KW_EXTERN = 8,
    KW_FALSE = 9,
    KW_FN = 10,
    KW_FOR = 11,
    KW_IF = 12,
    KW_IMPL = 13,
    KW_IN = 14,
    KW_LET = 15,
    KW_LOOP = 16,
    KW_MATCH = 17,
    KW_MOD = 18,
    KW_MOVE = 19,
    KW_MUT = 20,
    KW_PUB = 21,
    KW_REF = 22,
    KW_RETURN = 23,
    KW_SELFVALUE = 24,
    KW_SELFTYPE = 25,
    KW_STATIC = 26,
    KW_STRUCT = 27,
    KW_SUPER = 28,
    KW_TRAIT = 29,
    KW_TRUE = 30,
    KW_TYPE = 31,
    KW_UNSAFE = 32,
    KW_USE = 33,
    KW_WHERE = 34,
    KW_WHILE = 35,
    KW_ASYNC = 36,
    KW_AWAIT = 37,
    KW_DYN = 38,
    KW_ABSTRACT = 39,
    KW_BECOME = 40,
    KW_BOX = 41,
    KW_DO = 42,
    KW_FINAL = 43,
    KW_MACRO = 44,
    KW_OVERRIDE = 45,
    KW_PRIV = 46,
    KW_TYPEOF = 47,
    KW_UNSIZED = 48,
    KW_VIRTUAL = 49,
    KW_YIELD = 50,
    KW_TRY = 51,
    KW_UNION = 52,
    KW_STATICLIFETIME = 53,
    KW_MACRORULES = 54,
    KW_UNDERLINELIFETIME = 55,
    KW_DOLLARCRATE = 56,
    NON_KEYWORD_IDENTIFIER = 57,
    RAW_IDENTIFIER = 58,
    LINE_COMMENT = 59,
    BLOCK_COMMENT = 60,
    DOC_BLOCK_COMMENT = 61,
    INNER_LINE_DOC = 62,
    INNER_BLOCK_DOC = 63,
    OUTER_LINE_DOC = 64,
    OUTER_BLOCK_DOC = 65,
    BLOCK_COMMENT_OR_DOC = 66,
    SHEBANG = 67,
    WHITESPACE = 68,
    NEWLINE = 69,
    CHAR_LITERAL = 70,
    STRING_LITERAL = 71,
    RAW_STRING_LITERAL = 72,
    BYTE_LITERAL = 73,
    BYTE_STRING_LITERAL = 74,
    RAW_BYTE_STRING_LITERAL = 75,
    INTEGER_LITERAL = 76,
    DEC_LITERAL = 77,
    HEX_LITERAL = 78,
    OCT_LITERAL = 79,
    BIN_LITERAL = 80,
    FLOAT_LITERAL = 81,
    LIFETIME_OR_LABEL = 82,
    PLUS = 83,
    MINUS = 84,
    STAR = 85,
    SLASH = 86,
    PERCENT = 87,
    CARET = 88,
    NOT = 89,
    AND = 90,
    OR = 91,
    ANDAND = 92,
    OROR = 93,
    PLUSEQ = 94,
    MINUSEQ = 95,
    STAREQ = 96,
    SLASHEQ = 97,
    PERCENTEQ = 98,
    CARETEQ = 99,
    ANDEQ = 100,
    OREQ = 101,
    SHLEQ = 102,
    SHREQ = 103,
    EQ = 104,
    EQEQ = 105,
    NE = 106,
    GT = 107,
    LT = 108,
    GE = 109,
    LE = 110,
    AT = 111,
    UNDERSCORE = 112,
    DOT = 113,
    DOTDOT = 114,
    DOTDOTDOT = 115,
    DOTDOTEQ = 116,
    COMMA = 117,
    SEMI = 118,
    COLON = 119,
    PATHSEP = 120,
    RARROW = 121,
    FATARROW = 122,
    POUND = 123,
    DOLLAR = 124,
    QUESTION = 125,
    LCURLYBRACE = 126,
    RCURLYBRACE = 127,
    LSQUAREBRACKET = 128,
    RSQUAREBRACKET = 129,
    LPAREN = 130,
    RPAREN = 131
  };

  size_t lt1;
  size_t lt2;

  std::unique_ptr<antlr4::Token> nextToken() override {
    std::unique_ptr<antlr4::Token> next = antlr4::Lexer::nextToken();

    if (next->getChannel() == antlr4::Token::DEFAULT_CHANNEL) {
      // Keep track of the last token on the default channel.
      this->lt2 = this->lt1;
      this->lt1 = next->getType();
    }
    return next;
  }

  bool SOF() {
    size_t next = _input->LA(-1);
    return next == 0 || next == antlr4::Token::EOF;
  }

  bool next(char expect) { return _input->LA(1) == expect; }

  bool floatDotPossible() {
    size_t next = _input->LA(1);
    // only block . _ identifier after float
    if (next == '.' || next == '_')
      return false;
    if (next == 'f') {
      // 1.f32
      if (_input->LA(2) == '3' && _input->LA(3) == '2')
        return true;
      // 1.f64
      if (_input->LA(2) == '6' && _input->LA(3) == '4')
        return true;
      return false;
    }
    if (next >= 'a' && next <= 'z')
      return false;
    if (next >= 'A' && next <= 'Z')
      return false;
    return true;
  }

  bool floatLiteralPossible() {
    if (this->lt1 == antlr4::Token::INVALID_TYPE ||
        this->lt2 == antlr4::Token::INVALID_TYPE)
      return true;
    if (this->lt1 != RustLexer::DOT)
      return true;
    switch (this->lt2) {
    case RustLexer::CHAR_LITERAL:
    case RustLexer::STRING_LITERAL:
    case RustLexer::RAW_STRING_LITERAL:
    case RustLexer::BYTE_LITERAL:
    case RustLexer::BYTE_STRING_LITERAL:
    case RustLexer::RAW_BYTE_STRING_LITERAL:
    case RustLexer::INTEGER_LITERAL:
    case RustLexer::DEC_LITERAL:
    case RustLexer::HEX_LITERAL:
    case RustLexer::OCT_LITERAL:
    case RustLexer::BIN_LITERAL:

    case RustLexer::KW_SUPER:
    case RustLexer::KW_SELFVALUE:
    case RustLexer::KW_SELFTYPE:
    case RustLexer::KW_CRATE:
    case RustLexer::KW_DOLLARCRATE:

    case RustLexer::GT:
    case RustLexer::RCURLYBRACE:
    case RustLexer::RSQUAREBRACKET:
    case RustLexer::RPAREN:

    case RustLexer::KW_AWAIT:

    case RustLexer::NON_KEYWORD_IDENTIFIER:
    case RustLexer::RAW_IDENTIFIER:
    case RustLexer::KW_MACRORULES:
      return false;
    default:
      return true;
    }
  }

  explicit RustLexer(antlr4::CharStream *input);

  ~RustLexer() override;

  std::string getGrammarFileName() const override;

  const std::vector<std::string> &getRuleNames() const override;

  const std::vector<std::string> &getChannelNames() const override;

  const std::vector<std::string> &getModeNames() const override;

  const antlr4::dfa::Vocabulary &getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN &getATN() const override;

  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex,
               size_t predicateIndex) override;

  // By default the static state used to implement the lexer is lazily
  // initialized during the first call to the constructor. You can call this
  // function if you wish to initialize the static state ahead of time.
  static void initialize();

private:
  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.
  bool SHEBANGSempred(antlr4::RuleContext *_localctx, size_t predicateIndex);
  bool FLOAT_LITERALSempred(antlr4::RuleContext *_localctx,
                            size_t predicateIndex);
};
