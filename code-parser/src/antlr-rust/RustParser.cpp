
// Generated from RustParser.g4 by ANTLR 4.13.1

#include "RustParserListener.h"
#include "RustParserVisitor.h"

#include "RustParser.h"

using namespace antlrcpp;

using namespace antlr4;

namespace {

struct RustParserStaticData final {
  RustParserStaticData(std::vector<std::string> ruleNames,
                       std::vector<std::string> literalNames,
                       std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  RustParserStaticData(const RustParserStaticData &) = delete;
  RustParserStaticData(RustParserStaticData &&) = delete;
  RustParserStaticData &operator=(const RustParserStaticData &) = delete;
  RustParserStaticData &operator=(RustParserStaticData &&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag rustparserParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
    RustParserStaticData *rustparserParserStaticData = nullptr;

void rustparserParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (rustparserParserStaticData != nullptr) {
    return;
  }
#else
  assert(rustparserParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<RustParserStaticData>(
      std::vector<std::string>{"crate",
                               "macroInvocation",
                               "delimTokenTree",
                               "tokenTree",
                               "tokenTreeToken",
                               "macroInvocationSemi",
                               "macroRulesDefinition",
                               "macroRulesDef",
                               "macroRules",
                               "macroRule",
                               "macroMatcher",
                               "macroMatch",
                               "macroMatchToken",
                               "macroFragSpec",
                               "macroRepSep",
                               "macroRepOp",
                               "macroTranscriber",
                               "item",
                               "visItem",
                               "macroItem",
                               "module",
                               "externCrate",
                               "crateRef",
                               "asClause",
                               "useDeclaration",
                               "useTree",
                               "function_",
                               "functionQualifiers",
                               "abi",
                               "functionParameters",
                               "selfParam",
                               "shorthandSelf",
                               "typedSelf",
                               "functionParam",
                               "functionParamPattern",
                               "functionReturnType",
                               "typeAlias",
                               "struct_",
                               "structStruct",
                               "tupleStruct",
                               "structFields",
                               "structField",
                               "tupleFields",
                               "tupleField",
                               "enumeration",
                               "enumItems",
                               "enumItem",
                               "enumItemTuple",
                               "enumItemStruct",
                               "enumItemDiscriminant",
                               "union_",
                               "constantItem",
                               "staticItem",
                               "trait_",
                               "implementation",
                               "inherentImpl",
                               "traitImpl",
                               "externBlock",
                               "externalItem",
                               "genericParams",
                               "genericParam",
                               "lifetimeParam",
                               "typeParam",
                               "constParam",
                               "whereClause",
                               "whereClauseItem",
                               "lifetimeWhereClauseItem",
                               "typeBoundWhereClauseItem",
                               "forLifetimes",
                               "associatedItem",
                               "innerAttribute",
                               "outerAttribute",
                               "attr",
                               "attrInput",
                               "statement",
                               "letStatement",
                               "expressionStatement",
                               "expression",
                               "comparisonOperator",
                               "compoundAssignOperator",
                               "expressionWithBlock",
                               "literalExpression",
                               "pathExpression",
                               "blockExpression",
                               "statements",
                               "asyncBlockExpression",
                               "unsafeBlockExpression",
                               "arrayElements",
                               "tupleElements",
                               "tupleIndex",
                               "structExpression",
                               "structExprStruct",
                               "structExprFields",
                               "structExprField",
                               "structBase",
                               "structExprTuple",
                               "structExprUnit",
                               "enumerationVariantExpression",
                               "enumExprStruct",
                               "enumExprFields",
                               "enumExprField",
                               "enumExprTuple",
                               "enumExprFieldless",
                               "callParams",
                               "closureExpression",
                               "closureParameters",
                               "closureParam",
                               "loopExpression",
                               "infiniteLoopExpression",
                               "predicateLoopExpression",
                               "predicatePatternLoopExpression",
                               "iteratorLoopExpression",
                               "loopLabel",
                               "ifExpression",
                               "ifLetExpression",
                               "matchExpression",
                               "matchArms",
                               "matchArmExpression",
                               "matchArm",
                               "matchArmGuard",
                               "pattern",
                               "patternNoTopAlt",
                               "patternWithoutRange",
                               "literalPattern",
                               "identifierPattern",
                               "wildcardPattern",
                               "restPattern",
                               "rangePattern",
                               "rangePatternBound",
                               "referencePattern",
                               "structPattern",
                               "structPatternElements",
                               "structPatternFields",
                               "structPatternField",
                               "structPatternEtCetera",
                               "tupleStructPattern",
                               "tupleStructItems",
                               "tuplePattern",
                               "tuplePatternItems",
                               "groupedPattern",
                               "slicePattern",
                               "slicePatternItems",
                               "pathPattern",
                               "type_",
                               "typeNoBounds",
                               "parenthesizedType",
                               "neverType",
                               "tupleType",
                               "arrayType",
                               "sliceType",
                               "referenceType",
                               "rawPointerType",
                               "bareFunctionType",
                               "functionTypeQualifiers",
                               "bareFunctionReturnType",
                               "functionParametersMaybeNamedVariadic",
                               "maybeNamedFunctionParameters",
                               "maybeNamedParam",
                               "maybeNamedFunctionParametersVariadic",
                               "traitObjectType",
                               "traitObjectTypeOneBound",
                               "implTraitType",
                               "implTraitTypeOneBound",
                               "inferredType",
                               "typeParamBounds",
                               "typeParamBound",
                               "traitBound",
                               "lifetimeBounds",
                               "lifetime",
                               "simplePath",
                               "simplePathSegment",
                               "pathInExpression",
                               "pathExprSegment",
                               "pathIdentSegment",
                               "genericArgs",
                               "genericArg",
                               "genericArgsConst",
                               "genericArgsLifetimes",
                               "genericArgsTypes",
                               "genericArgsBindings",
                               "genericArgsBinding",
                               "qualifiedPathInExpression",
                               "qualifiedPathType",
                               "qualifiedPathInType",
                               "typePath",
                               "typePathSegment",
                               "typePathFn",
                               "typePathInputs",
                               "visibility",
                               "identifier",
                               "keyword",
                               "macroIdentifierLikeToken",
                               "macroLiteralToken",
                               "macroPunctuationToken",
                               "shl",
                               "shr"},
      std::vector<std::string>{
          "",           "'as'",     "'break'",  "'const'",   "'continue'",
          "'crate'",    "'else'",   "'enum'",   "'extern'",  "'false'",
          "'fn'",       "'for'",    "'if'",     "'impl'",    "'in'",
          "'let'",      "'loop'",   "'match'",  "'mod'",     "'move'",
          "'mut'",      "'pub'",    "'ref'",    "'return'",  "'self'",
          "'Self'",     "'static'", "'struct'", "'super'",   "'trait'",
          "'true'",     "'type'",   "'unsafe'", "'use'",     "'where'",
          "'while'",    "'async'",  "'await'",  "'dyn'",     "'abstract'",
          "'become'",   "'box'",    "'do'",     "'final'",   "'macro'",
          "'override'", "'priv'",   "'typeof'", "'unsized'", "'virtual'",
          "'yield'",    "'try'",    "'union'",  "''static'", "'macro_rules'",
          "''_'",       "'$crate'", "",         "",          "",
          "",           "",         "",         "",          "",
          "",           "",         "",         "",          "",
          "",           "",         "",         "",          "",
          "",           "",         "",         "",          "",
          "",           "",         "",         "'+'",       "'-'",
          "'*'",        "'/'",      "'%'",      "'^'",       "'!'",
          "'&'",        "'|'",      "'&&'",     "'||'",      "'+='",
          "'-='",       "'*='",     "'/='",     "'%='",      "'^='",
          "'&='",       "'|='",     "'<<='",    "'>>='",     "'='",
          "'=='",       "'!='",     "'>'",      "'<'",       "'>='",
          "'<='",       "'@'",      "'_'",      "'.'",       "'..'",
          "'...'",      "'..='",    "','",      "';'",       "':'",
          "'::'",       "'->'",     "'=>'",     "'#'",       "'$'",
          "'\\u003F'",  "'{'",      "'}'",      "'['",       "']'",
          "'('",        "')'"},
      std::vector<std::string>{"",
                               "KW_AS",
                               "KW_BREAK",
                               "KW_CONST",
                               "KW_CONTINUE",
                               "KW_CRATE",
                               "KW_ELSE",
                               "KW_ENUM",
                               "KW_EXTERN",
                               "KW_FALSE",
                               "KW_FN",
                               "KW_FOR",
                               "KW_IF",
                               "KW_IMPL",
                               "KW_IN",
                               "KW_LET",
                               "KW_LOOP",
                               "KW_MATCH",
                               "KW_MOD",
                               "KW_MOVE",
                               "KW_MUT",
                               "KW_PUB",
                               "KW_REF",
                               "KW_RETURN",
                               "KW_SELFVALUE",
                               "KW_SELFTYPE",
                               "KW_STATIC",
                               "KW_STRUCT",
                               "KW_SUPER",
                               "KW_TRAIT",
                               "KW_TRUE",
                               "KW_TYPE",
                               "KW_UNSAFE",
                               "KW_USE",
                               "KW_WHERE",
                               "KW_WHILE",
                               "KW_ASYNC",
                               "KW_AWAIT",
                               "KW_DYN",
                               "KW_ABSTRACT",
                               "KW_BECOME",
                               "KW_BOX",
                               "KW_DO",
                               "KW_FINAL",
                               "KW_MACRO",
                               "KW_OVERRIDE",
                               "KW_PRIV",
                               "KW_TYPEOF",
                               "KW_UNSIZED",
                               "KW_VIRTUAL",
                               "KW_YIELD",
                               "KW_TRY",
                               "KW_UNION",
                               "KW_STATICLIFETIME",
                               "KW_MACRORULES",
                               "KW_UNDERLINELIFETIME",
                               "KW_DOLLARCRATE",
                               "NON_KEYWORD_IDENTIFIER",
                               "RAW_IDENTIFIER",
                               "LINE_COMMENT",
                               "BLOCK_COMMENT",
                               "DOC_BLOCK_COMMENT",
                               "INNER_LINE_DOC",
                               "INNER_BLOCK_DOC",
                               "OUTER_LINE_DOC",
                               "OUTER_BLOCK_DOC",
                               "BLOCK_COMMENT_OR_DOC",
                               "SHEBANG",
                               "WHITESPACE",
                               "NEWLINE",
                               "CHAR_LITERAL",
                               "STRING_LITERAL",
                               "RAW_STRING_LITERAL",
                               "BYTE_LITERAL",
                               "BYTE_STRING_LITERAL",
                               "RAW_BYTE_STRING_LITERAL",
                               "INTEGER_LITERAL",
                               "DEC_LITERAL",
                               "HEX_LITERAL",
                               "OCT_LITERAL",
                               "BIN_LITERAL",
                               "FLOAT_LITERAL",
                               "LIFETIME_OR_LABEL",
                               "PLUS",
                               "MINUS",
                               "STAR",
                               "SLASH",
                               "PERCENT",
                               "CARET",
                               "NOT",
                               "AND",
                               "OR",
                               "ANDAND",
                               "OROR",
                               "PLUSEQ",
                               "MINUSEQ",
                               "STAREQ",
                               "SLASHEQ",
                               "PERCENTEQ",
                               "CARETEQ",
                               "ANDEQ",
                               "OREQ",
                               "SHLEQ",
                               "SHREQ",
                               "EQ",
                               "EQEQ",
                               "NE",
                               "GT",
                               "LT",
                               "GE",
                               "LE",
                               "AT",
                               "UNDERSCORE",
                               "DOT",
                               "DOTDOT",
                               "DOTDOTDOT",
                               "DOTDOTEQ",
                               "COMMA",
                               "SEMI",
                               "COLON",
                               "PATHSEP",
                               "RARROW",
                               "FATARROW",
                               "POUND",
                               "DOLLAR",
                               "QUESTION",
                               "LCURLYBRACE",
                               "RCURLYBRACE",
                               "LSQUAREBRACKET",
                               "RSQUAREBRACKET",
                               "LPAREN",
                               "RPAREN"});
  static const int32_t serializedATNSegment[] = {
      4,    1,    131,  2475, 2,    0,    7,    0,    2,    1,    7,    1,
      2,    2,    7,    2,    2,    3,    7,    3,    2,    4,    7,    4,
      2,    5,    7,    5,    2,    6,    7,    6,    2,    7,    7,    7,
      2,    8,    7,    8,    2,    9,    7,    9,    2,    10,   7,    10,
      2,    11,   7,    11,   2,    12,   7,    12,   2,    13,   7,    13,
      2,    14,   7,    14,   2,    15,   7,    15,   2,    16,   7,    16,
      2,    17,   7,    17,   2,    18,   7,    18,   2,    19,   7,    19,
      2,    20,   7,    20,   2,    21,   7,    21,   2,    22,   7,    22,
      2,    23,   7,    23,   2,    24,   7,    24,   2,    25,   7,    25,
      2,    26,   7,    26,   2,    27,   7,    27,   2,    28,   7,    28,
      2,    29,   7,    29,   2,    30,   7,    30,   2,    31,   7,    31,
      2,    32,   7,    32,   2,    33,   7,    33,   2,    34,   7,    34,
      2,    35,   7,    35,   2,    36,   7,    36,   2,    37,   7,    37,
      2,    38,   7,    38,   2,    39,   7,    39,   2,    40,   7,    40,
      2,    41,   7,    41,   2,    42,   7,    42,   2,    43,   7,    43,
      2,    44,   7,    44,   2,    45,   7,    45,   2,    46,   7,    46,
      2,    47,   7,    47,   2,    48,   7,    48,   2,    49,   7,    49,
      2,    50,   7,    50,   2,    51,   7,    51,   2,    52,   7,    52,
      2,    53,   7,    53,   2,    54,   7,    54,   2,    55,   7,    55,
      2,    56,   7,    56,   2,    57,   7,    57,   2,    58,   7,    58,
      2,    59,   7,    59,   2,    60,   7,    60,   2,    61,   7,    61,
      2,    62,   7,    62,   2,    63,   7,    63,   2,    64,   7,    64,
      2,    65,   7,    65,   2,    66,   7,    66,   2,    67,   7,    67,
      2,    68,   7,    68,   2,    69,   7,    69,   2,    70,   7,    70,
      2,    71,   7,    71,   2,    72,   7,    72,   2,    73,   7,    73,
      2,    74,   7,    74,   2,    75,   7,    75,   2,    76,   7,    76,
      2,    77,   7,    77,   2,    78,   7,    78,   2,    79,   7,    79,
      2,    80,   7,    80,   2,    81,   7,    81,   2,    82,   7,    82,
      2,    83,   7,    83,   2,    84,   7,    84,   2,    85,   7,    85,
      2,    86,   7,    86,   2,    87,   7,    87,   2,    88,   7,    88,
      2,    89,   7,    89,   2,    90,   7,    90,   2,    91,   7,    91,
      2,    92,   7,    92,   2,    93,   7,    93,   2,    94,   7,    94,
      2,    95,   7,    95,   2,    96,   7,    96,   2,    97,   7,    97,
      2,    98,   7,    98,   2,    99,   7,    99,   2,    100,  7,    100,
      2,    101,  7,    101,  2,    102,  7,    102,  2,    103,  7,    103,
      2,    104,  7,    104,  2,    105,  7,    105,  2,    106,  7,    106,
      2,    107,  7,    107,  2,    108,  7,    108,  2,    109,  7,    109,
      2,    110,  7,    110,  2,    111,  7,    111,  2,    112,  7,    112,
      2,    113,  7,    113,  2,    114,  7,    114,  2,    115,  7,    115,
      2,    116,  7,    116,  2,    117,  7,    117,  2,    118,  7,    118,
      2,    119,  7,    119,  2,    120,  7,    120,  2,    121,  7,    121,
      2,    122,  7,    122,  2,    123,  7,    123,  2,    124,  7,    124,
      2,    125,  7,    125,  2,    126,  7,    126,  2,    127,  7,    127,
      2,    128,  7,    128,  2,    129,  7,    129,  2,    130,  7,    130,
      2,    131,  7,    131,  2,    132,  7,    132,  2,    133,  7,    133,
      2,    134,  7,    134,  2,    135,  7,    135,  2,    136,  7,    136,
      2,    137,  7,    137,  2,    138,  7,    138,  2,    139,  7,    139,
      2,    140,  7,    140,  2,    141,  7,    141,  2,    142,  7,    142,
      2,    143,  7,    143,  2,    144,  7,    144,  2,    145,  7,    145,
      2,    146,  7,    146,  2,    147,  7,    147,  2,    148,  7,    148,
      2,    149,  7,    149,  2,    150,  7,    150,  2,    151,  7,    151,
      2,    152,  7,    152,  2,    153,  7,    153,  2,    154,  7,    154,
      2,    155,  7,    155,  2,    156,  7,    156,  2,    157,  7,    157,
      2,    158,  7,    158,  2,    159,  7,    159,  2,    160,  7,    160,
      2,    161,  7,    161,  2,    162,  7,    162,  2,    163,  7,    163,
      2,    164,  7,    164,  2,    165,  7,    165,  2,    166,  7,    166,
      2,    167,  7,    167,  2,    168,  7,    168,  2,    169,  7,    169,
      2,    170,  7,    170,  2,    171,  7,    171,  2,    172,  7,    172,
      2,    173,  7,    173,  2,    174,  7,    174,  2,    175,  7,    175,
      2,    176,  7,    176,  2,    177,  7,    177,  2,    178,  7,    178,
      2,    179,  7,    179,  2,    180,  7,    180,  2,    181,  7,    181,
      2,    182,  7,    182,  2,    183,  7,    183,  2,    184,  7,    184,
      2,    185,  7,    185,  2,    186,  7,    186,  2,    187,  7,    187,
      2,    188,  7,    188,  2,    189,  7,    189,  2,    190,  7,    190,
      2,    191,  7,    191,  2,    192,  7,    192,  2,    193,  7,    193,
      2,    194,  7,    194,  2,    195,  7,    195,  1,    0,    5,    0,
      394,  8,    0,    10,   0,    12,   0,    397,  9,    0,    1,    0,
      5,    0,    400,  8,    0,    10,   0,    12,   0,    403,  9,    0,
      1,    0,    1,    0,    1,    1,    1,    1,    1,    1,    1,    1,
      1,    2,    1,    2,    5,    2,    413,  8,    2,    10,   2,    12,
      2,    416,  9,    2,    1,    2,    1,    2,    1,    2,    5,    2,
      421,  8,    2,    10,   2,    12,   2,    424,  9,    2,    1,    2,
      1,    2,    1,    2,    5,    2,    429,  8,    2,    10,   2,    12,
      2,    432,  9,    2,    1,    2,    3,    2,    435,  8,    2,    1,
      3,    4,    3,    438,  8,    3,    11,   3,    12,   3,    439,  1,
      3,    3,    3,    443,  8,    3,    1,    4,    1,    4,    1,    4,
      1,    4,    1,    4,    3,    4,    450,  8,    4,    1,    5,    1,
      5,    1,    5,    1,    5,    5,    5,    456,  8,    5,    10,   5,
      12,   5,    459,  9,    5,    1,    5,    1,    5,    1,    5,    1,
      5,    1,    5,    1,    5,    1,    5,    5,    5,    468,  8,    5,
      10,   5,    12,   5,    471,  9,    5,    1,    5,    1,    5,    1,
      5,    1,    5,    1,    5,    1,    5,    1,    5,    5,    5,    480,
      8,    5,    10,   5,    12,   5,    483,  9,    5,    1,    5,    1,
      5,    3,    5,    487,  8,    5,    1,    6,    1,    6,    1,    6,
      1,    6,    1,    6,    1,    7,    1,    7,    1,    7,    1,    7,
      1,    7,    1,    7,    1,    7,    1,    7,    1,    7,    1,    7,
      1,    7,    1,    7,    1,    7,    1,    7,    3,    7,    508,  8,
      7,    1,    8,    1,    8,    1,    8,    5,    8,    513,  8,    8,
      10,   8,    12,   8,    516,  9,    8,    1,    8,    3,    8,    519,
      8,    8,    1,    9,    1,    9,    1,    9,    1,    9,    1,    10,
      1,    10,   5,    10,   527,  8,    10,   10,   10,   12,   10,   530,
      9,    10,   1,    10,   1,    10,   1,    10,   5,    10,   535,  8,
      10,   10,   10,   12,   10,   538,  9,    10,   1,    10,   1,    10,
      1,    10,   5,    10,   543,  8,    10,   10,   10,   12,   10,   546,
      9,    10,   1,    10,   3,    10,   549,  8,    10,   1,    11,   4,
      11,   552,  8,    11,   11,   11,   12,   11,   553,  1,    11,   1,
      11,   1,    11,   1,    11,   3,    11,   560,  8,    11,   1,    11,
      1,    11,   1,    11,   1,    11,   1,    11,   4,    11,   567,  8,
      11,   11,   11,   12,   11,   568,  1,    11,   1,    11,   3,    11,
      573,  8,    11,   1,    11,   1,    11,   3,    11,   577,  8,    11,
      1,    12,   1,    12,   1,    12,   1,    12,   3,    12,   583,  8,
      12,   1,    13,   1,    13,   1,    14,   1,    14,   1,    14,   1,
      14,   3,    14,   591,  8,    14,   1,    15,   1,    15,   1,    16,
      1,    16,   1,    17,   5,    17,   598,  8,    17,   10,   17,   12,
      17,   601,  9,    17,   1,    17,   1,    17,   3,    17,   605,  8,
      17,   1,    18,   3,    18,   608,  8,    18,   1,    18,   1,    18,
      1,    18,   1,    18,   1,    18,   1,    18,   1,    18,   1,    18,
      1,    18,   1,    18,   1,    18,   1,    18,   1,    18,   3,    18,
      623,  8,    18,   1,    19,   1,    19,   3,    19,   627,  8,    19,
      1,    20,   3,    20,   630,  8,    20,   1,    20,   1,    20,   1,
      20,   1,    20,   1,    20,   5,    20,   637,  8,    20,   10,   20,
      12,   20,   640,  9,    20,   1,    20,   5,    20,   643,  8,    20,
      10,   20,   12,   20,   646,  9,    20,   1,    20,   3,    20,   649,
      8,    20,   1,    21,   1,    21,   1,    21,   1,    21,   3,    21,
      655,  8,    21,   1,    21,   1,    21,   1,    22,   1,    22,   3,
      22,   661,  8,    22,   1,    23,   1,    23,   1,    23,   3,    23,
      666,  8,    23,   1,    24,   1,    24,   1,    24,   1,    24,   1,
      25,   3,    25,   673,  8,    25,   1,    25,   3,    25,   676,  8,
      25,   1,    25,   1,    25,   1,    25,   1,    25,   1,    25,   5,
      25,   683,  8,    25,   10,   25,   12,   25,   686,  9,    25,   1,
      25,   3,    25,   689,  8,    25,   3,    25,   691,  8,    25,   1,
      25,   3,    25,   694,  8,    25,   1,    25,   1,    25,   1,    25,
      1,    25,   3,    25,   700,  8,    25,   3,    25,   702,  8,    25,
      3,    25,   704,  8,    25,   1,    26,   1,    26,   1,    26,   1,
      26,   3,    26,   710,  8,    26,   1,    26,   1,    26,   3,    26,
      714,  8,    26,   1,    26,   1,    26,   3,    26,   718,  8,    26,
      1,    26,   3,    26,   721,  8,    26,   1,    26,   1,    26,   3,
      26,   725,  8,    26,   1,    27,   3,    27,   728,  8,    27,   1,
      27,   3,    27,   731,  8,    27,   1,    27,   3,    27,   734,  8,
      27,   1,    27,   1,    27,   3,    27,   738,  8,    27,   3,    27,
      740,  8,    27,   1,    28,   1,    28,   1,    29,   1,    29,   3,
      29,   746,  8,    29,   1,    29,   1,    29,   1,    29,   3,    29,
      751,  8,    29,   1,    29,   1,    29,   1,    29,   5,    29,   756,
      8,    29,   10,   29,   12,   29,   759,  9,    29,   1,    29,   3,
      29,   762,  8,    29,   3,    29,   764,  8,    29,   1,    30,   5,
      30,   767,  8,    30,   10,   30,   12,   30,   770,  9,    30,   1,
      30,   1,    30,   3,    30,   774,  8,    30,   1,    31,   1,    31,
      3,    31,   778,  8,    31,   3,    31,   780,  8,    31,   1,    31,
      3,    31,   783,  8,    31,   1,    31,   1,    31,   1,    32,   3,
      32,   788,  8,    32,   1,    32,   1,    32,   1,    32,   1,    32,
      1,    33,   5,    33,   795,  8,    33,   10,   33,   12,   33,   798,
      9,    33,   1,    33,   1,    33,   1,    33,   3,    33,   803,  8,
      33,   1,    34,   1,    34,   1,    34,   1,    34,   3,    34,   809,
      8,    34,   1,    35,   1,    35,   1,    35,   1,    36,   1,    36,
      1,    36,   3,    36,   817,  8,    36,   1,    36,   3,    36,   820,
      8,    36,   1,    36,   1,    36,   3,    36,   824,  8,    36,   1,
      36,   1,    36,   1,    37,   1,    37,   3,    37,   830,  8,    37,
      1,    38,   1,    38,   1,    38,   3,    38,   835,  8,    38,   1,
      38,   3,    38,   838,  8,    38,   1,    38,   1,    38,   3,    38,
      842,  8,    38,   1,    38,   1,    38,   3,    38,   846,  8,    38,
      1,    39,   1,    39,   1,    39,   3,    39,   851,  8,    39,   1,
      39,   1,    39,   3,    39,   855,  8,    39,   1,    39,   1,    39,
      3,    39,   859,  8,    39,   1,    39,   1,    39,   1,    40,   1,
      40,   1,    40,   5,    40,   866,  8,    40,   10,   40,   12,   40,
      869,  9,    40,   1,    40,   3,    40,   872,  8,    40,   1,    41,
      5,    41,   875,  8,    41,   10,   41,   12,   41,   878,  9,    41,
      1,    41,   3,    41,   881,  8,    41,   1,    41,   1,    41,   1,
      41,   1,    41,   1,    42,   1,    42,   1,    42,   5,    42,   890,
      8,    42,   10,   42,   12,   42,   893,  9,    42,   1,    42,   3,
      42,   896,  8,    42,   1,    43,   5,    43,   899,  8,    43,   10,
      43,   12,   43,   902,  9,    43,   1,    43,   3,    43,   905,  8,
      43,   1,    43,   1,    43,   1,    44,   1,    44,   1,    44,   3,
      44,   912,  8,    44,   1,    44,   3,    44,   915,  8,    44,   1,
      44,   1,    44,   3,    44,   919,  8,    44,   1,    44,   1,    44,
      1,    45,   1,    45,   1,    45,   5,    45,   926,  8,    45,   10,
      45,   12,   45,   929,  9,    45,   1,    45,   3,    45,   932,  8,
      45,   1,    46,   5,    46,   935,  8,    46,   10,   46,   12,   46,
      938,  9,    46,   1,    46,   3,    46,   941,  8,    46,   1,    46,
      1,    46,   1,    46,   1,    46,   3,    46,   947,  8,    46,   1,
      47,   1,    47,   3,    47,   951,  8,    47,   1,    47,   1,    47,
      1,    48,   1,    48,   3,    48,   957,  8,    48,   1,    48,   1,
      48,   1,    49,   1,    49,   1,    49,   1,    50,   1,    50,   1,
      50,   3,    50,   967,  8,    50,   1,    50,   3,    50,   970,  8,
      50,   1,    50,   1,    50,   1,    50,   1,    50,   1,    51,   1,
      51,   1,    51,   3,    51,   979,  8,    51,   1,    51,   1,    51,
      1,    51,   1,    51,   3,    51,   985,  8,    51,   1,    51,   1,
      51,   1,    52,   1,    52,   3,    52,   991,  8,    52,   1,    52,
      1,    52,   1,    52,   1,    52,   1,    52,   3,    52,   998,  8,
      52,   1,    52,   1,    52,   1,    53,   3,    53,   1003, 8,    53,
      1,    53,   1,    53,   1,    53,   3,    53,   1008, 8,    53,   1,
      53,   1,    53,   3,    53,   1012, 8,    53,   3,    53,   1014, 8,
      53,   1,    53,   3,    53,   1017, 8,    53,   1,    53,   1,    53,
      5,    53,   1021, 8,    53,   10,   53,   12,   53,   1024, 9,    53,
      1,    53,   5,    53,   1027, 8,    53,   10,   53,   12,   53,   1030,
      9,    53,   1,    53,   1,    53,   1,    54,   1,    54,   3,    54,
      1036, 8,    54,   1,    55,   1,    55,   3,    55,   1040, 8,    55,
      1,    55,   1,    55,   3,    55,   1044, 8,    55,   1,    55,   1,
      55,   5,    55,   1048, 8,    55,   10,   55,   12,   55,   1051, 9,
      55,   1,    55,   5,    55,   1054, 8,    55,   10,   55,   12,   55,
      1057, 9,    55,   1,    55,   1,    55,   1,    56,   3,    56,   1062,
      8,    56,   1,    56,   1,    56,   3,    56,   1066, 8,    56,   1,
      56,   3,    56,   1069, 8,    56,   1,    56,   1,    56,   1,    56,
      1,    56,   3,    56,   1075, 8,    56,   1,    56,   1,    56,   5,
      56,   1079, 8,    56,   10,   56,   12,   56,   1082, 9,    56,   1,
      56,   5,    56,   1085, 8,    56,   10,   56,   12,   56,   1088, 9,
      56,   1,    56,   1,    56,   1,    57,   3,    57,   1093, 8,    57,
      1,    57,   1,    57,   3,    57,   1097, 8,    57,   1,    57,   1,
      57,   5,    57,   1101, 8,    57,   10,   57,   12,   57,   1104, 9,
      57,   1,    57,   5,    57,   1107, 8,    57,   10,   57,   12,   57,
      1110, 9,    57,   1,    57,   1,    57,   1,    58,   5,    58,   1115,
      8,    58,   10,   58,   12,   58,   1118, 9,    58,   1,    58,   1,
      58,   3,    58,   1122, 8,    58,   1,    58,   1,    58,   3,    58,
      1126, 8,    58,   3,    58,   1128, 8,    58,   1,    59,   1,    59,
      1,    59,   1,    59,   5,    59,   1134, 8,    59,   10,   59,   12,
      59,   1137, 9,    59,   1,    59,   1,    59,   3,    59,   1141, 8,
      59,   3,    59,   1143, 8,    59,   1,    59,   1,    59,   1,    60,
      5,    60,   1148, 8,    60,   10,   60,   12,   60,   1151, 9,    60,
      1,    60,   1,    60,   1,    60,   3,    60,   1156, 8,    60,   1,
      61,   3,    61,   1159, 8,    61,   1,    61,   1,    61,   1,    61,
      3,    61,   1164, 8,    61,   1,    62,   3,    62,   1167, 8,    62,
      1,    62,   1,    62,   1,    62,   3,    62,   1172, 8,    62,   3,
      62,   1174, 8,    62,   1,    62,   1,    62,   3,    62,   1178, 8,
      62,   1,    63,   1,    63,   1,    63,   1,    63,   1,    63,   1,
      64,   1,    64,   1,    64,   1,    64,   5,    64,   1189, 8,    64,
      10,   64,   12,   64,   1192, 9,    64,   1,    64,   3,    64,   1195,
      8,    64,   1,    65,   1,    65,   3,    65,   1199, 8,    65,   1,
      66,   1,    66,   1,    66,   1,    66,   1,    67,   3,    67,   1206,
      8,    67,   1,    67,   1,    67,   1,    67,   3,    67,   1211, 8,
      67,   1,    68,   1,    68,   1,    68,   1,    69,   5,    69,   1217,
      8,    69,   10,   69,   12,   69,   1220, 9,    69,   1,    69,   1,
      69,   3,    69,   1224, 8,    69,   1,    69,   1,    69,   1,    69,
      3,    69,   1229, 8,    69,   3,    69,   1231, 8,    69,   1,    70,
      1,    70,   1,    70,   1,    70,   1,    70,   1,    70,   1,    71,
      1,    71,   1,    71,   1,    71,   1,    71,   1,    72,   1,    72,
      3,    72,   1246, 8,    72,   1,    73,   1,    73,   1,    73,   3,
      73,   1251, 8,    73,   1,    74,   1,    74,   1,    74,   1,    74,
      1,    74,   3,    74,   1258, 8,    74,   1,    75,   5,    75,   1261,
      8,    75,   10,   75,   12,   75,   1264, 9,    75,   1,    75,   1,
      75,   1,    75,   1,    75,   3,    75,   1270, 8,    75,   1,    75,
      1,    75,   3,    75,   1274, 8,    75,   1,    75,   1,    75,   1,
      76,   1,    76,   1,    76,   1,    76,   1,    76,   3,    76,   1283,
      8,    76,   3,    76,   1285, 8,    76,   1,    77,   1,    77,   4,
      77,   1289, 8,    77,   11,   77,   12,   77,   1290, 1,    77,   1,
      77,   1,    77,   1,    77,   1,    77,   1,    77,   3,    77,   1299,
      8,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   3,    77,   1308, 8,    77,   1,    77,   1,
      77,   1,    77,   1,    77,   3,    77,   1314, 8,    77,   1,    77,
      3,    77,   1317, 8,    77,   1,    77,   1,    77,   3,    77,   1321,
      8,    77,   1,    77,   3,    77,   1324, 8,    77,   1,    77,   1,
      77,   3,    77,   1328, 8,    77,   1,    77,   1,    77,   5,    77,
      1332, 8,    77,   10,   77,   12,   77,   1335, 9,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   5,    77,   1342, 8,
      77,   10,   77,   12,   77,   1345, 9,    77,   1,    77,   3,    77,
      1348, 8,    77,   1,    77,   1,    77,   1,    77,   5,    77,   1353,
      8,    77,   10,   77,   12,   77,   1356, 9,    77,   1,    77,   3,
      77,   1359, 8,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   3,    77,   1367, 8,    77,   1,    77,   1,
      77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,
      77,   1,    77,   3,    77,   1378, 8,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   3,    77,   1416, 8,
      77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,
      77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,
      77,   1,    77,   1,    77,   3,    77,   1432, 8,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   1,    77,   1,    77,   1,    77,   1,    77,   1,    77,
      1,    77,   3,    77,   1448, 8,    77,   5,    77,   1450, 8,    77,
      10,   77,   12,   77,   1453, 9,    77,   1,    78,   1,    78,   1,
      79,   1,    79,   1,    80,   4,    80,   1460, 8,    80,   11,   80,
      12,   80,   1461, 1,    80,   1,    80,   1,    80,   1,    80,   1,
      80,   1,    80,   1,    80,   1,    80,   1,    80,   3,    80,   1473,
      8,    80,   1,    81,   1,    81,   1,    82,   1,    82,   3,    82,
      1479, 8,    82,   1,    83,   1,    83,   5,    83,   1483, 8,    83,
      10,   83,   12,   83,   1486, 9,    83,   1,    83,   3,    83,   1489,
      8,    83,   1,    83,   1,    83,   1,    84,   4,    84,   1494, 8,
      84,   11,   84,   12,   84,   1495, 1,    84,   3,    84,   1499, 8,
      84,   1,    84,   3,    84,   1502, 8,    84,   1,    85,   1,    85,
      3,    85,   1506, 8,    85,   1,    85,   1,    85,   1,    86,   1,
      86,   1,    86,   1,    87,   1,    87,   1,    87,   5,    87,   1516,
      8,    87,   10,   87,   12,   87,   1519, 9,    87,   1,    87,   3,
      87,   1522, 8,    87,   1,    87,   1,    87,   1,    87,   1,    87,
      3,    87,   1528, 8,    87,   1,    88,   1,    88,   1,    88,   4,
      88,   1533, 8,    88,   11,   88,   12,   88,   1534, 1,    88,   3,
      88,   1538, 8,    88,   1,    89,   1,    89,   1,    90,   1,    90,
      1,    90,   3,    90,   1545, 8,    90,   1,    91,   1,    91,   1,
      91,   5,    91,   1550, 8,    91,   10,   91,   12,   91,   1553, 9,
      91,   1,    91,   1,    91,   3,    91,   1557, 8,    91,   1,    91,
      1,    91,   1,    92,   1,    92,   1,    92,   5,    92,   1564, 8,
      92,   10,   92,   12,   92,   1567, 9,    92,   1,    92,   1,    92,
      1,    92,   3,    92,   1572, 8,    92,   3,    92,   1574, 8,    92,
      1,    93,   5,    93,   1577, 8,    93,   10,   93,   12,   93,   1580,
      9,    93,   1,    93,   1,    93,   1,    93,   3,    93,   1585, 8,
      93,   1,    93,   1,    93,   1,    93,   3,    93,   1590, 8,    93,
      1,    94,   1,    94,   1,    94,   1,    95,   1,    95,   1,    95,
      5,    95,   1598, 8,    95,   10,   95,   12,   95,   1601, 9,    95,
      1,    95,   1,    95,   1,    95,   5,    95,   1606, 8,    95,   10,
      95,   12,   95,   1609, 9,    95,   1,    95,   3,    95,   1612, 8,
      95,   3,    95,   1614, 8,    95,   1,    95,   1,    95,   1,    96,
      1,    96,   1,    97,   1,    97,   1,    97,   3,    97,   1623, 8,
      97,   1,    98,   1,    98,   1,    98,   3,    98,   1628, 8,    98,
      1,    98,   1,    98,   1,    99,   1,    99,   1,    99,   5,    99,
      1635, 8,    99,   10,   99,   12,   99,   1638, 9,    99,   1,    99,
      3,    99,   1641, 8,    99,   1,    100,  1,    100,  1,    100,  3,
      100,  1646, 8,    100,  1,    100,  1,    100,  1,    100,  3,    100,
      1651, 8,    100,  1,    101,  1,    101,  1,    101,  1,    101,  1,
      101,  5,    101,  1658, 8,    101,  10,   101,  12,   101,  1661, 9,
      101,  1,    101,  3,    101,  1664, 8,    101,  3,    101,  1666, 8,
      101,  1,    101,  1,    101,  1,    102,  1,    102,  1,    103,  1,
      103,  1,    103,  5,    103,  1675, 8,    103,  10,   103,  12,   103,
      1678, 9,    103,  1,    103,  3,    103,  1681, 8,    103,  1,    104,
      3,    104,  1684, 8,    104,  1,    104,  1,    104,  1,    104,  3,
      104,  1689, 8,    104,  1,    104,  3,    104,  1692, 8,    104,  1,
      104,  1,    104,  1,    104,  1,    104,  1,    104,  3,    104,  1699,
      8,    104,  1,    105,  1,    105,  1,    105,  5,    105,  1704, 8,
      105,  10,   105,  12,   105,  1707, 9,    105,  1,    105,  3,    105,
      1710, 8,    105,  1,    106,  5,    106,  1713, 8,    106,  10,   106,
      12,   106,  1716, 9,    106,  1,    106,  1,    106,  1,    106,  3,
      106,  1721, 8,    106,  1,    107,  3,    107,  1724, 8,    107,  1,
      107,  1,    107,  1,    107,  1,    107,  3,    107,  1730, 8,    107,
      1,    108,  1,    108,  1,    108,  1,    109,  1,    109,  1,    109,
      1,    109,  1,    110,  1,    110,  1,    110,  1,    110,  1,    110,
      1,    110,  1,    110,  1,    111,  1,    111,  1,    111,  1,    111,
      1,    111,  1,    111,  1,    112,  1,    112,  1,    112,  1,    113,
      1,    113,  1,    113,  1,    113,  1,    113,  1,    113,  1,    113,
      3,    113,  1762, 8,    113,  3,    113,  1764, 8,    113,  1,    114,
      1,    114,  1,    114,  1,    114,  1,    114,  1,    114,  1,    114,
      1,    114,  1,    114,  1,    114,  3,    114,  1776, 8,    114,  3,
      114,  1778, 8,    114,  1,    115,  1,    115,  1,    115,  1,    115,
      5,    115,  1784, 8,    115,  10,   115,  12,   115,  1787, 9,    115,
      1,    115,  3,    115,  1790, 8,    115,  1,    115,  1,    115,  1,
      116,  1,    116,  1,    116,  1,    116,  5,    116,  1798, 8,    116,
      10,   116,  12,   116,  1801, 9,    116,  1,    116,  1,    116,  1,
      116,  1,    116,  3,    116,  1807, 8,    116,  1,    117,  1,    117,
      1,    117,  1,    117,  1,    117,  3,    117,  1814, 8,    117,  3,
      117,  1816, 8,    117,  1,    118,  5,    118,  1819, 8,    118,  10,
      118,  12,   118,  1822, 9,    118,  1,    118,  1,    118,  3,    118,
      1826, 8,    118,  1,    119,  1,    119,  1,    119,  1,    120,  3,
      120,  1832, 8,    120,  1,    120,  1,    120,  1,    120,  5,    120,
      1837, 8,    120,  10,   120,  12,   120,  1840, 9,    120,  1,    121,
      1,    121,  3,    121,  1844, 8,    121,  1,    122,  1,    122,  1,
      122,  1,    122,  1,    122,  1,    122,  1,    122,  1,    122,  1,
      122,  1,    122,  1,    122,  1,    122,  3,    122,  1858, 8,    122,
      1,    123,  1,    123,  1,    123,  1,    123,  1,    123,  1,    123,
      1,    123,  1,    123,  1,    123,  3,    123,  1869, 8,    123,  1,
      123,  1,    123,  3,    123,  1873, 8,    123,  1,    123,  3,    123,
      1876, 8,    123,  1,    124,  3,    124,  1879, 8,    124,  1,    124,
      3,    124,  1882, 8,    124,  1,    124,  1,    124,  1,    124,  3,
      124,  1887, 8,    124,  1,    125,  1,    125,  1,    126,  1,    126,
      1,    127,  1,    127,  1,    127,  1,    127,  1,    127,  1,    127,
      1,    127,  1,    127,  1,    127,  1,    127,  1,    127,  3,    127,
      1904, 8,    127,  1,    128,  1,    128,  1,    128,  3,    128,  1909,
      8,    128,  1,    128,  1,    128,  3,    128,  1913, 8,    128,  1,
      128,  1,    128,  3,    128,  1917, 8,    128,  1,    129,  1,    129,
      3,    129,  1921, 8,    129,  1,    129,  1,    129,  1,    130,  1,
      130,  1,    130,  3,    130,  1928, 8,    130,  1,    130,  1,    130,
      1,    131,  1,    131,  1,    131,  3,    131,  1935, 8,    131,  3,
      131,  1937, 8,    131,  1,    131,  3,    131,  1940, 8,    131,  1,
      132,  1,    132,  1,    132,  5,    132,  1945, 8,    132,  10,   132,
      12,   132,  1948, 9,    132,  1,    133,  5,    133,  1951, 8,    133,
      10,   133,  12,   133,  1954, 9,    133,  1,    133,  1,    133,  1,
      133,  1,    133,  1,    133,  1,    133,  1,    133,  1,    133,  1,
      133,  3,    133,  1965, 8,    133,  1,    133,  3,    133,  1968, 8,
      133,  1,    133,  3,    133,  1971, 8,    133,  1,    134,  5,    134,
      1974, 8,    134,  10,   134,  12,   134,  1977, 9,    134,  1,    134,
      1,    134,  1,    135,  1,    135,  1,    135,  3,    135,  1984, 8,
      135,  1,    135,  1,    135,  1,    136,  1,    136,  1,    136,  5,
      136,  1991, 8,    136,  10,   136,  12,   136,  1994, 9,    136,  1,
      136,  3,    136,  1997, 8,    136,  1,    137,  1,    137,  3,    137,
      2001, 8,    137,  1,    137,  1,    137,  1,    138,  1,    138,  1,
      138,  1,    138,  1,    138,  1,    138,  1,    138,  4,    138,  2012,
      8,    138,  11,   138,  12,   138,  2013, 1,    138,  3,    138,  2017,
      8,    138,  3,    138,  2019, 8,    138,  1,    139,  1,    139,  1,
      139,  1,    139,  1,    140,  1,    140,  3,    140,  2027, 8,    140,
      1,    140,  1,    140,  1,    141,  1,    141,  1,    141,  5,    141,
      2034, 8,    141,  10,   141,  12,   141,  2037, 9,    141,  1,    141,
      3,    141,  2040, 8,    141,  1,    142,  1,    142,  3,    142,  2044,
      8,    142,  1,    143,  1,    143,  1,    143,  3,    143,  2049, 8,
      143,  1,    144,  1,    144,  1,    144,  1,    144,  1,    144,  1,
      144,  1,    144,  1,    144,  1,    144,  1,    144,  1,    144,  1,
      144,  1,    144,  1,    144,  3,    144,  2065, 8,    144,  1,    145,
      1,    145,  1,    145,  1,    145,  1,    146,  1,    146,  1,    147,
      1,    147,  1,    147,  1,    147,  4,    147,  2077, 8,    147,  11,
      147,  12,   147,  2078, 1,    147,  3,    147,  2082, 8,    147,  3,
      147,  2084, 8,    147,  1,    147,  1,    147,  1,    148,  1,    148,
      1,    148,  1,    148,  1,    148,  1,    148,  1,    149,  1,    149,
      1,    149,  1,    149,  1,    150,  1,    150,  3,    150,  2100, 8,
      150,  1,    150,  3,    150,  2103, 8,    150,  1,    150,  1,    150,
      1,    151,  1,    151,  1,    151,  1,    151,  1,    152,  3,    152,
      2112, 8,    152,  1,    152,  1,    152,  1,    152,  1,    152,  3,
      152,  2118, 8,    152,  1,    152,  1,    152,  3,    152,  2122, 8,
      152,  1,    153,  3,    153,  2125, 8,    153,  1,    153,  1,    153,
      3,    153,  2129, 8,    153,  3,    153,  2131, 8,    153,  1,    154,
      1,    154,  1,    154,  1,    155,  1,    155,  3,    155,  2138, 8,
      155,  1,    156,  1,    156,  1,    156,  5,    156,  2143, 8,    156,
      10,   156,  12,   156,  2146, 9,    156,  1,    156,  3,    156,  2149,
      8,    156,  1,    157,  5,    157,  2152, 8,    157,  10,   157,  12,
      157,  2155, 9,    157,  1,    157,  1,    157,  3,    157,  2159, 8,
      157,  1,    157,  3,    157,  2162, 8,    157,  1,    157,  1,    157,
      1,    158,  1,    158,  1,    158,  5,    158,  2169, 8,    158,  10,
      158,  12,   158,  2172, 9,    158,  1,    158,  1,    158,  1,    158,
      5,    158,  2177, 8,    158,  10,   158,  12,   158,  2180, 9,    158,
      1,    158,  1,    158,  1,    159,  3,    159,  2185, 8,    159,  1,
      159,  1,    159,  1,    160,  3,    160,  2190, 8,    160,  1,    160,
      1,    160,  1,    161,  1,    161,  1,    161,  1,    162,  1,    162,
      1,    162,  1,    163,  1,    163,  1,    164,  1,    164,  1,    164,
      5,    164,  2205, 8,    164,  10,   164,  12,   164,  2208, 9,    164,
      1,    164,  3,    164,  2211, 8,    164,  1,    165,  1,    165,  3,
      165,  2215, 8,    165,  1,    166,  3,    166,  2218, 8,    166,  1,
      166,  3,    166,  2221, 8,    166,  1,    166,  1,    166,  1,    166,
      3,    166,  2226, 8,    166,  1,    166,  3,    166,  2229, 8,    166,
      1,    166,  1,    166,  1,    166,  3,    166,  2234, 8,    166,  1,
      167,  1,    167,  1,    167,  5,    167,  2239, 8,    167,  10,   167,
      12,   167,  2242, 9,    167,  1,    167,  3,    167,  2245, 8,    167,
      1,    168,  1,    168,  1,    169,  3,    169,  2250, 8,    169,  1,
      169,  1,    169,  1,    169,  5,    169,  2255, 8,    169,  10,   169,
      12,   169,  2258, 9,    169,  1,    170,  1,    170,  1,    170,  1,
      170,  1,    170,  3,    170,  2265, 8,    170,  1,    171,  3,    171,
      2268, 8,    171,  1,    171,  1,    171,  1,    171,  5,    171,  2273,
      8,    171,  10,   171,  12,   171,  2276, 9,    171,  1,    172,  1,
      172,  1,    172,  3,    172,  2281, 8,    172,  1,    173,  1,    173,
      1,    173,  1,    173,  1,    173,  1,    173,  3,    173,  2289, 8,
      173,  1,    174,  1,    174,  1,    174,  1,    174,  1,    174,  1,
      174,  3,    174,  2297, 8,    174,  1,    174,  1,    174,  3,    174,
      2301, 8,    174,  1,    174,  3,    174,  2304, 8,    174,  1,    174,
      1,    174,  1,    174,  1,    174,  1,    174,  1,    174,  3,    174,
      2312, 8,    174,  1,    174,  3,    174,  2315, 8,    174,  1,    174,
      1,    174,  1,    174,  1,    174,  1,    174,  1,    174,  5,    174,
      2323, 8,    174,  10,   174,  12,   174,  2326, 9,    174,  1,    174,
      1,    174,  3,    174,  2330, 8,    174,  1,    174,  1,    174,  3,
      174,  2334, 8,    174,  1,    175,  1,    175,  1,    175,  1,    175,
      3,    175,  2340, 8,    175,  1,    176,  1,    176,  3,    176,  2344,
      8,    176,  1,    176,  1,    176,  3,    176,  2348, 8,    176,  1,
      177,  1,    177,  1,    177,  5,    177,  2353, 8,    177,  10,   177,
      12,   177,  2356, 9,    177,  1,    178,  1,    178,  1,    178,  5,
      178,  2361, 8,    178,  10,   178,  12,   178,  2364, 9,    178,  1,
      179,  1,    179,  1,    179,  5,    179,  2369, 8,    179,  10,   179,
      12,   179,  2372, 9,    179,  1,    180,  1,    180,  1,    180,  1,
      180,  1,    181,  1,    181,  1,    181,  4,    181,  2381, 8,    181,
      11,   181,  12,   181,  2382, 1,    182,  1,    182,  1,    182,  1,
      182,  3,    182,  2389, 8,    182,  1,    182,  1,    182,  1,    183,
      1,    183,  1,    183,  4,    183,  2396, 8,    183,  11,   183,  12,
      183,  2397, 1,    184,  3,    184,  2401, 8,    184,  1,    184,  1,
      184,  1,    184,  5,    184,  2406, 8,    184,  10,   184,  12,   184,
      2409, 9,    184,  1,    185,  1,    185,  3,    185,  2413, 8,    185,
      1,    185,  1,    185,  3,    185,  2417, 8,    185,  1,    186,  1,
      186,  3,    186,  2421, 8,    186,  1,    186,  1,    186,  1,    186,
      3,    186,  2426, 8,    186,  1,    187,  1,    187,  1,    187,  5,
      187,  2431, 8,    187,  10,   187,  12,   187,  2434, 9,    187,  1,
      187,  3,    187,  2437, 8,    187,  1,    188,  1,    188,  1,    188,
      1,    188,  1,    188,  1,    188,  1,    188,  3,    188,  2446, 8,
      188,  1,    188,  3,    188,  2449, 8,    188,  1,    189,  1,    189,
      1,    190,  1,    190,  1,    191,  1,    191,  1,    191,  1,    191,
      1,    191,  1,    191,  3,    191,  2461, 8,    191,  1,    192,  1,
      192,  1,    193,  1,    193,  1,    194,  1,    194,  1,    194,  1,
      194,  1,    195,  1,    195,  1,    195,  1,    195,  1,    195,  0,
      1,    154,  196,  0,    2,    4,    6,    8,    10,   12,   14,   16,
      18,   20,   22,   24,   26,   28,   30,   32,   34,   36,   38,   40,
      42,   44,   46,   48,   50,   52,   54,   56,   58,   60,   62,   64,
      66,   68,   70,   72,   74,   76,   78,   80,   82,   84,   86,   88,
      90,   92,   94,   96,   98,   100,  102,  104,  106,  108,  110,  112,
      114,  116,  118,  120,  122,  124,  126,  128,  130,  132,  134,  136,
      138,  140,  142,  144,  146,  148,  150,  152,  154,  156,  158,  160,
      162,  164,  166,  168,  170,  172,  174,  176,  178,  180,  182,  184,
      186,  188,  190,  192,  194,  196,  198,  200,  202,  204,  206,  208,
      210,  212,  214,  216,  218,  220,  222,  224,  226,  228,  230,  232,
      234,  236,  238,  240,  242,  244,  246,  248,  250,  252,  254,  256,
      258,  260,  262,  264,  266,  268,  270,  272,  274,  276,  278,  280,
      282,  284,  286,  288,  290,  292,  294,  296,  298,  300,  302,  304,
      306,  308,  310,  312,  314,  316,  318,  320,  322,  324,  326,  328,
      330,  332,  334,  336,  338,  340,  342,  344,  346,  348,  350,  352,
      354,  356,  358,  360,  362,  364,  366,  368,  370,  372,  374,  376,
      378,  380,  382,  384,  386,  388,  390,  0,    14,   3,    0,    83,
      83,   85,   85,   125,  125,  1,    0,    71,   72,   2,    0,    90,
      90,   92,   92,   2,    0,    84,   84,   89,   89,   1,    0,    85,
      87,   1,    0,    83,   84,   1,    0,    105,  110,  1,    0,    94,
      103,  4,    0,    9,    9,    30,   30,   70,   76,   81,   81,   2,
      0,    3,    3,    20,   20,   3,    0,    53,   53,   55,   55,   82,
      82,   2,    0,    54,   54,   57,   58,   1,    0,    1,    53,   2,
      0,    84,   84,   86,   123,  2760, 0,    395,  1,    0,    0,    0,
      2,    406,  1,    0,    0,    0,    4,    434,  1,    0,    0,    0,
      6,    442,  1,    0,    0,    0,    8,    449,  1,    0,    0,    0,
      10,   486,  1,    0,    0,    0,    12,   488,  1,    0,    0,    0,
      14,   507,  1,    0,    0,    0,    16,   509,  1,    0,    0,    0,
      18,   520,  1,    0,    0,    0,    20,   548,  1,    0,    0,    0,
      22,   576,  1,    0,    0,    0,    24,   582,  1,    0,    0,    0,
      26,   584,  1,    0,    0,    0,    28,   590,  1,    0,    0,    0,
      30,   592,  1,    0,    0,    0,    32,   594,  1,    0,    0,    0,
      34,   599,  1,    0,    0,    0,    36,   607,  1,    0,    0,    0,
      38,   626,  1,    0,    0,    0,    40,   629,  1,    0,    0,    0,
      42,   650,  1,    0,    0,    0,    44,   660,  1,    0,    0,    0,
      46,   662,  1,    0,    0,    0,    48,   667,  1,    0,    0,    0,
      50,   703,  1,    0,    0,    0,    52,   705,  1,    0,    0,    0,
      54,   727,  1,    0,    0,    0,    56,   741,  1,    0,    0,    0,
      58,   763,  1,    0,    0,    0,    60,   768,  1,    0,    0,    0,
      62,   779,  1,    0,    0,    0,    64,   787,  1,    0,    0,    0,
      66,   796,  1,    0,    0,    0,    68,   804,  1,    0,    0,    0,
      70,   810,  1,    0,    0,    0,    72,   813,  1,    0,    0,    0,
      74,   829,  1,    0,    0,    0,    76,   831,  1,    0,    0,    0,
      78,   847,  1,    0,    0,    0,    80,   862,  1,    0,    0,    0,
      82,   876,  1,    0,    0,    0,    84,   886,  1,    0,    0,    0,
      86,   900,  1,    0,    0,    0,    88,   908,  1,    0,    0,    0,
      90,   922,  1,    0,    0,    0,    92,   936,  1,    0,    0,    0,
      94,   948,  1,    0,    0,    0,    96,   954,  1,    0,    0,    0,
      98,   960,  1,    0,    0,    0,    100,  963,  1,    0,    0,    0,
      102,  975,  1,    0,    0,    0,    104,  988,  1,    0,    0,    0,
      106,  1002, 1,    0,    0,    0,    108,  1035, 1,    0,    0,    0,
      110,  1037, 1,    0,    0,    0,    112,  1061, 1,    0,    0,    0,
      114,  1092, 1,    0,    0,    0,    116,  1116, 1,    0,    0,    0,
      118,  1129, 1,    0,    0,    0,    120,  1149, 1,    0,    0,    0,
      122,  1158, 1,    0,    0,    0,    124,  1166, 1,    0,    0,    0,
      126,  1179, 1,    0,    0,    0,    128,  1184, 1,    0,    0,    0,
      130,  1198, 1,    0,    0,    0,    132,  1200, 1,    0,    0,    0,
      134,  1205, 1,    0,    0,    0,    136,  1212, 1,    0,    0,    0,
      138,  1218, 1,    0,    0,    0,    140,  1232, 1,    0,    0,    0,
      142,  1238, 1,    0,    0,    0,    144,  1243, 1,    0,    0,    0,
      146,  1250, 1,    0,    0,    0,    148,  1257, 1,    0,    0,    0,
      150,  1262, 1,    0,    0,    0,    152,  1284, 1,    0,    0,    0,
      154,  1366, 1,    0,    0,    0,    156,  1454, 1,    0,    0,    0,
      158,  1456, 1,    0,    0,    0,    160,  1472, 1,    0,    0,    0,
      162,  1474, 1,    0,    0,    0,    164,  1478, 1,    0,    0,    0,
      166,  1480, 1,    0,    0,    0,    168,  1501, 1,    0,    0,    0,
      170,  1503, 1,    0,    0,    0,    172,  1509, 1,    0,    0,    0,
      174,  1527, 1,    0,    0,    0,    176,  1532, 1,    0,    0,    0,
      178,  1539, 1,    0,    0,    0,    180,  1544, 1,    0,    0,    0,
      182,  1546, 1,    0,    0,    0,    184,  1560, 1,    0,    0,    0,
      186,  1578, 1,    0,    0,    0,    188,  1591, 1,    0,    0,    0,
      190,  1594, 1,    0,    0,    0,    192,  1617, 1,    0,    0,    0,
      194,  1622, 1,    0,    0,    0,    196,  1624, 1,    0,    0,    0,
      198,  1631, 1,    0,    0,    0,    200,  1650, 1,    0,    0,    0,
      202,  1652, 1,    0,    0,    0,    204,  1669, 1,    0,    0,    0,
      206,  1671, 1,    0,    0,    0,    208,  1683, 1,    0,    0,    0,
      210,  1700, 1,    0,    0,    0,    212,  1714, 1,    0,    0,    0,
      214,  1723, 1,    0,    0,    0,    216,  1731, 1,    0,    0,    0,
      218,  1734, 1,    0,    0,    0,    220,  1738, 1,    0,    0,    0,
      222,  1745, 1,    0,    0,    0,    224,  1751, 1,    0,    0,    0,
      226,  1754, 1,    0,    0,    0,    228,  1765, 1,    0,    0,    0,
      230,  1779, 1,    0,    0,    0,    232,  1799, 1,    0,    0,    0,
      234,  1815, 1,    0,    0,    0,    236,  1820, 1,    0,    0,    0,
      238,  1827, 1,    0,    0,    0,    240,  1831, 1,    0,    0,    0,
      242,  1843, 1,    0,    0,    0,    244,  1857, 1,    0,    0,    0,
      246,  1875, 1,    0,    0,    0,    248,  1878, 1,    0,    0,    0,
      250,  1888, 1,    0,    0,    0,    252,  1890, 1,    0,    0,    0,
      254,  1903, 1,    0,    0,    0,    256,  1916, 1,    0,    0,    0,
      258,  1918, 1,    0,    0,    0,    260,  1924, 1,    0,    0,    0,
      262,  1939, 1,    0,    0,    0,    264,  1941, 1,    0,    0,    0,
      266,  1952, 1,    0,    0,    0,    268,  1975, 1,    0,    0,    0,
      270,  1980, 1,    0,    0,    0,    272,  1987, 1,    0,    0,    0,
      274,  1998, 1,    0,    0,    0,    276,  2018, 1,    0,    0,    0,
      278,  2020, 1,    0,    0,    0,    280,  2024, 1,    0,    0,    0,
      282,  2030, 1,    0,    0,    0,    284,  2043, 1,    0,    0,    0,
      286,  2048, 1,    0,    0,    0,    288,  2064, 1,    0,    0,    0,
      290,  2066, 1,    0,    0,    0,    292,  2070, 1,    0,    0,    0,
      294,  2072, 1,    0,    0,    0,    296,  2087, 1,    0,    0,    0,
      298,  2093, 1,    0,    0,    0,    300,  2097, 1,    0,    0,    0,
      302,  2106, 1,    0,    0,    0,    304,  2111, 1,    0,    0,    0,
      306,  2124, 1,    0,    0,    0,    308,  2132, 1,    0,    0,    0,
      310,  2137, 1,    0,    0,    0,    312,  2139, 1,    0,    0,    0,
      314,  2153, 1,    0,    0,    0,    316,  2170, 1,    0,    0,    0,
      318,  2184, 1,    0,    0,    0,    320,  2189, 1,    0,    0,    0,
      322,  2193, 1,    0,    0,    0,    324,  2196, 1,    0,    0,    0,
      326,  2199, 1,    0,    0,    0,    328,  2201, 1,    0,    0,    0,
      330,  2214, 1,    0,    0,    0,    332,  2233, 1,    0,    0,    0,
      334,  2240, 1,    0,    0,    0,    336,  2246, 1,    0,    0,    0,
      338,  2249, 1,    0,    0,    0,    340,  2264, 1,    0,    0,    0,
      342,  2267, 1,    0,    0,    0,    344,  2277, 1,    0,    0,    0,
      346,  2288, 1,    0,    0,    0,    348,  2333, 1,    0,    0,    0,
      350,  2339, 1,    0,    0,    0,    352,  2347, 1,    0,    0,    0,
      354,  2349, 1,    0,    0,    0,    356,  2357, 1,    0,    0,    0,
      358,  2365, 1,    0,    0,    0,    360,  2373, 1,    0,    0,    0,
      362,  2377, 1,    0,    0,    0,    364,  2384, 1,    0,    0,    0,
      366,  2392, 1,    0,    0,    0,    368,  2400, 1,    0,    0,    0,
      370,  2410, 1,    0,    0,    0,    372,  2418, 1,    0,    0,    0,
      374,  2427, 1,    0,    0,    0,    376,  2438, 1,    0,    0,    0,
      378,  2450, 1,    0,    0,    0,    380,  2452, 1,    0,    0,    0,
      382,  2460, 1,    0,    0,    0,    384,  2462, 1,    0,    0,    0,
      386,  2464, 1,    0,    0,    0,    388,  2466, 1,    0,    0,    0,
      390,  2470, 1,    0,    0,    0,    392,  394,  3,    140,  70,   0,
      393,  392,  1,    0,    0,    0,    394,  397,  1,    0,    0,    0,
      395,  393,  1,    0,    0,    0,    395,  396,  1,    0,    0,    0,
      396,  401,  1,    0,    0,    0,    397,  395,  1,    0,    0,    0,
      398,  400,  3,    34,   17,   0,    399,  398,  1,    0,    0,    0,
      400,  403,  1,    0,    0,    0,    401,  399,  1,    0,    0,    0,
      401,  402,  1,    0,    0,    0,    402,  404,  1,    0,    0,    0,
      403,  401,  1,    0,    0,    0,    404,  405,  5,    0,    0,    1,
      405,  1,    1,    0,    0,    0,    406,  407,  3,    338,  169,  0,
      407,  408,  5,    89,   0,    0,    408,  409,  3,    4,    2,    0,
      409,  3,    1,    0,    0,    0,    410,  414,  5,    130,  0,    0,
      411,  413,  3,    6,    3,    0,    412,  411,  1,    0,    0,    0,
      413,  416,  1,    0,    0,    0,    414,  412,  1,    0,    0,    0,
      414,  415,  1,    0,    0,    0,    415,  417,  1,    0,    0,    0,
      416,  414,  1,    0,    0,    0,    417,  435,  5,    131,  0,    0,
      418,  422,  5,    128,  0,    0,    419,  421,  3,    6,    3,    0,
      420,  419,  1,    0,    0,    0,    421,  424,  1,    0,    0,    0,
      422,  420,  1,    0,    0,    0,    422,  423,  1,    0,    0,    0,
      423,  425,  1,    0,    0,    0,    424,  422,  1,    0,    0,    0,
      425,  435,  5,    129,  0,    0,    426,  430,  5,    126,  0,    0,
      427,  429,  3,    6,    3,    0,    428,  427,  1,    0,    0,    0,
      429,  432,  1,    0,    0,    0,    430,  428,  1,    0,    0,    0,
      430,  431,  1,    0,    0,    0,    431,  433,  1,    0,    0,    0,
      432,  430,  1,    0,    0,    0,    433,  435,  5,    127,  0,    0,
      434,  410,  1,    0,    0,    0,    434,  418,  1,    0,    0,    0,
      434,  426,  1,    0,    0,    0,    435,  5,    1,    0,    0,    0,
      436,  438,  3,    8,    4,    0,    437,  436,  1,    0,    0,    0,
      438,  439,  1,    0,    0,    0,    439,  437,  1,    0,    0,    0,
      439,  440,  1,    0,    0,    0,    440,  443,  1,    0,    0,    0,
      441,  443,  3,    4,    2,    0,    442,  437,  1,    0,    0,    0,
      442,  441,  1,    0,    0,    0,    443,  7,    1,    0,    0,    0,
      444,  450,  3,    382,  191,  0,    445,  450,  3,    384,  192,  0,
      446,  450,  3,    386,  193,  0,    447,  450,  3,    30,   15,   0,
      448,  450,  5,    124,  0,    0,    449,  444,  1,    0,    0,    0,
      449,  445,  1,    0,    0,    0,    449,  446,  1,    0,    0,    0,
      449,  447,  1,    0,    0,    0,    449,  448,  1,    0,    0,    0,
      450,  9,    1,    0,    0,    0,    451,  452,  3,    338,  169,  0,
      452,  453,  5,    89,   0,    0,    453,  457,  5,    130,  0,    0,
      454,  456,  3,    6,    3,    0,    455,  454,  1,    0,    0,    0,
      456,  459,  1,    0,    0,    0,    457,  455,  1,    0,    0,    0,
      457,  458,  1,    0,    0,    0,    458,  460,  1,    0,    0,    0,
      459,  457,  1,    0,    0,    0,    460,  461,  5,    131,  0,    0,
      461,  462,  5,    118,  0,    0,    462,  487,  1,    0,    0,    0,
      463,  464,  3,    338,  169,  0,    464,  465,  5,    89,   0,    0,
      465,  469,  5,    128,  0,    0,    466,  468,  3,    6,    3,    0,
      467,  466,  1,    0,    0,    0,    468,  471,  1,    0,    0,    0,
      469,  467,  1,    0,    0,    0,    469,  470,  1,    0,    0,    0,
      470,  472,  1,    0,    0,    0,    471,  469,  1,    0,    0,    0,
      472,  473,  5,    129,  0,    0,    473,  474,  5,    118,  0,    0,
      474,  487,  1,    0,    0,    0,    475,  476,  3,    338,  169,  0,
      476,  477,  5,    89,   0,    0,    477,  481,  5,    126,  0,    0,
      478,  480,  3,    6,    3,    0,    479,  478,  1,    0,    0,    0,
      480,  483,  1,    0,    0,    0,    481,  479,  1,    0,    0,    0,
      481,  482,  1,    0,    0,    0,    482,  484,  1,    0,    0,    0,
      483,  481,  1,    0,    0,    0,    484,  485,  5,    127,  0,    0,
      485,  487,  1,    0,    0,    0,    486,  451,  1,    0,    0,    0,
      486,  463,  1,    0,    0,    0,    486,  475,  1,    0,    0,    0,
      487,  11,   1,    0,    0,    0,    488,  489,  5,    54,   0,    0,
      489,  490,  5,    89,   0,    0,    490,  491,  3,    378,  189,  0,
      491,  492,  3,    14,   7,    0,    492,  13,   1,    0,    0,    0,
      493,  494,  5,    130,  0,    0,    494,  495,  3,    16,   8,    0,
      495,  496,  5,    131,  0,    0,    496,  497,  5,    118,  0,    0,
      497,  508,  1,    0,    0,    0,    498,  499,  5,    128,  0,    0,
      499,  500,  3,    16,   8,    0,    500,  501,  5,    129,  0,    0,
      501,  502,  5,    118,  0,    0,    502,  508,  1,    0,    0,    0,
      503,  504,  5,    126,  0,    0,    504,  505,  3,    16,   8,    0,
      505,  506,  5,    127,  0,    0,    506,  508,  1,    0,    0,    0,
      507,  493,  1,    0,    0,    0,    507,  498,  1,    0,    0,    0,
      507,  503,  1,    0,    0,    0,    508,  15,   1,    0,    0,    0,
      509,  514,  3,    18,   9,    0,    510,  511,  5,    118,  0,    0,
      511,  513,  3,    18,   9,    0,    512,  510,  1,    0,    0,    0,
      513,  516,  1,    0,    0,    0,    514,  512,  1,    0,    0,    0,
      514,  515,  1,    0,    0,    0,    515,  518,  1,    0,    0,    0,
      516,  514,  1,    0,    0,    0,    517,  519,  5,    118,  0,    0,
      518,  517,  1,    0,    0,    0,    518,  519,  1,    0,    0,    0,
      519,  17,   1,    0,    0,    0,    520,  521,  3,    20,   10,   0,
      521,  522,  5,    122,  0,    0,    522,  523,  3,    32,   16,   0,
      523,  19,   1,    0,    0,    0,    524,  528,  5,    130,  0,    0,
      525,  527,  3,    22,   11,   0,    526,  525,  1,    0,    0,    0,
      527,  530,  1,    0,    0,    0,    528,  526,  1,    0,    0,    0,
      528,  529,  1,    0,    0,    0,    529,  531,  1,    0,    0,    0,
      530,  528,  1,    0,    0,    0,    531,  549,  5,    131,  0,    0,
      532,  536,  5,    128,  0,    0,    533,  535,  3,    22,   11,   0,
      534,  533,  1,    0,    0,    0,    535,  538,  1,    0,    0,    0,
      536,  534,  1,    0,    0,    0,    536,  537,  1,    0,    0,    0,
      537,  539,  1,    0,    0,    0,    538,  536,  1,    0,    0,    0,
      539,  549,  5,    129,  0,    0,    540,  544,  5,    126,  0,    0,
      541,  543,  3,    22,   11,   0,    542,  541,  1,    0,    0,    0,
      543,  546,  1,    0,    0,    0,    544,  542,  1,    0,    0,    0,
      544,  545,  1,    0,    0,    0,    545,  547,  1,    0,    0,    0,
      546,  544,  1,    0,    0,    0,    547,  549,  5,    127,  0,    0,
      548,  524,  1,    0,    0,    0,    548,  532,  1,    0,    0,    0,
      548,  540,  1,    0,    0,    0,    549,  21,   1,    0,    0,    0,
      550,  552,  3,    24,   12,   0,    551,  550,  1,    0,    0,    0,
      552,  553,  1,    0,    0,    0,    553,  551,  1,    0,    0,    0,
      553,  554,  1,    0,    0,    0,    554,  577,  1,    0,    0,    0,
      555,  577,  3,    20,   10,   0,    556,  559,  5,    124,  0,    0,
      557,  560,  3,    378,  189,  0,    558,  560,  5,    24,   0,    0,
      559,  557,  1,    0,    0,    0,    559,  558,  1,    0,    0,    0,
      560,  561,  1,    0,    0,    0,    561,  562,  5,    119,  0,    0,
      562,  577,  3,    26,   13,   0,    563,  564,  5,    124,  0,    0,
      564,  566,  5,    130,  0,    0,    565,  567,  3,    22,   11,   0,
      566,  565,  1,    0,    0,    0,    567,  568,  1,    0,    0,    0,
      568,  566,  1,    0,    0,    0,    568,  569,  1,    0,    0,    0,
      569,  570,  1,    0,    0,    0,    570,  572,  5,    131,  0,    0,
      571,  573,  3,    28,   14,   0,    572,  571,  1,    0,    0,    0,
      572,  573,  1,    0,    0,    0,    573,  574,  1,    0,    0,    0,
      574,  575,  3,    30,   15,   0,    575,  577,  1,    0,    0,    0,
      576,  551,  1,    0,    0,    0,    576,  555,  1,    0,    0,    0,
      576,  556,  1,    0,    0,    0,    576,  563,  1,    0,    0,    0,
      577,  23,   1,    0,    0,    0,    578,  583,  3,    382,  191,  0,
      579,  583,  3,    384,  192,  0,    580,  583,  3,    386,  193,  0,
      581,  583,  3,    30,   15,   0,    582,  578,  1,    0,    0,    0,
      582,  579,  1,    0,    0,    0,    582,  580,  1,    0,    0,    0,
      582,  581,  1,    0,    0,    0,    583,  25,   1,    0,    0,    0,
      584,  585,  3,    378,  189,  0,    585,  27,   1,    0,    0,    0,
      586,  591,  3,    382,  191,  0,    587,  591,  3,    384,  192,  0,
      588,  591,  3,    386,  193,  0,    589,  591,  5,    124,  0,    0,
      590,  586,  1,    0,    0,    0,    590,  587,  1,    0,    0,    0,
      590,  588,  1,    0,    0,    0,    590,  589,  1,    0,    0,    0,
      591,  29,   1,    0,    0,    0,    592,  593,  7,    0,    0,    0,
      593,  31,   1,    0,    0,    0,    594,  595,  3,    4,    2,    0,
      595,  33,   1,    0,    0,    0,    596,  598,  3,    142,  71,   0,
      597,  596,  1,    0,    0,    0,    598,  601,  1,    0,    0,    0,
      599,  597,  1,    0,    0,    0,    599,  600,  1,    0,    0,    0,
      600,  604,  1,    0,    0,    0,    601,  599,  1,    0,    0,    0,
      602,  605,  3,    36,   18,   0,    603,  605,  3,    38,   19,   0,
      604,  602,  1,    0,    0,    0,    604,  603,  1,    0,    0,    0,
      605,  35,   1,    0,    0,    0,    606,  608,  3,    376,  188,  0,
      607,  606,  1,    0,    0,    0,    607,  608,  1,    0,    0,    0,
      608,  622,  1,    0,    0,    0,    609,  623,  3,    40,   20,   0,
      610,  623,  3,    42,   21,   0,    611,  623,  3,    48,   24,   0,
      612,  623,  3,    52,   26,   0,    613,  623,  3,    72,   36,   0,
      614,  623,  3,    74,   37,   0,    615,  623,  3,    88,   44,   0,
      616,  623,  3,    100,  50,   0,    617,  623,  3,    102,  51,   0,
      618,  623,  3,    104,  52,   0,    619,  623,  3,    106,  53,   0,
      620,  623,  3,    108,  54,   0,    621,  623,  3,    114,  57,   0,
      622,  609,  1,    0,    0,    0,    622,  610,  1,    0,    0,    0,
      622,  611,  1,    0,    0,    0,    622,  612,  1,    0,    0,    0,
      622,  613,  1,    0,    0,    0,    622,  614,  1,    0,    0,    0,
      622,  615,  1,    0,    0,    0,    622,  616,  1,    0,    0,    0,
      622,  617,  1,    0,    0,    0,    622,  618,  1,    0,    0,    0,
      622,  619,  1,    0,    0,    0,    622,  620,  1,    0,    0,    0,
      622,  621,  1,    0,    0,    0,    623,  37,   1,    0,    0,    0,
      624,  627,  3,    10,   5,    0,    625,  627,  3,    12,   6,    0,
      626,  624,  1,    0,    0,    0,    626,  625,  1,    0,    0,    0,
      627,  39,   1,    0,    0,    0,    628,  630,  5,    32,   0,    0,
      629,  628,  1,    0,    0,    0,    629,  630,  1,    0,    0,    0,
      630,  631,  1,    0,    0,    0,    631,  632,  5,    18,   0,    0,
      632,  648,  3,    378,  189,  0,    633,  649,  5,    118,  0,    0,
      634,  638,  5,    126,  0,    0,    635,  637,  3,    140,  70,   0,
      636,  635,  1,    0,    0,    0,    637,  640,  1,    0,    0,    0,
      638,  636,  1,    0,    0,    0,    638,  639,  1,    0,    0,    0,
      639,  644,  1,    0,    0,    0,    640,  638,  1,    0,    0,    0,
      641,  643,  3,    34,   17,   0,    642,  641,  1,    0,    0,    0,
      643,  646,  1,    0,    0,    0,    644,  642,  1,    0,    0,    0,
      644,  645,  1,    0,    0,    0,    645,  647,  1,    0,    0,    0,
      646,  644,  1,    0,    0,    0,    647,  649,  5,    127,  0,    0,
      648,  633,  1,    0,    0,    0,    648,  634,  1,    0,    0,    0,
      649,  41,   1,    0,    0,    0,    650,  651,  5,    8,    0,    0,
      651,  652,  5,    5,    0,    0,    652,  654,  3,    44,   22,   0,
      653,  655,  3,    46,   23,   0,    654,  653,  1,    0,    0,    0,
      654,  655,  1,    0,    0,    0,    655,  656,  1,    0,    0,    0,
      656,  657,  5,    118,  0,    0,    657,  43,   1,    0,    0,    0,
      658,  661,  3,    378,  189,  0,    659,  661,  5,    24,   0,    0,
      660,  658,  1,    0,    0,    0,    660,  659,  1,    0,    0,    0,
      661,  45,   1,    0,    0,    0,    662,  665,  5,    1,    0,    0,
      663,  666,  3,    378,  189,  0,    664,  666,  5,    112,  0,    0,
      665,  663,  1,    0,    0,    0,    665,  664,  1,    0,    0,    0,
      666,  47,   1,    0,    0,    0,    667,  668,  5,    33,   0,    0,
      668,  669,  3,    50,   25,   0,    669,  670,  5,    118,  0,    0,
      670,  49,   1,    0,    0,    0,    671,  673,  3,    338,  169,  0,
      672,  671,  1,    0,    0,    0,    672,  673,  1,    0,    0,    0,
      673,  674,  1,    0,    0,    0,    674,  676,  5,    120,  0,    0,
      675,  672,  1,    0,    0,    0,    675,  676,  1,    0,    0,    0,
      676,  693,  1,    0,    0,    0,    677,  694,  5,    85,   0,    0,
      678,  690,  5,    126,  0,    0,    679,  684,  3,    50,   25,   0,
      680,  681,  5,    117,  0,    0,    681,  683,  3,    50,   25,   0,
      682,  680,  1,    0,    0,    0,    683,  686,  1,    0,    0,    0,
      684,  682,  1,    0,    0,    0,    684,  685,  1,    0,    0,    0,
      685,  688,  1,    0,    0,    0,    686,  684,  1,    0,    0,    0,
      687,  689,  5,    117,  0,    0,    688,  687,  1,    0,    0,    0,
      688,  689,  1,    0,    0,    0,    689,  691,  1,    0,    0,    0,
      690,  679,  1,    0,    0,    0,    690,  691,  1,    0,    0,    0,
      691,  692,  1,    0,    0,    0,    692,  694,  5,    127,  0,    0,
      693,  677,  1,    0,    0,    0,    693,  678,  1,    0,    0,    0,
      694,  704,  1,    0,    0,    0,    695,  701,  3,    338,  169,  0,
      696,  699,  5,    1,    0,    0,    697,  700,  3,    378,  189,  0,
      698,  700,  5,    112,  0,    0,    699,  697,  1,    0,    0,    0,
      699,  698,  1,    0,    0,    0,    700,  702,  1,    0,    0,    0,
      701,  696,  1,    0,    0,    0,    701,  702,  1,    0,    0,    0,
      702,  704,  1,    0,    0,    0,    703,  675,  1,    0,    0,    0,
      703,  695,  1,    0,    0,    0,    704,  51,   1,    0,    0,    0,
      705,  706,  3,    54,   27,   0,    706,  707,  5,    10,   0,    0,
      707,  709,  3,    378,  189,  0,    708,  710,  3,    118,  59,   0,
      709,  708,  1,    0,    0,    0,    709,  710,  1,    0,    0,    0,
      710,  711,  1,    0,    0,    0,    711,  713,  5,    130,  0,    0,
      712,  714,  3,    58,   29,   0,    713,  712,  1,    0,    0,    0,
      713,  714,  1,    0,    0,    0,    714,  715,  1,    0,    0,    0,
      715,  717,  5,    131,  0,    0,    716,  718,  3,    70,   35,   0,
      717,  716,  1,    0,    0,    0,    717,  718,  1,    0,    0,    0,
      718,  720,  1,    0,    0,    0,    719,  721,  3,    128,  64,   0,
      720,  719,  1,    0,    0,    0,    720,  721,  1,    0,    0,    0,
      721,  724,  1,    0,    0,    0,    722,  725,  3,    166,  83,   0,
      723,  725,  5,    118,  0,    0,    724,  722,  1,    0,    0,    0,
      724,  723,  1,    0,    0,    0,    725,  53,   1,    0,    0,    0,
      726,  728,  5,    3,    0,    0,    727,  726,  1,    0,    0,    0,
      727,  728,  1,    0,    0,    0,    728,  730,  1,    0,    0,    0,
      729,  731,  5,    36,   0,    0,    730,  729,  1,    0,    0,    0,
      730,  731,  1,    0,    0,    0,    731,  733,  1,    0,    0,    0,
      732,  734,  5,    32,   0,    0,    733,  732,  1,    0,    0,    0,
      733,  734,  1,    0,    0,    0,    734,  739,  1,    0,    0,    0,
      735,  737,  5,    8,    0,    0,    736,  738,  3,    56,   28,   0,
      737,  736,  1,    0,    0,    0,    737,  738,  1,    0,    0,    0,
      738,  740,  1,    0,    0,    0,    739,  735,  1,    0,    0,    0,
      739,  740,  1,    0,    0,    0,    740,  55,   1,    0,    0,    0,
      741,  742,  7,    1,    0,    0,    742,  57,   1,    0,    0,    0,
      743,  745,  3,    60,   30,   0,    744,  746,  5,    117,  0,    0,
      745,  744,  1,    0,    0,    0,    745,  746,  1,    0,    0,    0,
      746,  764,  1,    0,    0,    0,    747,  748,  3,    60,   30,   0,
      748,  749,  5,    117,  0,    0,    749,  751,  1,    0,    0,    0,
      750,  747,  1,    0,    0,    0,    750,  751,  1,    0,    0,    0,
      751,  752,  1,    0,    0,    0,    752,  757,  3,    66,   33,   0,
      753,  754,  5,    117,  0,    0,    754,  756,  3,    66,   33,   0,
      755,  753,  1,    0,    0,    0,    756,  759,  1,    0,    0,    0,
      757,  755,  1,    0,    0,    0,    757,  758,  1,    0,    0,    0,
      758,  761,  1,    0,    0,    0,    759,  757,  1,    0,    0,    0,
      760,  762,  5,    117,  0,    0,    761,  760,  1,    0,    0,    0,
      761,  762,  1,    0,    0,    0,    762,  764,  1,    0,    0,    0,
      763,  743,  1,    0,    0,    0,    763,  750,  1,    0,    0,    0,
      764,  59,   1,    0,    0,    0,    765,  767,  3,    142,  71,   0,
      766,  765,  1,    0,    0,    0,    767,  770,  1,    0,    0,    0,
      768,  766,  1,    0,    0,    0,    768,  769,  1,    0,    0,    0,
      769,  773,  1,    0,    0,    0,    770,  768,  1,    0,    0,    0,
      771,  774,  3,    62,   31,   0,    772,  774,  3,    64,   32,   0,
      773,  771,  1,    0,    0,    0,    773,  772,  1,    0,    0,    0,
      774,  61,   1,    0,    0,    0,    775,  777,  5,    90,   0,    0,
      776,  778,  3,    336,  168,  0,    777,  776,  1,    0,    0,    0,
      777,  778,  1,    0,    0,    0,    778,  780,  1,    0,    0,    0,
      779,  775,  1,    0,    0,    0,    779,  780,  1,    0,    0,    0,
      780,  782,  1,    0,    0,    0,    781,  783,  5,    20,   0,    0,
      782,  781,  1,    0,    0,    0,    782,  783,  1,    0,    0,    0,
      783,  784,  1,    0,    0,    0,    784,  785,  5,    24,   0,    0,
      785,  63,   1,    0,    0,    0,    786,  788,  5,    20,   0,    0,
      787,  786,  1,    0,    0,    0,    787,  788,  1,    0,    0,    0,
      788,  789,  1,    0,    0,    0,    789,  790,  5,    24,   0,    0,
      790,  791,  5,    119,  0,    0,    791,  792,  3,    286,  143,  0,
      792,  65,   1,    0,    0,    0,    793,  795,  3,    142,  71,   0,
      794,  793,  1,    0,    0,    0,    795,  798,  1,    0,    0,    0,
      796,  794,  1,    0,    0,    0,    796,  797,  1,    0,    0,    0,
      797,  802,  1,    0,    0,    0,    798,  796,  1,    0,    0,    0,
      799,  803,  3,    68,   34,   0,    800,  803,  5,    115,  0,    0,
      801,  803,  3,    286,  143,  0,    802,  799,  1,    0,    0,    0,
      802,  800,  1,    0,    0,    0,    802,  801,  1,    0,    0,    0,
      803,  67,   1,    0,    0,    0,    804,  805,  3,    240,  120,  0,
      805,  808,  5,    119,  0,    0,    806,  809,  3,    286,  143,  0,
      807,  809,  5,    115,  0,    0,    808,  806,  1,    0,    0,    0,
      808,  807,  1,    0,    0,    0,    809,  69,   1,    0,    0,    0,
      810,  811,  5,    121,  0,    0,    811,  812,  3,    286,  143,  0,
      812,  71,   1,    0,    0,    0,    813,  814,  5,    31,   0,    0,
      814,  816,  3,    378,  189,  0,    815,  817,  3,    118,  59,   0,
      816,  815,  1,    0,    0,    0,    816,  817,  1,    0,    0,    0,
      817,  819,  1,    0,    0,    0,    818,  820,  3,    128,  64,   0,
      819,  818,  1,    0,    0,    0,    819,  820,  1,    0,    0,    0,
      820,  823,  1,    0,    0,    0,    821,  822,  5,    104,  0,    0,
      822,  824,  3,    286,  143,  0,    823,  821,  1,    0,    0,    0,
      823,  824,  1,    0,    0,    0,    824,  825,  1,    0,    0,    0,
      825,  826,  5,    118,  0,    0,    826,  73,   1,    0,    0,    0,
      827,  830,  3,    76,   38,   0,    828,  830,  3,    78,   39,   0,
      829,  827,  1,    0,    0,    0,    829,  828,  1,    0,    0,    0,
      830,  75,   1,    0,    0,    0,    831,  832,  5,    27,   0,    0,
      832,  834,  3,    378,  189,  0,    833,  835,  3,    118,  59,   0,
      834,  833,  1,    0,    0,    0,    834,  835,  1,    0,    0,    0,
      835,  837,  1,    0,    0,    0,    836,  838,  3,    128,  64,   0,
      837,  836,  1,    0,    0,    0,    837,  838,  1,    0,    0,    0,
      838,  845,  1,    0,    0,    0,    839,  841,  5,    126,  0,    0,
      840,  842,  3,    80,   40,   0,    841,  840,  1,    0,    0,    0,
      841,  842,  1,    0,    0,    0,    842,  843,  1,    0,    0,    0,
      843,  846,  5,    127,  0,    0,    844,  846,  5,    118,  0,    0,
      845,  839,  1,    0,    0,    0,    845,  844,  1,    0,    0,    0,
      846,  77,   1,    0,    0,    0,    847,  848,  5,    27,   0,    0,
      848,  850,  3,    378,  189,  0,    849,  851,  3,    118,  59,   0,
      850,  849,  1,    0,    0,    0,    850,  851,  1,    0,    0,    0,
      851,  852,  1,    0,    0,    0,    852,  854,  5,    130,  0,    0,
      853,  855,  3,    84,   42,   0,    854,  853,  1,    0,    0,    0,
      854,  855,  1,    0,    0,    0,    855,  856,  1,    0,    0,    0,
      856,  858,  5,    131,  0,    0,    857,  859,  3,    128,  64,   0,
      858,  857,  1,    0,    0,    0,    858,  859,  1,    0,    0,    0,
      859,  860,  1,    0,    0,    0,    860,  861,  5,    118,  0,    0,
      861,  79,   1,    0,    0,    0,    862,  867,  3,    82,   41,   0,
      863,  864,  5,    117,  0,    0,    864,  866,  3,    82,   41,   0,
      865,  863,  1,    0,    0,    0,    866,  869,  1,    0,    0,    0,
      867,  865,  1,    0,    0,    0,    867,  868,  1,    0,    0,    0,
      868,  871,  1,    0,    0,    0,    869,  867,  1,    0,    0,    0,
      870,  872,  5,    117,  0,    0,    871,  870,  1,    0,    0,    0,
      871,  872,  1,    0,    0,    0,    872,  81,   1,    0,    0,    0,
      873,  875,  3,    142,  71,   0,    874,  873,  1,    0,    0,    0,
      875,  878,  1,    0,    0,    0,    876,  874,  1,    0,    0,    0,
      876,  877,  1,    0,    0,    0,    877,  880,  1,    0,    0,    0,
      878,  876,  1,    0,    0,    0,    879,  881,  3,    376,  188,  0,
      880,  879,  1,    0,    0,    0,    880,  881,  1,    0,    0,    0,
      881,  882,  1,    0,    0,    0,    882,  883,  3,    378,  189,  0,
      883,  884,  5,    119,  0,    0,    884,  885,  3,    286,  143,  0,
      885,  83,   1,    0,    0,    0,    886,  891,  3,    86,   43,   0,
      887,  888,  5,    117,  0,    0,    888,  890,  3,    86,   43,   0,
      889,  887,  1,    0,    0,    0,    890,  893,  1,    0,    0,    0,
      891,  889,  1,    0,    0,    0,    891,  892,  1,    0,    0,    0,
      892,  895,  1,    0,    0,    0,    893,  891,  1,    0,    0,    0,
      894,  896,  5,    117,  0,    0,    895,  894,  1,    0,    0,    0,
      895,  896,  1,    0,    0,    0,    896,  85,   1,    0,    0,    0,
      897,  899,  3,    142,  71,   0,    898,  897,  1,    0,    0,    0,
      899,  902,  1,    0,    0,    0,    900,  898,  1,    0,    0,    0,
      900,  901,  1,    0,    0,    0,    901,  904,  1,    0,    0,    0,
      902,  900,  1,    0,    0,    0,    903,  905,  3,    376,  188,  0,
      904,  903,  1,    0,    0,    0,    904,  905,  1,    0,    0,    0,
      905,  906,  1,    0,    0,    0,    906,  907,  3,    286,  143,  0,
      907,  87,   1,    0,    0,    0,    908,  909,  5,    7,    0,    0,
      909,  911,  3,    378,  189,  0,    910,  912,  3,    118,  59,   0,
      911,  910,  1,    0,    0,    0,    911,  912,  1,    0,    0,    0,
      912,  914,  1,    0,    0,    0,    913,  915,  3,    128,  64,   0,
      914,  913,  1,    0,    0,    0,    914,  915,  1,    0,    0,    0,
      915,  916,  1,    0,    0,    0,    916,  918,  5,    126,  0,    0,
      917,  919,  3,    90,   45,   0,    918,  917,  1,    0,    0,    0,
      918,  919,  1,    0,    0,    0,    919,  920,  1,    0,    0,    0,
      920,  921,  5,    127,  0,    0,    921,  89,   1,    0,    0,    0,
      922,  927,  3,    92,   46,   0,    923,  924,  5,    117,  0,    0,
      924,  926,  3,    92,   46,   0,    925,  923,  1,    0,    0,    0,
      926,  929,  1,    0,    0,    0,    927,  925,  1,    0,    0,    0,
      927,  928,  1,    0,    0,    0,    928,  931,  1,    0,    0,    0,
      929,  927,  1,    0,    0,    0,    930,  932,  5,    117,  0,    0,
      931,  930,  1,    0,    0,    0,    931,  932,  1,    0,    0,    0,
      932,  91,   1,    0,    0,    0,    933,  935,  3,    142,  71,   0,
      934,  933,  1,    0,    0,    0,    935,  938,  1,    0,    0,    0,
      936,  934,  1,    0,    0,    0,    936,  937,  1,    0,    0,    0,
      937,  940,  1,    0,    0,    0,    938,  936,  1,    0,    0,    0,
      939,  941,  3,    376,  188,  0,    940,  939,  1,    0,    0,    0,
      940,  941,  1,    0,    0,    0,    941,  942,  1,    0,    0,    0,
      942,  946,  3,    378,  189,  0,    943,  947,  3,    94,   47,   0,
      944,  947,  3,    96,   48,   0,    945,  947,  3,    98,   49,   0,
      946,  943,  1,    0,    0,    0,    946,  944,  1,    0,    0,    0,
      946,  945,  1,    0,    0,    0,    946,  947,  1,    0,    0,    0,
      947,  93,   1,    0,    0,    0,    948,  950,  5,    130,  0,    0,
      949,  951,  3,    84,   42,   0,    950,  949,  1,    0,    0,    0,
      950,  951,  1,    0,    0,    0,    951,  952,  1,    0,    0,    0,
      952,  953,  5,    131,  0,    0,    953,  95,   1,    0,    0,    0,
      954,  956,  5,    126,  0,    0,    955,  957,  3,    80,   40,   0,
      956,  955,  1,    0,    0,    0,    956,  957,  1,    0,    0,    0,
      957,  958,  1,    0,    0,    0,    958,  959,  5,    127,  0,    0,
      959,  97,   1,    0,    0,    0,    960,  961,  5,    104,  0,    0,
      961,  962,  3,    154,  77,   0,    962,  99,   1,    0,    0,    0,
      963,  964,  5,    52,   0,    0,    964,  966,  3,    378,  189,  0,
      965,  967,  3,    118,  59,   0,    966,  965,  1,    0,    0,    0,
      966,  967,  1,    0,    0,    0,    967,  969,  1,    0,    0,    0,
      968,  970,  3,    128,  64,   0,    969,  968,  1,    0,    0,    0,
      969,  970,  1,    0,    0,    0,    970,  971,  1,    0,    0,    0,
      971,  972,  5,    126,  0,    0,    972,  973,  3,    80,   40,   0,
      973,  974,  5,    127,  0,    0,    974,  101,  1,    0,    0,    0,
      975,  978,  5,    3,    0,    0,    976,  979,  3,    378,  189,  0,
      977,  979,  5,    112,  0,    0,    978,  976,  1,    0,    0,    0,
      978,  977,  1,    0,    0,    0,    979,  980,  1,    0,    0,    0,
      980,  981,  5,    119,  0,    0,    981,  984,  3,    286,  143,  0,
      982,  983,  5,    104,  0,    0,    983,  985,  3,    154,  77,   0,
      984,  982,  1,    0,    0,    0,    984,  985,  1,    0,    0,    0,
      985,  986,  1,    0,    0,    0,    986,  987,  5,    118,  0,    0,
      987,  103,  1,    0,    0,    0,    988,  990,  5,    26,   0,    0,
      989,  991,  5,    20,   0,    0,    990,  989,  1,    0,    0,    0,
      990,  991,  1,    0,    0,    0,    991,  992,  1,    0,    0,    0,
      992,  993,  3,    378,  189,  0,    993,  994,  5,    119,  0,    0,
      994,  997,  3,    286,  143,  0,    995,  996,  5,    104,  0,    0,
      996,  998,  3,    154,  77,   0,    997,  995,  1,    0,    0,    0,
      997,  998,  1,    0,    0,    0,    998,  999,  1,    0,    0,    0,
      999,  1000, 5,    118,  0,    0,    1000, 105,  1,    0,    0,    0,
      1001, 1003, 5,    32,   0,    0,    1002, 1001, 1,    0,    0,    0,
      1002, 1003, 1,    0,    0,    0,    1003, 1004, 1,    0,    0,    0,
      1004, 1005, 5,    29,   0,    0,    1005, 1007, 3,    378,  189,  0,
      1006, 1008, 3,    118,  59,   0,    1007, 1006, 1,    0,    0,    0,
      1007, 1008, 1,    0,    0,    0,    1008, 1013, 1,    0,    0,    0,
      1009, 1011, 5,    119,  0,    0,    1010, 1012, 3,    328,  164,  0,
      1011, 1010, 1,    0,    0,    0,    1011, 1012, 1,    0,    0,    0,
      1012, 1014, 1,    0,    0,    0,    1013, 1009, 1,    0,    0,    0,
      1013, 1014, 1,    0,    0,    0,    1014, 1016, 1,    0,    0,    0,
      1015, 1017, 3,    128,  64,   0,    1016, 1015, 1,    0,    0,    0,
      1016, 1017, 1,    0,    0,    0,    1017, 1018, 1,    0,    0,    0,
      1018, 1022, 5,    126,  0,    0,    1019, 1021, 3,    140,  70,   0,
      1020, 1019, 1,    0,    0,    0,    1021, 1024, 1,    0,    0,    0,
      1022, 1020, 1,    0,    0,    0,    1022, 1023, 1,    0,    0,    0,
      1023, 1028, 1,    0,    0,    0,    1024, 1022, 1,    0,    0,    0,
      1025, 1027, 3,    138,  69,   0,    1026, 1025, 1,    0,    0,    0,
      1027, 1030, 1,    0,    0,    0,    1028, 1026, 1,    0,    0,    0,
      1028, 1029, 1,    0,    0,    0,    1029, 1031, 1,    0,    0,    0,
      1030, 1028, 1,    0,    0,    0,    1031, 1032, 5,    127,  0,    0,
      1032, 107,  1,    0,    0,    0,    1033, 1036, 3,    110,  55,   0,
      1034, 1036, 3,    112,  56,   0,    1035, 1033, 1,    0,    0,    0,
      1035, 1034, 1,    0,    0,    0,    1036, 109,  1,    0,    0,    0,
      1037, 1039, 5,    13,   0,    0,    1038, 1040, 3,    118,  59,   0,
      1039, 1038, 1,    0,    0,    0,    1039, 1040, 1,    0,    0,    0,
      1040, 1041, 1,    0,    0,    0,    1041, 1043, 3,    286,  143,  0,
      1042, 1044, 3,    128,  64,   0,    1043, 1042, 1,    0,    0,    0,
      1043, 1044, 1,    0,    0,    0,    1044, 1045, 1,    0,    0,    0,
      1045, 1049, 5,    126,  0,    0,    1046, 1048, 3,    140,  70,   0,
      1047, 1046, 1,    0,    0,    0,    1048, 1051, 1,    0,    0,    0,
      1049, 1047, 1,    0,    0,    0,    1049, 1050, 1,    0,    0,    0,
      1050, 1055, 1,    0,    0,    0,    1051, 1049, 1,    0,    0,    0,
      1052, 1054, 3,    138,  69,   0,    1053, 1052, 1,    0,    0,    0,
      1054, 1057, 1,    0,    0,    0,    1055, 1053, 1,    0,    0,    0,
      1055, 1056, 1,    0,    0,    0,    1056, 1058, 1,    0,    0,    0,
      1057, 1055, 1,    0,    0,    0,    1058, 1059, 5,    127,  0,    0,
      1059, 111,  1,    0,    0,    0,    1060, 1062, 5,    32,   0,    0,
      1061, 1060, 1,    0,    0,    0,    1061, 1062, 1,    0,    0,    0,
      1062, 1063, 1,    0,    0,    0,    1063, 1065, 5,    13,   0,    0,
      1064, 1066, 3,    118,  59,   0,    1065, 1064, 1,    0,    0,    0,
      1065, 1066, 1,    0,    0,    0,    1066, 1068, 1,    0,    0,    0,
      1067, 1069, 5,    89,   0,    0,    1068, 1067, 1,    0,    0,    0,
      1068, 1069, 1,    0,    0,    0,    1069, 1070, 1,    0,    0,    0,
      1070, 1071, 3,    368,  184,  0,    1071, 1072, 5,    11,   0,    0,
      1072, 1074, 3,    286,  143,  0,    1073, 1075, 3,    128,  64,   0,
      1074, 1073, 1,    0,    0,    0,    1074, 1075, 1,    0,    0,    0,
      1075, 1076, 1,    0,    0,    0,    1076, 1080, 5,    126,  0,    0,
      1077, 1079, 3,    140,  70,   0,    1078, 1077, 1,    0,    0,    0,
      1079, 1082, 1,    0,    0,    0,    1080, 1078, 1,    0,    0,    0,
      1080, 1081, 1,    0,    0,    0,    1081, 1086, 1,    0,    0,    0,
      1082, 1080, 1,    0,    0,    0,    1083, 1085, 3,    138,  69,   0,
      1084, 1083, 1,    0,    0,    0,    1085, 1088, 1,    0,    0,    0,
      1086, 1084, 1,    0,    0,    0,    1086, 1087, 1,    0,    0,    0,
      1087, 1089, 1,    0,    0,    0,    1088, 1086, 1,    0,    0,    0,
      1089, 1090, 5,    127,  0,    0,    1090, 113,  1,    0,    0,    0,
      1091, 1093, 5,    32,   0,    0,    1092, 1091, 1,    0,    0,    0,
      1092, 1093, 1,    0,    0,    0,    1093, 1094, 1,    0,    0,    0,
      1094, 1096, 5,    8,    0,    0,    1095, 1097, 3,    56,   28,   0,
      1096, 1095, 1,    0,    0,    0,    1096, 1097, 1,    0,    0,    0,
      1097, 1098, 1,    0,    0,    0,    1098, 1102, 5,    126,  0,    0,
      1099, 1101, 3,    140,  70,   0,    1100, 1099, 1,    0,    0,    0,
      1101, 1104, 1,    0,    0,    0,    1102, 1100, 1,    0,    0,    0,
      1102, 1103, 1,    0,    0,    0,    1103, 1108, 1,    0,    0,    0,
      1104, 1102, 1,    0,    0,    0,    1105, 1107, 3,    116,  58,   0,
      1106, 1105, 1,    0,    0,    0,    1107, 1110, 1,    0,    0,    0,
      1108, 1106, 1,    0,    0,    0,    1108, 1109, 1,    0,    0,    0,
      1109, 1111, 1,    0,    0,    0,    1110, 1108, 1,    0,    0,    0,
      1111, 1112, 5,    127,  0,    0,    1112, 115,  1,    0,    0,    0,
      1113, 1115, 3,    142,  71,   0,    1114, 1113, 1,    0,    0,    0,
      1115, 1118, 1,    0,    0,    0,    1116, 1114, 1,    0,    0,    0,
      1116, 1117, 1,    0,    0,    0,    1117, 1127, 1,    0,    0,    0,
      1118, 1116, 1,    0,    0,    0,    1119, 1128, 3,    10,   5,    0,
      1120, 1122, 3,    376,  188,  0,    1121, 1120, 1,    0,    0,    0,
      1121, 1122, 1,    0,    0,    0,    1122, 1125, 1,    0,    0,    0,
      1123, 1126, 3,    104,  52,   0,    1124, 1126, 3,    52,   26,   0,
      1125, 1123, 1,    0,    0,    0,    1125, 1124, 1,    0,    0,    0,
      1126, 1128, 1,    0,    0,    0,    1127, 1119, 1,    0,    0,    0,
      1127, 1121, 1,    0,    0,    0,    1128, 117,  1,    0,    0,    0,
      1129, 1142, 5,    108,  0,    0,    1130, 1131, 3,    120,  60,   0,
      1131, 1132, 5,    117,  0,    0,    1132, 1134, 1,    0,    0,    0,
      1133, 1130, 1,    0,    0,    0,    1134, 1137, 1,    0,    0,    0,
      1135, 1133, 1,    0,    0,    0,    1135, 1136, 1,    0,    0,    0,
      1136, 1138, 1,    0,    0,    0,    1137, 1135, 1,    0,    0,    0,
      1138, 1140, 3,    120,  60,   0,    1139, 1141, 5,    117,  0,    0,
      1140, 1139, 1,    0,    0,    0,    1140, 1141, 1,    0,    0,    0,
      1141, 1143, 1,    0,    0,    0,    1142, 1135, 1,    0,    0,    0,
      1142, 1143, 1,    0,    0,    0,    1143, 1144, 1,    0,    0,    0,
      1144, 1145, 5,    107,  0,    0,    1145, 119,  1,    0,    0,    0,
      1146, 1148, 3,    142,  71,   0,    1147, 1146, 1,    0,    0,    0,
      1148, 1151, 1,    0,    0,    0,    1149, 1147, 1,    0,    0,    0,
      1149, 1150, 1,    0,    0,    0,    1150, 1155, 1,    0,    0,    0,
      1151, 1149, 1,    0,    0,    0,    1152, 1156, 3,    122,  61,   0,
      1153, 1156, 3,    124,  62,   0,    1154, 1156, 3,    126,  63,   0,
      1155, 1152, 1,    0,    0,    0,    1155, 1153, 1,    0,    0,    0,
      1155, 1154, 1,    0,    0,    0,    1156, 121,  1,    0,    0,    0,
      1157, 1159, 3,    142,  71,   0,    1158, 1157, 1,    0,    0,    0,
      1158, 1159, 1,    0,    0,    0,    1159, 1160, 1,    0,    0,    0,
      1160, 1163, 5,    82,   0,    0,    1161, 1162, 5,    119,  0,    0,
      1162, 1164, 3,    334,  167,  0,    1163, 1161, 1,    0,    0,    0,
      1163, 1164, 1,    0,    0,    0,    1164, 123,  1,    0,    0,    0,
      1165, 1167, 3,    142,  71,   0,    1166, 1165, 1,    0,    0,    0,
      1166, 1167, 1,    0,    0,    0,    1167, 1168, 1,    0,    0,    0,
      1168, 1173, 3,    378,  189,  0,    1169, 1171, 5,    119,  0,    0,
      1170, 1172, 3,    328,  164,  0,    1171, 1170, 1,    0,    0,    0,
      1171, 1172, 1,    0,    0,    0,    1172, 1174, 1,    0,    0,    0,
      1173, 1169, 1,    0,    0,    0,    1173, 1174, 1,    0,    0,    0,
      1174, 1177, 1,    0,    0,    0,    1175, 1176, 5,    104,  0,    0,
      1176, 1178, 3,    286,  143,  0,    1177, 1175, 1,    0,    0,    0,
      1177, 1178, 1,    0,    0,    0,    1178, 125,  1,    0,    0,    0,
      1179, 1180, 5,    3,    0,    0,    1180, 1181, 3,    378,  189,  0,
      1181, 1182, 5,    119,  0,    0,    1182, 1183, 3,    286,  143,  0,
      1183, 127,  1,    0,    0,    0,    1184, 1190, 5,    34,   0,    0,
      1185, 1186, 3,    130,  65,   0,    1186, 1187, 5,    117,  0,    0,
      1187, 1189, 1,    0,    0,    0,    1188, 1185, 1,    0,    0,    0,
      1189, 1192, 1,    0,    0,    0,    1190, 1188, 1,    0,    0,    0,
      1190, 1191, 1,    0,    0,    0,    1191, 1194, 1,    0,    0,    0,
      1192, 1190, 1,    0,    0,    0,    1193, 1195, 3,    130,  65,   0,
      1194, 1193, 1,    0,    0,    0,    1194, 1195, 1,    0,    0,    0,
      1195, 129,  1,    0,    0,    0,    1196, 1199, 3,    132,  66,   0,
      1197, 1199, 3,    134,  67,   0,    1198, 1196, 1,    0,    0,    0,
      1198, 1197, 1,    0,    0,    0,    1199, 131,  1,    0,    0,    0,
      1200, 1201, 3,    336,  168,  0,    1201, 1202, 5,    119,  0,    0,
      1202, 1203, 3,    334,  167,  0,    1203, 133,  1,    0,    0,    0,
      1204, 1206, 3,    136,  68,   0,    1205, 1204, 1,    0,    0,    0,
      1205, 1206, 1,    0,    0,    0,    1206, 1207, 1,    0,    0,    0,
      1207, 1208, 3,    286,  143,  0,    1208, 1210, 5,    119,  0,    0,
      1209, 1211, 3,    328,  164,  0,    1210, 1209, 1,    0,    0,    0,
      1210, 1211, 1,    0,    0,    0,    1211, 135,  1,    0,    0,    0,
      1212, 1213, 5,    11,   0,    0,    1213, 1214, 3,    118,  59,   0,
      1214, 137,  1,    0,    0,    0,    1215, 1217, 3,    142,  71,   0,
      1216, 1215, 1,    0,    0,    0,    1217, 1220, 1,    0,    0,    0,
      1218, 1216, 1,    0,    0,    0,    1218, 1219, 1,    0,    0,    0,
      1219, 1230, 1,    0,    0,    0,    1220, 1218, 1,    0,    0,    0,
      1221, 1231, 3,    10,   5,    0,    1222, 1224, 3,    376,  188,  0,
      1223, 1222, 1,    0,    0,    0,    1223, 1224, 1,    0,    0,    0,
      1224, 1228, 1,    0,    0,    0,    1225, 1229, 3,    72,   36,   0,
      1226, 1229, 3,    102,  51,   0,    1227, 1229, 3,    52,   26,   0,
      1228, 1225, 1,    0,    0,    0,    1228, 1226, 1,    0,    0,    0,
      1228, 1227, 1,    0,    0,    0,    1229, 1231, 1,    0,    0,    0,
      1230, 1221, 1,    0,    0,    0,    1230, 1223, 1,    0,    0,    0,
      1231, 139,  1,    0,    0,    0,    1232, 1233, 5,    123,  0,    0,
      1233, 1234, 5,    89,   0,    0,    1234, 1235, 5,    128,  0,    0,
      1235, 1236, 3,    144,  72,   0,    1236, 1237, 5,    129,  0,    0,
      1237, 141,  1,    0,    0,    0,    1238, 1239, 5,    123,  0,    0,
      1239, 1240, 5,    128,  0,    0,    1240, 1241, 3,    144,  72,   0,
      1241, 1242, 5,    129,  0,    0,    1242, 143,  1,    0,    0,    0,
      1243, 1245, 3,    338,  169,  0,    1244, 1246, 3,    146,  73,   0,
      1245, 1244, 1,    0,    0,    0,    1245, 1246, 1,    0,    0,    0,
      1246, 145,  1,    0,    0,    0,    1247, 1251, 3,    4,    2,    0,
      1248, 1249, 5,    104,  0,    0,    1249, 1251, 3,    162,  81,   0,
      1250, 1247, 1,    0,    0,    0,    1250, 1248, 1,    0,    0,    0,
      1251, 147,  1,    0,    0,    0,    1252, 1258, 5,    118,  0,    0,
      1253, 1258, 3,    34,   17,   0,    1254, 1258, 3,    150,  75,   0,
      1255, 1258, 3,    152,  76,   0,    1256, 1258, 3,    10,   5,    0,
      1257, 1252, 1,    0,    0,    0,    1257, 1253, 1,    0,    0,    0,
      1257, 1254, 1,    0,    0,    0,    1257, 1255, 1,    0,    0,    0,
      1257, 1256, 1,    0,    0,    0,    1258, 149,  1,    0,    0,    0,
      1259, 1261, 3,    142,  71,   0,    1260, 1259, 1,    0,    0,    0,
      1261, 1264, 1,    0,    0,    0,    1262, 1260, 1,    0,    0,    0,
      1262, 1263, 1,    0,    0,    0,    1263, 1265, 1,    0,    0,    0,
      1264, 1262, 1,    0,    0,    0,    1265, 1266, 5,    15,   0,    0,
      1266, 1269, 3,    242,  121,  0,    1267, 1268, 5,    119,  0,    0,
      1268, 1270, 3,    286,  143,  0,    1269, 1267, 1,    0,    0,    0,
      1269, 1270, 1,    0,    0,    0,    1270, 1273, 1,    0,    0,    0,
      1271, 1272, 5,    104,  0,    0,    1272, 1274, 3,    154,  77,   0,
      1273, 1271, 1,    0,    0,    0,    1273, 1274, 1,    0,    0,    0,
      1274, 1275, 1,    0,    0,    0,    1275, 1276, 5,    118,  0,    0,
      1276, 151,  1,    0,    0,    0,    1277, 1278, 3,    154,  77,   0,
      1278, 1279, 5,    118,  0,    0,    1279, 1285, 1,    0,    0,    0,
      1280, 1282, 3,    160,  80,   0,    1281, 1283, 5,    118,  0,    0,
      1282, 1281, 1,    0,    0,    0,    1282, 1283, 1,    0,    0,    0,
      1283, 1285, 1,    0,    0,    0,    1284, 1277, 1,    0,    0,    0,
      1284, 1280, 1,    0,    0,    0,    1285, 153,  1,    0,    0,    0,
      1286, 1288, 6,    77,   -1,   0,    1287, 1289, 3,    142,  71,   0,
      1288, 1287, 1,    0,    0,    0,    1289, 1290, 1,    0,    0,    0,
      1290, 1288, 1,    0,    0,    0,    1290, 1291, 1,    0,    0,    0,
      1291, 1292, 1,    0,    0,    0,    1292, 1293, 3,    154,  77,   40,
      1293, 1367, 1,    0,    0,    0,    1294, 1367, 3,    162,  81,   0,
      1295, 1367, 3,    164,  82,   0,    1296, 1298, 7,    2,    0,    0,
      1297, 1299, 5,    20,   0,    0,    1298, 1297, 1,    0,    0,    0,
      1298, 1299, 1,    0,    0,    0,    1299, 1300, 1,    0,    0,    0,
      1300, 1367, 3,    154,  77,   30,   1301, 1302, 5,    85,   0,    0,
      1302, 1367, 3,    154,  77,   29,   1303, 1304, 7,    3,    0,    0,
      1304, 1367, 3,    154,  77,   28,   1305, 1307, 5,    114,  0,    0,
      1306, 1308, 3,    154,  77,   0,    1307, 1306, 1,    0,    0,    0,
      1307, 1308, 1,    0,    0,    0,    1308, 1367, 1,    0,    0,    0,
      1309, 1310, 5,    116,  0,    0,    1310, 1367, 3,    154,  77,   15,
      1311, 1313, 5,    4,    0,    0,    1312, 1314, 5,    82,   0,    0,
      1313, 1312, 1,    0,    0,    0,    1313, 1314, 1,    0,    0,    0,
      1314, 1316, 1,    0,    0,    0,    1315, 1317, 3,    154,  77,   0,
      1316, 1315, 1,    0,    0,    0,    1316, 1317, 1,    0,    0,    0,
      1317, 1367, 1,    0,    0,    0,    1318, 1320, 5,    2,    0,    0,
      1319, 1321, 5,    82,   0,    0,    1320, 1319, 1,    0,    0,    0,
      1320, 1321, 1,    0,    0,    0,    1321, 1323, 1,    0,    0,    0,
      1322, 1324, 3,    154,  77,   0,    1323, 1322, 1,    0,    0,    0,
      1323, 1324, 1,    0,    0,    0,    1324, 1367, 1,    0,    0,    0,
      1325, 1327, 5,    23,   0,    0,    1326, 1328, 3,    154,  77,   0,
      1327, 1326, 1,    0,    0,    0,    1327, 1328, 1,    0,    0,    0,
      1328, 1367, 1,    0,    0,    0,    1329, 1333, 5,    130,  0,    0,
      1330, 1332, 3,    140,  70,   0,    1331, 1330, 1,    0,    0,    0,
      1332, 1335, 1,    0,    0,    0,    1333, 1331, 1,    0,    0,    0,
      1333, 1334, 1,    0,    0,    0,    1334, 1336, 1,    0,    0,    0,
      1335, 1333, 1,    0,    0,    0,    1336, 1337, 3,    154,  77,   0,
      1337, 1338, 5,    131,  0,    0,    1338, 1367, 1,    0,    0,    0,
      1339, 1343, 5,    128,  0,    0,    1340, 1342, 3,    140,  70,   0,
      1341, 1340, 1,    0,    0,    0,    1342, 1345, 1,    0,    0,    0,
      1343, 1341, 1,    0,    0,    0,    1343, 1344, 1,    0,    0,    0,
      1344, 1347, 1,    0,    0,    0,    1345, 1343, 1,    0,    0,    0,
      1346, 1348, 3,    174,  87,   0,    1347, 1346, 1,    0,    0,    0,
      1347, 1348, 1,    0,    0,    0,    1348, 1349, 1,    0,    0,    0,
      1349, 1367, 5,    129,  0,    0,    1350, 1354, 5,    130,  0,    0,
      1351, 1353, 3,    140,  70,   0,    1352, 1351, 1,    0,    0,    0,
      1353, 1356, 1,    0,    0,    0,    1354, 1352, 1,    0,    0,    0,
      1354, 1355, 1,    0,    0,    0,    1355, 1358, 1,    0,    0,    0,
      1356, 1354, 1,    0,    0,    0,    1357, 1359, 3,    176,  88,   0,
      1358, 1357, 1,    0,    0,    0,    1358, 1359, 1,    0,    0,    0,
      1359, 1360, 1,    0,    0,    0,    1360, 1367, 5,    131,  0,    0,
      1361, 1367, 3,    180,  90,   0,    1362, 1367, 3,    194,  97,   0,
      1363, 1367, 3,    208,  104,  0,    1364, 1367, 3,    160,  80,   0,
      1365, 1367, 3,    2,    1,    0,    1366, 1286, 1,    0,    0,    0,
      1366, 1294, 1,    0,    0,    0,    1366, 1295, 1,    0,    0,    0,
      1366, 1296, 1,    0,    0,    0,    1366, 1301, 1,    0,    0,    0,
      1366, 1303, 1,    0,    0,    0,    1366, 1305, 1,    0,    0,    0,
      1366, 1309, 1,    0,    0,    0,    1366, 1311, 1,    0,    0,    0,
      1366, 1318, 1,    0,    0,    0,    1366, 1325, 1,    0,    0,    0,
      1366, 1329, 1,    0,    0,    0,    1366, 1339, 1,    0,    0,    0,
      1366, 1350, 1,    0,    0,    0,    1366, 1361, 1,    0,    0,    0,
      1366, 1362, 1,    0,    0,    0,    1366, 1363, 1,    0,    0,    0,
      1366, 1364, 1,    0,    0,    0,    1366, 1365, 1,    0,    0,    0,
      1367, 1451, 1,    0,    0,    0,    1368, 1369, 10,   26,   0,    0,
      1369, 1370, 7,    4,    0,    0,    1370, 1450, 3,    154,  77,   27,
      1371, 1372, 10,   25,   0,    0,    1372, 1373, 7,    5,    0,    0,
      1373, 1450, 3,    154,  77,   26,   1374, 1377, 10,   24,   0,    0,
      1375, 1378, 3,    388,  194,  0,    1376, 1378, 3,    390,  195,  0,
      1377, 1375, 1,    0,    0,    0,    1377, 1376, 1,    0,    0,    0,
      1378, 1379, 1,    0,    0,    0,    1379, 1380, 3,    154,  77,   25,
      1380, 1450, 1,    0,    0,    0,    1381, 1382, 10,   23,   0,    0,
      1382, 1383, 5,    90,   0,    0,    1383, 1450, 3,    154,  77,   24,
      1384, 1385, 10,   22,   0,    0,    1385, 1386, 5,    88,   0,    0,
      1386, 1450, 3,    154,  77,   23,   1387, 1388, 10,   21,   0,    0,
      1388, 1389, 5,    91,   0,    0,    1389, 1450, 3,    154,  77,   22,
      1390, 1391, 10,   20,   0,    0,    1391, 1392, 3,    156,  78,   0,
      1392, 1393, 3,    154,  77,   21,   1393, 1450, 1,    0,    0,    0,
      1394, 1395, 10,   19,   0,    0,    1395, 1396, 5,    92,   0,    0,
      1396, 1450, 3,    154,  77,   20,   1397, 1398, 10,   18,   0,    0,
      1398, 1399, 5,    93,   0,    0,    1399, 1450, 3,    154,  77,   19,
      1400, 1401, 10,   14,   0,    0,    1401, 1402, 5,    116,  0,    0,
      1402, 1450, 3,    154,  77,   15,   1403, 1404, 10,   13,   0,    0,
      1404, 1405, 5,    104,  0,    0,    1405, 1450, 3,    154,  77,   14,
      1406, 1407, 10,   12,   0,    0,    1407, 1408, 3,    158,  79,   0,
      1408, 1409, 3,    154,  77,   13,   1409, 1450, 1,    0,    0,    0,
      1410, 1411, 10,   37,   0,    0,    1411, 1412, 5,    113,  0,    0,
      1412, 1413, 3,    344,  172,  0,    1413, 1415, 5,    130,  0,    0,
      1414, 1416, 3,    206,  103,  0,    1415, 1414, 1,    0,    0,    0,
      1415, 1416, 1,    0,    0,    0,    1416, 1417, 1,    0,    0,    0,
      1417, 1418, 5,    131,  0,    0,    1418, 1450, 1,    0,    0,    0,
      1419, 1420, 10,   36,   0,    0,    1420, 1421, 5,    113,  0,    0,
      1421, 1450, 3,    378,  189,  0,    1422, 1423, 10,   35,   0,    0,
      1423, 1424, 5,    113,  0,    0,    1424, 1450, 3,    178,  89,   0,
      1425, 1426, 10,   34,   0,    0,    1426, 1427, 5,    113,  0,    0,
      1427, 1450, 5,    37,   0,    0,    1428, 1429, 10,   33,   0,    0,
      1429, 1431, 5,    130,  0,    0,    1430, 1432, 3,    206,  103,  0,
      1431, 1430, 1,    0,    0,    0,    1431, 1432, 1,    0,    0,    0,
      1432, 1433, 1,    0,    0,    0,    1433, 1450, 5,    131,  0,    0,
      1434, 1435, 10,   32,   0,    0,    1435, 1436, 5,    128,  0,    0,
      1436, 1437, 3,    154,  77,   0,    1437, 1438, 5,    129,  0,    0,
      1438, 1450, 1,    0,    0,    0,    1439, 1440, 10,   31,   0,    0,
      1440, 1450, 5,    125,  0,    0,    1441, 1442, 10,   27,   0,    0,
      1442, 1443, 5,    1,    0,    0,    1443, 1450, 3,    288,  144,  0,
      1444, 1445, 10,   17,   0,    0,    1445, 1447, 5,    114,  0,    0,
      1446, 1448, 3,    154,  77,   0,    1447, 1446, 1,    0,    0,    0,
      1447, 1448, 1,    0,    0,    0,    1448, 1450, 1,    0,    0,    0,
      1449, 1368, 1,    0,    0,    0,    1449, 1371, 1,    0,    0,    0,
      1449, 1374, 1,    0,    0,    0,    1449, 1381, 1,    0,    0,    0,
      1449, 1384, 1,    0,    0,    0,    1449, 1387, 1,    0,    0,    0,
      1449, 1390, 1,    0,    0,    0,    1449, 1394, 1,    0,    0,    0,
      1449, 1397, 1,    0,    0,    0,    1449, 1400, 1,    0,    0,    0,
      1449, 1403, 1,    0,    0,    0,    1449, 1406, 1,    0,    0,    0,
      1449, 1410, 1,    0,    0,    0,    1449, 1419, 1,    0,    0,    0,
      1449, 1422, 1,    0,    0,    0,    1449, 1425, 1,    0,    0,    0,
      1449, 1428, 1,    0,    0,    0,    1449, 1434, 1,    0,    0,    0,
      1449, 1439, 1,    0,    0,    0,    1449, 1441, 1,    0,    0,    0,
      1449, 1444, 1,    0,    0,    0,    1450, 1453, 1,    0,    0,    0,
      1451, 1449, 1,    0,    0,    0,    1451, 1452, 1,    0,    0,    0,
      1452, 155,  1,    0,    0,    0,    1453, 1451, 1,    0,    0,    0,
      1454, 1455, 7,    6,    0,    0,    1455, 157,  1,    0,    0,    0,
      1456, 1457, 7,    7,    0,    0,    1457, 159,  1,    0,    0,    0,
      1458, 1460, 3,    142,  71,   0,    1459, 1458, 1,    0,    0,    0,
      1460, 1461, 1,    0,    0,    0,    1461, 1459, 1,    0,    0,    0,
      1461, 1462, 1,    0,    0,    0,    1462, 1463, 1,    0,    0,    0,
      1463, 1464, 3,    160,  80,   0,    1464, 1473, 1,    0,    0,    0,
      1465, 1473, 3,    166,  83,   0,    1466, 1473, 3,    170,  85,   0,
      1467, 1473, 3,    172,  86,   0,    1468, 1473, 3,    214,  107,  0,
      1469, 1473, 3,    226,  113,  0,    1470, 1473, 3,    228,  114,  0,
      1471, 1473, 3,    230,  115,  0,    1472, 1459, 1,    0,    0,    0,
      1472, 1465, 1,    0,    0,    0,    1472, 1466, 1,    0,    0,    0,
      1472, 1467, 1,    0,    0,    0,    1472, 1468, 1,    0,    0,    0,
      1472, 1469, 1,    0,    0,    0,    1472, 1470, 1,    0,    0,    0,
      1472, 1471, 1,    0,    0,    0,    1473, 161,  1,    0,    0,    0,
      1474, 1475, 7,    8,    0,    0,    1475, 163,  1,    0,    0,    0,
      1476, 1479, 3,    342,  171,  0,    1477, 1479, 3,    362,  181,  0,
      1478, 1476, 1,    0,    0,    0,    1478, 1477, 1,    0,    0,    0,
      1479, 165,  1,    0,    0,    0,    1480, 1484, 5,    126,  0,    0,
      1481, 1483, 3,    140,  70,   0,    1482, 1481, 1,    0,    0,    0,
      1483, 1486, 1,    0,    0,    0,    1484, 1482, 1,    0,    0,    0,
      1484, 1485, 1,    0,    0,    0,    1485, 1488, 1,    0,    0,    0,
      1486, 1484, 1,    0,    0,    0,    1487, 1489, 3,    168,  84,   0,
      1488, 1487, 1,    0,    0,    0,    1488, 1489, 1,    0,    0,    0,
      1489, 1490, 1,    0,    0,    0,    1490, 1491, 5,    127,  0,    0,
      1491, 167,  1,    0,    0,    0,    1492, 1494, 3,    148,  74,   0,
      1493, 1492, 1,    0,    0,    0,    1494, 1495, 1,    0,    0,    0,
      1495, 1493, 1,    0,    0,    0,    1495, 1496, 1,    0,    0,    0,
      1496, 1498, 1,    0,    0,    0,    1497, 1499, 3,    154,  77,   0,
      1498, 1497, 1,    0,    0,    0,    1498, 1499, 1,    0,    0,    0,
      1499, 1502, 1,    0,    0,    0,    1500, 1502, 3,    154,  77,   0,
      1501, 1493, 1,    0,    0,    0,    1501, 1500, 1,    0,    0,    0,
      1502, 169,  1,    0,    0,    0,    1503, 1505, 5,    36,   0,    0,
      1504, 1506, 5,    19,   0,    0,    1505, 1504, 1,    0,    0,    0,
      1505, 1506, 1,    0,    0,    0,    1506, 1507, 1,    0,    0,    0,
      1507, 1508, 3,    166,  83,   0,    1508, 171,  1,    0,    0,    0,
      1509, 1510, 5,    32,   0,    0,    1510, 1511, 3,    166,  83,   0,
      1511, 173,  1,    0,    0,    0,    1512, 1517, 3,    154,  77,   0,
      1513, 1514, 5,    117,  0,    0,    1514, 1516, 3,    154,  77,   0,
      1515, 1513, 1,    0,    0,    0,    1516, 1519, 1,    0,    0,    0,
      1517, 1515, 1,    0,    0,    0,    1517, 1518, 1,    0,    0,    0,
      1518, 1521, 1,    0,    0,    0,    1519, 1517, 1,    0,    0,    0,
      1520, 1522, 5,    117,  0,    0,    1521, 1520, 1,    0,    0,    0,
      1521, 1522, 1,    0,    0,    0,    1522, 1528, 1,    0,    0,    0,
      1523, 1524, 3,    154,  77,   0,    1524, 1525, 5,    118,  0,    0,
      1525, 1526, 3,    154,  77,   0,    1526, 1528, 1,    0,    0,    0,
      1527, 1512, 1,    0,    0,    0,    1527, 1523, 1,    0,    0,    0,
      1528, 175,  1,    0,    0,    0,    1529, 1530, 3,    154,  77,   0,
      1530, 1531, 5,    117,  0,    0,    1531, 1533, 1,    0,    0,    0,
      1532, 1529, 1,    0,    0,    0,    1533, 1534, 1,    0,    0,    0,
      1534, 1532, 1,    0,    0,    0,    1534, 1535, 1,    0,    0,    0,
      1535, 1537, 1,    0,    0,    0,    1536, 1538, 3,    154,  77,   0,
      1537, 1536, 1,    0,    0,    0,    1537, 1538, 1,    0,    0,    0,
      1538, 177,  1,    0,    0,    0,    1539, 1540, 5,    76,   0,    0,
      1540, 179,  1,    0,    0,    0,    1541, 1545, 3,    182,  91,   0,
      1542, 1545, 3,    190,  95,   0,    1543, 1545, 3,    192,  96,   0,
      1544, 1541, 1,    0,    0,    0,    1544, 1542, 1,    0,    0,    0,
      1544, 1543, 1,    0,    0,    0,    1545, 181,  1,    0,    0,    0,
      1546, 1547, 3,    342,  171,  0,    1547, 1551, 5,    126,  0,    0,
      1548, 1550, 3,    140,  70,   0,    1549, 1548, 1,    0,    0,    0,
      1550, 1553, 1,    0,    0,    0,    1551, 1549, 1,    0,    0,    0,
      1551, 1552, 1,    0,    0,    0,    1552, 1556, 1,    0,    0,    0,
      1553, 1551, 1,    0,    0,    0,    1554, 1557, 3,    184,  92,   0,
      1555, 1557, 3,    188,  94,   0,    1556, 1554, 1,    0,    0,    0,
      1556, 1555, 1,    0,    0,    0,    1556, 1557, 1,    0,    0,    0,
      1557, 1558, 1,    0,    0,    0,    1558, 1559, 5,    127,  0,    0,
      1559, 183,  1,    0,    0,    0,    1560, 1565, 3,    186,  93,   0,
      1561, 1562, 5,    117,  0,    0,    1562, 1564, 3,    186,  93,   0,
      1563, 1561, 1,    0,    0,    0,    1564, 1567, 1,    0,    0,    0,
      1565, 1563, 1,    0,    0,    0,    1565, 1566, 1,    0,    0,    0,
      1566, 1573, 1,    0,    0,    0,    1567, 1565, 1,    0,    0,    0,
      1568, 1569, 5,    117,  0,    0,    1569, 1574, 3,    188,  94,   0,
      1570, 1572, 5,    117,  0,    0,    1571, 1570, 1,    0,    0,    0,
      1571, 1572, 1,    0,    0,    0,    1572, 1574, 1,    0,    0,    0,
      1573, 1568, 1,    0,    0,    0,    1573, 1571, 1,    0,    0,    0,
      1574, 185,  1,    0,    0,    0,    1575, 1577, 3,    142,  71,   0,
      1576, 1575, 1,    0,    0,    0,    1577, 1580, 1,    0,    0,    0,
      1578, 1576, 1,    0,    0,    0,    1578, 1579, 1,    0,    0,    0,
      1579, 1589, 1,    0,    0,    0,    1580, 1578, 1,    0,    0,    0,
      1581, 1590, 3,    378,  189,  0,    1582, 1585, 3,    378,  189,  0,
      1583, 1585, 3,    178,  89,   0,    1584, 1582, 1,    0,    0,    0,
      1584, 1583, 1,    0,    0,    0,    1585, 1586, 1,    0,    0,    0,
      1586, 1587, 5,    119,  0,    0,    1587, 1588, 3,    154,  77,   0,
      1588, 1590, 1,    0,    0,    0,    1589, 1581, 1,    0,    0,    0,
      1589, 1584, 1,    0,    0,    0,    1590, 187,  1,    0,    0,    0,
      1591, 1592, 5,    114,  0,    0,    1592, 1593, 3,    154,  77,   0,
      1593, 189,  1,    0,    0,    0,    1594, 1595, 3,    342,  171,  0,
      1595, 1599, 5,    130,  0,    0,    1596, 1598, 3,    140,  70,   0,
      1597, 1596, 1,    0,    0,    0,    1598, 1601, 1,    0,    0,    0,
      1599, 1597, 1,    0,    0,    0,    1599, 1600, 1,    0,    0,    0,
      1600, 1613, 1,    0,    0,    0,    1601, 1599, 1,    0,    0,    0,
      1602, 1607, 3,    154,  77,   0,    1603, 1604, 5,    117,  0,    0,
      1604, 1606, 3,    154,  77,   0,    1605, 1603, 1,    0,    0,    0,
      1606, 1609, 1,    0,    0,    0,    1607, 1605, 1,    0,    0,    0,
      1607, 1608, 1,    0,    0,    0,    1608, 1611, 1,    0,    0,    0,
      1609, 1607, 1,    0,    0,    0,    1610, 1612, 5,    117,  0,    0,
      1611, 1610, 1,    0,    0,    0,    1611, 1612, 1,    0,    0,    0,
      1612, 1614, 1,    0,    0,    0,    1613, 1602, 1,    0,    0,    0,
      1613, 1614, 1,    0,    0,    0,    1614, 1615, 1,    0,    0,    0,
      1615, 1616, 5,    131,  0,    0,    1616, 191,  1,    0,    0,    0,
      1617, 1618, 3,    342,  171,  0,    1618, 193,  1,    0,    0,    0,
      1619, 1623, 3,    196,  98,   0,    1620, 1623, 3,    202,  101,  0,
      1621, 1623, 3,    204,  102,  0,    1622, 1619, 1,    0,    0,    0,
      1622, 1620, 1,    0,    0,    0,    1622, 1621, 1,    0,    0,    0,
      1623, 195,  1,    0,    0,    0,    1624, 1625, 3,    342,  171,  0,
      1625, 1627, 5,    126,  0,    0,    1626, 1628, 3,    198,  99,   0,
      1627, 1626, 1,    0,    0,    0,    1627, 1628, 1,    0,    0,    0,
      1628, 1629, 1,    0,    0,    0,    1629, 1630, 5,    127,  0,    0,
      1630, 197,  1,    0,    0,    0,    1631, 1636, 3,    200,  100,  0,
      1632, 1633, 5,    117,  0,    0,    1633, 1635, 3,    200,  100,  0,
      1634, 1632, 1,    0,    0,    0,    1635, 1638, 1,    0,    0,    0,
      1636, 1634, 1,    0,    0,    0,    1636, 1637, 1,    0,    0,    0,
      1637, 1640, 1,    0,    0,    0,    1638, 1636, 1,    0,    0,    0,
      1639, 1641, 5,    117,  0,    0,    1640, 1639, 1,    0,    0,    0,
      1640, 1641, 1,    0,    0,    0,    1641, 199,  1,    0,    0,    0,
      1642, 1651, 3,    378,  189,  0,    1643, 1646, 3,    378,  189,  0,
      1644, 1646, 3,    178,  89,   0,    1645, 1643, 1,    0,    0,    0,
      1645, 1644, 1,    0,    0,    0,    1646, 1647, 1,    0,    0,    0,
      1647, 1648, 5,    119,  0,    0,    1648, 1649, 3,    154,  77,   0,
      1649, 1651, 1,    0,    0,    0,    1650, 1642, 1,    0,    0,    0,
      1650, 1645, 1,    0,    0,    0,    1651, 201,  1,    0,    0,    0,
      1652, 1653, 3,    342,  171,  0,    1653, 1665, 5,    130,  0,    0,
      1654, 1659, 3,    154,  77,   0,    1655, 1656, 5,    117,  0,    0,
      1656, 1658, 3,    154,  77,   0,    1657, 1655, 1,    0,    0,    0,
      1658, 1661, 1,    0,    0,    0,    1659, 1657, 1,    0,    0,    0,
      1659, 1660, 1,    0,    0,    0,    1660, 1663, 1,    0,    0,    0,
      1661, 1659, 1,    0,    0,    0,    1662, 1664, 5,    117,  0,    0,
      1663, 1662, 1,    0,    0,    0,    1663, 1664, 1,    0,    0,    0,
      1664, 1666, 1,    0,    0,    0,    1665, 1654, 1,    0,    0,    0,
      1665, 1666, 1,    0,    0,    0,    1666, 1667, 1,    0,    0,    0,
      1667, 1668, 5,    131,  0,    0,    1668, 203,  1,    0,    0,    0,
      1669, 1670, 3,    342,  171,  0,    1670, 205,  1,    0,    0,    0,
      1671, 1676, 3,    154,  77,   0,    1672, 1673, 5,    117,  0,    0,
      1673, 1675, 3,    154,  77,   0,    1674, 1672, 1,    0,    0,    0,
      1675, 1678, 1,    0,    0,    0,    1676, 1674, 1,    0,    0,    0,
      1676, 1677, 1,    0,    0,    0,    1677, 1680, 1,    0,    0,    0,
      1678, 1676, 1,    0,    0,    0,    1679, 1681, 5,    117,  0,    0,
      1680, 1679, 1,    0,    0,    0,    1680, 1681, 1,    0,    0,    0,
      1681, 207,  1,    0,    0,    0,    1682, 1684, 5,    19,   0,    0,
      1683, 1682, 1,    0,    0,    0,    1683, 1684, 1,    0,    0,    0,
      1684, 1691, 1,    0,    0,    0,    1685, 1692, 5,    93,   0,    0,
      1686, 1688, 5,    91,   0,    0,    1687, 1689, 3,    210,  105,  0,
      1688, 1687, 1,    0,    0,    0,    1688, 1689, 1,    0,    0,    0,
      1689, 1690, 1,    0,    0,    0,    1690, 1692, 5,    91,   0,    0,
      1691, 1685, 1,    0,    0,    0,    1691, 1686, 1,    0,    0,    0,
      1692, 1698, 1,    0,    0,    0,    1693, 1699, 3,    154,  77,   0,
      1694, 1695, 5,    121,  0,    0,    1695, 1696, 3,    288,  144,  0,
      1696, 1697, 3,    166,  83,   0,    1697, 1699, 1,    0,    0,    0,
      1698, 1693, 1,    0,    0,    0,    1698, 1694, 1,    0,    0,    0,
      1699, 209,  1,    0,    0,    0,    1700, 1705, 3,    212,  106,  0,
      1701, 1702, 5,    117,  0,    0,    1702, 1704, 3,    212,  106,  0,
      1703, 1701, 1,    0,    0,    0,    1704, 1707, 1,    0,    0,    0,
      1705, 1703, 1,    0,    0,    0,    1705, 1706, 1,    0,    0,    0,
      1706, 1709, 1,    0,    0,    0,    1707, 1705, 1,    0,    0,    0,
      1708, 1710, 5,    117,  0,    0,    1709, 1708, 1,    0,    0,    0,
      1709, 1710, 1,    0,    0,    0,    1710, 211,  1,    0,    0,    0,
      1711, 1713, 3,    142,  71,   0,    1712, 1711, 1,    0,    0,    0,
      1713, 1716, 1,    0,    0,    0,    1714, 1712, 1,    0,    0,    0,
      1714, 1715, 1,    0,    0,    0,    1715, 1717, 1,    0,    0,    0,
      1716, 1714, 1,    0,    0,    0,    1717, 1720, 3,    240,  120,  0,
      1718, 1719, 5,    119,  0,    0,    1719, 1721, 3,    286,  143,  0,
      1720, 1718, 1,    0,    0,    0,    1720, 1721, 1,    0,    0,    0,
      1721, 213,  1,    0,    0,    0,    1722, 1724, 3,    224,  112,  0,
      1723, 1722, 1,    0,    0,    0,    1723, 1724, 1,    0,    0,    0,
      1724, 1729, 1,    0,    0,    0,    1725, 1730, 3,    216,  108,  0,
      1726, 1730, 3,    218,  109,  0,    1727, 1730, 3,    220,  110,  0,
      1728, 1730, 3,    222,  111,  0,    1729, 1725, 1,    0,    0,    0,
      1729, 1726, 1,    0,    0,    0,    1729, 1727, 1,    0,    0,    0,
      1729, 1728, 1,    0,    0,    0,    1730, 215,  1,    0,    0,    0,
      1731, 1732, 5,    16,   0,    0,    1732, 1733, 3,    166,  83,   0,
      1733, 217,  1,    0,    0,    0,    1734, 1735, 5,    35,   0,    0,
      1735, 1736, 3,    154,  77,   0,    1736, 1737, 3,    166,  83,   0,
      1737, 219,  1,    0,    0,    0,    1738, 1739, 5,    35,   0,    0,
      1739, 1740, 5,    15,   0,    0,    1740, 1741, 3,    240,  120,  0,
      1741, 1742, 5,    104,  0,    0,    1742, 1743, 3,    154,  77,   0,
      1743, 1744, 3,    166,  83,   0,    1744, 221,  1,    0,    0,    0,
      1745, 1746, 5,    11,   0,    0,    1746, 1747, 3,    240,  120,  0,
      1747, 1748, 5,    14,   0,    0,    1748, 1749, 3,    154,  77,   0,
      1749, 1750, 3,    166,  83,   0,    1750, 223,  1,    0,    0,    0,
      1751, 1752, 5,    82,   0,    0,    1752, 1753, 5,    119,  0,    0,
      1753, 225,  1,    0,    0,    0,    1754, 1755, 5,    12,   0,    0,
      1755, 1756, 3,    154,  77,   0,    1756, 1763, 3,    166,  83,   0,
      1757, 1761, 5,    6,    0,    0,    1758, 1762, 3,    166,  83,   0,
      1759, 1762, 3,    226,  113,  0,    1760, 1762, 3,    228,  114,  0,
      1761, 1758, 1,    0,    0,    0,    1761, 1759, 1,    0,    0,    0,
      1761, 1760, 1,    0,    0,    0,    1762, 1764, 1,    0,    0,    0,
      1763, 1757, 1,    0,    0,    0,    1763, 1764, 1,    0,    0,    0,
      1764, 227,  1,    0,    0,    0,    1765, 1766, 5,    12,   0,    0,
      1766, 1767, 5,    15,   0,    0,    1767, 1768, 3,    240,  120,  0,
      1768, 1769, 5,    104,  0,    0,    1769, 1770, 3,    154,  77,   0,
      1770, 1777, 3,    166,  83,   0,    1771, 1775, 5,    6,    0,    0,
      1772, 1776, 3,    166,  83,   0,    1773, 1776, 3,    226,  113,  0,
      1774, 1776, 3,    228,  114,  0,    1775, 1772, 1,    0,    0,    0,
      1775, 1773, 1,    0,    0,    0,    1775, 1774, 1,    0,    0,    0,
      1776, 1778, 1,    0,    0,    0,    1777, 1771, 1,    0,    0,    0,
      1777, 1778, 1,    0,    0,    0,    1778, 229,  1,    0,    0,    0,
      1779, 1780, 5,    17,   0,    0,    1780, 1781, 3,    154,  77,   0,
      1781, 1785, 5,    126,  0,    0,    1782, 1784, 3,    140,  70,   0,
      1783, 1782, 1,    0,    0,    0,    1784, 1787, 1,    0,    0,    0,
      1785, 1783, 1,    0,    0,    0,    1785, 1786, 1,    0,    0,    0,
      1786, 1789, 1,    0,    0,    0,    1787, 1785, 1,    0,    0,    0,
      1788, 1790, 3,    232,  116,  0,    1789, 1788, 1,    0,    0,    0,
      1789, 1790, 1,    0,    0,    0,    1790, 1791, 1,    0,    0,    0,
      1791, 1792, 5,    127,  0,    0,    1792, 231,  1,    0,    0,    0,
      1793, 1794, 3,    236,  118,  0,    1794, 1795, 5,    122,  0,    0,
      1795, 1796, 3,    234,  117,  0,    1796, 1798, 1,    0,    0,    0,
      1797, 1793, 1,    0,    0,    0,    1798, 1801, 1,    0,    0,    0,
      1799, 1797, 1,    0,    0,    0,    1799, 1800, 1,    0,    0,    0,
      1800, 1802, 1,    0,    0,    0,    1801, 1799, 1,    0,    0,    0,
      1802, 1803, 3,    236,  118,  0,    1803, 1804, 5,    122,  0,    0,
      1804, 1806, 3,    154,  77,   0,    1805, 1807, 5,    117,  0,    0,
      1806, 1805, 1,    0,    0,    0,    1806, 1807, 1,    0,    0,    0,
      1807, 233,  1,    0,    0,    0,    1808, 1809, 3,    154,  77,   0,
      1809, 1810, 5,    117,  0,    0,    1810, 1816, 1,    0,    0,    0,
      1811, 1813, 3,    160,  80,   0,    1812, 1814, 5,    117,  0,    0,
      1813, 1812, 1,    0,    0,    0,    1813, 1814, 1,    0,    0,    0,
      1814, 1816, 1,    0,    0,    0,    1815, 1808, 1,    0,    0,    0,
      1815, 1811, 1,    0,    0,    0,    1816, 235,  1,    0,    0,    0,
      1817, 1819, 3,    142,  71,   0,    1818, 1817, 1,    0,    0,    0,
      1819, 1822, 1,    0,    0,    0,    1820, 1818, 1,    0,    0,    0,
      1820, 1821, 1,    0,    0,    0,    1821, 1823, 1,    0,    0,    0,
      1822, 1820, 1,    0,    0,    0,    1823, 1825, 3,    240,  120,  0,
      1824, 1826, 3,    238,  119,  0,    1825, 1824, 1,    0,    0,    0,
      1825, 1826, 1,    0,    0,    0,    1826, 237,  1,    0,    0,    0,
      1827, 1828, 5,    12,   0,    0,    1828, 1829, 3,    154,  77,   0,
      1829, 239,  1,    0,    0,    0,    1830, 1832, 5,    91,   0,    0,
      1831, 1830, 1,    0,    0,    0,    1831, 1832, 1,    0,    0,    0,
      1832, 1833, 1,    0,    0,    0,    1833, 1838, 3,    242,  121,  0,
      1834, 1835, 5,    91,   0,    0,    1835, 1837, 3,    242,  121,  0,
      1836, 1834, 1,    0,    0,    0,    1837, 1840, 1,    0,    0,    0,
      1838, 1836, 1,    0,    0,    0,    1838, 1839, 1,    0,    0,    0,
      1839, 241,  1,    0,    0,    0,    1840, 1838, 1,    0,    0,    0,
      1841, 1844, 3,    244,  122,  0,    1842, 1844, 3,    254,  127,  0,
      1843, 1841, 1,    0,    0,    0,    1843, 1842, 1,    0,    0,    0,
      1844, 243,  1,    0,    0,    0,    1845, 1858, 3,    246,  123,  0,
      1846, 1858, 3,    248,  124,  0,    1847, 1858, 3,    250,  125,  0,
      1848, 1858, 3,    252,  126,  0,    1849, 1858, 3,    258,  129,  0,
      1850, 1858, 3,    260,  130,  0,    1851, 1858, 3,    270,  135,  0,
      1852, 1858, 3,    274,  137,  0,    1853, 1858, 3,    278,  139,  0,
      1854, 1858, 3,    280,  140,  0,    1855, 1858, 3,    284,  142,  0,
      1856, 1858, 3,    2,    1,    0,    1857, 1845, 1,    0,    0,    0,
      1857, 1846, 1,    0,    0,    0,    1857, 1847, 1,    0,    0,    0,
      1857, 1848, 1,    0,    0,    0,    1857, 1849, 1,    0,    0,    0,
      1857, 1850, 1,    0,    0,    0,    1857, 1851, 1,    0,    0,    0,
      1857, 1852, 1,    0,    0,    0,    1857, 1853, 1,    0,    0,    0,
      1857, 1854, 1,    0,    0,    0,    1857, 1855, 1,    0,    0,    0,
      1857, 1856, 1,    0,    0,    0,    1858, 245,  1,    0,    0,    0,
      1859, 1876, 5,    30,   0,    0,    1860, 1876, 5,    9,    0,    0,
      1861, 1876, 5,    70,   0,    0,    1862, 1876, 5,    73,   0,    0,
      1863, 1876, 5,    71,   0,    0,    1864, 1876, 5,    72,   0,    0,
      1865, 1876, 5,    74,   0,    0,    1866, 1876, 5,    75,   0,    0,
      1867, 1869, 5,    84,   0,    0,    1868, 1867, 1,    0,    0,    0,
      1868, 1869, 1,    0,    0,    0,    1869, 1870, 1,    0,    0,    0,
      1870, 1876, 5,    76,   0,    0,    1871, 1873, 5,    84,   0,    0,
      1872, 1871, 1,    0,    0,    0,    1872, 1873, 1,    0,    0,    0,
      1873, 1874, 1,    0,    0,    0,    1874, 1876, 5,    81,   0,    0,
      1875, 1859, 1,    0,    0,    0,    1875, 1860, 1,    0,    0,    0,
      1875, 1861, 1,    0,    0,    0,    1875, 1862, 1,    0,    0,    0,
      1875, 1863, 1,    0,    0,    0,    1875, 1864, 1,    0,    0,    0,
      1875, 1865, 1,    0,    0,    0,    1875, 1866, 1,    0,    0,    0,
      1875, 1868, 1,    0,    0,    0,    1875, 1872, 1,    0,    0,    0,
      1876, 247,  1,    0,    0,    0,    1877, 1879, 5,    22,   0,    0,
      1878, 1877, 1,    0,    0,    0,    1878, 1879, 1,    0,    0,    0,
      1879, 1881, 1,    0,    0,    0,    1880, 1882, 5,    20,   0,    0,
      1881, 1880, 1,    0,    0,    0,    1881, 1882, 1,    0,    0,    0,
      1882, 1883, 1,    0,    0,    0,    1883, 1886, 3,    378,  189,  0,
      1884, 1885, 5,    111,  0,    0,    1885, 1887, 3,    240,  120,  0,
      1886, 1884, 1,    0,    0,    0,    1886, 1887, 1,    0,    0,    0,
      1887, 249,  1,    0,    0,    0,    1888, 1889, 5,    112,  0,    0,
      1889, 251,  1,    0,    0,    0,    1890, 1891, 5,    114,  0,    0,
      1891, 253,  1,    0,    0,    0,    1892, 1893, 3,    256,  128,  0,
      1893, 1894, 5,    116,  0,    0,    1894, 1895, 3,    256,  128,  0,
      1895, 1904, 1,    0,    0,    0,    1896, 1897, 3,    256,  128,  0,
      1897, 1898, 5,    114,  0,    0,    1898, 1904, 1,    0,    0,    0,
      1899, 1900, 3,    256,  128,  0,    1900, 1901, 5,    115,  0,    0,
      1901, 1902, 3,    256,  128,  0,    1902, 1904, 1,    0,    0,    0,
      1903, 1892, 1,    0,    0,    0,    1903, 1896, 1,    0,    0,    0,
      1903, 1899, 1,    0,    0,    0,    1904, 255,  1,    0,    0,    0,
      1905, 1917, 5,    70,   0,    0,    1906, 1917, 5,    73,   0,    0,
      1907, 1909, 5,    84,   0,    0,    1908, 1907, 1,    0,    0,    0,
      1908, 1909, 1,    0,    0,    0,    1909, 1910, 1,    0,    0,    0,
      1910, 1917, 5,    76,   0,    0,    1911, 1913, 5,    84,   0,    0,
      1912, 1911, 1,    0,    0,    0,    1912, 1913, 1,    0,    0,    0,
      1913, 1914, 1,    0,    0,    0,    1914, 1917, 5,    81,   0,    0,
      1915, 1917, 3,    284,  142,  0,    1916, 1905, 1,    0,    0,    0,
      1916, 1906, 1,    0,    0,    0,    1916, 1908, 1,    0,    0,    0,
      1916, 1912, 1,    0,    0,    0,    1916, 1915, 1,    0,    0,    0,
      1917, 257,  1,    0,    0,    0,    1918, 1920, 7,    2,    0,    0,
      1919, 1921, 5,    20,   0,    0,    1920, 1919, 1,    0,    0,    0,
      1920, 1921, 1,    0,    0,    0,    1921, 1922, 1,    0,    0,    0,
      1922, 1923, 3,    244,  122,  0,    1923, 259,  1,    0,    0,    0,
      1924, 1925, 3,    342,  171,  0,    1925, 1927, 5,    126,  0,    0,
      1926, 1928, 3,    262,  131,  0,    1927, 1926, 1,    0,    0,    0,
      1927, 1928, 1,    0,    0,    0,    1928, 1929, 1,    0,    0,    0,
      1929, 1930, 5,    127,  0,    0,    1930, 261,  1,    0,    0,    0,
      1931, 1936, 3,    264,  132,  0,    1932, 1934, 5,    117,  0,    0,
      1933, 1935, 3,    268,  134,  0,    1934, 1933, 1,    0,    0,    0,
      1934, 1935, 1,    0,    0,    0,    1935, 1937, 1,    0,    0,    0,
      1936, 1932, 1,    0,    0,    0,    1936, 1937, 1,    0,    0,    0,
      1937, 1940, 1,    0,    0,    0,    1938, 1940, 3,    268,  134,  0,
      1939, 1931, 1,    0,    0,    0,    1939, 1938, 1,    0,    0,    0,
      1940, 263,  1,    0,    0,    0,    1941, 1946, 3,    266,  133,  0,
      1942, 1943, 5,    117,  0,    0,    1943, 1945, 3,    266,  133,  0,
      1944, 1942, 1,    0,    0,    0,    1945, 1948, 1,    0,    0,    0,
      1946, 1944, 1,    0,    0,    0,    1946, 1947, 1,    0,    0,    0,
      1947, 265,  1,    0,    0,    0,    1948, 1946, 1,    0,    0,    0,
      1949, 1951, 3,    142,  71,   0,    1950, 1949, 1,    0,    0,    0,
      1951, 1954, 1,    0,    0,    0,    1952, 1950, 1,    0,    0,    0,
      1952, 1953, 1,    0,    0,    0,    1953, 1970, 1,    0,    0,    0,
      1954, 1952, 1,    0,    0,    0,    1955, 1956, 3,    178,  89,   0,
      1956, 1957, 5,    119,  0,    0,    1957, 1958, 3,    240,  120,  0,
      1958, 1971, 1,    0,    0,    0,    1959, 1960, 3,    378,  189,  0,
      1960, 1961, 5,    119,  0,    0,    1961, 1962, 3,    240,  120,  0,
      1962, 1971, 1,    0,    0,    0,    1963, 1965, 5,    22,   0,    0,
      1964, 1963, 1,    0,    0,    0,    1964, 1965, 1,    0,    0,    0,
      1965, 1967, 1,    0,    0,    0,    1966, 1968, 5,    20,   0,    0,
      1967, 1966, 1,    0,    0,    0,    1967, 1968, 1,    0,    0,    0,
      1968, 1969, 1,    0,    0,    0,    1969, 1971, 3,    378,  189,  0,
      1970, 1955, 1,    0,    0,    0,    1970, 1959, 1,    0,    0,    0,
      1970, 1964, 1,    0,    0,    0,    1971, 267,  1,    0,    0,    0,
      1972, 1974, 3,    142,  71,   0,    1973, 1972, 1,    0,    0,    0,
      1974, 1977, 1,    0,    0,    0,    1975, 1973, 1,    0,    0,    0,
      1975, 1976, 1,    0,    0,    0,    1976, 1978, 1,    0,    0,    0,
      1977, 1975, 1,    0,    0,    0,    1978, 1979, 5,    114,  0,    0,
      1979, 269,  1,    0,    0,    0,    1980, 1981, 3,    342,  171,  0,
      1981, 1983, 5,    130,  0,    0,    1982, 1984, 3,    272,  136,  0,
      1983, 1982, 1,    0,    0,    0,    1983, 1984, 1,    0,    0,    0,
      1984, 1985, 1,    0,    0,    0,    1985, 1986, 5,    131,  0,    0,
      1986, 271,  1,    0,    0,    0,    1987, 1992, 3,    240,  120,  0,
      1988, 1989, 5,    117,  0,    0,    1989, 1991, 3,    240,  120,  0,
      1990, 1988, 1,    0,    0,    0,    1991, 1994, 1,    0,    0,    0,
      1992, 1990, 1,    0,    0,    0,    1992, 1993, 1,    0,    0,    0,
      1993, 1996, 1,    0,    0,    0,    1994, 1992, 1,    0,    0,    0,
      1995, 1997, 5,    117,  0,    0,    1996, 1995, 1,    0,    0,    0,
      1996, 1997, 1,    0,    0,    0,    1997, 273,  1,    0,    0,    0,
      1998, 2000, 5,    130,  0,    0,    1999, 2001, 3,    276,  138,  0,
      2000, 1999, 1,    0,    0,    0,    2000, 2001, 1,    0,    0,    0,
      2001, 2002, 1,    0,    0,    0,    2002, 2003, 5,    131,  0,    0,
      2003, 275,  1,    0,    0,    0,    2004, 2005, 3,    240,  120,  0,
      2005, 2006, 5,    117,  0,    0,    2006, 2019, 1,    0,    0,    0,
      2007, 2019, 3,    252,  126,  0,    2008, 2011, 3,    240,  120,  0,
      2009, 2010, 5,    117,  0,    0,    2010, 2012, 3,    240,  120,  0,
      2011, 2009, 1,    0,    0,    0,    2012, 2013, 1,    0,    0,    0,
      2013, 2011, 1,    0,    0,    0,    2013, 2014, 1,    0,    0,    0,
      2014, 2016, 1,    0,    0,    0,    2015, 2017, 5,    117,  0,    0,
      2016, 2015, 1,    0,    0,    0,    2016, 2017, 1,    0,    0,    0,
      2017, 2019, 1,    0,    0,    0,    2018, 2004, 1,    0,    0,    0,
      2018, 2007, 1,    0,    0,    0,    2018, 2008, 1,    0,    0,    0,
      2019, 277,  1,    0,    0,    0,    2020, 2021, 5,    130,  0,    0,
      2021, 2022, 3,    240,  120,  0,    2022, 2023, 5,    131,  0,    0,
      2023, 279,  1,    0,    0,    0,    2024, 2026, 5,    128,  0,    0,
      2025, 2027, 3,    282,  141,  0,    2026, 2025, 1,    0,    0,    0,
      2026, 2027, 1,    0,    0,    0,    2027, 2028, 1,    0,    0,    0,
      2028, 2029, 5,    129,  0,    0,    2029, 281,  1,    0,    0,    0,
      2030, 2035, 3,    240,  120,  0,    2031, 2032, 5,    117,  0,    0,
      2032, 2034, 3,    240,  120,  0,    2033, 2031, 1,    0,    0,    0,
      2034, 2037, 1,    0,    0,    0,    2035, 2033, 1,    0,    0,    0,
      2035, 2036, 1,    0,    0,    0,    2036, 2039, 1,    0,    0,    0,
      2037, 2035, 1,    0,    0,    0,    2038, 2040, 5,    117,  0,    0,
      2039, 2038, 1,    0,    0,    0,    2039, 2040, 1,    0,    0,    0,
      2040, 283,  1,    0,    0,    0,    2041, 2044, 3,    342,  171,  0,
      2042, 2044, 3,    362,  181,  0,    2043, 2041, 1,    0,    0,    0,
      2043, 2042, 1,    0,    0,    0,    2044, 285,  1,    0,    0,    0,
      2045, 2049, 3,    288,  144,  0,    2046, 2049, 3,    322,  161,  0,
      2047, 2049, 3,    318,  159,  0,    2048, 2045, 1,    0,    0,    0,
      2048, 2046, 1,    0,    0,    0,    2048, 2047, 1,    0,    0,    0,
      2049, 287,  1,    0,    0,    0,    2050, 2065, 3,    290,  145,  0,
      2051, 2065, 3,    324,  162,  0,    2052, 2065, 3,    320,  160,  0,
      2053, 2065, 3,    368,  184,  0,    2054, 2065, 3,    294,  147,  0,
      2055, 2065, 3,    292,  146,  0,    2056, 2065, 3,    302,  151,  0,
      2057, 2065, 3,    300,  150,  0,    2058, 2065, 3,    296,  148,  0,
      2059, 2065, 3,    298,  149,  0,    2060, 2065, 3,    326,  163,  0,
      2061, 2065, 3,    366,  183,  0,    2062, 2065, 3,    304,  152,  0,
      2063, 2065, 3,    2,    1,    0,    2064, 2050, 1,    0,    0,    0,
      2064, 2051, 1,    0,    0,    0,    2064, 2052, 1,    0,    0,    0,
      2064, 2053, 1,    0,    0,    0,    2064, 2054, 1,    0,    0,    0,
      2064, 2055, 1,    0,    0,    0,    2064, 2056, 1,    0,    0,    0,
      2064, 2057, 1,    0,    0,    0,    2064, 2058, 1,    0,    0,    0,
      2064, 2059, 1,    0,    0,    0,    2064, 2060, 1,    0,    0,    0,
      2064, 2061, 1,    0,    0,    0,    2064, 2062, 1,    0,    0,    0,
      2064, 2063, 1,    0,    0,    0,    2065, 289,  1,    0,    0,    0,
      2066, 2067, 5,    130,  0,    0,    2067, 2068, 3,    286,  143,  0,
      2068, 2069, 5,    131,  0,    0,    2069, 291,  1,    0,    0,    0,
      2070, 2071, 5,    89,   0,    0,    2071, 293,  1,    0,    0,    0,
      2072, 2083, 5,    130,  0,    0,    2073, 2074, 3,    286,  143,  0,
      2074, 2075, 5,    117,  0,    0,    2075, 2077, 1,    0,    0,    0,
      2076, 2073, 1,    0,    0,    0,    2077, 2078, 1,    0,    0,    0,
      2078, 2076, 1,    0,    0,    0,    2078, 2079, 1,    0,    0,    0,
      2079, 2081, 1,    0,    0,    0,    2080, 2082, 3,    286,  143,  0,
      2081, 2080, 1,    0,    0,    0,    2081, 2082, 1,    0,    0,    0,
      2082, 2084, 1,    0,    0,    0,    2083, 2076, 1,    0,    0,    0,
      2083, 2084, 1,    0,    0,    0,    2084, 2085, 1,    0,    0,    0,
      2085, 2086, 5,    131,  0,    0,    2086, 295,  1,    0,    0,    0,
      2087, 2088, 5,    128,  0,    0,    2088, 2089, 3,    286,  143,  0,
      2089, 2090, 5,    118,  0,    0,    2090, 2091, 3,    154,  77,   0,
      2091, 2092, 5,    129,  0,    0,    2092, 297,  1,    0,    0,    0,
      2093, 2094, 5,    128,  0,    0,    2094, 2095, 3,    286,  143,  0,
      2095, 2096, 5,    129,  0,    0,    2096, 299,  1,    0,    0,    0,
      2097, 2099, 5,    90,   0,    0,    2098, 2100, 3,    336,  168,  0,
      2099, 2098, 1,    0,    0,    0,    2099, 2100, 1,    0,    0,    0,
      2100, 2102, 1,    0,    0,    0,    2101, 2103, 5,    20,   0,    0,
      2102, 2101, 1,    0,    0,    0,    2102, 2103, 1,    0,    0,    0,
      2103, 2104, 1,    0,    0,    0,    2104, 2105, 3,    288,  144,  0,
      2105, 301,  1,    0,    0,    0,    2106, 2107, 5,    85,   0,    0,
      2107, 2108, 7,    9,    0,    0,    2108, 2109, 3,    288,  144,  0,
      2109, 303,  1,    0,    0,    0,    2110, 2112, 3,    136,  68,   0,
      2111, 2110, 1,    0,    0,    0,    2111, 2112, 1,    0,    0,    0,
      2112, 2113, 1,    0,    0,    0,    2113, 2114, 3,    306,  153,  0,
      2114, 2115, 5,    10,   0,    0,    2115, 2117, 5,    130,  0,    0,
      2116, 2118, 3,    310,  155,  0,    2117, 2116, 1,    0,    0,    0,
      2117, 2118, 1,    0,    0,    0,    2118, 2119, 1,    0,    0,    0,
      2119, 2121, 5,    131,  0,    0,    2120, 2122, 3,    308,  154,  0,
      2121, 2120, 1,    0,    0,    0,    2121, 2122, 1,    0,    0,    0,
      2122, 305,  1,    0,    0,    0,    2123, 2125, 5,    32,   0,    0,
      2124, 2123, 1,    0,    0,    0,    2124, 2125, 1,    0,    0,    0,
      2125, 2130, 1,    0,    0,    0,    2126, 2128, 5,    8,    0,    0,
      2127, 2129, 3,    56,   28,   0,    2128, 2127, 1,    0,    0,    0,
      2128, 2129, 1,    0,    0,    0,    2129, 2131, 1,    0,    0,    0,
      2130, 2126, 1,    0,    0,    0,    2130, 2131, 1,    0,    0,    0,
      2131, 307,  1,    0,    0,    0,    2132, 2133, 5,    121,  0,    0,
      2133, 2134, 3,    288,  144,  0,    2134, 309,  1,    0,    0,    0,
      2135, 2138, 3,    312,  156,  0,    2136, 2138, 3,    316,  158,  0,
      2137, 2135, 1,    0,    0,    0,    2137, 2136, 1,    0,    0,    0,
      2138, 311,  1,    0,    0,    0,    2139, 2144, 3,    314,  157,  0,
      2140, 2141, 5,    117,  0,    0,    2141, 2143, 3,    314,  157,  0,
      2142, 2140, 1,    0,    0,    0,    2143, 2146, 1,    0,    0,    0,
      2144, 2142, 1,    0,    0,    0,    2144, 2145, 1,    0,    0,    0,
      2145, 2148, 1,    0,    0,    0,    2146, 2144, 1,    0,    0,    0,
      2147, 2149, 5,    117,  0,    0,    2148, 2147, 1,    0,    0,    0,
      2148, 2149, 1,    0,    0,    0,    2149, 313,  1,    0,    0,    0,
      2150, 2152, 3,    142,  71,   0,    2151, 2150, 1,    0,    0,    0,
      2152, 2155, 1,    0,    0,    0,    2153, 2151, 1,    0,    0,    0,
      2153, 2154, 1,    0,    0,    0,    2154, 2161, 1,    0,    0,    0,
      2155, 2153, 1,    0,    0,    0,    2156, 2159, 3,    378,  189,  0,
      2157, 2159, 5,    112,  0,    0,    2158, 2156, 1,    0,    0,    0,
      2158, 2157, 1,    0,    0,    0,    2159, 2160, 1,    0,    0,    0,
      2160, 2162, 5,    119,  0,    0,    2161, 2158, 1,    0,    0,    0,
      2161, 2162, 1,    0,    0,    0,    2162, 2163, 1,    0,    0,    0,
      2163, 2164, 3,    286,  143,  0,    2164, 315,  1,    0,    0,    0,
      2165, 2166, 3,    314,  157,  0,    2166, 2167, 5,    117,  0,    0,
      2167, 2169, 1,    0,    0,    0,    2168, 2165, 1,    0,    0,    0,
      2169, 2172, 1,    0,    0,    0,    2170, 2168, 1,    0,    0,    0,
      2170, 2171, 1,    0,    0,    0,    2171, 2173, 1,    0,    0,    0,
      2172, 2170, 1,    0,    0,    0,    2173, 2174, 3,    314,  157,  0,
      2174, 2178, 5,    117,  0,    0,    2175, 2177, 3,    142,  71,   0,
      2176, 2175, 1,    0,    0,    0,    2177, 2180, 1,    0,    0,    0,
      2178, 2176, 1,    0,    0,    0,    2178, 2179, 1,    0,    0,    0,
      2179, 2181, 1,    0,    0,    0,    2180, 2178, 1,    0,    0,    0,
      2181, 2182, 5,    115,  0,    0,    2182, 317,  1,    0,    0,    0,
      2183, 2185, 5,    38,   0,    0,    2184, 2183, 1,    0,    0,    0,
      2184, 2185, 1,    0,    0,    0,    2185, 2186, 1,    0,    0,    0,
      2186, 2187, 3,    328,  164,  0,    2187, 319,  1,    0,    0,    0,
      2188, 2190, 5,    38,   0,    0,    2189, 2188, 1,    0,    0,    0,
      2189, 2190, 1,    0,    0,    0,    2190, 2191, 1,    0,    0,    0,
      2191, 2192, 3,    332,  166,  0,    2192, 321,  1,    0,    0,    0,
      2193, 2194, 5,    13,   0,    0,    2194, 2195, 3,    328,  164,  0,
      2195, 323,  1,    0,    0,    0,    2196, 2197, 5,    13,   0,    0,
      2197, 2198, 3,    332,  166,  0,    2198, 325,  1,    0,    0,    0,
      2199, 2200, 5,    112,  0,    0,    2200, 327,  1,    0,    0,    0,
      2201, 2206, 3,    330,  165,  0,    2202, 2203, 5,    83,   0,    0,
      2203, 2205, 3,    330,  165,  0,    2204, 2202, 1,    0,    0,    0,
      2205, 2208, 1,    0,    0,    0,    2206, 2204, 1,    0,    0,    0,
      2206, 2207, 1,    0,    0,    0,    2207, 2210, 1,    0,    0,    0,
      2208, 2206, 1,    0,    0,    0,    2209, 2211, 5,    83,   0,    0,
      2210, 2209, 1,    0,    0,    0,    2210, 2211, 1,    0,    0,    0,
      2211, 329,  1,    0,    0,    0,    2212, 2215, 3,    336,  168,  0,
      2213, 2215, 3,    332,  166,  0,    2214, 2212, 1,    0,    0,    0,
      2214, 2213, 1,    0,    0,    0,    2215, 331,  1,    0,    0,    0,
      2216, 2218, 5,    125,  0,    0,    2217, 2216, 1,    0,    0,    0,
      2217, 2218, 1,    0,    0,    0,    2218, 2220, 1,    0,    0,    0,
      2219, 2221, 3,    136,  68,   0,    2220, 2219, 1,    0,    0,    0,
      2220, 2221, 1,    0,    0,    0,    2221, 2222, 1,    0,    0,    0,
      2222, 2234, 3,    368,  184,  0,    2223, 2225, 5,    130,  0,    0,
      2224, 2226, 5,    125,  0,    0,    2225, 2224, 1,    0,    0,    0,
      2225, 2226, 1,    0,    0,    0,    2226, 2228, 1,    0,    0,    0,
      2227, 2229, 3,    136,  68,   0,    2228, 2227, 1,    0,    0,    0,
      2228, 2229, 1,    0,    0,    0,    2229, 2230, 1,    0,    0,    0,
      2230, 2231, 3,    368,  184,  0,    2231, 2232, 5,    131,  0,    0,
      2232, 2234, 1,    0,    0,    0,    2233, 2217, 1,    0,    0,    0,
      2233, 2223, 1,    0,    0,    0,    2234, 333,  1,    0,    0,    0,
      2235, 2236, 3,    336,  168,  0,    2236, 2237, 5,    83,   0,    0,
      2237, 2239, 1,    0,    0,    0,    2238, 2235, 1,    0,    0,    0,
      2239, 2242, 1,    0,    0,    0,    2240, 2238, 1,    0,    0,    0,
      2240, 2241, 1,    0,    0,    0,    2241, 2244, 1,    0,    0,    0,
      2242, 2240, 1,    0,    0,    0,    2243, 2245, 3,    336,  168,  0,
      2244, 2243, 1,    0,    0,    0,    2244, 2245, 1,    0,    0,    0,
      2245, 335,  1,    0,    0,    0,    2246, 2247, 7,    10,   0,    0,
      2247, 337,  1,    0,    0,    0,    2248, 2250, 5,    120,  0,    0,
      2249, 2248, 1,    0,    0,    0,    2249, 2250, 1,    0,    0,    0,
      2250, 2251, 1,    0,    0,    0,    2251, 2256, 3,    340,  170,  0,
      2252, 2253, 5,    120,  0,    0,    2253, 2255, 3,    340,  170,  0,
      2254, 2252, 1,    0,    0,    0,    2255, 2258, 1,    0,    0,    0,
      2256, 2254, 1,    0,    0,    0,    2256, 2257, 1,    0,    0,    0,
      2257, 339,  1,    0,    0,    0,    2258, 2256, 1,    0,    0,    0,
      2259, 2265, 3,    378,  189,  0,    2260, 2265, 5,    28,   0,    0,
      2261, 2265, 5,    24,   0,    0,    2262, 2265, 5,    5,    0,    0,
      2263, 2265, 5,    56,   0,    0,    2264, 2259, 1,    0,    0,    0,
      2264, 2260, 1,    0,    0,    0,    2264, 2261, 1,    0,    0,    0,
      2264, 2262, 1,    0,    0,    0,    2264, 2263, 1,    0,    0,    0,
      2265, 341,  1,    0,    0,    0,    2266, 2268, 5,    120,  0,    0,
      2267, 2266, 1,    0,    0,    0,    2267, 2268, 1,    0,    0,    0,
      2268, 2269, 1,    0,    0,    0,    2269, 2274, 3,    344,  172,  0,
      2270, 2271, 5,    120,  0,    0,    2271, 2273, 3,    344,  172,  0,
      2272, 2270, 1,    0,    0,    0,    2273, 2276, 1,    0,    0,    0,
      2274, 2272, 1,    0,    0,    0,    2274, 2275, 1,    0,    0,    0,
      2275, 343,  1,    0,    0,    0,    2276, 2274, 1,    0,    0,    0,
      2277, 2280, 3,    346,  173,  0,    2278, 2279, 5,    120,  0,    0,
      2279, 2281, 3,    348,  174,  0,    2280, 2278, 1,    0,    0,    0,
      2280, 2281, 1,    0,    0,    0,    2281, 345,  1,    0,    0,    0,
      2282, 2289, 3,    378,  189,  0,    2283, 2289, 5,    28,   0,    0,
      2284, 2289, 5,    24,   0,    0,    2285, 2289, 5,    25,   0,    0,
      2286, 2289, 5,    5,    0,    0,    2287, 2289, 5,    56,   0,    0,
      2288, 2282, 1,    0,    0,    0,    2288, 2283, 1,    0,    0,    0,
      2288, 2284, 1,    0,    0,    0,    2288, 2285, 1,    0,    0,    0,
      2288, 2286, 1,    0,    0,    0,    2288, 2287, 1,    0,    0,    0,
      2289, 347,  1,    0,    0,    0,    2290, 2291, 5,    108,  0,    0,
      2291, 2334, 5,    107,  0,    0,    2292, 2293, 5,    108,  0,    0,
      2293, 2296, 3,    354,  177,  0,    2294, 2295, 5,    117,  0,    0,
      2295, 2297, 3,    356,  178,  0,    2296, 2294, 1,    0,    0,    0,
      2296, 2297, 1,    0,    0,    0,    2297, 2300, 1,    0,    0,    0,
      2298, 2299, 5,    117,  0,    0,    2299, 2301, 3,    358,  179,  0,
      2300, 2298, 1,    0,    0,    0,    2300, 2301, 1,    0,    0,    0,
      2301, 2303, 1,    0,    0,    0,    2302, 2304, 5,    117,  0,    0,
      2303, 2302, 1,    0,    0,    0,    2303, 2304, 1,    0,    0,    0,
      2304, 2305, 1,    0,    0,    0,    2305, 2306, 5,    107,  0,    0,
      2306, 2334, 1,    0,    0,    0,    2307, 2308, 5,    108,  0,    0,
      2308, 2311, 3,    356,  178,  0,    2309, 2310, 5,    117,  0,    0,
      2310, 2312, 3,    358,  179,  0,    2311, 2309, 1,    0,    0,    0,
      2311, 2312, 1,    0,    0,    0,    2312, 2314, 1,    0,    0,    0,
      2313, 2315, 5,    117,  0,    0,    2314, 2313, 1,    0,    0,    0,
      2314, 2315, 1,    0,    0,    0,    2315, 2316, 1,    0,    0,    0,
      2316, 2317, 5,    107,  0,    0,    2317, 2334, 1,    0,    0,    0,
      2318, 2324, 5,    108,  0,    0,    2319, 2320, 3,    350,  175,  0,
      2320, 2321, 5,    117,  0,    0,    2321, 2323, 1,    0,    0,    0,
      2322, 2319, 1,    0,    0,    0,    2323, 2326, 1,    0,    0,    0,
      2324, 2322, 1,    0,    0,    0,    2324, 2325, 1,    0,    0,    0,
      2325, 2327, 1,    0,    0,    0,    2326, 2324, 1,    0,    0,    0,
      2327, 2329, 3,    350,  175,  0,    2328, 2330, 5,    117,  0,    0,
      2329, 2328, 1,    0,    0,    0,    2329, 2330, 1,    0,    0,    0,
      2330, 2331, 1,    0,    0,    0,    2331, 2332, 5,    107,  0,    0,
      2332, 2334, 1,    0,    0,    0,    2333, 2290, 1,    0,    0,    0,
      2333, 2292, 1,    0,    0,    0,    2333, 2307, 1,    0,    0,    0,
      2333, 2318, 1,    0,    0,    0,    2334, 349,  1,    0,    0,    0,
      2335, 2340, 3,    336,  168,  0,    2336, 2340, 3,    286,  143,  0,
      2337, 2340, 3,    352,  176,  0,    2338, 2340, 3,    360,  180,  0,
      2339, 2335, 1,    0,    0,    0,    2339, 2336, 1,    0,    0,    0,
      2339, 2337, 1,    0,    0,    0,    2339, 2338, 1,    0,    0,    0,
      2340, 351,  1,    0,    0,    0,    2341, 2348, 3,    166,  83,   0,
      2342, 2344, 5,    84,   0,    0,    2343, 2342, 1,    0,    0,    0,
      2343, 2344, 1,    0,    0,    0,    2344, 2345, 1,    0,    0,    0,
      2345, 2348, 3,    162,  81,   0,    2346, 2348, 3,    340,  170,  0,
      2347, 2341, 1,    0,    0,    0,    2347, 2343, 1,    0,    0,    0,
      2347, 2346, 1,    0,    0,    0,    2348, 353,  1,    0,    0,    0,
      2349, 2354, 3,    336,  168,  0,    2350, 2351, 5,    117,  0,    0,
      2351, 2353, 3,    336,  168,  0,    2352, 2350, 1,    0,    0,    0,
      2353, 2356, 1,    0,    0,    0,    2354, 2352, 1,    0,    0,    0,
      2354, 2355, 1,    0,    0,    0,    2355, 355,  1,    0,    0,    0,
      2356, 2354, 1,    0,    0,    0,    2357, 2362, 3,    286,  143,  0,
      2358, 2359, 5,    117,  0,    0,    2359, 2361, 3,    286,  143,  0,
      2360, 2358, 1,    0,    0,    0,    2361, 2364, 1,    0,    0,    0,
      2362, 2360, 1,    0,    0,    0,    2362, 2363, 1,    0,    0,    0,
      2363, 357,  1,    0,    0,    0,    2364, 2362, 1,    0,    0,    0,
      2365, 2370, 3,    360,  180,  0,    2366, 2367, 5,    117,  0,    0,
      2367, 2369, 3,    360,  180,  0,    2368, 2366, 1,    0,    0,    0,
      2369, 2372, 1,    0,    0,    0,    2370, 2368, 1,    0,    0,    0,
      2370, 2371, 1,    0,    0,    0,    2371, 359,  1,    0,    0,    0,
      2372, 2370, 1,    0,    0,    0,    2373, 2374, 3,    378,  189,  0,
      2374, 2375, 5,    104,  0,    0,    2375, 2376, 3,    286,  143,  0,
      2376, 361,  1,    0,    0,    0,    2377, 2380, 3,    364,  182,  0,
      2378, 2379, 5,    120,  0,    0,    2379, 2381, 3,    344,  172,  0,
      2380, 2378, 1,    0,    0,    0,    2381, 2382, 1,    0,    0,    0,
      2382, 2380, 1,    0,    0,    0,    2382, 2383, 1,    0,    0,    0,
      2383, 363,  1,    0,    0,    0,    2384, 2385, 5,    108,  0,    0,
      2385, 2388, 3,    286,  143,  0,    2386, 2387, 5,    1,    0,    0,
      2387, 2389, 3,    368,  184,  0,    2388, 2386, 1,    0,    0,    0,
      2388, 2389, 1,    0,    0,    0,    2389, 2390, 1,    0,    0,    0,
      2390, 2391, 5,    107,  0,    0,    2391, 365,  1,    0,    0,    0,
      2392, 2395, 3,    364,  182,  0,    2393, 2394, 5,    120,  0,    0,
      2394, 2396, 3,    370,  185,  0,    2395, 2393, 1,    0,    0,    0,
      2396, 2397, 1,    0,    0,    0,    2397, 2395, 1,    0,    0,    0,
      2397, 2398, 1,    0,    0,    0,    2398, 367,  1,    0,    0,    0,
      2399, 2401, 5,    120,  0,    0,    2400, 2399, 1,    0,    0,    0,
      2400, 2401, 1,    0,    0,    0,    2401, 2402, 1,    0,    0,    0,
      2402, 2407, 3,    370,  185,  0,    2403, 2404, 5,    120,  0,    0,
      2404, 2406, 3,    370,  185,  0,    2405, 2403, 1,    0,    0,    0,
      2406, 2409, 1,    0,    0,    0,    2407, 2405, 1,    0,    0,    0,
      2407, 2408, 1,    0,    0,    0,    2408, 369,  1,    0,    0,    0,
      2409, 2407, 1,    0,    0,    0,    2410, 2412, 3,    346,  173,  0,
      2411, 2413, 5,    120,  0,    0,    2412, 2411, 1,    0,    0,    0,
      2412, 2413, 1,    0,    0,    0,    2413, 2416, 1,    0,    0,    0,
      2414, 2417, 3,    348,  174,  0,    2415, 2417, 3,    372,  186,  0,
      2416, 2414, 1,    0,    0,    0,    2416, 2415, 1,    0,    0,    0,
      2416, 2417, 1,    0,    0,    0,    2417, 371,  1,    0,    0,    0,
      2418, 2420, 5,    130,  0,    0,    2419, 2421, 3,    374,  187,  0,
      2420, 2419, 1,    0,    0,    0,    2420, 2421, 1,    0,    0,    0,
      2421, 2422, 1,    0,    0,    0,    2422, 2425, 5,    131,  0,    0,
      2423, 2424, 5,    121,  0,    0,    2424, 2426, 3,    286,  143,  0,
      2425, 2423, 1,    0,    0,    0,    2425, 2426, 1,    0,    0,    0,
      2426, 373,  1,    0,    0,    0,    2427, 2432, 3,    286,  143,  0,
      2428, 2429, 5,    117,  0,    0,    2429, 2431, 3,    286,  143,  0,
      2430, 2428, 1,    0,    0,    0,    2431, 2434, 1,    0,    0,    0,
      2432, 2430, 1,    0,    0,    0,    2432, 2433, 1,    0,    0,    0,
      2433, 2436, 1,    0,    0,    0,    2434, 2432, 1,    0,    0,    0,
      2435, 2437, 5,    117,  0,    0,    2436, 2435, 1,    0,    0,    0,
      2436, 2437, 1,    0,    0,    0,    2437, 375,  1,    0,    0,    0,
      2438, 2448, 5,    21,   0,    0,    2439, 2445, 5,    130,  0,    0,
      2440, 2446, 5,    5,    0,    0,    2441, 2446, 5,    24,   0,    0,
      2442, 2446, 5,    28,   0,    0,    2443, 2444, 5,    14,   0,    0,
      2444, 2446, 3,    338,  169,  0,    2445, 2440, 1,    0,    0,    0,
      2445, 2441, 1,    0,    0,    0,    2445, 2442, 1,    0,    0,    0,
      2445, 2443, 1,    0,    0,    0,    2446, 2447, 1,    0,    0,    0,
      2447, 2449, 5,    131,  0,    0,    2448, 2439, 1,    0,    0,    0,
      2448, 2449, 1,    0,    0,    0,    2449, 377,  1,    0,    0,    0,
      2450, 2451, 7,    11,   0,    0,    2451, 379,  1,    0,    0,    0,
      2452, 2453, 7,    12,   0,    0,    2453, 381,  1,    0,    0,    0,
      2454, 2461, 3,    380,  190,  0,    2455, 2461, 3,    378,  189,  0,
      2456, 2461, 5,    54,   0,    0,    2457, 2461, 5,    55,   0,    0,
      2458, 2461, 5,    56,   0,    0,    2459, 2461, 5,    82,   0,    0,
      2460, 2454, 1,    0,    0,    0,    2460, 2455, 1,    0,    0,    0,
      2460, 2456, 1,    0,    0,    0,    2460, 2457, 1,    0,    0,    0,
      2460, 2458, 1,    0,    0,    0,    2460, 2459, 1,    0,    0,    0,
      2461, 383,  1,    0,    0,    0,    2462, 2463, 3,    162,  81,   0,
      2463, 385,  1,    0,    0,    0,    2464, 2465, 7,    13,   0,    0,
      2465, 387,  1,    0,    0,    0,    2466, 2467, 5,    108,  0,    0,
      2467, 2468, 4,    194,  21,   0,    2468, 2469, 5,    108,  0,    0,
      2469, 389,  1,    0,    0,    0,    2470, 2471, 5,    107,  0,    0,
      2471, 2472, 4,    195,  22,   0,    2472, 2473, 5,    107,  0,    0,
      2473, 391,  1,    0,    0,    0,    345,  395,  401,  414,  422,  430,
      434,  439,  442,  449,  457,  469,  481,  486,  507,  514,  518,  528,
      536,  544,  548,  553,  559,  568,  572,  576,  582,  590,  599,  604,
      607,  622,  626,  629,  638,  644,  648,  654,  660,  665,  672,  675,
      684,  688,  690,  693,  699,  701,  703,  709,  713,  717,  720,  724,
      727,  730,  733,  737,  739,  745,  750,  757,  761,  763,  768,  773,
      777,  779,  782,  787,  796,  802,  808,  816,  819,  823,  829,  834,
      837,  841,  845,  850,  854,  858,  867,  871,  876,  880,  891,  895,
      900,  904,  911,  914,  918,  927,  931,  936,  940,  946,  950,  956,
      966,  969,  978,  984,  990,  997,  1002, 1007, 1011, 1013, 1016, 1022,
      1028, 1035, 1039, 1043, 1049, 1055, 1061, 1065, 1068, 1074, 1080, 1086,
      1092, 1096, 1102, 1108, 1116, 1121, 1125, 1127, 1135, 1140, 1142, 1149,
      1155, 1158, 1163, 1166, 1171, 1173, 1177, 1190, 1194, 1198, 1205, 1210,
      1218, 1223, 1228, 1230, 1245, 1250, 1257, 1262, 1269, 1273, 1282, 1284,
      1290, 1298, 1307, 1313, 1316, 1320, 1323, 1327, 1333, 1343, 1347, 1354,
      1358, 1366, 1377, 1415, 1431, 1447, 1449, 1451, 1461, 1472, 1478, 1484,
      1488, 1495, 1498, 1501, 1505, 1517, 1521, 1527, 1534, 1537, 1544, 1551,
      1556, 1565, 1571, 1573, 1578, 1584, 1589, 1599, 1607, 1611, 1613, 1622,
      1627, 1636, 1640, 1645, 1650, 1659, 1663, 1665, 1676, 1680, 1683, 1688,
      1691, 1698, 1705, 1709, 1714, 1720, 1723, 1729, 1761, 1763, 1775, 1777,
      1785, 1789, 1799, 1806, 1813, 1815, 1820, 1825, 1831, 1838, 1843, 1857,
      1868, 1872, 1875, 1878, 1881, 1886, 1903, 1908, 1912, 1916, 1920, 1927,
      1934, 1936, 1939, 1946, 1952, 1964, 1967, 1970, 1975, 1983, 1992, 1996,
      2000, 2013, 2016, 2018, 2026, 2035, 2039, 2043, 2048, 2064, 2078, 2081,
      2083, 2099, 2102, 2111, 2117, 2121, 2124, 2128, 2130, 2137, 2144, 2148,
      2153, 2158, 2161, 2170, 2178, 2184, 2189, 2206, 2210, 2214, 2217, 2220,
      2225, 2228, 2233, 2240, 2244, 2249, 2256, 2264, 2267, 2274, 2280, 2288,
      2296, 2300, 2303, 2311, 2314, 2324, 2329, 2333, 2339, 2343, 2347, 2354,
      2362, 2370, 2382, 2388, 2397, 2400, 2407, 2412, 2416, 2420, 2425, 2432,
      2436, 2445, 2448, 2460};
  staticData->serializedATN = antlr4::atn::SerializedATNView(
      serializedATNSegment,
      sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) {
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i),
                                           i);
  }
  rustparserParserStaticData = staticData.release();
}

} // namespace

RustParser::RustParser(TokenStream *input)
    : RustParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

RustParser::RustParser(TokenStream *input,
                       const antlr4::atn::ParserATNSimulatorOptions &options)
    : Parser(input) {
  RustParser::initialize();
  _interpreter = new atn::ParserATNSimulator(
      this, *rustparserParserStaticData->atn,
      rustparserParserStaticData->decisionToDFA,
      rustparserParserStaticData->sharedContextCache, options);
}

RustParser::~RustParser() { delete _interpreter; }

const atn::ATN &RustParser::getATN() const {
  return *rustparserParserStaticData->atn;
}

std::string RustParser::getGrammarFileName() const { return "RustParser.g4"; }

const std::vector<std::string> &RustParser::getRuleNames() const {
  return rustparserParserStaticData->ruleNames;
}

const dfa::Vocabulary &RustParser::getVocabulary() const {
  return rustparserParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView RustParser::getSerializedATN() const {
  return rustparserParserStaticData->serializedATN;
}

//----------------- CrateContext
//------------------------------------------------------------------

RustParser::CrateContext::CrateContext(ParserRuleContext *parent,
                                       size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::CrateContext::EOF() {
  return getToken(RustParser::EOF, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::CrateContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::CrateContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

std::vector<RustParser::ItemContext *> RustParser::CrateContext::item() {
  return getRuleContexts<RustParser::ItemContext>();
}

RustParser::ItemContext *RustParser::CrateContext::item(size_t i) {
  return getRuleContext<RustParser::ItemContext>(i);
}

size_t RustParser::CrateContext::getRuleIndex() const {
  return RustParser::RuleCrate;
}

void RustParser::CrateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCrate(this);
}

void RustParser::CrateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCrate(this);
}

std::any RustParser::CrateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitCrate(this);
  else
    return visitor->visitChildren(this);
}

RustParser::CrateContext *RustParser::crate() {
  CrateContext *_localctx =
      _tracker.createInstance<CrateContext>(_ctx, getState());
  enterRule(_localctx, 0, RustParser::RuleCrate);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(395);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0,
                                                                     _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(392);
        innerAttribute();
      }
      setState(397);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                       0, _ctx);
    }
    setState(401);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~0x3fULL) == 0) &&
            ((1ULL << _la) & 526921241179989416) != 0) ||
           _la == RustParser::PATHSEP

           || _la == RustParser::POUND) {
      setState(398);
      item();
      setState(403);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(404);
    match(RustParser::EOF);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroInvocationContext
//------------------------------------------------------------------

RustParser::MacroInvocationContext::MacroInvocationContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::SimplePathContext *
RustParser::MacroInvocationContext::simplePath() {
  return getRuleContext<RustParser::SimplePathContext>(0);
}

tree::TerminalNode *RustParser::MacroInvocationContext::NOT() {
  return getToken(RustParser::NOT, 0);
}

RustParser::DelimTokenTreeContext *
RustParser::MacroInvocationContext::delimTokenTree() {
  return getRuleContext<RustParser::DelimTokenTreeContext>(0);
}

size_t RustParser::MacroInvocationContext::getRuleIndex() const {
  return RustParser::RuleMacroInvocation;
}

void RustParser::MacroInvocationContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroInvocation(this);
}

void RustParser::MacroInvocationContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroInvocation(this);
}

std::any
RustParser::MacroInvocationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroInvocation(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroInvocationContext *RustParser::macroInvocation() {
  MacroInvocationContext *_localctx =
      _tracker.createInstance<MacroInvocationContext>(_ctx, getState());
  enterRule(_localctx, 2, RustParser::RuleMacroInvocation);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(406);
    simplePath();
    setState(407);
    match(RustParser::NOT);
    setState(408);
    delimTokenTree();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DelimTokenTreeContext
//------------------------------------------------------------------

RustParser::DelimTokenTreeContext::DelimTokenTreeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::DelimTokenTreeContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::DelimTokenTreeContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

std::vector<RustParser::TokenTreeContext *>
RustParser::DelimTokenTreeContext::tokenTree() {
  return getRuleContexts<RustParser::TokenTreeContext>();
}

RustParser::TokenTreeContext *
RustParser::DelimTokenTreeContext::tokenTree(size_t i) {
  return getRuleContext<RustParser::TokenTreeContext>(i);
}

tree::TerminalNode *RustParser::DelimTokenTreeContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::DelimTokenTreeContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::DelimTokenTreeContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::DelimTokenTreeContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

size_t RustParser::DelimTokenTreeContext::getRuleIndex() const {
  return RustParser::RuleDelimTokenTree;
}

void RustParser::DelimTokenTreeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDelimTokenTree(this);
}

void RustParser::DelimTokenTreeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDelimTokenTree(this);
}

std::any
RustParser::DelimTokenTreeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitDelimTokenTree(this);
  else
    return visitor->visitChildren(this);
}

RustParser::DelimTokenTreeContext *RustParser::delimTokenTree() {
  DelimTokenTreeContext *_localctx =
      _tracker.createInstance<DelimTokenTreeContext>(_ctx, getState());
  enterRule(_localctx, 4, RustParser::RuleDelimTokenTree);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(434);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::LPAREN: {
      enterOuterAlt(_localctx, 1);
      setState(410);
      match(RustParser::LPAREN);
      setState(414);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(411);
        tokenTree();
        setState(416);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(417);
      match(RustParser::RPAREN);
      break;
    }

    case RustParser::LSQUAREBRACKET: {
      enterOuterAlt(_localctx, 2);
      setState(418);
      match(RustParser::LSQUAREBRACKET);
      setState(422);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(419);
        tokenTree();
        setState(424);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(425);
      match(RustParser::RSQUAREBRACKET);
      break;
    }

    case RustParser::LCURLYBRACE: {
      enterOuterAlt(_localctx, 3);
      setState(426);
      match(RustParser::LCURLYBRACE);
      setState(430);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(427);
        tokenTree();
        setState(432);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(433);
      match(RustParser::RCURLYBRACE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TokenTreeContext
//------------------------------------------------------------------

RustParser::TokenTreeContext::TokenTreeContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::TokenTreeTokenContext *>
RustParser::TokenTreeContext::tokenTreeToken() {
  return getRuleContexts<RustParser::TokenTreeTokenContext>();
}

RustParser::TokenTreeTokenContext *
RustParser::TokenTreeContext::tokenTreeToken(size_t i) {
  return getRuleContext<RustParser::TokenTreeTokenContext>(i);
}

RustParser::DelimTokenTreeContext *
RustParser::TokenTreeContext::delimTokenTree() {
  return getRuleContext<RustParser::DelimTokenTreeContext>(0);
}

size_t RustParser::TokenTreeContext::getRuleIndex() const {
  return RustParser::RuleTokenTree;
}

void RustParser::TokenTreeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTokenTree(this);
}

void RustParser::TokenTreeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTokenTree(this);
}

std::any RustParser::TokenTreeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTokenTree(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TokenTreeContext *RustParser::tokenTree() {
  TokenTreeContext *_localctx =
      _tracker.createInstance<TokenTreeContext>(_ctx, getState());
  enterRule(_localctx, 6, RustParser::RuleTokenTree);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(442);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_AS:
    case RustParser::KW_BREAK:
    case RustParser::KW_CONST:
    case RustParser::KW_CONTINUE:
    case RustParser::KW_CRATE:
    case RustParser::KW_ELSE:
    case RustParser::KW_ENUM:
    case RustParser::KW_EXTERN:
    case RustParser::KW_FALSE:
    case RustParser::KW_FN:
    case RustParser::KW_FOR:
    case RustParser::KW_IF:
    case RustParser::KW_IMPL:
    case RustParser::KW_IN:
    case RustParser::KW_LET:
    case RustParser::KW_LOOP:
    case RustParser::KW_MATCH:
    case RustParser::KW_MOD:
    case RustParser::KW_MOVE:
    case RustParser::KW_MUT:
    case RustParser::KW_PUB:
    case RustParser::KW_REF:
    case RustParser::KW_RETURN:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SELFTYPE:
    case RustParser::KW_STATIC:
    case RustParser::KW_STRUCT:
    case RustParser::KW_SUPER:
    case RustParser::KW_TRAIT:
    case RustParser::KW_TRUE:
    case RustParser::KW_TYPE:
    case RustParser::KW_UNSAFE:
    case RustParser::KW_USE:
    case RustParser::KW_WHERE:
    case RustParser::KW_WHILE:
    case RustParser::KW_ASYNC:
    case RustParser::KW_AWAIT:
    case RustParser::KW_DYN:
    case RustParser::KW_ABSTRACT:
    case RustParser::KW_BECOME:
    case RustParser::KW_BOX:
    case RustParser::KW_DO:
    case RustParser::KW_FINAL:
    case RustParser::KW_MACRO:
    case RustParser::KW_OVERRIDE:
    case RustParser::KW_PRIV:
    case RustParser::KW_TYPEOF:
    case RustParser::KW_UNSIZED:
    case RustParser::KW_VIRTUAL:
    case RustParser::KW_YIELD:
    case RustParser::KW_TRY:
    case RustParser::KW_UNION:
    case RustParser::KW_STATICLIFETIME:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_UNDERLINELIFETIME:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::CHAR_LITERAL:
    case RustParser::STRING_LITERAL:
    case RustParser::RAW_STRING_LITERAL:
    case RustParser::BYTE_LITERAL:
    case RustParser::BYTE_STRING_LITERAL:
    case RustParser::RAW_BYTE_STRING_LITERAL:
    case RustParser::INTEGER_LITERAL:
    case RustParser::FLOAT_LITERAL:
    case RustParser::LIFETIME_OR_LABEL:
    case RustParser::PLUS:
    case RustParser::MINUS:
    case RustParser::STAR:
    case RustParser::SLASH:
    case RustParser::PERCENT:
    case RustParser::CARET:
    case RustParser::NOT:
    case RustParser::AND:
    case RustParser::OR:
    case RustParser::ANDAND:
    case RustParser::OROR:
    case RustParser::PLUSEQ:
    case RustParser::MINUSEQ:
    case RustParser::STAREQ:
    case RustParser::SLASHEQ:
    case RustParser::PERCENTEQ:
    case RustParser::CARETEQ:
    case RustParser::ANDEQ:
    case RustParser::OREQ:
    case RustParser::SHLEQ:
    case RustParser::SHREQ:
    case RustParser::EQ:
    case RustParser::EQEQ:
    case RustParser::NE:
    case RustParser::GT:
    case RustParser::LT:
    case RustParser::GE:
    case RustParser::LE:
    case RustParser::AT:
    case RustParser::UNDERSCORE:
    case RustParser::DOT:
    case RustParser::DOTDOT:
    case RustParser::DOTDOTDOT:
    case RustParser::DOTDOTEQ:
    case RustParser::COMMA:
    case RustParser::SEMI:
    case RustParser::COLON:
    case RustParser::PATHSEP:
    case RustParser::RARROW:
    case RustParser::FATARROW:
    case RustParser::POUND:
    case RustParser::DOLLAR:
    case RustParser::QUESTION: {
      enterOuterAlt(_localctx, 1);
      setState(437);
      _errHandler->sync(this);
      alt = 1;
      do {
        switch (alt) {
        case 1: {
          setState(436);
          tokenTreeToken();
          break;
        }

        default:
          throw NoViableAltException(this);
        }
        setState(439);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 6, _ctx);
      } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
      break;
    }

    case RustParser::LCURLYBRACE:
    case RustParser::LSQUAREBRACKET:
    case RustParser::LPAREN: {
      enterOuterAlt(_localctx, 2);
      setState(441);
      delimTokenTree();
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TokenTreeTokenContext
//------------------------------------------------------------------

RustParser::TokenTreeTokenContext::TokenTreeTokenContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::MacroIdentifierLikeTokenContext *
RustParser::TokenTreeTokenContext::macroIdentifierLikeToken() {
  return getRuleContext<RustParser::MacroIdentifierLikeTokenContext>(0);
}

RustParser::MacroLiteralTokenContext *
RustParser::TokenTreeTokenContext::macroLiteralToken() {
  return getRuleContext<RustParser::MacroLiteralTokenContext>(0);
}

RustParser::MacroPunctuationTokenContext *
RustParser::TokenTreeTokenContext::macroPunctuationToken() {
  return getRuleContext<RustParser::MacroPunctuationTokenContext>(0);
}

RustParser::MacroRepOpContext *RustParser::TokenTreeTokenContext::macroRepOp() {
  return getRuleContext<RustParser::MacroRepOpContext>(0);
}

tree::TerminalNode *RustParser::TokenTreeTokenContext::DOLLAR() {
  return getToken(RustParser::DOLLAR, 0);
}

size_t RustParser::TokenTreeTokenContext::getRuleIndex() const {
  return RustParser::RuleTokenTreeToken;
}

void RustParser::TokenTreeTokenContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTokenTreeToken(this);
}

void RustParser::TokenTreeTokenContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTokenTreeToken(this);
}

std::any
RustParser::TokenTreeTokenContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTokenTreeToken(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TokenTreeTokenContext *RustParser::tokenTreeToken() {
  TokenTreeTokenContext *_localctx =
      _tracker.createInstance<TokenTreeTokenContext>(_ctx, getState());
  enterRule(_localctx, 8, RustParser::RuleTokenTreeToken);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(449);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 8, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(444);
      macroIdentifierLikeToken();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(445);
      macroLiteralToken();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(446);
      macroPunctuationToken();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(447);
      macroRepOp();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(448);
      match(RustParser::DOLLAR);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroInvocationSemiContext
//------------------------------------------------------------------

RustParser::MacroInvocationSemiContext::MacroInvocationSemiContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::SimplePathContext *
RustParser::MacroInvocationSemiContext::simplePath() {
  return getRuleContext<RustParser::SimplePathContext>(0);
}

tree::TerminalNode *RustParser::MacroInvocationSemiContext::NOT() {
  return getToken(RustParser::NOT, 0);
}

tree::TerminalNode *RustParser::MacroInvocationSemiContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::MacroInvocationSemiContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

tree::TerminalNode *RustParser::MacroInvocationSemiContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

std::vector<RustParser::TokenTreeContext *>
RustParser::MacroInvocationSemiContext::tokenTree() {
  return getRuleContexts<RustParser::TokenTreeContext>();
}

RustParser::TokenTreeContext *
RustParser::MacroInvocationSemiContext::tokenTree(size_t i) {
  return getRuleContext<RustParser::TokenTreeContext>(i);
}

tree::TerminalNode *RustParser::MacroInvocationSemiContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::MacroInvocationSemiContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::MacroInvocationSemiContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::MacroInvocationSemiContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

size_t RustParser::MacroInvocationSemiContext::getRuleIndex() const {
  return RustParser::RuleMacroInvocationSemi;
}

void RustParser::MacroInvocationSemiContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroInvocationSemi(this);
}

void RustParser::MacroInvocationSemiContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroInvocationSemi(this);
}

std::any RustParser::MacroInvocationSemiContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroInvocationSemi(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroInvocationSemiContext *RustParser::macroInvocationSemi() {
  MacroInvocationSemiContext *_localctx =
      _tracker.createInstance<MacroInvocationSemiContext>(_ctx, getState());
  enterRule(_localctx, 10, RustParser::RuleMacroInvocationSemi);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(486);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 12, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(451);
      simplePath();
      setState(452);
      match(RustParser::NOT);
      setState(453);
      match(RustParser::LPAREN);
      setState(457);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(454);
        tokenTree();
        setState(459);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(460);
      match(RustParser::RPAREN);
      setState(461);
      match(RustParser::SEMI);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(463);
      simplePath();
      setState(464);
      match(RustParser::NOT);
      setState(465);
      match(RustParser::LSQUAREBRACKET);
      setState(469);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(466);
        tokenTree();
        setState(471);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(472);
      match(RustParser::RSQUAREBRACKET);
      setState(473);
      match(RustParser::SEMI);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(475);
      simplePath();
      setState(476);
      match(RustParser::NOT);
      setState(477);
      match(RustParser::LCURLYBRACE);
      setState(481);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(478);
        tokenTree();
        setState(483);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(484);
      match(RustParser::RCURLYBRACE);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroRulesDefinitionContext
//------------------------------------------------------------------

RustParser::MacroRulesDefinitionContext::MacroRulesDefinitionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::MacroRulesDefinitionContext::KW_MACRORULES() {
  return getToken(RustParser::KW_MACRORULES, 0);
}

tree::TerminalNode *RustParser::MacroRulesDefinitionContext::NOT() {
  return getToken(RustParser::NOT, 0);
}

RustParser::IdentifierContext *
RustParser::MacroRulesDefinitionContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

RustParser::MacroRulesDefContext *
RustParser::MacroRulesDefinitionContext::macroRulesDef() {
  return getRuleContext<RustParser::MacroRulesDefContext>(0);
}

size_t RustParser::MacroRulesDefinitionContext::getRuleIndex() const {
  return RustParser::RuleMacroRulesDefinition;
}

void RustParser::MacroRulesDefinitionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroRulesDefinition(this);
}

void RustParser::MacroRulesDefinitionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroRulesDefinition(this);
}

std::any RustParser::MacroRulesDefinitionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroRulesDefinition(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroRulesDefinitionContext *RustParser::macroRulesDefinition() {
  MacroRulesDefinitionContext *_localctx =
      _tracker.createInstance<MacroRulesDefinitionContext>(_ctx, getState());
  enterRule(_localctx, 12, RustParser::RuleMacroRulesDefinition);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(488);
    match(RustParser::KW_MACRORULES);
    setState(489);
    match(RustParser::NOT);
    setState(490);
    identifier();
    setState(491);
    macroRulesDef();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroRulesDefContext
//------------------------------------------------------------------

RustParser::MacroRulesDefContext::MacroRulesDefContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::MacroRulesDefContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

RustParser::MacroRulesContext *RustParser::MacroRulesDefContext::macroRules() {
  return getRuleContext<RustParser::MacroRulesContext>(0);
}

tree::TerminalNode *RustParser::MacroRulesDefContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

tree::TerminalNode *RustParser::MacroRulesDefContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

tree::TerminalNode *RustParser::MacroRulesDefContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::MacroRulesDefContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::MacroRulesDefContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::MacroRulesDefContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

size_t RustParser::MacroRulesDefContext::getRuleIndex() const {
  return RustParser::RuleMacroRulesDef;
}

void RustParser::MacroRulesDefContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroRulesDef(this);
}

void RustParser::MacroRulesDefContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroRulesDef(this);
}

std::any
RustParser::MacroRulesDefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroRulesDef(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroRulesDefContext *RustParser::macroRulesDef() {
  MacroRulesDefContext *_localctx =
      _tracker.createInstance<MacroRulesDefContext>(_ctx, getState());
  enterRule(_localctx, 14, RustParser::RuleMacroRulesDef);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(507);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::LPAREN: {
      enterOuterAlt(_localctx, 1);
      setState(493);
      match(RustParser::LPAREN);
      setState(494);
      macroRules();
      setState(495);
      match(RustParser::RPAREN);
      setState(496);
      match(RustParser::SEMI);
      break;
    }

    case RustParser::LSQUAREBRACKET: {
      enterOuterAlt(_localctx, 2);
      setState(498);
      match(RustParser::LSQUAREBRACKET);
      setState(499);
      macroRules();
      setState(500);
      match(RustParser::RSQUAREBRACKET);
      setState(501);
      match(RustParser::SEMI);
      break;
    }

    case RustParser::LCURLYBRACE: {
      enterOuterAlt(_localctx, 3);
      setState(503);
      match(RustParser::LCURLYBRACE);
      setState(504);
      macroRules();
      setState(505);
      match(RustParser::RCURLYBRACE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroRulesContext
//------------------------------------------------------------------

RustParser::MacroRulesContext::MacroRulesContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::MacroRuleContext *>
RustParser::MacroRulesContext::macroRule() {
  return getRuleContexts<RustParser::MacroRuleContext>();
}

RustParser::MacroRuleContext *
RustParser::MacroRulesContext::macroRule(size_t i) {
  return getRuleContext<RustParser::MacroRuleContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::MacroRulesContext::SEMI() {
  return getTokens(RustParser::SEMI);
}

tree::TerminalNode *RustParser::MacroRulesContext::SEMI(size_t i) {
  return getToken(RustParser::SEMI, i);
}

size_t RustParser::MacroRulesContext::getRuleIndex() const {
  return RustParser::RuleMacroRules;
}

void RustParser::MacroRulesContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroRules(this);
}

void RustParser::MacroRulesContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroRules(this);
}

std::any
RustParser::MacroRulesContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroRules(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroRulesContext *RustParser::macroRules() {
  MacroRulesContext *_localctx =
      _tracker.createInstance<MacroRulesContext>(_ctx, getState());
  enterRule(_localctx, 16, RustParser::RuleMacroRules);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(509);
    macroRule();
    setState(514);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14,
                                                                     _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(510);
        match(RustParser::SEMI);
        setState(511);
        macroRule();
      }
      setState(516);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 14, _ctx);
    }
    setState(518);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::SEMI) {
      setState(517);
      match(RustParser::SEMI);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroRuleContext
//------------------------------------------------------------------

RustParser::MacroRuleContext::MacroRuleContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::MacroMatcherContext *RustParser::MacroRuleContext::macroMatcher() {
  return getRuleContext<RustParser::MacroMatcherContext>(0);
}

tree::TerminalNode *RustParser::MacroRuleContext::FATARROW() {
  return getToken(RustParser::FATARROW, 0);
}

RustParser::MacroTranscriberContext *
RustParser::MacroRuleContext::macroTranscriber() {
  return getRuleContext<RustParser::MacroTranscriberContext>(0);
}

size_t RustParser::MacroRuleContext::getRuleIndex() const {
  return RustParser::RuleMacroRule;
}

void RustParser::MacroRuleContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroRule(this);
}

void RustParser::MacroRuleContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroRule(this);
}

std::any RustParser::MacroRuleContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroRule(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroRuleContext *RustParser::macroRule() {
  MacroRuleContext *_localctx =
      _tracker.createInstance<MacroRuleContext>(_ctx, getState());
  enterRule(_localctx, 18, RustParser::RuleMacroRule);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(520);
    macroMatcher();
    setState(521);
    match(RustParser::FATARROW);
    setState(522);
    macroTranscriber();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroMatcherContext
//------------------------------------------------------------------

RustParser::MacroMatcherContext::MacroMatcherContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::MacroMatcherContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::MacroMatcherContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

std::vector<RustParser::MacroMatchContext *>
RustParser::MacroMatcherContext::macroMatch() {
  return getRuleContexts<RustParser::MacroMatchContext>();
}

RustParser::MacroMatchContext *
RustParser::MacroMatcherContext::macroMatch(size_t i) {
  return getRuleContext<RustParser::MacroMatchContext>(i);
}

tree::TerminalNode *RustParser::MacroMatcherContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::MacroMatcherContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::MacroMatcherContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::MacroMatcherContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

size_t RustParser::MacroMatcherContext::getRuleIndex() const {
  return RustParser::RuleMacroMatcher;
}

void RustParser::MacroMatcherContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroMatcher(this);
}

void RustParser::MacroMatcherContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroMatcher(this);
}

std::any
RustParser::MacroMatcherContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroMatcher(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroMatcherContext *RustParser::macroMatcher() {
  MacroMatcherContext *_localctx =
      _tracker.createInstance<MacroMatcherContext>(_ctx, getState());
  enterRule(_localctx, 20, RustParser::RuleMacroMatcher);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(548);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::LPAREN: {
      enterOuterAlt(_localctx, 1);
      setState(524);
      match(RustParser::LPAREN);
      setState(528);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(525);
        macroMatch();
        setState(530);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(531);
      match(RustParser::RPAREN);
      break;
    }

    case RustParser::LSQUAREBRACKET: {
      enterOuterAlt(_localctx, 2);
      setState(532);
      match(RustParser::LSQUAREBRACKET);
      setState(536);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(533);
        macroMatch();
        setState(538);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(539);
      match(RustParser::RSQUAREBRACKET);
      break;
    }

    case RustParser::LCURLYBRACE: {
      enterOuterAlt(_localctx, 3);
      setState(540);
      match(RustParser::LCURLYBRACE);
      setState(544);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 576460752303423486) != 0) ||
             ((((_la - 70) & ~0x3fULL) == 0) &&
              ((1ULL << (_la - 70)) & 1585267068834412671) != 0)) {
        setState(541);
        macroMatch();
        setState(546);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(547);
      match(RustParser::RCURLYBRACE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroMatchContext
//------------------------------------------------------------------

RustParser::MacroMatchContext::MacroMatchContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::MacroMatchTokenContext *>
RustParser::MacroMatchContext::macroMatchToken() {
  return getRuleContexts<RustParser::MacroMatchTokenContext>();
}

RustParser::MacroMatchTokenContext *
RustParser::MacroMatchContext::macroMatchToken(size_t i) {
  return getRuleContext<RustParser::MacroMatchTokenContext>(i);
}

RustParser::MacroMatcherContext *RustParser::MacroMatchContext::macroMatcher() {
  return getRuleContext<RustParser::MacroMatcherContext>(0);
}

tree::TerminalNode *RustParser::MacroMatchContext::DOLLAR() {
  return getToken(RustParser::DOLLAR, 0);
}

tree::TerminalNode *RustParser::MacroMatchContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::MacroFragSpecContext *
RustParser::MacroMatchContext::macroFragSpec() {
  return getRuleContext<RustParser::MacroFragSpecContext>(0);
}

RustParser::IdentifierContext *RustParser::MacroMatchContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::MacroMatchContext::KW_SELFVALUE() {
  return getToken(RustParser::KW_SELFVALUE, 0);
}

tree::TerminalNode *RustParser::MacroMatchContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::MacroMatchContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::MacroRepOpContext *RustParser::MacroMatchContext::macroRepOp() {
  return getRuleContext<RustParser::MacroRepOpContext>(0);
}

std::vector<RustParser::MacroMatchContext *>
RustParser::MacroMatchContext::macroMatch() {
  return getRuleContexts<RustParser::MacroMatchContext>();
}

RustParser::MacroMatchContext *
RustParser::MacroMatchContext::macroMatch(size_t i) {
  return getRuleContext<RustParser::MacroMatchContext>(i);
}

RustParser::MacroRepSepContext *RustParser::MacroMatchContext::macroRepSep() {
  return getRuleContext<RustParser::MacroRepSepContext>(0);
}

size_t RustParser::MacroMatchContext::getRuleIndex() const {
  return RustParser::RuleMacroMatch;
}

void RustParser::MacroMatchContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroMatch(this);
}

void RustParser::MacroMatchContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroMatch(this);
}

std::any
RustParser::MacroMatchContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroMatch(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroMatchContext *RustParser::macroMatch() {
  MacroMatchContext *_localctx =
      _tracker.createInstance<MacroMatchContext>(_ctx, getState());
  enterRule(_localctx, 22, RustParser::RuleMacroMatch);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(576);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 24, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(551);
      _errHandler->sync(this);
      alt = 1;
      do {
        switch (alt) {
        case 1: {
          setState(550);
          macroMatchToken();
          break;
        }

        default:
          throw NoViableAltException(this);
        }
        setState(553);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 20, _ctx);
      } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(555);
      macroMatcher();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(556);
      match(RustParser::DOLLAR);
      setState(559);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
      case RustParser::KW_MACRORULES:
      case RustParser::NON_KEYWORD_IDENTIFIER:
      case RustParser::RAW_IDENTIFIER: {
        setState(557);
        identifier();
        break;
      }

      case RustParser::KW_SELFVALUE: {
        setState(558);
        match(RustParser::KW_SELFVALUE);
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      setState(561);
      match(RustParser::COLON);
      setState(562);
      macroFragSpec();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(563);
      match(RustParser::DOLLAR);
      setState(564);
      match(RustParser::LPAREN);
      setState(566);
      _errHandler->sync(this);
      _la = _input->LA(1);
      do {
        setState(565);
        macroMatch();
        setState(568);
        _errHandler->sync(this);
        _la = _input->LA(1);
      } while ((((_la & ~0x3fULL) == 0) &&
                ((1ULL << _la) & 576460752303423486) != 0) ||
               ((((_la - 70) & ~0x3fULL) == 0) &&
                ((1ULL << (_la - 70)) & 1585267068834412671) != 0));
      setState(570);
      match(RustParser::RPAREN);
      setState(572);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 576460752303423486) != 0) ||
          ((((_la - 70) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 70)) & 36028797018921087) != 0)) {
        setState(571);
        macroRepSep();
      }
      setState(574);
      macroRepOp();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroMatchTokenContext
//------------------------------------------------------------------

RustParser::MacroMatchTokenContext::MacroMatchTokenContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::MacroIdentifierLikeTokenContext *
RustParser::MacroMatchTokenContext::macroIdentifierLikeToken() {
  return getRuleContext<RustParser::MacroIdentifierLikeTokenContext>(0);
}

RustParser::MacroLiteralTokenContext *
RustParser::MacroMatchTokenContext::macroLiteralToken() {
  return getRuleContext<RustParser::MacroLiteralTokenContext>(0);
}

RustParser::MacroPunctuationTokenContext *
RustParser::MacroMatchTokenContext::macroPunctuationToken() {
  return getRuleContext<RustParser::MacroPunctuationTokenContext>(0);
}

RustParser::MacroRepOpContext *
RustParser::MacroMatchTokenContext::macroRepOp() {
  return getRuleContext<RustParser::MacroRepOpContext>(0);
}

size_t RustParser::MacroMatchTokenContext::getRuleIndex() const {
  return RustParser::RuleMacroMatchToken;
}

void RustParser::MacroMatchTokenContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroMatchToken(this);
}

void RustParser::MacroMatchTokenContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroMatchToken(this);
}

std::any
RustParser::MacroMatchTokenContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroMatchToken(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroMatchTokenContext *RustParser::macroMatchToken() {
  MacroMatchTokenContext *_localctx =
      _tracker.createInstance<MacroMatchTokenContext>(_ctx, getState());
  enterRule(_localctx, 24, RustParser::RuleMacroMatchToken);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(582);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 25, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(578);
      macroIdentifierLikeToken();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(579);
      macroLiteralToken();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(580);
      macroPunctuationToken();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(581);
      macroRepOp();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroFragSpecContext
//------------------------------------------------------------------

RustParser::MacroFragSpecContext::MacroFragSpecContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *RustParser::MacroFragSpecContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

size_t RustParser::MacroFragSpecContext::getRuleIndex() const {
  return RustParser::RuleMacroFragSpec;
}

void RustParser::MacroFragSpecContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroFragSpec(this);
}

void RustParser::MacroFragSpecContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroFragSpec(this);
}

std::any
RustParser::MacroFragSpecContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroFragSpec(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroFragSpecContext *RustParser::macroFragSpec() {
  MacroFragSpecContext *_localctx =
      _tracker.createInstance<MacroFragSpecContext>(_ctx, getState());
  enterRule(_localctx, 26, RustParser::RuleMacroFragSpec);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(584);
    identifier();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroRepSepContext
//------------------------------------------------------------------

RustParser::MacroRepSepContext::MacroRepSepContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::MacroIdentifierLikeTokenContext *
RustParser::MacroRepSepContext::macroIdentifierLikeToken() {
  return getRuleContext<RustParser::MacroIdentifierLikeTokenContext>(0);
}

RustParser::MacroLiteralTokenContext *
RustParser::MacroRepSepContext::macroLiteralToken() {
  return getRuleContext<RustParser::MacroLiteralTokenContext>(0);
}

RustParser::MacroPunctuationTokenContext *
RustParser::MacroRepSepContext::macroPunctuationToken() {
  return getRuleContext<RustParser::MacroPunctuationTokenContext>(0);
}

tree::TerminalNode *RustParser::MacroRepSepContext::DOLLAR() {
  return getToken(RustParser::DOLLAR, 0);
}

size_t RustParser::MacroRepSepContext::getRuleIndex() const {
  return RustParser::RuleMacroRepSep;
}

void RustParser::MacroRepSepContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroRepSep(this);
}

void RustParser::MacroRepSepContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroRepSep(this);
}

std::any
RustParser::MacroRepSepContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroRepSep(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroRepSepContext *RustParser::macroRepSep() {
  MacroRepSepContext *_localctx =
      _tracker.createInstance<MacroRepSepContext>(_ctx, getState());
  enterRule(_localctx, 28, RustParser::RuleMacroRepSep);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(590);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 26, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(586);
      macroIdentifierLikeToken();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(587);
      macroLiteralToken();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(588);
      macroPunctuationToken();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(589);
      match(RustParser::DOLLAR);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroRepOpContext
//------------------------------------------------------------------

RustParser::MacroRepOpContext::MacroRepOpContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::MacroRepOpContext::STAR() {
  return getToken(RustParser::STAR, 0);
}

tree::TerminalNode *RustParser::MacroRepOpContext::PLUS() {
  return getToken(RustParser::PLUS, 0);
}

tree::TerminalNode *RustParser::MacroRepOpContext::QUESTION() {
  return getToken(RustParser::QUESTION, 0);
}

size_t RustParser::MacroRepOpContext::getRuleIndex() const {
  return RustParser::RuleMacroRepOp;
}

void RustParser::MacroRepOpContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroRepOp(this);
}

void RustParser::MacroRepOpContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroRepOp(this);
}

std::any
RustParser::MacroRepOpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroRepOp(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroRepOpContext *RustParser::macroRepOp() {
  MacroRepOpContext *_localctx =
      _tracker.createInstance<MacroRepOpContext>(_ctx, getState());
  enterRule(_localctx, 30, RustParser::RuleMacroRepOp);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(592);
    _la = _input->LA(1);
    if (!(((((_la - 83) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 83)) & 4398046511109) != 0))) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroTranscriberContext
//------------------------------------------------------------------

RustParser::MacroTranscriberContext::MacroTranscriberContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::DelimTokenTreeContext *
RustParser::MacroTranscriberContext::delimTokenTree() {
  return getRuleContext<RustParser::DelimTokenTreeContext>(0);
}

size_t RustParser::MacroTranscriberContext::getRuleIndex() const {
  return RustParser::RuleMacroTranscriber;
}

void RustParser::MacroTranscriberContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroTranscriber(this);
}

void RustParser::MacroTranscriberContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroTranscriber(this);
}

std::any
RustParser::MacroTranscriberContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroTranscriber(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroTranscriberContext *RustParser::macroTranscriber() {
  MacroTranscriberContext *_localctx =
      _tracker.createInstance<MacroTranscriberContext>(_ctx, getState());
  enterRule(_localctx, 32, RustParser::RuleMacroTranscriber);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(594);
    delimTokenTree();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ItemContext
//------------------------------------------------------------------

RustParser::ItemContext::ItemContext(ParserRuleContext *parent,
                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::VisItemContext *RustParser::ItemContext::visItem() {
  return getRuleContext<RustParser::VisItemContext>(0);
}

RustParser::MacroItemContext *RustParser::ItemContext::macroItem() {
  return getRuleContext<RustParser::MacroItemContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::ItemContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::ItemContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

size_t RustParser::ItemContext::getRuleIndex() const {
  return RustParser::RuleItem;
}

void RustParser::ItemContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterItem(this);
}

void RustParser::ItemContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitItem(this);
}

std::any RustParser::ItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ItemContext *RustParser::item() {
  ItemContext *_localctx =
      _tracker.createInstance<ItemContext>(_ctx, getState());
  enterRule(_localctx, 34, RustParser::RuleItem);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(599);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(596);
      outerAttribute();
      setState(601);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(604);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_CONST:
    case RustParser::KW_ENUM:
    case RustParser::KW_EXTERN:
    case RustParser::KW_FN:
    case RustParser::KW_IMPL:
    case RustParser::KW_MOD:
    case RustParser::KW_PUB:
    case RustParser::KW_STATIC:
    case RustParser::KW_STRUCT:
    case RustParser::KW_TRAIT:
    case RustParser::KW_TYPE:
    case RustParser::KW_UNSAFE:
    case RustParser::KW_USE:
    case RustParser::KW_ASYNC:
    case RustParser::KW_UNION: {
      setState(602);
      visItem();
      break;
    }

    case RustParser::KW_CRATE:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SUPER:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::PATHSEP: {
      setState(603);
      macroItem();
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VisItemContext
//------------------------------------------------------------------

RustParser::VisItemContext::VisItemContext(ParserRuleContext *parent,
                                           size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::ModuleContext *RustParser::VisItemContext::module() {
  return getRuleContext<RustParser::ModuleContext>(0);
}

RustParser::ExternCrateContext *RustParser::VisItemContext::externCrate() {
  return getRuleContext<RustParser::ExternCrateContext>(0);
}

RustParser::UseDeclarationContext *
RustParser::VisItemContext::useDeclaration() {
  return getRuleContext<RustParser::UseDeclarationContext>(0);
}

RustParser::Function_Context *RustParser::VisItemContext::function_() {
  return getRuleContext<RustParser::Function_Context>(0);
}

RustParser::TypeAliasContext *RustParser::VisItemContext::typeAlias() {
  return getRuleContext<RustParser::TypeAliasContext>(0);
}

RustParser::Struct_Context *RustParser::VisItemContext::struct_() {
  return getRuleContext<RustParser::Struct_Context>(0);
}

RustParser::EnumerationContext *RustParser::VisItemContext::enumeration() {
  return getRuleContext<RustParser::EnumerationContext>(0);
}

RustParser::Union_Context *RustParser::VisItemContext::union_() {
  return getRuleContext<RustParser::Union_Context>(0);
}

RustParser::ConstantItemContext *RustParser::VisItemContext::constantItem() {
  return getRuleContext<RustParser::ConstantItemContext>(0);
}

RustParser::StaticItemContext *RustParser::VisItemContext::staticItem() {
  return getRuleContext<RustParser::StaticItemContext>(0);
}

RustParser::Trait_Context *RustParser::VisItemContext::trait_() {
  return getRuleContext<RustParser::Trait_Context>(0);
}

RustParser::ImplementationContext *
RustParser::VisItemContext::implementation() {
  return getRuleContext<RustParser::ImplementationContext>(0);
}

RustParser::ExternBlockContext *RustParser::VisItemContext::externBlock() {
  return getRuleContext<RustParser::ExternBlockContext>(0);
}

RustParser::VisibilityContext *RustParser::VisItemContext::visibility() {
  return getRuleContext<RustParser::VisibilityContext>(0);
}

size_t RustParser::VisItemContext::getRuleIndex() const {
  return RustParser::RuleVisItem;
}

void RustParser::VisItemContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVisItem(this);
}

void RustParser::VisItemContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVisItem(this);
}

std::any RustParser::VisItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitVisItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::VisItemContext *RustParser::visItem() {
  VisItemContext *_localctx =
      _tracker.createInstance<VisItemContext>(_ctx, getState());
  enterRule(_localctx, 36, RustParser::RuleVisItem);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(607);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_PUB) {
      setState(606);
      visibility();
    }
    setState(622);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 30, _ctx)) {
    case 1: {
      setState(609);
      module();
      break;
    }

    case 2: {
      setState(610);
      externCrate();
      break;
    }

    case 3: {
      setState(611);
      useDeclaration();
      break;
    }

    case 4: {
      setState(612);
      function_();
      break;
    }

    case 5: {
      setState(613);
      typeAlias();
      break;
    }

    case 6: {
      setState(614);
      struct_();
      break;
    }

    case 7: {
      setState(615);
      enumeration();
      break;
    }

    case 8: {
      setState(616);
      union_();
      break;
    }

    case 9: {
      setState(617);
      constantItem();
      break;
    }

    case 10: {
      setState(618);
      staticItem();
      break;
    }

    case 11: {
      setState(619);
      trait_();
      break;
    }

    case 12: {
      setState(620);
      implementation();
      break;
    }

    case 13: {
      setState(621);
      externBlock();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroItemContext
//------------------------------------------------------------------

RustParser::MacroItemContext::MacroItemContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::MacroInvocationSemiContext *
RustParser::MacroItemContext::macroInvocationSemi() {
  return getRuleContext<RustParser::MacroInvocationSemiContext>(0);
}

RustParser::MacroRulesDefinitionContext *
RustParser::MacroItemContext::macroRulesDefinition() {
  return getRuleContext<RustParser::MacroRulesDefinitionContext>(0);
}

size_t RustParser::MacroItemContext::getRuleIndex() const {
  return RustParser::RuleMacroItem;
}

void RustParser::MacroItemContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroItem(this);
}

void RustParser::MacroItemContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroItem(this);
}

std::any RustParser::MacroItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroItemContext *RustParser::macroItem() {
  MacroItemContext *_localctx =
      _tracker.createInstance<MacroItemContext>(_ctx, getState());
  enterRule(_localctx, 38, RustParser::RuleMacroItem);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(626);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 31, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(624);
      macroInvocationSemi();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(625);
      macroRulesDefinition();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ModuleContext
//------------------------------------------------------------------

RustParser::ModuleContext::ModuleContext(ParserRuleContext *parent,
                                         size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ModuleContext::KW_MOD() {
  return getToken(RustParser::KW_MOD, 0);
}

RustParser::IdentifierContext *RustParser::ModuleContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::ModuleContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

tree::TerminalNode *RustParser::ModuleContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::ModuleContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::ModuleContext::KW_UNSAFE() {
  return getToken(RustParser::KW_UNSAFE, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::ModuleContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::ModuleContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

std::vector<RustParser::ItemContext *> RustParser::ModuleContext::item() {
  return getRuleContexts<RustParser::ItemContext>();
}

RustParser::ItemContext *RustParser::ModuleContext::item(size_t i) {
  return getRuleContext<RustParser::ItemContext>(i);
}

size_t RustParser::ModuleContext::getRuleIndex() const {
  return RustParser::RuleModule;
}

void RustParser::ModuleContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterModule(this);
}

void RustParser::ModuleContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitModule(this);
}

std::any RustParser::ModuleContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitModule(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ModuleContext *RustParser::module() {
  ModuleContext *_localctx =
      _tracker.createInstance<ModuleContext>(_ctx, getState());
  enterRule(_localctx, 40, RustParser::RuleModule);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(629);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_UNSAFE) {
      setState(628);
      match(RustParser::KW_UNSAFE);
    }
    setState(631);
    match(RustParser::KW_MOD);
    setState(632);
    identifier();
    setState(648);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::SEMI: {
      setState(633);
      match(RustParser::SEMI);
      break;
    }

    case RustParser::LCURLYBRACE: {
      setState(634);
      match(RustParser::LCURLYBRACE);
      setState(638);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 33, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(635);
          innerAttribute();
        }
        setState(640);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 33, _ctx);
      }
      setState(644);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~0x3fULL) == 0) &&
              ((1ULL << _la) & 526921241179989416) != 0) ||
             _la == RustParser::PATHSEP

             || _la == RustParser::POUND) {
        setState(641);
        item();
        setState(646);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(647);
      match(RustParser::RCURLYBRACE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExternCrateContext
//------------------------------------------------------------------

RustParser::ExternCrateContext::ExternCrateContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ExternCrateContext::KW_EXTERN() {
  return getToken(RustParser::KW_EXTERN, 0);
}

tree::TerminalNode *RustParser::ExternCrateContext::KW_CRATE() {
  return getToken(RustParser::KW_CRATE, 0);
}

RustParser::CrateRefContext *RustParser::ExternCrateContext::crateRef() {
  return getRuleContext<RustParser::CrateRefContext>(0);
}

tree::TerminalNode *RustParser::ExternCrateContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::AsClauseContext *RustParser::ExternCrateContext::asClause() {
  return getRuleContext<RustParser::AsClauseContext>(0);
}

size_t RustParser::ExternCrateContext::getRuleIndex() const {
  return RustParser::RuleExternCrate;
}

void RustParser::ExternCrateContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExternCrate(this);
}

void RustParser::ExternCrateContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExternCrate(this);
}

std::any
RustParser::ExternCrateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitExternCrate(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ExternCrateContext *RustParser::externCrate() {
  ExternCrateContext *_localctx =
      _tracker.createInstance<ExternCrateContext>(_ctx, getState());
  enterRule(_localctx, 42, RustParser::RuleExternCrate);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(650);
    match(RustParser::KW_EXTERN);
    setState(651);
    match(RustParser::KW_CRATE);
    setState(652);
    crateRef();
    setState(654);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_AS) {
      setState(653);
      asClause();
    }
    setState(656);
    match(RustParser::SEMI);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CrateRefContext
//------------------------------------------------------------------

RustParser::CrateRefContext::CrateRefContext(ParserRuleContext *parent,
                                             size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *RustParser::CrateRefContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::CrateRefContext::KW_SELFVALUE() {
  return getToken(RustParser::KW_SELFVALUE, 0);
}

size_t RustParser::CrateRefContext::getRuleIndex() const {
  return RustParser::RuleCrateRef;
}

void RustParser::CrateRefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCrateRef(this);
}

void RustParser::CrateRefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCrateRef(this);
}

std::any RustParser::CrateRefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitCrateRef(this);
  else
    return visitor->visitChildren(this);
}

RustParser::CrateRefContext *RustParser::crateRef() {
  CrateRefContext *_localctx =
      _tracker.createInstance<CrateRefContext>(_ctx, getState());
  enterRule(_localctx, 44, RustParser::RuleCrateRef);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(660);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_MACRORULES:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER: {
      enterOuterAlt(_localctx, 1);
      setState(658);
      identifier();
      break;
    }

    case RustParser::KW_SELFVALUE: {
      enterOuterAlt(_localctx, 2);
      setState(659);
      match(RustParser::KW_SELFVALUE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AsClauseContext
//------------------------------------------------------------------

RustParser::AsClauseContext::AsClauseContext(ParserRuleContext *parent,
                                             size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::AsClauseContext::KW_AS() {
  return getToken(RustParser::KW_AS, 0);
}

RustParser::IdentifierContext *RustParser::AsClauseContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::AsClauseContext::UNDERSCORE() {
  return getToken(RustParser::UNDERSCORE, 0);
}

size_t RustParser::AsClauseContext::getRuleIndex() const {
  return RustParser::RuleAsClause;
}

void RustParser::AsClauseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAsClause(this);
}

void RustParser::AsClauseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAsClause(this);
}

std::any RustParser::AsClauseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAsClause(this);
  else
    return visitor->visitChildren(this);
}

RustParser::AsClauseContext *RustParser::asClause() {
  AsClauseContext *_localctx =
      _tracker.createInstance<AsClauseContext>(_ctx, getState());
  enterRule(_localctx, 46, RustParser::RuleAsClause);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(662);
    match(RustParser::KW_AS);
    setState(665);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_MACRORULES:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER: {
      setState(663);
      identifier();
      break;
    }

    case RustParser::UNDERSCORE: {
      setState(664);
      match(RustParser::UNDERSCORE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UseDeclarationContext
//------------------------------------------------------------------

RustParser::UseDeclarationContext::UseDeclarationContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::UseDeclarationContext::KW_USE() {
  return getToken(RustParser::KW_USE, 0);
}

RustParser::UseTreeContext *RustParser::UseDeclarationContext::useTree() {
  return getRuleContext<RustParser::UseTreeContext>(0);
}

tree::TerminalNode *RustParser::UseDeclarationContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

size_t RustParser::UseDeclarationContext::getRuleIndex() const {
  return RustParser::RuleUseDeclaration;
}

void RustParser::UseDeclarationContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUseDeclaration(this);
}

void RustParser::UseDeclarationContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUseDeclaration(this);
}

std::any
RustParser::UseDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitUseDeclaration(this);
  else
    return visitor->visitChildren(this);
}

RustParser::UseDeclarationContext *RustParser::useDeclaration() {
  UseDeclarationContext *_localctx =
      _tracker.createInstance<UseDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 48, RustParser::RuleUseDeclaration);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(667);
    match(RustParser::KW_USE);
    setState(668);
    useTree();
    setState(669);
    match(RustParser::SEMI);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UseTreeContext
//------------------------------------------------------------------

RustParser::UseTreeContext::UseTreeContext(ParserRuleContext *parent,
                                           size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::UseTreeContext::STAR() {
  return getToken(RustParser::STAR, 0);
}

tree::TerminalNode *RustParser::UseTreeContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::UseTreeContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::UseTreeContext::PATHSEP() {
  return getToken(RustParser::PATHSEP, 0);
}

std::vector<RustParser::UseTreeContext *>
RustParser::UseTreeContext::useTree() {
  return getRuleContexts<RustParser::UseTreeContext>();
}

RustParser::UseTreeContext *RustParser::UseTreeContext::useTree(size_t i) {
  return getRuleContext<RustParser::UseTreeContext>(i);
}

RustParser::SimplePathContext *RustParser::UseTreeContext::simplePath() {
  return getRuleContext<RustParser::SimplePathContext>(0);
}

std::vector<tree::TerminalNode *> RustParser::UseTreeContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::UseTreeContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

tree::TerminalNode *RustParser::UseTreeContext::KW_AS() {
  return getToken(RustParser::KW_AS, 0);
}

RustParser::IdentifierContext *RustParser::UseTreeContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::UseTreeContext::UNDERSCORE() {
  return getToken(RustParser::UNDERSCORE, 0);
}

size_t RustParser::UseTreeContext::getRuleIndex() const {
  return RustParser::RuleUseTree;
}

void RustParser::UseTreeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUseTree(this);
}

void RustParser::UseTreeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUseTree(this);
}

std::any RustParser::UseTreeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitUseTree(this);
  else
    return visitor->visitChildren(this);
}

RustParser::UseTreeContext *RustParser::useTree() {
  UseTreeContext *_localctx =
      _tracker.createInstance<UseTreeContext>(_ctx, getState());
  enterRule(_localctx, 50, RustParser::RuleUseTree);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(703);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 47, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(675);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 522417557060190240) != 0) ||
          _la == RustParser::PATHSEP) {
        setState(672);
        _errHandler->sync(this);

        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 39, _ctx)) {
        case 1: {
          setState(671);
          simplePath();
          break;
        }

        default:
          break;
        }
        setState(674);
        match(RustParser::PATHSEP);
      }
      setState(693);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
      case RustParser::STAR: {
        setState(677);
        match(RustParser::STAR);
        break;
      }

      case RustParser::LCURLYBRACE: {
        setState(678);
        match(RustParser::LCURLYBRACE);
        setState(690);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~0x3fULL) == 0) &&
             ((1ULL << _la) & 522417557060190240) != 0) ||
            ((((_la - 85) & ~0x3fULL) == 0) &&
             ((1ULL << (_la - 85)) & 2233382993921) != 0)) {
          setState(679);
          useTree();
          setState(684);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
              _input, 41, _ctx);
          while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
            if (alt == 1) {
              setState(680);
              match(RustParser::COMMA);
              setState(681);
              useTree();
            }
            setState(686);
            _errHandler->sync(this);
            alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
                _input, 41, _ctx);
          }
          setState(688);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == RustParser::COMMA) {
            setState(687);
            match(RustParser::COMMA);
          }
        }
        setState(692);
        match(RustParser::RCURLYBRACE);
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(695);
      simplePath();
      setState(701);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::KW_AS) {
        setState(696);
        match(RustParser::KW_AS);
        setState(699);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
        case RustParser::KW_MACRORULES:
        case RustParser::NON_KEYWORD_IDENTIFIER:
        case RustParser::RAW_IDENTIFIER: {
          setState(697);
          identifier();
          break;
        }

        case RustParser::UNDERSCORE: {
          setState(698);
          match(RustParser::UNDERSCORE);
          break;
        }

        default:
          throw NoViableAltException(this);
        }
      }
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Function_Context
//------------------------------------------------------------------

RustParser::Function_Context::Function_Context(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::FunctionQualifiersContext *
RustParser::Function_Context::functionQualifiers() {
  return getRuleContext<RustParser::FunctionQualifiersContext>(0);
}

tree::TerminalNode *RustParser::Function_Context::KW_FN() {
  return getToken(RustParser::KW_FN, 0);
}

RustParser::IdentifierContext *RustParser::Function_Context::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::Function_Context::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::Function_Context::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::BlockExpressionContext *
RustParser::Function_Context::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

tree::TerminalNode *RustParser::Function_Context::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::GenericParamsContext *
RustParser::Function_Context::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

RustParser::FunctionParametersContext *
RustParser::Function_Context::functionParameters() {
  return getRuleContext<RustParser::FunctionParametersContext>(0);
}

RustParser::FunctionReturnTypeContext *
RustParser::Function_Context::functionReturnType() {
  return getRuleContext<RustParser::FunctionReturnTypeContext>(0);
}

RustParser::WhereClauseContext *RustParser::Function_Context::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

size_t RustParser::Function_Context::getRuleIndex() const {
  return RustParser::RuleFunction_;
}

void RustParser::Function_Context::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunction_(this);
}

void RustParser::Function_Context::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunction_(this);
}

std::any RustParser::Function_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFunction_(this);
  else
    return visitor->visitChildren(this);
}

RustParser::Function_Context *RustParser::function_() {
  Function_Context *_localctx =
      _tracker.createInstance<Function_Context>(_ctx, getState());
  enterRule(_localctx, 52, RustParser::RuleFunction_);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(705);
    functionQualifiers();
    setState(706);
    match(RustParser::KW_FN);
    setState(707);
    identifier();
    setState(709);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LT) {
      setState(708);
      genericParams();
    }
    setState(711);
    match(RustParser::LPAREN);
    setState(713);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 567453833619320608) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1487371226429577343) != 0)) {
      setState(712);
      functionParameters();
    }
    setState(715);
    match(RustParser::RPAREN);
    setState(717);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::RARROW) {
      setState(716);
      functionReturnType();
    }
    setState(720);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(719);
      whereClause();
    }
    setState(724);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::LCURLYBRACE: {
      setState(722);
      blockExpression();
      break;
    }

    case RustParser::SEMI: {
      setState(723);
      match(RustParser::SEMI);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionQualifiersContext
//------------------------------------------------------------------

RustParser::FunctionQualifiersContext::FunctionQualifiersContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::FunctionQualifiersContext::KW_CONST() {
  return getToken(RustParser::KW_CONST, 0);
}

tree::TerminalNode *RustParser::FunctionQualifiersContext::KW_ASYNC() {
  return getToken(RustParser::KW_ASYNC, 0);
}

tree::TerminalNode *RustParser::FunctionQualifiersContext::KW_UNSAFE() {
  return getToken(RustParser::KW_UNSAFE, 0);
}

tree::TerminalNode *RustParser::FunctionQualifiersContext::KW_EXTERN() {
  return getToken(RustParser::KW_EXTERN, 0);
}

RustParser::AbiContext *RustParser::FunctionQualifiersContext::abi() {
  return getRuleContext<RustParser::AbiContext>(0);
}

size_t RustParser::FunctionQualifiersContext::getRuleIndex() const {
  return RustParser::RuleFunctionQualifiers;
}

void RustParser::FunctionQualifiersContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionQualifiers(this);
}

void RustParser::FunctionQualifiersContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionQualifiers(this);
}

std::any
RustParser::FunctionQualifiersContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFunctionQualifiers(this);
  else
    return visitor->visitChildren(this);
}

RustParser::FunctionQualifiersContext *RustParser::functionQualifiers() {
  FunctionQualifiersContext *_localctx =
      _tracker.createInstance<FunctionQualifiersContext>(_ctx, getState());
  enterRule(_localctx, 54, RustParser::RuleFunctionQualifiers);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(727);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_CONST) {
      setState(726);
      match(RustParser::KW_CONST);
    }
    setState(730);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_ASYNC) {
      setState(729);
      match(RustParser::KW_ASYNC);
    }
    setState(733);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_UNSAFE) {
      setState(732);
      match(RustParser::KW_UNSAFE);
    }
    setState(739);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_EXTERN) {
      setState(735);
      match(RustParser::KW_EXTERN);
      setState(737);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::STRING_LITERAL

          || _la == RustParser::RAW_STRING_LITERAL) {
        setState(736);
        abi();
      }
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AbiContext
//------------------------------------------------------------------

RustParser::AbiContext::AbiContext(ParserRuleContext *parent,
                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::AbiContext::STRING_LITERAL() {
  return getToken(RustParser::STRING_LITERAL, 0);
}

tree::TerminalNode *RustParser::AbiContext::RAW_STRING_LITERAL() {
  return getToken(RustParser::RAW_STRING_LITERAL, 0);
}

size_t RustParser::AbiContext::getRuleIndex() const {
  return RustParser::RuleAbi;
}

void RustParser::AbiContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAbi(this);
}

void RustParser::AbiContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAbi(this);
}

std::any RustParser::AbiContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAbi(this);
  else
    return visitor->visitChildren(this);
}

RustParser::AbiContext *RustParser::abi() {
  AbiContext *_localctx = _tracker.createInstance<AbiContext>(_ctx, getState());
  enterRule(_localctx, 56, RustParser::RuleAbi);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(741);
    _la = _input->LA(1);
    if (!(_la == RustParser::STRING_LITERAL

          || _la == RustParser::RAW_STRING_LITERAL)) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionParametersContext
//------------------------------------------------------------------

RustParser::FunctionParametersContext::FunctionParametersContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::SelfParamContext *
RustParser::FunctionParametersContext::selfParam() {
  return getRuleContext<RustParser::SelfParamContext>(0);
}

std::vector<tree::TerminalNode *>
RustParser::FunctionParametersContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::FunctionParametersContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

std::vector<RustParser::FunctionParamContext *>
RustParser::FunctionParametersContext::functionParam() {
  return getRuleContexts<RustParser::FunctionParamContext>();
}

RustParser::FunctionParamContext *
RustParser::FunctionParametersContext::functionParam(size_t i) {
  return getRuleContext<RustParser::FunctionParamContext>(i);
}

size_t RustParser::FunctionParametersContext::getRuleIndex() const {
  return RustParser::RuleFunctionParameters;
}

void RustParser::FunctionParametersContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionParameters(this);
}

void RustParser::FunctionParametersContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionParameters(this);
}

std::any
RustParser::FunctionParametersContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFunctionParameters(this);
  else
    return visitor->visitChildren(this);
}

RustParser::FunctionParametersContext *RustParser::functionParameters() {
  FunctionParametersContext *_localctx =
      _tracker.createInstance<FunctionParametersContext>(_ctx, getState());
  enterRule(_localctx, 58, RustParser::RuleFunctionParameters);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(763);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 62, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(743);
      selfParam();
      setState(745);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(744);
        match(RustParser::COMMA);
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(750);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 59, _ctx)) {
      case 1: {
        setState(747);
        selfParam();
        setState(748);
        match(RustParser::COMMA);
        break;
      }

      default:
        break;
      }
      setState(752);
      functionParam();
      setState(757);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 60, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(753);
          match(RustParser::COMMA);
          setState(754);
          functionParam();
        }
        setState(759);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 60, _ctx);
      }
      setState(761);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(760);
        match(RustParser::COMMA);
      }
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SelfParamContext
//------------------------------------------------------------------

RustParser::SelfParamContext::SelfParamContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::ShorthandSelfContext *
RustParser::SelfParamContext::shorthandSelf() {
  return getRuleContext<RustParser::ShorthandSelfContext>(0);
}

RustParser::TypedSelfContext *RustParser::SelfParamContext::typedSelf() {
  return getRuleContext<RustParser::TypedSelfContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::SelfParamContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::SelfParamContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

size_t RustParser::SelfParamContext::getRuleIndex() const {
  return RustParser::RuleSelfParam;
}

void RustParser::SelfParamContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSelfParam(this);
}

void RustParser::SelfParamContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSelfParam(this);
}

std::any RustParser::SelfParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitSelfParam(this);
  else
    return visitor->visitChildren(this);
}

RustParser::SelfParamContext *RustParser::selfParam() {
  SelfParamContext *_localctx =
      _tracker.createInstance<SelfParamContext>(_ctx, getState());
  enterRule(_localctx, 60, RustParser::RuleSelfParam);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(768);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(765);
      outerAttribute();
      setState(770);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(773);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 64, _ctx)) {
    case 1: {
      setState(771);
      shorthandSelf();
      break;
    }

    case 2: {
      setState(772);
      typedSelf();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShorthandSelfContext
//------------------------------------------------------------------

RustParser::ShorthandSelfContext::ShorthandSelfContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ShorthandSelfContext::KW_SELFVALUE() {
  return getToken(RustParser::KW_SELFVALUE, 0);
}

tree::TerminalNode *RustParser::ShorthandSelfContext::AND() {
  return getToken(RustParser::AND, 0);
}

tree::TerminalNode *RustParser::ShorthandSelfContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

RustParser::LifetimeContext *RustParser::ShorthandSelfContext::lifetime() {
  return getRuleContext<RustParser::LifetimeContext>(0);
}

size_t RustParser::ShorthandSelfContext::getRuleIndex() const {
  return RustParser::RuleShorthandSelf;
}

void RustParser::ShorthandSelfContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShorthandSelf(this);
}

void RustParser::ShorthandSelfContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShorthandSelf(this);
}

std::any
RustParser::ShorthandSelfContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitShorthandSelf(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ShorthandSelfContext *RustParser::shorthandSelf() {
  ShorthandSelfContext *_localctx =
      _tracker.createInstance<ShorthandSelfContext>(_ctx, getState());
  enterRule(_localctx, 62, RustParser::RuleShorthandSelf);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(779);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::AND) {
      setState(775);
      match(RustParser::AND);
      setState(777);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (((((_la - 53) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 53)) & 536870917) != 0)) {
        setState(776);
        lifetime();
      }
    }
    setState(782);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_MUT) {
      setState(781);
      match(RustParser::KW_MUT);
    }
    setState(784);
    match(RustParser::KW_SELFVALUE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypedSelfContext
//------------------------------------------------------------------

RustParser::TypedSelfContext::TypedSelfContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::TypedSelfContext::KW_SELFVALUE() {
  return getToken(RustParser::KW_SELFVALUE, 0);
}

tree::TerminalNode *RustParser::TypedSelfContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::Type_Context *RustParser::TypedSelfContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::TypedSelfContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

size_t RustParser::TypedSelfContext::getRuleIndex() const {
  return RustParser::RuleTypedSelf;
}

void RustParser::TypedSelfContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypedSelf(this);
}

void RustParser::TypedSelfContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypedSelf(this);
}

std::any RustParser::TypedSelfContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypedSelf(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypedSelfContext *RustParser::typedSelf() {
  TypedSelfContext *_localctx =
      _tracker.createInstance<TypedSelfContext>(_ctx, getState());
  enterRule(_localctx, 64, RustParser::RuleTypedSelf);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(787);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_MUT) {
      setState(786);
      match(RustParser::KW_MUT);
    }
    setState(789);
    match(RustParser::KW_SELFVALUE);
    setState(790);
    match(RustParser::COLON);
    setState(791);
    type_();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionParamContext
//------------------------------------------------------------------

RustParser::FunctionParamContext::FunctionParamContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::FunctionParamPatternContext *
RustParser::FunctionParamContext::functionParamPattern() {
  return getRuleContext<RustParser::FunctionParamPatternContext>(0);
}

tree::TerminalNode *RustParser::FunctionParamContext::DOTDOTDOT() {
  return getToken(RustParser::DOTDOTDOT, 0);
}

RustParser::Type_Context *RustParser::FunctionParamContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::FunctionParamContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::FunctionParamContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

size_t RustParser::FunctionParamContext::getRuleIndex() const {
  return RustParser::RuleFunctionParam;
}

void RustParser::FunctionParamContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionParam(this);
}

void RustParser::FunctionParamContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionParam(this);
}

std::any
RustParser::FunctionParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFunctionParam(this);
  else
    return visitor->visitChildren(this);
}

RustParser::FunctionParamContext *RustParser::functionParam() {
  FunctionParamContext *_localctx =
      _tracker.createInstance<FunctionParamContext>(_ctx, getState());
  enterRule(_localctx, 66, RustParser::RuleFunctionParam);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(796);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(793);
      outerAttribute();
      setState(798);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(802);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 70, _ctx)) {
    case 1: {
      setState(799);
      functionParamPattern();
      break;
    }

    case 2: {
      setState(800);
      match(RustParser::DOTDOTDOT);
      break;
    }

    case 3: {
      setState(801);
      type_();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionParamPatternContext
//------------------------------------------------------------------

RustParser::FunctionParamPatternContext::FunctionParamPatternContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PatternContext *RustParser::FunctionParamPatternContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

tree::TerminalNode *RustParser::FunctionParamPatternContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::Type_Context *RustParser::FunctionParamPatternContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::FunctionParamPatternContext::DOTDOTDOT() {
  return getToken(RustParser::DOTDOTDOT, 0);
}

size_t RustParser::FunctionParamPatternContext::getRuleIndex() const {
  return RustParser::RuleFunctionParamPattern;
}

void RustParser::FunctionParamPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionParamPattern(this);
}

void RustParser::FunctionParamPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionParamPattern(this);
}

std::any RustParser::FunctionParamPatternContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFunctionParamPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::FunctionParamPatternContext *RustParser::functionParamPattern() {
  FunctionParamPatternContext *_localctx =
      _tracker.createInstance<FunctionParamPatternContext>(_ctx, getState());
  enterRule(_localctx, 68, RustParser::RuleFunctionParamPattern);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(804);
    pattern();
    setState(805);
    match(RustParser::COLON);
    setState(808);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_CRATE:
    case RustParser::KW_EXTERN:
    case RustParser::KW_FN:
    case RustParser::KW_FOR:
    case RustParser::KW_IMPL:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SELFTYPE:
    case RustParser::KW_SUPER:
    case RustParser::KW_UNSAFE:
    case RustParser::KW_DYN:
    case RustParser::KW_STATICLIFETIME:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_UNDERLINELIFETIME:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::LIFETIME_OR_LABEL:
    case RustParser::STAR:
    case RustParser::NOT:
    case RustParser::AND:
    case RustParser::LT:
    case RustParser::UNDERSCORE:
    case RustParser::PATHSEP:
    case RustParser::QUESTION:
    case RustParser::LSQUAREBRACKET:
    case RustParser::LPAREN: {
      setState(806);
      type_();
      break;
    }

    case RustParser::DOTDOTDOT: {
      setState(807);
      match(RustParser::DOTDOTDOT);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionReturnTypeContext
//------------------------------------------------------------------

RustParser::FunctionReturnTypeContext::FunctionReturnTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::FunctionReturnTypeContext::RARROW() {
  return getToken(RustParser::RARROW, 0);
}

RustParser::Type_Context *RustParser::FunctionReturnTypeContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

size_t RustParser::FunctionReturnTypeContext::getRuleIndex() const {
  return RustParser::RuleFunctionReturnType;
}

void RustParser::FunctionReturnTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionReturnType(this);
}

void RustParser::FunctionReturnTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionReturnType(this);
}

std::any
RustParser::FunctionReturnTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFunctionReturnType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::FunctionReturnTypeContext *RustParser::functionReturnType() {
  FunctionReturnTypeContext *_localctx =
      _tracker.createInstance<FunctionReturnTypeContext>(_ctx, getState());
  enterRule(_localctx, 70, RustParser::RuleFunctionReturnType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(810);
    match(RustParser::RARROW);
    setState(811);
    type_();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeAliasContext
//------------------------------------------------------------------

RustParser::TypeAliasContext::TypeAliasContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::TypeAliasContext::KW_TYPE() {
  return getToken(RustParser::KW_TYPE, 0);
}

RustParser::IdentifierContext *RustParser::TypeAliasContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::TypeAliasContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::GenericParamsContext *
RustParser::TypeAliasContext::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

RustParser::WhereClauseContext *RustParser::TypeAliasContext::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

tree::TerminalNode *RustParser::TypeAliasContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::Type_Context *RustParser::TypeAliasContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

size_t RustParser::TypeAliasContext::getRuleIndex() const {
  return RustParser::RuleTypeAlias;
}

void RustParser::TypeAliasContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeAlias(this);
}

void RustParser::TypeAliasContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeAlias(this);
}

std::any RustParser::TypeAliasContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypeAlias(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypeAliasContext *RustParser::typeAlias() {
  TypeAliasContext *_localctx =
      _tracker.createInstance<TypeAliasContext>(_ctx, getState());
  enterRule(_localctx, 72, RustParser::RuleTypeAlias);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(813);
    match(RustParser::KW_TYPE);
    setState(814);
    identifier();
    setState(816);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LT) {
      setState(815);
      genericParams();
    }
    setState(819);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(818);
      whereClause();
    }
    setState(823);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::EQ) {
      setState(821);
      match(RustParser::EQ);
      setState(822);
      type_();
    }
    setState(825);
    match(RustParser::SEMI);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Struct_Context
//------------------------------------------------------------------

RustParser::Struct_Context::Struct_Context(ParserRuleContext *parent,
                                           size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::StructStructContext *RustParser::Struct_Context::structStruct() {
  return getRuleContext<RustParser::StructStructContext>(0);
}

RustParser::TupleStructContext *RustParser::Struct_Context::tupleStruct() {
  return getRuleContext<RustParser::TupleStructContext>(0);
}

size_t RustParser::Struct_Context::getRuleIndex() const {
  return RustParser::RuleStruct_;
}

void RustParser::Struct_Context::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStruct_(this);
}

void RustParser::Struct_Context::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStruct_(this);
}

std::any RustParser::Struct_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStruct_(this);
  else
    return visitor->visitChildren(this);
}

RustParser::Struct_Context *RustParser::struct_() {
  Struct_Context *_localctx =
      _tracker.createInstance<Struct_Context>(_ctx, getState());
  enterRule(_localctx, 74, RustParser::RuleStruct_);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(829);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 75, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(827);
      structStruct();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(828);
      tupleStruct();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructStructContext
//------------------------------------------------------------------

RustParser::StructStructContext::StructStructContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::StructStructContext::KW_STRUCT() {
  return getToken(RustParser::KW_STRUCT, 0);
}

RustParser::IdentifierContext *RustParser::StructStructContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::StructStructContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::StructStructContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::StructStructContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::GenericParamsContext *
RustParser::StructStructContext::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

RustParser::WhereClauseContext *RustParser::StructStructContext::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

RustParser::StructFieldsContext *
RustParser::StructStructContext::structFields() {
  return getRuleContext<RustParser::StructFieldsContext>(0);
}

size_t RustParser::StructStructContext::getRuleIndex() const {
  return RustParser::RuleStructStruct;
}

void RustParser::StructStructContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructStruct(this);
}

void RustParser::StructStructContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructStruct(this);
}

std::any
RustParser::StructStructContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructStruct(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructStructContext *RustParser::structStruct() {
  StructStructContext *_localctx =
      _tracker.createInstance<StructStructContext>(_ctx, getState());
  enterRule(_localctx, 76, RustParser::RuleStructStruct);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(831);
    match(RustParser::KW_STRUCT);
    setState(832);
    identifier();
    setState(834);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LT) {
      setState(833);
      genericParams();
    }
    setState(837);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(836);
      whereClause();
    }
    setState(845);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::LCURLYBRACE: {
      setState(839);
      match(RustParser::LCURLYBRACE);
      setState(841);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 450359962739146752) != 0) ||
          _la == RustParser::POUND) {
        setState(840);
        structFields();
      }
      setState(843);
      match(RustParser::RCURLYBRACE);
      break;
    }

    case RustParser::SEMI: {
      setState(844);
      match(RustParser::SEMI);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TupleStructContext
//------------------------------------------------------------------

RustParser::TupleStructContext::TupleStructContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::TupleStructContext::KW_STRUCT() {
  return getToken(RustParser::KW_STRUCT, 0);
}

RustParser::IdentifierContext *RustParser::TupleStructContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::TupleStructContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::TupleStructContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

tree::TerminalNode *RustParser::TupleStructContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::GenericParamsContext *
RustParser::TupleStructContext::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

RustParser::TupleFieldsContext *RustParser::TupleStructContext::tupleFields() {
  return getRuleContext<RustParser::TupleFieldsContext>(0);
}

RustParser::WhereClauseContext *RustParser::TupleStructContext::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

size_t RustParser::TupleStructContext::getRuleIndex() const {
  return RustParser::RuleTupleStruct;
}

void RustParser::TupleStructContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleStruct(this);
}

void RustParser::TupleStructContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleStruct(this);
}

std::any
RustParser::TupleStructContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleStruct(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TupleStructContext *RustParser::tupleStruct() {
  TupleStructContext *_localctx =
      _tracker.createInstance<TupleStructContext>(_ctx, getState());
  enterRule(_localctx, 78, RustParser::RuleTupleStruct);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(847);
    match(RustParser::KW_STRUCT);
    setState(848);
    identifier();
    setState(850);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LT) {
      setState(849);
      genericParams();
    }
    setState(852);
    match(RustParser::LPAREN);
    setState(854);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 567453832542432544) != 0) ||
        ((((_la - 82) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 82)) & 363114855924105) != 0)) {
      setState(853);
      tupleFields();
    }
    setState(856);
    match(RustParser::RPAREN);
    setState(858);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(857);
      whereClause();
    }
    setState(860);
    match(RustParser::SEMI);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructFieldsContext
//------------------------------------------------------------------

RustParser::StructFieldsContext::StructFieldsContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::StructFieldContext *>
RustParser::StructFieldsContext::structField() {
  return getRuleContexts<RustParser::StructFieldContext>();
}

RustParser::StructFieldContext *
RustParser::StructFieldsContext::structField(size_t i) {
  return getRuleContext<RustParser::StructFieldContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::StructFieldsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::StructFieldsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::StructFieldsContext::getRuleIndex() const {
  return RustParser::RuleStructFields;
}

void RustParser::StructFieldsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructFields(this);
}

void RustParser::StructFieldsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructFields(this);
}

std::any
RustParser::StructFieldsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructFields(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructFieldsContext *RustParser::structFields() {
  StructFieldsContext *_localctx =
      _tracker.createInstance<StructFieldsContext>(_ctx, getState());
  enterRule(_localctx, 80, RustParser::RuleStructFields);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(862);
    structField();
    setState(867);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 83,
                                                                     _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(863);
        match(RustParser::COMMA);
        setState(864);
        structField();
      }
      setState(869);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 83, _ctx);
    }
    setState(871);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(870);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructFieldContext
//------------------------------------------------------------------

RustParser::StructFieldContext::StructFieldContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *RustParser::StructFieldContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::StructFieldContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::Type_Context *RustParser::StructFieldContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::StructFieldContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::StructFieldContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::VisibilityContext *RustParser::StructFieldContext::visibility() {
  return getRuleContext<RustParser::VisibilityContext>(0);
}

size_t RustParser::StructFieldContext::getRuleIndex() const {
  return RustParser::RuleStructField;
}

void RustParser::StructFieldContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructField(this);
}

void RustParser::StructFieldContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructField(this);
}

std::any
RustParser::StructFieldContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructField(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructFieldContext *RustParser::structField() {
  StructFieldContext *_localctx =
      _tracker.createInstance<StructFieldContext>(_ctx, getState());
  enterRule(_localctx, 82, RustParser::RuleStructField);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(876);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(873);
      outerAttribute();
      setState(878);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(880);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_PUB) {
      setState(879);
      visibility();
    }
    setState(882);
    identifier();
    setState(883);
    match(RustParser::COLON);
    setState(884);
    type_();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TupleFieldsContext
//------------------------------------------------------------------

RustParser::TupleFieldsContext::TupleFieldsContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::TupleFieldContext *>
RustParser::TupleFieldsContext::tupleField() {
  return getRuleContexts<RustParser::TupleFieldContext>();
}

RustParser::TupleFieldContext *
RustParser::TupleFieldsContext::tupleField(size_t i) {
  return getRuleContext<RustParser::TupleFieldContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::TupleFieldsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::TupleFieldsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::TupleFieldsContext::getRuleIndex() const {
  return RustParser::RuleTupleFields;
}

void RustParser::TupleFieldsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleFields(this);
}

void RustParser::TupleFieldsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleFields(this);
}

std::any
RustParser::TupleFieldsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleFields(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TupleFieldsContext *RustParser::tupleFields() {
  TupleFieldsContext *_localctx =
      _tracker.createInstance<TupleFieldsContext>(_ctx, getState());
  enterRule(_localctx, 84, RustParser::RuleTupleFields);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(886);
    tupleField();
    setState(891);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 87,
                                                                     _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(887);
        match(RustParser::COMMA);
        setState(888);
        tupleField();
      }
      setState(893);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 87, _ctx);
    }
    setState(895);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(894);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TupleFieldContext
//------------------------------------------------------------------

RustParser::TupleFieldContext::TupleFieldContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::Type_Context *RustParser::TupleFieldContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::TupleFieldContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::TupleFieldContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::VisibilityContext *RustParser::TupleFieldContext::visibility() {
  return getRuleContext<RustParser::VisibilityContext>(0);
}

size_t RustParser::TupleFieldContext::getRuleIndex() const {
  return RustParser::RuleTupleField;
}

void RustParser::TupleFieldContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleField(this);
}

void RustParser::TupleFieldContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleField(this);
}

std::any
RustParser::TupleFieldContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleField(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TupleFieldContext *RustParser::tupleField() {
  TupleFieldContext *_localctx =
      _tracker.createInstance<TupleFieldContext>(_ctx, getState());
  enterRule(_localctx, 86, RustParser::RuleTupleField);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(900);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(897);
      outerAttribute();
      setState(902);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(904);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_PUB) {
      setState(903);
      visibility();
    }
    setState(906);
    type_();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumerationContext
//------------------------------------------------------------------

RustParser::EnumerationContext::EnumerationContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::EnumerationContext::KW_ENUM() {
  return getToken(RustParser::KW_ENUM, 0);
}

RustParser::IdentifierContext *RustParser::EnumerationContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::EnumerationContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::EnumerationContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

RustParser::GenericParamsContext *
RustParser::EnumerationContext::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

RustParser::WhereClauseContext *RustParser::EnumerationContext::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

RustParser::EnumItemsContext *RustParser::EnumerationContext::enumItems() {
  return getRuleContext<RustParser::EnumItemsContext>(0);
}

size_t RustParser::EnumerationContext::getRuleIndex() const {
  return RustParser::RuleEnumeration;
}

void RustParser::EnumerationContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumeration(this);
}

void RustParser::EnumerationContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumeration(this);
}

std::any
RustParser::EnumerationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumeration(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumerationContext *RustParser::enumeration() {
  EnumerationContext *_localctx =
      _tracker.createInstance<EnumerationContext>(_ctx, getState());
  enterRule(_localctx, 88, RustParser::RuleEnumeration);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(908);
    match(RustParser::KW_ENUM);
    setState(909);
    identifier();
    setState(911);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LT) {
      setState(910);
      genericParams();
    }
    setState(914);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(913);
      whereClause();
    }
    setState(916);
    match(RustParser::LCURLYBRACE);
    setState(918);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 450359962739146752) != 0) ||
        _la == RustParser::POUND) {
      setState(917);
      enumItems();
    }
    setState(920);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumItemsContext
//------------------------------------------------------------------

RustParser::EnumItemsContext::EnumItemsContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::EnumItemContext *>
RustParser::EnumItemsContext::enumItem() {
  return getRuleContexts<RustParser::EnumItemContext>();
}

RustParser::EnumItemContext *RustParser::EnumItemsContext::enumItem(size_t i) {
  return getRuleContext<RustParser::EnumItemContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::EnumItemsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::EnumItemsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::EnumItemsContext::getRuleIndex() const {
  return RustParser::RuleEnumItems;
}

void RustParser::EnumItemsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumItems(this);
}

void RustParser::EnumItemsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumItems(this);
}

std::any RustParser::EnumItemsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumItems(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumItemsContext *RustParser::enumItems() {
  EnumItemsContext *_localctx =
      _tracker.createInstance<EnumItemsContext>(_ctx, getState());
  enterRule(_localctx, 90, RustParser::RuleEnumItems);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(922);
    enumItem();
    setState(927);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 94,
                                                                     _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(923);
        match(RustParser::COMMA);
        setState(924);
        enumItem();
      }
      setState(929);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 94, _ctx);
    }
    setState(931);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(930);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumItemContext
//------------------------------------------------------------------

RustParser::EnumItemContext::EnumItemContext(ParserRuleContext *parent,
                                             size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *RustParser::EnumItemContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::EnumItemContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::EnumItemContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::VisibilityContext *RustParser::EnumItemContext::visibility() {
  return getRuleContext<RustParser::VisibilityContext>(0);
}

RustParser::EnumItemTupleContext *RustParser::EnumItemContext::enumItemTuple() {
  return getRuleContext<RustParser::EnumItemTupleContext>(0);
}

RustParser::EnumItemStructContext *
RustParser::EnumItemContext::enumItemStruct() {
  return getRuleContext<RustParser::EnumItemStructContext>(0);
}

RustParser::EnumItemDiscriminantContext *
RustParser::EnumItemContext::enumItemDiscriminant() {
  return getRuleContext<RustParser::EnumItemDiscriminantContext>(0);
}

size_t RustParser::EnumItemContext::getRuleIndex() const {
  return RustParser::RuleEnumItem;
}

void RustParser::EnumItemContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumItem(this);
}

void RustParser::EnumItemContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumItem(this);
}

std::any RustParser::EnumItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumItemContext *RustParser::enumItem() {
  EnumItemContext *_localctx =
      _tracker.createInstance<EnumItemContext>(_ctx, getState());
  enterRule(_localctx, 92, RustParser::RuleEnumItem);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(936);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(933);
      outerAttribute();
      setState(938);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(940);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_PUB) {
      setState(939);
      visibility();
    }
    setState(942);
    identifier();
    setState(946);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::LPAREN: {
      setState(943);
      enumItemTuple();
      break;
    }

    case RustParser::LCURLYBRACE: {
      setState(944);
      enumItemStruct();
      break;
    }

    case RustParser::EQ: {
      setState(945);
      enumItemDiscriminant();
      break;
    }

    case RustParser::COMMA:
    case RustParser::RCURLYBRACE: {
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumItemTupleContext
//------------------------------------------------------------------

RustParser::EnumItemTupleContext::EnumItemTupleContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::EnumItemTupleContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::EnumItemTupleContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::TupleFieldsContext *
RustParser::EnumItemTupleContext::tupleFields() {
  return getRuleContext<RustParser::TupleFieldsContext>(0);
}

size_t RustParser::EnumItemTupleContext::getRuleIndex() const {
  return RustParser::RuleEnumItemTuple;
}

void RustParser::EnumItemTupleContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumItemTuple(this);
}

void RustParser::EnumItemTupleContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumItemTuple(this);
}

std::any
RustParser::EnumItemTupleContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumItemTuple(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumItemTupleContext *RustParser::enumItemTuple() {
  EnumItemTupleContext *_localctx =
      _tracker.createInstance<EnumItemTupleContext>(_ctx, getState());
  enterRule(_localctx, 94, RustParser::RuleEnumItemTuple);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(948);
    match(RustParser::LPAREN);
    setState(950);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 567453832542432544) != 0) ||
        ((((_la - 82) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 82)) & 363114855924105) != 0)) {
      setState(949);
      tupleFields();
    }
    setState(952);
    match(RustParser::RPAREN);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumItemStructContext
//------------------------------------------------------------------

RustParser::EnumItemStructContext::EnumItemStructContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::EnumItemStructContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::EnumItemStructContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

RustParser::StructFieldsContext *
RustParser::EnumItemStructContext::structFields() {
  return getRuleContext<RustParser::StructFieldsContext>(0);
}

size_t RustParser::EnumItemStructContext::getRuleIndex() const {
  return RustParser::RuleEnumItemStruct;
}

void RustParser::EnumItemStructContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumItemStruct(this);
}

void RustParser::EnumItemStructContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumItemStruct(this);
}

std::any
RustParser::EnumItemStructContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumItemStruct(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumItemStructContext *RustParser::enumItemStruct() {
  EnumItemStructContext *_localctx =
      _tracker.createInstance<EnumItemStructContext>(_ctx, getState());
  enterRule(_localctx, 96, RustParser::RuleEnumItemStruct);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(954);
    match(RustParser::LCURLYBRACE);
    setState(956);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 450359962739146752) != 0) ||
        _la == RustParser::POUND) {
      setState(955);
      structFields();
    }
    setState(958);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumItemDiscriminantContext
//------------------------------------------------------------------

RustParser::EnumItemDiscriminantContext::EnumItemDiscriminantContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::EnumItemDiscriminantContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::ExpressionContext *
RustParser::EnumItemDiscriminantContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

size_t RustParser::EnumItemDiscriminantContext::getRuleIndex() const {
  return RustParser::RuleEnumItemDiscriminant;
}

void RustParser::EnumItemDiscriminantContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumItemDiscriminant(this);
}

void RustParser::EnumItemDiscriminantContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumItemDiscriminant(this);
}

std::any RustParser::EnumItemDiscriminantContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumItemDiscriminant(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumItemDiscriminantContext *RustParser::enumItemDiscriminant() {
  EnumItemDiscriminantContext *_localctx =
      _tracker.createInstance<EnumItemDiscriminantContext>(_ctx, getState());
  enterRule(_localctx, 98, RustParser::RuleEnumItemDiscriminant);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(960);
    match(RustParser::EQ);
    setState(961);
    expression(0);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Union_Context
//------------------------------------------------------------------

RustParser::Union_Context::Union_Context(ParserRuleContext *parent,
                                         size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::Union_Context::KW_UNION() {
  return getToken(RustParser::KW_UNION, 0);
}

RustParser::IdentifierContext *RustParser::Union_Context::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::Union_Context::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

RustParser::StructFieldsContext *RustParser::Union_Context::structFields() {
  return getRuleContext<RustParser::StructFieldsContext>(0);
}

tree::TerminalNode *RustParser::Union_Context::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

RustParser::GenericParamsContext *RustParser::Union_Context::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

RustParser::WhereClauseContext *RustParser::Union_Context::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

size_t RustParser::Union_Context::getRuleIndex() const {
  return RustParser::RuleUnion_;
}

void RustParser::Union_Context::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnion_(this);
}

void RustParser::Union_Context::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnion_(this);
}

std::any RustParser::Union_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitUnion_(this);
  else
    return visitor->visitChildren(this);
}

RustParser::Union_Context *RustParser::union_() {
  Union_Context *_localctx =
      _tracker.createInstance<Union_Context>(_ctx, getState());
  enterRule(_localctx, 100, RustParser::RuleUnion_);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(963);
    match(RustParser::KW_UNION);
    setState(964);
    identifier();
    setState(966);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LT) {
      setState(965);
      genericParams();
    }
    setState(969);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(968);
      whereClause();
    }
    setState(971);
    match(RustParser::LCURLYBRACE);
    setState(972);
    structFields();
    setState(973);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstantItemContext
//------------------------------------------------------------------

RustParser::ConstantItemContext::ConstantItemContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ConstantItemContext::KW_CONST() {
  return getToken(RustParser::KW_CONST, 0);
}

tree::TerminalNode *RustParser::ConstantItemContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::Type_Context *RustParser::ConstantItemContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::ConstantItemContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::IdentifierContext *RustParser::ConstantItemContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::ConstantItemContext::UNDERSCORE() {
  return getToken(RustParser::UNDERSCORE, 0);
}

tree::TerminalNode *RustParser::ConstantItemContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::ExpressionContext *RustParser::ConstantItemContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

size_t RustParser::ConstantItemContext::getRuleIndex() const {
  return RustParser::RuleConstantItem;
}

void RustParser::ConstantItemContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstantItem(this);
}

void RustParser::ConstantItemContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstantItem(this);
}

std::any
RustParser::ConstantItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitConstantItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ConstantItemContext *RustParser::constantItem() {
  ConstantItemContext *_localctx =
      _tracker.createInstance<ConstantItemContext>(_ctx, getState());
  enterRule(_localctx, 102, RustParser::RuleConstantItem);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(975);
    match(RustParser::KW_CONST);
    setState(978);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_MACRORULES:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER: {
      setState(976);
      identifier();
      break;
    }

    case RustParser::UNDERSCORE: {
      setState(977);
      match(RustParser::UNDERSCORE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }
    setState(980);
    match(RustParser::COLON);
    setState(981);
    type_();
    setState(984);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::EQ) {
      setState(982);
      match(RustParser::EQ);
      setState(983);
      expression(0);
    }
    setState(986);
    match(RustParser::SEMI);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StaticItemContext
//------------------------------------------------------------------

RustParser::StaticItemContext::StaticItemContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::StaticItemContext::KW_STATIC() {
  return getToken(RustParser::KW_STATIC, 0);
}

RustParser::IdentifierContext *RustParser::StaticItemContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::StaticItemContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::Type_Context *RustParser::StaticItemContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::StaticItemContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

tree::TerminalNode *RustParser::StaticItemContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

tree::TerminalNode *RustParser::StaticItemContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::ExpressionContext *RustParser::StaticItemContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

size_t RustParser::StaticItemContext::getRuleIndex() const {
  return RustParser::RuleStaticItem;
}

void RustParser::StaticItemContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStaticItem(this);
}

void RustParser::StaticItemContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStaticItem(this);
}

std::any
RustParser::StaticItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStaticItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StaticItemContext *RustParser::staticItem() {
  StaticItemContext *_localctx =
      _tracker.createInstance<StaticItemContext>(_ctx, getState());
  enterRule(_localctx, 104, RustParser::RuleStaticItem);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(988);
    match(RustParser::KW_STATIC);
    setState(990);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_MUT) {
      setState(989);
      match(RustParser::KW_MUT);
    }
    setState(992);
    identifier();
    setState(993);
    match(RustParser::COLON);
    setState(994);
    type_();
    setState(997);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::EQ) {
      setState(995);
      match(RustParser::EQ);
      setState(996);
      expression(0);
    }
    setState(999);
    match(RustParser::SEMI);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Trait_Context
//------------------------------------------------------------------

RustParser::Trait_Context::Trait_Context(ParserRuleContext *parent,
                                         size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::Trait_Context::KW_TRAIT() {
  return getToken(RustParser::KW_TRAIT, 0);
}

RustParser::IdentifierContext *RustParser::Trait_Context::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::Trait_Context::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::Trait_Context::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::Trait_Context::KW_UNSAFE() {
  return getToken(RustParser::KW_UNSAFE, 0);
}

RustParser::GenericParamsContext *RustParser::Trait_Context::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

tree::TerminalNode *RustParser::Trait_Context::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::WhereClauseContext *RustParser::Trait_Context::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::Trait_Context::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::Trait_Context::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

std::vector<RustParser::AssociatedItemContext *>
RustParser::Trait_Context::associatedItem() {
  return getRuleContexts<RustParser::AssociatedItemContext>();
}

RustParser::AssociatedItemContext *
RustParser::Trait_Context::associatedItem(size_t i) {
  return getRuleContext<RustParser::AssociatedItemContext>(i);
}

RustParser::TypeParamBoundsContext *
RustParser::Trait_Context::typeParamBounds() {
  return getRuleContext<RustParser::TypeParamBoundsContext>(0);
}

size_t RustParser::Trait_Context::getRuleIndex() const {
  return RustParser::RuleTrait_;
}

void RustParser::Trait_Context::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTrait_(this);
}

void RustParser::Trait_Context::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTrait_(this);
}

std::any RustParser::Trait_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTrait_(this);
  else
    return visitor->visitChildren(this);
}

RustParser::Trait_Context *RustParser::trait_() {
  Trait_Context *_localctx =
      _tracker.createInstance<Trait_Context>(_ctx, getState());
  enterRule(_localctx, 106, RustParser::RuleTrait_);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1002);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_UNSAFE) {
      setState(1001);
      match(RustParser::KW_UNSAFE);
    }
    setState(1004);
    match(RustParser::KW_TRAIT);
    setState(1005);
    identifier();
    setState(1007);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LT) {
      setState(1006);
      genericParams();
    }
    setState(1013);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COLON) {
      setState(1009);
      match(RustParser::COLON);
      setState(1011);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 567453553367451680) != 0) ||
          ((((_la - 82) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 82)) & 290545947639809) != 0)) {
        setState(1010);
        typeParamBounds();
      }
    }
    setState(1016);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(1015);
      whereClause();
    }
    setState(1018);
    match(RustParser::LCURLYBRACE);
    setState(1022);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     112, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1019);
        innerAttribute();
      }
      setState(1024);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 112, _ctx);
    }
    setState(1028);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~0x3fULL) == 0) &&
            ((1ULL << _la) & 522417632224216360) != 0) ||
           _la == RustParser::PATHSEP

           || _la == RustParser::POUND) {
      setState(1025);
      associatedItem();
      setState(1030);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1031);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ImplementationContext
//------------------------------------------------------------------

RustParser::ImplementationContext::ImplementationContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::InherentImplContext *
RustParser::ImplementationContext::inherentImpl() {
  return getRuleContext<RustParser::InherentImplContext>(0);
}

RustParser::TraitImplContext *RustParser::ImplementationContext::traitImpl() {
  return getRuleContext<RustParser::TraitImplContext>(0);
}

size_t RustParser::ImplementationContext::getRuleIndex() const {
  return RustParser::RuleImplementation;
}

void RustParser::ImplementationContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterImplementation(this);
}

void RustParser::ImplementationContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitImplementation(this);
}

std::any
RustParser::ImplementationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitImplementation(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ImplementationContext *RustParser::implementation() {
  ImplementationContext *_localctx =
      _tracker.createInstance<ImplementationContext>(_ctx, getState());
  enterRule(_localctx, 108, RustParser::RuleImplementation);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1035);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 114, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1033);
      inherentImpl();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1034);
      traitImpl();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- InherentImplContext
//------------------------------------------------------------------

RustParser::InherentImplContext::InherentImplContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::InherentImplContext::KW_IMPL() {
  return getToken(RustParser::KW_IMPL, 0);
}

RustParser::Type_Context *RustParser::InherentImplContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::InherentImplContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::InherentImplContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

RustParser::GenericParamsContext *
RustParser::InherentImplContext::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

RustParser::WhereClauseContext *RustParser::InherentImplContext::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::InherentImplContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::InherentImplContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

std::vector<RustParser::AssociatedItemContext *>
RustParser::InherentImplContext::associatedItem() {
  return getRuleContexts<RustParser::AssociatedItemContext>();
}

RustParser::AssociatedItemContext *
RustParser::InherentImplContext::associatedItem(size_t i) {
  return getRuleContext<RustParser::AssociatedItemContext>(i);
}

size_t RustParser::InherentImplContext::getRuleIndex() const {
  return RustParser::RuleInherentImpl;
}

void RustParser::InherentImplContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInherentImpl(this);
}

void RustParser::InherentImplContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInherentImpl(this);
}

std::any
RustParser::InherentImplContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitInherentImpl(this);
  else
    return visitor->visitChildren(this);
}

RustParser::InherentImplContext *RustParser::inherentImpl() {
  InherentImplContext *_localctx =
      _tracker.createInstance<InherentImplContext>(_ctx, getState());
  enterRule(_localctx, 110, RustParser::RuleInherentImpl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1037);
    match(RustParser::KW_IMPL);
    setState(1039);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 115, _ctx)) {
    case 1: {
      setState(1038);
      genericParams();
      break;
    }

    default:
      break;
    }
    setState(1041);
    type_();
    setState(1043);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(1042);
      whereClause();
    }
    setState(1045);
    match(RustParser::LCURLYBRACE);
    setState(1049);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     117, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1046);
        innerAttribute();
      }
      setState(1051);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 117, _ctx);
    }
    setState(1055);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~0x3fULL) == 0) &&
            ((1ULL << _la) & 522417632224216360) != 0) ||
           _la == RustParser::PATHSEP

           || _la == RustParser::POUND) {
      setState(1052);
      associatedItem();
      setState(1057);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1058);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TraitImplContext
//------------------------------------------------------------------

RustParser::TraitImplContext::TraitImplContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::TraitImplContext::KW_IMPL() {
  return getToken(RustParser::KW_IMPL, 0);
}

RustParser::TypePathContext *RustParser::TraitImplContext::typePath() {
  return getRuleContext<RustParser::TypePathContext>(0);
}

tree::TerminalNode *RustParser::TraitImplContext::KW_FOR() {
  return getToken(RustParser::KW_FOR, 0);
}

RustParser::Type_Context *RustParser::TraitImplContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::TraitImplContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::TraitImplContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::TraitImplContext::KW_UNSAFE() {
  return getToken(RustParser::KW_UNSAFE, 0);
}

RustParser::GenericParamsContext *
RustParser::TraitImplContext::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

tree::TerminalNode *RustParser::TraitImplContext::NOT() {
  return getToken(RustParser::NOT, 0);
}

RustParser::WhereClauseContext *RustParser::TraitImplContext::whereClause() {
  return getRuleContext<RustParser::WhereClauseContext>(0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::TraitImplContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::TraitImplContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

std::vector<RustParser::AssociatedItemContext *>
RustParser::TraitImplContext::associatedItem() {
  return getRuleContexts<RustParser::AssociatedItemContext>();
}

RustParser::AssociatedItemContext *
RustParser::TraitImplContext::associatedItem(size_t i) {
  return getRuleContext<RustParser::AssociatedItemContext>(i);
}

size_t RustParser::TraitImplContext::getRuleIndex() const {
  return RustParser::RuleTraitImpl;
}

void RustParser::TraitImplContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTraitImpl(this);
}

void RustParser::TraitImplContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTraitImpl(this);
}

std::any RustParser::TraitImplContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTraitImpl(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TraitImplContext *RustParser::traitImpl() {
  TraitImplContext *_localctx =
      _tracker.createInstance<TraitImplContext>(_ctx, getState());
  enterRule(_localctx, 112, RustParser::RuleTraitImpl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1061);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_UNSAFE) {
      setState(1060);
      match(RustParser::KW_UNSAFE);
    }
    setState(1063);
    match(RustParser::KW_IMPL);
    setState(1065);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LT) {
      setState(1064);
      genericParams();
    }
    setState(1068);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::NOT) {
      setState(1067);
      match(RustParser::NOT);
    }
    setState(1070);
    typePath();
    setState(1071);
    match(RustParser::KW_FOR);
    setState(1072);
    type_();
    setState(1074);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_WHERE) {
      setState(1073);
      whereClause();
    }
    setState(1076);
    match(RustParser::LCURLYBRACE);
    setState(1080);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     123, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1077);
        innerAttribute();
      }
      setState(1082);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 123, _ctx);
    }
    setState(1086);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~0x3fULL) == 0) &&
            ((1ULL << _la) & 522417632224216360) != 0) ||
           _la == RustParser::PATHSEP

           || _la == RustParser::POUND) {
      setState(1083);
      associatedItem();
      setState(1088);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1089);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExternBlockContext
//------------------------------------------------------------------

RustParser::ExternBlockContext::ExternBlockContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ExternBlockContext::KW_EXTERN() {
  return getToken(RustParser::KW_EXTERN, 0);
}

tree::TerminalNode *RustParser::ExternBlockContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::ExternBlockContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::ExternBlockContext::KW_UNSAFE() {
  return getToken(RustParser::KW_UNSAFE, 0);
}

RustParser::AbiContext *RustParser::ExternBlockContext::abi() {
  return getRuleContext<RustParser::AbiContext>(0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::ExternBlockContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::ExternBlockContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

std::vector<RustParser::ExternalItemContext *>
RustParser::ExternBlockContext::externalItem() {
  return getRuleContexts<RustParser::ExternalItemContext>();
}

RustParser::ExternalItemContext *
RustParser::ExternBlockContext::externalItem(size_t i) {
  return getRuleContext<RustParser::ExternalItemContext>(i);
}

size_t RustParser::ExternBlockContext::getRuleIndex() const {
  return RustParser::RuleExternBlock;
}

void RustParser::ExternBlockContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExternBlock(this);
}

void RustParser::ExternBlockContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExternBlock(this);
}

std::any
RustParser::ExternBlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitExternBlock(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ExternBlockContext *RustParser::externBlock() {
  ExternBlockContext *_localctx =
      _tracker.createInstance<ExternBlockContext>(_ctx, getState());
  enterRule(_localctx, 114, RustParser::RuleExternBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1092);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_UNSAFE) {
      setState(1091);
      match(RustParser::KW_UNSAFE);
    }
    setState(1094);
    match(RustParser::KW_EXTERN);
    setState(1096);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::STRING_LITERAL

        || _la == RustParser::RAW_STRING_LITERAL) {
      setState(1095);
      abi();
    }
    setState(1098);
    match(RustParser::LCURLYBRACE);
    setState(1102);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     127, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1099);
        innerAttribute();
      }
      setState(1104);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 127, _ctx);
    }
    setState(1108);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~0x3fULL) == 0) &&
            ((1ULL << _la) & 522417630143841576) != 0) ||
           _la == RustParser::PATHSEP

           || _la == RustParser::POUND) {
      setState(1105);
      externalItem();
      setState(1110);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1111);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExternalItemContext
//------------------------------------------------------------------

RustParser::ExternalItemContext::ExternalItemContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::MacroInvocationSemiContext *
RustParser::ExternalItemContext::macroInvocationSemi() {
  return getRuleContext<RustParser::MacroInvocationSemiContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::ExternalItemContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::ExternalItemContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::StaticItemContext *RustParser::ExternalItemContext::staticItem() {
  return getRuleContext<RustParser::StaticItemContext>(0);
}

RustParser::Function_Context *RustParser::ExternalItemContext::function_() {
  return getRuleContext<RustParser::Function_Context>(0);
}

RustParser::VisibilityContext *RustParser::ExternalItemContext::visibility() {
  return getRuleContext<RustParser::VisibilityContext>(0);
}

size_t RustParser::ExternalItemContext::getRuleIndex() const {
  return RustParser::RuleExternalItem;
}

void RustParser::ExternalItemContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExternalItem(this);
}

void RustParser::ExternalItemContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExternalItem(this);
}

std::any
RustParser::ExternalItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitExternalItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ExternalItemContext *RustParser::externalItem() {
  ExternalItemContext *_localctx =
      _tracker.createInstance<ExternalItemContext>(_ctx, getState());
  enterRule(_localctx, 116, RustParser::RuleExternalItem);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1116);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(1113);
      outerAttribute();
      setState(1118);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1127);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_CRATE:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SUPER:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::PATHSEP: {
      setState(1119);
      macroInvocationSemi();
      break;
    }

    case RustParser::KW_CONST:
    case RustParser::KW_EXTERN:
    case RustParser::KW_FN:
    case RustParser::KW_PUB:
    case RustParser::KW_STATIC:
    case RustParser::KW_UNSAFE:
    case RustParser::KW_ASYNC: {
      setState(1121);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::KW_PUB) {
        setState(1120);
        visibility();
      }
      setState(1125);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
      case RustParser::KW_STATIC: {
        setState(1123);
        staticItem();
        break;
      }

      case RustParser::KW_CONST:
      case RustParser::KW_EXTERN:
      case RustParser::KW_FN:
      case RustParser::KW_UNSAFE:
      case RustParser::KW_ASYNC: {
        setState(1124);
        function_();
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericParamsContext
//------------------------------------------------------------------

RustParser::GenericParamsContext::GenericParamsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::GenericParamsContext::LT() {
  return getToken(RustParser::LT, 0);
}

tree::TerminalNode *RustParser::GenericParamsContext::GT() {
  return getToken(RustParser::GT, 0);
}

std::vector<RustParser::GenericParamContext *>
RustParser::GenericParamsContext::genericParam() {
  return getRuleContexts<RustParser::GenericParamContext>();
}

RustParser::GenericParamContext *
RustParser::GenericParamsContext::genericParam(size_t i) {
  return getRuleContext<RustParser::GenericParamContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::GenericParamsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::GenericParamsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::GenericParamsContext::getRuleIndex() const {
  return RustParser::RuleGenericParams;
}

void RustParser::GenericParamsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericParams(this);
}

void RustParser::GenericParamsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericParams(this);
}

std::any
RustParser::GenericParamsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericParams(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericParamsContext *RustParser::genericParams() {
  GenericParamsContext *_localctx =
      _tracker.createInstance<GenericParamsContext>(_ctx, getState());
  enterRule(_localctx, 118, RustParser::RuleGenericParams);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1129);
    match(RustParser::LT);
    setState(1142);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 450359962737049608) != 0) ||
        _la == RustParser::LIFETIME_OR_LABEL

        || _la == RustParser::POUND) {
      setState(1135);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 133, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(1130);
          genericParam();
          setState(1131);
          match(RustParser::COMMA);
        }
        setState(1137);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 133, _ctx);
      }
      setState(1138);
      genericParam();
      setState(1140);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(1139);
        match(RustParser::COMMA);
      }
    }
    setState(1144);
    match(RustParser::GT);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericParamContext
//------------------------------------------------------------------

RustParser::GenericParamContext::GenericParamContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::LifetimeParamContext *
RustParser::GenericParamContext::lifetimeParam() {
  return getRuleContext<RustParser::LifetimeParamContext>(0);
}

RustParser::TypeParamContext *RustParser::GenericParamContext::typeParam() {
  return getRuleContext<RustParser::TypeParamContext>(0);
}

RustParser::ConstParamContext *RustParser::GenericParamContext::constParam() {
  return getRuleContext<RustParser::ConstParamContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::GenericParamContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::GenericParamContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

size_t RustParser::GenericParamContext::getRuleIndex() const {
  return RustParser::RuleGenericParam;
}

void RustParser::GenericParamContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericParam(this);
}

void RustParser::GenericParamContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericParam(this);
}

std::any
RustParser::GenericParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericParam(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericParamContext *RustParser::genericParam() {
  GenericParamContext *_localctx =
      _tracker.createInstance<GenericParamContext>(_ctx, getState());
  enterRule(_localctx, 120, RustParser::RuleGenericParam);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1149);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     136, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1146);
        outerAttribute();
      }
      setState(1151);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 136, _ctx);
    }
    setState(1155);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 137, _ctx)) {
    case 1: {
      setState(1152);
      lifetimeParam();
      break;
    }

    case 2: {
      setState(1153);
      typeParam();
      break;
    }

    case 3: {
      setState(1154);
      constParam();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LifetimeParamContext
//------------------------------------------------------------------

RustParser::LifetimeParamContext::LifetimeParamContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::LifetimeParamContext::LIFETIME_OR_LABEL() {
  return getToken(RustParser::LIFETIME_OR_LABEL, 0);
}

RustParser::OuterAttributeContext *
RustParser::LifetimeParamContext::outerAttribute() {
  return getRuleContext<RustParser::OuterAttributeContext>(0);
}

tree::TerminalNode *RustParser::LifetimeParamContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::LifetimeBoundsContext *
RustParser::LifetimeParamContext::lifetimeBounds() {
  return getRuleContext<RustParser::LifetimeBoundsContext>(0);
}

size_t RustParser::LifetimeParamContext::getRuleIndex() const {
  return RustParser::RuleLifetimeParam;
}

void RustParser::LifetimeParamContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLifetimeParam(this);
}

void RustParser::LifetimeParamContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLifetimeParam(this);
}

std::any
RustParser::LifetimeParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLifetimeParam(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LifetimeParamContext *RustParser::lifetimeParam() {
  LifetimeParamContext *_localctx =
      _tracker.createInstance<LifetimeParamContext>(_ctx, getState());
  enterRule(_localctx, 122, RustParser::RuleLifetimeParam);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1158);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::POUND) {
      setState(1157);
      outerAttribute();
    }
    setState(1160);
    match(RustParser::LIFETIME_OR_LABEL);
    setState(1163);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COLON) {
      setState(1161);
      match(RustParser::COLON);
      setState(1162);
      lifetimeBounds();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeParamContext
//------------------------------------------------------------------

RustParser::TypeParamContext::TypeParamContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *RustParser::TypeParamContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

RustParser::OuterAttributeContext *
RustParser::TypeParamContext::outerAttribute() {
  return getRuleContext<RustParser::OuterAttributeContext>(0);
}

tree::TerminalNode *RustParser::TypeParamContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

tree::TerminalNode *RustParser::TypeParamContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::Type_Context *RustParser::TypeParamContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

RustParser::TypeParamBoundsContext *
RustParser::TypeParamContext::typeParamBounds() {
  return getRuleContext<RustParser::TypeParamBoundsContext>(0);
}

size_t RustParser::TypeParamContext::getRuleIndex() const {
  return RustParser::RuleTypeParam;
}

void RustParser::TypeParamContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeParam(this);
}

void RustParser::TypeParamContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeParam(this);
}

std::any RustParser::TypeParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypeParam(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypeParamContext *RustParser::typeParam() {
  TypeParamContext *_localctx =
      _tracker.createInstance<TypeParamContext>(_ctx, getState());
  enterRule(_localctx, 124, RustParser::RuleTypeParam);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1166);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::POUND) {
      setState(1165);
      outerAttribute();
    }
    setState(1168);
    identifier();
    setState(1173);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COLON) {
      setState(1169);
      match(RustParser::COLON);
      setState(1171);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 567453553367451680) != 0) ||
          ((((_la - 82) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 82)) & 290545947639809) != 0)) {
        setState(1170);
        typeParamBounds();
      }
    }
    setState(1177);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::EQ) {
      setState(1175);
      match(RustParser::EQ);
      setState(1176);
      type_();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstParamContext
//------------------------------------------------------------------

RustParser::ConstParamContext::ConstParamContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ConstParamContext::KW_CONST() {
  return getToken(RustParser::KW_CONST, 0);
}

RustParser::IdentifierContext *RustParser::ConstParamContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::ConstParamContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::Type_Context *RustParser::ConstParamContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

size_t RustParser::ConstParamContext::getRuleIndex() const {
  return RustParser::RuleConstParam;
}

void RustParser::ConstParamContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstParam(this);
}

void RustParser::ConstParamContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstParam(this);
}

std::any
RustParser::ConstParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitConstParam(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ConstParamContext *RustParser::constParam() {
  ConstParamContext *_localctx =
      _tracker.createInstance<ConstParamContext>(_ctx, getState());
  enterRule(_localctx, 126, RustParser::RuleConstParam);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1179);
    match(RustParser::KW_CONST);
    setState(1180);
    identifier();
    setState(1181);
    match(RustParser::COLON);
    setState(1182);
    type_();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- WhereClauseContext
//------------------------------------------------------------------

RustParser::WhereClauseContext::WhereClauseContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::WhereClauseContext::KW_WHERE() {
  return getToken(RustParser::KW_WHERE, 0);
}

std::vector<RustParser::WhereClauseItemContext *>
RustParser::WhereClauseContext::whereClauseItem() {
  return getRuleContexts<RustParser::WhereClauseItemContext>();
}

RustParser::WhereClauseItemContext *
RustParser::WhereClauseContext::whereClauseItem(size_t i) {
  return getRuleContext<RustParser::WhereClauseItemContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::WhereClauseContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::WhereClauseContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::WhereClauseContext::getRuleIndex() const {
  return RustParser::RuleWhereClause;
}

void RustParser::WhereClauseContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWhereClause(this);
}

void RustParser::WhereClauseContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWhereClause(this);
}

std::any
RustParser::WhereClauseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitWhereClause(this);
  else
    return visitor->visitChildren(this);
}

RustParser::WhereClauseContext *RustParser::whereClause() {
  WhereClauseContext *_localctx =
      _tracker.createInstance<WhereClauseContext>(_ctx, getState());
  enterRule(_localctx, 128, RustParser::RuleWhereClause);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1184);
    match(RustParser::KW_WHERE);
    setState(1190);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     144, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1185);
        whereClauseItem();
        setState(1186);
        match(RustParser::COMMA);
      }
      setState(1192);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 144, _ctx);
    }
    setState(1194);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 567453832540335392) != 0) ||
        ((((_la - 82) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 82)) & 360915832668553) != 0)) {
      setState(1193);
      whereClauseItem();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- WhereClauseItemContext
//------------------------------------------------------------------

RustParser::WhereClauseItemContext::WhereClauseItemContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::LifetimeWhereClauseItemContext *
RustParser::WhereClauseItemContext::lifetimeWhereClauseItem() {
  return getRuleContext<RustParser::LifetimeWhereClauseItemContext>(0);
}

RustParser::TypeBoundWhereClauseItemContext *
RustParser::WhereClauseItemContext::typeBoundWhereClauseItem() {
  return getRuleContext<RustParser::TypeBoundWhereClauseItemContext>(0);
}

size_t RustParser::WhereClauseItemContext::getRuleIndex() const {
  return RustParser::RuleWhereClauseItem;
}

void RustParser::WhereClauseItemContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWhereClauseItem(this);
}

void RustParser::WhereClauseItemContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWhereClauseItem(this);
}

std::any
RustParser::WhereClauseItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitWhereClauseItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::WhereClauseItemContext *RustParser::whereClauseItem() {
  WhereClauseItemContext *_localctx =
      _tracker.createInstance<WhereClauseItemContext>(_ctx, getState());
  enterRule(_localctx, 130, RustParser::RuleWhereClauseItem);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1198);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 146, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1196);
      lifetimeWhereClauseItem();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1197);
      typeBoundWhereClauseItem();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LifetimeWhereClauseItemContext
//------------------------------------------------------------------

RustParser::LifetimeWhereClauseItemContext::LifetimeWhereClauseItemContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::LifetimeContext *
RustParser::LifetimeWhereClauseItemContext::lifetime() {
  return getRuleContext<RustParser::LifetimeContext>(0);
}

tree::TerminalNode *RustParser::LifetimeWhereClauseItemContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::LifetimeBoundsContext *
RustParser::LifetimeWhereClauseItemContext::lifetimeBounds() {
  return getRuleContext<RustParser::LifetimeBoundsContext>(0);
}

size_t RustParser::LifetimeWhereClauseItemContext::getRuleIndex() const {
  return RustParser::RuleLifetimeWhereClauseItem;
}

void RustParser::LifetimeWhereClauseItemContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLifetimeWhereClauseItem(this);
}

void RustParser::LifetimeWhereClauseItemContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLifetimeWhereClauseItem(this);
}

std::any RustParser::LifetimeWhereClauseItemContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLifetimeWhereClauseItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LifetimeWhereClauseItemContext *
RustParser::lifetimeWhereClauseItem() {
  LifetimeWhereClauseItemContext *_localctx =
      _tracker.createInstance<LifetimeWhereClauseItemContext>(_ctx, getState());
  enterRule(_localctx, 132, RustParser::RuleLifetimeWhereClauseItem);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1200);
    lifetime();
    setState(1201);
    match(RustParser::COLON);
    setState(1202);
    lifetimeBounds();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeBoundWhereClauseItemContext
//------------------------------------------------------------------

RustParser::TypeBoundWhereClauseItemContext::TypeBoundWhereClauseItemContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::Type_Context *RustParser::TypeBoundWhereClauseItemContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::TypeBoundWhereClauseItemContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::ForLifetimesContext *
RustParser::TypeBoundWhereClauseItemContext::forLifetimes() {
  return getRuleContext<RustParser::ForLifetimesContext>(0);
}

RustParser::TypeParamBoundsContext *
RustParser::TypeBoundWhereClauseItemContext::typeParamBounds() {
  return getRuleContext<RustParser::TypeParamBoundsContext>(0);
}

size_t RustParser::TypeBoundWhereClauseItemContext::getRuleIndex() const {
  return RustParser::RuleTypeBoundWhereClauseItem;
}

void RustParser::TypeBoundWhereClauseItemContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeBoundWhereClauseItem(this);
}

void RustParser::TypeBoundWhereClauseItemContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeBoundWhereClauseItem(this);
}

std::any RustParser::TypeBoundWhereClauseItemContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypeBoundWhereClauseItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypeBoundWhereClauseItemContext *
RustParser::typeBoundWhereClauseItem() {
  TypeBoundWhereClauseItemContext *_localctx =
      _tracker.createInstance<TypeBoundWhereClauseItemContext>(_ctx,
                                                               getState());
  enterRule(_localctx, 134, RustParser::RuleTypeBoundWhereClauseItem);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1205);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 147, _ctx)) {
    case 1: {
      setState(1204);
      forLifetimes();
      break;
    }

    default:
      break;
    }
    setState(1207);
    type_();
    setState(1208);
    match(RustParser::COLON);
    setState(1210);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 567453553367451680) != 0) ||
        ((((_la - 82) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 82)) & 290545947639809) != 0)) {
      setState(1209);
      typeParamBounds();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ForLifetimesContext
//------------------------------------------------------------------

RustParser::ForLifetimesContext::ForLifetimesContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ForLifetimesContext::KW_FOR() {
  return getToken(RustParser::KW_FOR, 0);
}

RustParser::GenericParamsContext *
RustParser::ForLifetimesContext::genericParams() {
  return getRuleContext<RustParser::GenericParamsContext>(0);
}

size_t RustParser::ForLifetimesContext::getRuleIndex() const {
  return RustParser::RuleForLifetimes;
}

void RustParser::ForLifetimesContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterForLifetimes(this);
}

void RustParser::ForLifetimesContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitForLifetimes(this);
}

std::any
RustParser::ForLifetimesContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitForLifetimes(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ForLifetimesContext *RustParser::forLifetimes() {
  ForLifetimesContext *_localctx =
      _tracker.createInstance<ForLifetimesContext>(_ctx, getState());
  enterRule(_localctx, 136, RustParser::RuleForLifetimes);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1212);
    match(RustParser::KW_FOR);
    setState(1213);
    genericParams();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssociatedItemContext
//------------------------------------------------------------------

RustParser::AssociatedItemContext::AssociatedItemContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::MacroInvocationSemiContext *
RustParser::AssociatedItemContext::macroInvocationSemi() {
  return getRuleContext<RustParser::MacroInvocationSemiContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::AssociatedItemContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::AssociatedItemContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::TypeAliasContext *RustParser::AssociatedItemContext::typeAlias() {
  return getRuleContext<RustParser::TypeAliasContext>(0);
}

RustParser::ConstantItemContext *
RustParser::AssociatedItemContext::constantItem() {
  return getRuleContext<RustParser::ConstantItemContext>(0);
}

RustParser::Function_Context *RustParser::AssociatedItemContext::function_() {
  return getRuleContext<RustParser::Function_Context>(0);
}

RustParser::VisibilityContext *RustParser::AssociatedItemContext::visibility() {
  return getRuleContext<RustParser::VisibilityContext>(0);
}

size_t RustParser::AssociatedItemContext::getRuleIndex() const {
  return RustParser::RuleAssociatedItem;
}

void RustParser::AssociatedItemContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssociatedItem(this);
}

void RustParser::AssociatedItemContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssociatedItem(this);
}

std::any
RustParser::AssociatedItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAssociatedItem(this);
  else
    return visitor->visitChildren(this);
}

RustParser::AssociatedItemContext *RustParser::associatedItem() {
  AssociatedItemContext *_localctx =
      _tracker.createInstance<AssociatedItemContext>(_ctx, getState());
  enterRule(_localctx, 138, RustParser::RuleAssociatedItem);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1218);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(1215);
      outerAttribute();
      setState(1220);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1230);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_CRATE:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SUPER:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::PATHSEP: {
      setState(1221);
      macroInvocationSemi();
      break;
    }

    case RustParser::KW_CONST:
    case RustParser::KW_EXTERN:
    case RustParser::KW_FN:
    case RustParser::KW_PUB:
    case RustParser::KW_TYPE:
    case RustParser::KW_UNSAFE:
    case RustParser::KW_ASYNC: {
      setState(1223);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::KW_PUB) {
        setState(1222);
        visibility();
      }
      setState(1228);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 151, _ctx)) {
      case 1: {
        setState(1225);
        typeAlias();
        break;
      }

      case 2: {
        setState(1226);
        constantItem();
        break;
      }

      case 3: {
        setState(1227);
        function_();
        break;
      }

      default:
        break;
      }
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- InnerAttributeContext
//------------------------------------------------------------------

RustParser::InnerAttributeContext::InnerAttributeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::InnerAttributeContext::POUND() {
  return getToken(RustParser::POUND, 0);
}

tree::TerminalNode *RustParser::InnerAttributeContext::NOT() {
  return getToken(RustParser::NOT, 0);
}

tree::TerminalNode *RustParser::InnerAttributeContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

RustParser::AttrContext *RustParser::InnerAttributeContext::attr() {
  return getRuleContext<RustParser::AttrContext>(0);
}

tree::TerminalNode *RustParser::InnerAttributeContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

size_t RustParser::InnerAttributeContext::getRuleIndex() const {
  return RustParser::RuleInnerAttribute;
}

void RustParser::InnerAttributeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInnerAttribute(this);
}

void RustParser::InnerAttributeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInnerAttribute(this);
}

std::any
RustParser::InnerAttributeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitInnerAttribute(this);
  else
    return visitor->visitChildren(this);
}

RustParser::InnerAttributeContext *RustParser::innerAttribute() {
  InnerAttributeContext *_localctx =
      _tracker.createInstance<InnerAttributeContext>(_ctx, getState());
  enterRule(_localctx, 140, RustParser::RuleInnerAttribute);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1232);
    match(RustParser::POUND);
    setState(1233);
    match(RustParser::NOT);
    setState(1234);
    match(RustParser::LSQUAREBRACKET);
    setState(1235);
    attr();
    setState(1236);
    match(RustParser::RSQUAREBRACKET);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OuterAttributeContext
//------------------------------------------------------------------

RustParser::OuterAttributeContext::OuterAttributeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::OuterAttributeContext::POUND() {
  return getToken(RustParser::POUND, 0);
}

tree::TerminalNode *RustParser::OuterAttributeContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

RustParser::AttrContext *RustParser::OuterAttributeContext::attr() {
  return getRuleContext<RustParser::AttrContext>(0);
}

tree::TerminalNode *RustParser::OuterAttributeContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

size_t RustParser::OuterAttributeContext::getRuleIndex() const {
  return RustParser::RuleOuterAttribute;
}

void RustParser::OuterAttributeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOuterAttribute(this);
}

void RustParser::OuterAttributeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOuterAttribute(this);
}

std::any
RustParser::OuterAttributeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitOuterAttribute(this);
  else
    return visitor->visitChildren(this);
}

RustParser::OuterAttributeContext *RustParser::outerAttribute() {
  OuterAttributeContext *_localctx =
      _tracker.createInstance<OuterAttributeContext>(_ctx, getState());
  enterRule(_localctx, 142, RustParser::RuleOuterAttribute);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1238);
    match(RustParser::POUND);
    setState(1239);
    match(RustParser::LSQUAREBRACKET);
    setState(1240);
    attr();
    setState(1241);
    match(RustParser::RSQUAREBRACKET);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AttrContext
//------------------------------------------------------------------

RustParser::AttrContext::AttrContext(ParserRuleContext *parent,
                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::SimplePathContext *RustParser::AttrContext::simplePath() {
  return getRuleContext<RustParser::SimplePathContext>(0);
}

RustParser::AttrInputContext *RustParser::AttrContext::attrInput() {
  return getRuleContext<RustParser::AttrInputContext>(0);
}

size_t RustParser::AttrContext::getRuleIndex() const {
  return RustParser::RuleAttr;
}

void RustParser::AttrContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAttr(this);
}

void RustParser::AttrContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAttr(this);
}

std::any RustParser::AttrContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAttr(this);
  else
    return visitor->visitChildren(this);
}

RustParser::AttrContext *RustParser::attr() {
  AttrContext *_localctx =
      _tracker.createInstance<AttrContext>(_ctx, getState());
  enterRule(_localctx, 144, RustParser::RuleAttr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1243);
    simplePath();
    setState(1245);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 104) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 104)) & 88080385) != 0)) {
      setState(1244);
      attrInput();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AttrInputContext
//------------------------------------------------------------------

RustParser::AttrInputContext::AttrInputContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::DelimTokenTreeContext *
RustParser::AttrInputContext::delimTokenTree() {
  return getRuleContext<RustParser::DelimTokenTreeContext>(0);
}

tree::TerminalNode *RustParser::AttrInputContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::LiteralExpressionContext *
RustParser::AttrInputContext::literalExpression() {
  return getRuleContext<RustParser::LiteralExpressionContext>(0);
}

size_t RustParser::AttrInputContext::getRuleIndex() const {
  return RustParser::RuleAttrInput;
}

void RustParser::AttrInputContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAttrInput(this);
}

void RustParser::AttrInputContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAttrInput(this);
}

std::any RustParser::AttrInputContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAttrInput(this);
  else
    return visitor->visitChildren(this);
}

RustParser::AttrInputContext *RustParser::attrInput() {
  AttrInputContext *_localctx =
      _tracker.createInstance<AttrInputContext>(_ctx, getState());
  enterRule(_localctx, 146, RustParser::RuleAttrInput);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1250);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::LCURLYBRACE:
    case RustParser::LSQUAREBRACKET:
    case RustParser::LPAREN: {
      enterOuterAlt(_localctx, 1);
      setState(1247);
      delimTokenTree();
      break;
    }

    case RustParser::EQ: {
      enterOuterAlt(_localctx, 2);
      setState(1248);
      match(RustParser::EQ);
      setState(1249);
      literalExpression();
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext
//------------------------------------------------------------------

RustParser::StatementContext::StatementContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::StatementContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::ItemContext *RustParser::StatementContext::item() {
  return getRuleContext<RustParser::ItemContext>(0);
}

RustParser::LetStatementContext *RustParser::StatementContext::letStatement() {
  return getRuleContext<RustParser::LetStatementContext>(0);
}

RustParser::ExpressionStatementContext *
RustParser::StatementContext::expressionStatement() {
  return getRuleContext<RustParser::ExpressionStatementContext>(0);
}

RustParser::MacroInvocationSemiContext *
RustParser::StatementContext::macroInvocationSemi() {
  return getRuleContext<RustParser::MacroInvocationSemiContext>(0);
}

size_t RustParser::StatementContext::getRuleIndex() const {
  return RustParser::RuleStatement;
}

void RustParser::StatementContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void RustParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}

std::any RustParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StatementContext *RustParser::statement() {
  StatementContext *_localctx =
      _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 148, RustParser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1257);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 155, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1252);
      match(RustParser::SEMI);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1253);
      item();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(1254);
      letStatement();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(1255);
      expressionStatement();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(1256);
      macroInvocationSemi();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LetStatementContext
//------------------------------------------------------------------

RustParser::LetStatementContext::LetStatementContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::LetStatementContext::KW_LET() {
  return getToken(RustParser::KW_LET, 0);
}

RustParser::PatternNoTopAltContext *
RustParser::LetStatementContext::patternNoTopAlt() {
  return getRuleContext<RustParser::PatternNoTopAltContext>(0);
}

tree::TerminalNode *RustParser::LetStatementContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::LetStatementContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::LetStatementContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

tree::TerminalNode *RustParser::LetStatementContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::Type_Context *RustParser::LetStatementContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::LetStatementContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::ExpressionContext *RustParser::LetStatementContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

size_t RustParser::LetStatementContext::getRuleIndex() const {
  return RustParser::RuleLetStatement;
}

void RustParser::LetStatementContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLetStatement(this);
}

void RustParser::LetStatementContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLetStatement(this);
}

std::any
RustParser::LetStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLetStatement(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LetStatementContext *RustParser::letStatement() {
  LetStatementContext *_localctx =
      _tracker.createInstance<LetStatementContext>(_ctx, getState());
  enterRule(_localctx, 150, RustParser::RuleLetStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1262);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(1259);
      outerAttribute();
      setState(1264);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1265);
    match(RustParser::KW_LET);
    setState(1266);
    patternNoTopAlt();
    setState(1269);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COLON) {
      setState(1267);
      match(RustParser::COLON);
      setState(1268);
      type_();
    }
    setState(1273);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::EQ) {
      setState(1271);
      match(RustParser::EQ);
      setState(1272);
      expression(0);
    }
    setState(1275);
    match(RustParser::SEMI);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionStatementContext
//------------------------------------------------------------------

RustParser::ExpressionStatementContext::ExpressionStatementContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::ExpressionContext *
RustParser::ExpressionStatementContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::ExpressionStatementContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::ExpressionWithBlockContext *
RustParser::ExpressionStatementContext::expressionWithBlock() {
  return getRuleContext<RustParser::ExpressionWithBlockContext>(0);
}

size_t RustParser::ExpressionStatementContext::getRuleIndex() const {
  return RustParser::RuleExpressionStatement;
}

void RustParser::ExpressionStatementContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressionStatement(this);
}

void RustParser::ExpressionStatementContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressionStatement(this);
}

std::any RustParser::ExpressionStatementContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitExpressionStatement(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ExpressionStatementContext *RustParser::expressionStatement() {
  ExpressionStatementContext *_localctx =
      _tracker.createInstance<ExpressionStatementContext>(_ctx, getState());
  enterRule(_localctx, 152, RustParser::RuleExpressionStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1284);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 160, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1277);
      expression(0);
      setState(1278);
      match(RustParser::SEMI);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1280);
      expressionWithBlock();
      setState(1282);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 159, _ctx)) {
      case 1: {
        setState(1281);
        match(RustParser::SEMI);
        break;
      }

      default:
        break;
      }
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

size_t RustParser::ExpressionContext::getRuleIndex() const {
  return RustParser::RuleExpression;
}

void RustParser::ExpressionContext::copyFrom(ExpressionContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- TypeCastExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::TypeCastExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::TypeCastExpressionContext::KW_AS() {
  return getToken(RustParser::KW_AS, 0);
}

RustParser::TypeNoBoundsContext *
RustParser::TypeCastExpressionContext::typeNoBounds() {
  return getRuleContext<RustParser::TypeNoBoundsContext>(0);
}

RustParser::TypeCastExpressionContext::TypeCastExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::TypeCastExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeCastExpression(this);
}
void RustParser::TypeCastExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeCastExpression(this);
}

std::any
RustParser::TypeCastExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypeCastExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- PathExpression_Context
//------------------------------------------------------------------

RustParser::PathExpressionContext *
RustParser::PathExpression_Context::pathExpression() {
  return getRuleContext<RustParser::PathExpressionContext>(0);
}

RustParser::PathExpression_Context::PathExpression_Context(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::PathExpression_Context::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPathExpression_(this);
}
void RustParser::PathExpression_Context::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPathExpression_(this);
}

std::any
RustParser::PathExpression_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPathExpression_(this);
  else
    return visitor->visitChildren(this);
}
//----------------- TupleExpressionContext
//------------------------------------------------------------------

tree::TerminalNode *RustParser::TupleExpressionContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::TupleExpressionContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::TupleExpressionContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::TupleExpressionContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

RustParser::TupleElementsContext *
RustParser::TupleExpressionContext::tupleElements() {
  return getRuleContext<RustParser::TupleElementsContext>(0);
}

RustParser::TupleExpressionContext::TupleExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::TupleExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleExpression(this);
}
void RustParser::TupleExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleExpression(this);
}

std::any
RustParser::TupleExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IndexExpressionContext
//------------------------------------------------------------------

std::vector<RustParser::ExpressionContext *>
RustParser::IndexExpressionContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::IndexExpressionContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

tree::TerminalNode *RustParser::IndexExpressionContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::IndexExpressionContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

RustParser::IndexExpressionContext::IndexExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::IndexExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndexExpression(this);
}
void RustParser::IndexExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndexExpression(this);
}

std::any
RustParser::IndexExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitIndexExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- RangeExpressionContext
//------------------------------------------------------------------

tree::TerminalNode *RustParser::RangeExpressionContext::DOTDOT() {
  return getToken(RustParser::DOTDOT, 0);
}

std::vector<RustParser::ExpressionContext *>
RustParser::RangeExpressionContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::RangeExpressionContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

tree::TerminalNode *RustParser::RangeExpressionContext::DOTDOTEQ() {
  return getToken(RustParser::DOTDOTEQ, 0);
}

RustParser::RangeExpressionContext::RangeExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::RangeExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRangeExpression(this);
}
void RustParser::RangeExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRangeExpression(this);
}

std::any
RustParser::RangeExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitRangeExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- MacroInvocationAsExpressionContext
//------------------------------------------------------------------

RustParser::MacroInvocationContext *
RustParser::MacroInvocationAsExpressionContext::macroInvocation() {
  return getRuleContext<RustParser::MacroInvocationContext>(0);
}

RustParser::MacroInvocationAsExpressionContext::
    MacroInvocationAsExpressionContext(ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::MacroInvocationAsExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroInvocationAsExpression(this);
}
void RustParser::MacroInvocationAsExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroInvocationAsExpression(this);
}

std::any RustParser::MacroInvocationAsExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroInvocationAsExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ReturnExpressionContext
//------------------------------------------------------------------

tree::TerminalNode *RustParser::ReturnExpressionContext::KW_RETURN() {
  return getToken(RustParser::KW_RETURN, 0);
}

RustParser::ExpressionContext *
RustParser::ReturnExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

RustParser::ReturnExpressionContext::ReturnExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ReturnExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReturnExpression(this);
}
void RustParser::ReturnExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReturnExpression(this);
}

std::any
RustParser::ReturnExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitReturnExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AwaitExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::AwaitExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::AwaitExpressionContext::DOT() {
  return getToken(RustParser::DOT, 0);
}

tree::TerminalNode *RustParser::AwaitExpressionContext::KW_AWAIT() {
  return getToken(RustParser::KW_AWAIT, 0);
}

RustParser::AwaitExpressionContext::AwaitExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::AwaitExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAwaitExpression(this);
}
void RustParser::AwaitExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAwaitExpression(this);
}

std::any
RustParser::AwaitExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAwaitExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ErrorPropagationExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::ErrorPropagationExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::ErrorPropagationExpressionContext::QUESTION() {
  return getToken(RustParser::QUESTION, 0);
}

RustParser::ErrorPropagationExpressionContext::
    ErrorPropagationExpressionContext(ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ErrorPropagationExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterErrorPropagationExpression(this);
}
void RustParser::ErrorPropagationExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitErrorPropagationExpression(this);
}

std::any RustParser::ErrorPropagationExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitErrorPropagationExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ContinueExpressionContext
//------------------------------------------------------------------

tree::TerminalNode *RustParser::ContinueExpressionContext::KW_CONTINUE() {
  return getToken(RustParser::KW_CONTINUE, 0);
}

tree::TerminalNode *RustParser::ContinueExpressionContext::LIFETIME_OR_LABEL() {
  return getToken(RustParser::LIFETIME_OR_LABEL, 0);
}

RustParser::ExpressionContext *
RustParser::ContinueExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

RustParser::ContinueExpressionContext::ContinueExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ContinueExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterContinueExpression(this);
}
void RustParser::ContinueExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitContinueExpression(this);
}

std::any
RustParser::ContinueExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitContinueExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AssignmentExpressionContext
//------------------------------------------------------------------

std::vector<RustParser::ExpressionContext *>
RustParser::AssignmentExpressionContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::AssignmentExpressionContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

tree::TerminalNode *RustParser::AssignmentExpressionContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::AssignmentExpressionContext::AssignmentExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::AssignmentExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignmentExpression(this);
}
void RustParser::AssignmentExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignmentExpression(this);
}

std::any RustParser::AssignmentExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAssignmentExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- MethodCallExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::MethodCallExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::MethodCallExpressionContext::DOT() {
  return getToken(RustParser::DOT, 0);
}

RustParser::PathExprSegmentContext *
RustParser::MethodCallExpressionContext::pathExprSegment() {
  return getRuleContext<RustParser::PathExprSegmentContext>(0);
}

tree::TerminalNode *RustParser::MethodCallExpressionContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::MethodCallExpressionContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::CallParamsContext *
RustParser::MethodCallExpressionContext::callParams() {
  return getRuleContext<RustParser::CallParamsContext>(0);
}

RustParser::MethodCallExpressionContext::MethodCallExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::MethodCallExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMethodCallExpression(this);
}
void RustParser::MethodCallExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMethodCallExpression(this);
}

std::any RustParser::MethodCallExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMethodCallExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LiteralExpression_Context
//------------------------------------------------------------------

RustParser::LiteralExpressionContext *
RustParser::LiteralExpression_Context::literalExpression() {
  return getRuleContext<RustParser::LiteralExpressionContext>(0);
}

RustParser::LiteralExpression_Context::LiteralExpression_Context(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::LiteralExpression_Context::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLiteralExpression_(this);
}
void RustParser::LiteralExpression_Context::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLiteralExpression_(this);
}

std::any
RustParser::LiteralExpression_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLiteralExpression_(this);
  else
    return visitor->visitChildren(this);
}
//----------------- StructExpression_Context
//------------------------------------------------------------------

RustParser::StructExpressionContext *
RustParser::StructExpression_Context::structExpression() {
  return getRuleContext<RustParser::StructExpressionContext>(0);
}

RustParser::StructExpression_Context::StructExpression_Context(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::StructExpression_Context::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructExpression_(this);
}
void RustParser::StructExpression_Context::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructExpression_(this);
}

std::any
RustParser::StructExpression_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructExpression_(this);
  else
    return visitor->visitChildren(this);
}
//----------------- TupleIndexingExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::TupleIndexingExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::TupleIndexingExpressionContext::DOT() {
  return getToken(RustParser::DOT, 0);
}

RustParser::TupleIndexContext *
RustParser::TupleIndexingExpressionContext::tupleIndex() {
  return getRuleContext<RustParser::TupleIndexContext>(0);
}

RustParser::TupleIndexingExpressionContext::TupleIndexingExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::TupleIndexingExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleIndexingExpression(this);
}
void RustParser::TupleIndexingExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleIndexingExpression(this);
}

std::any RustParser::TupleIndexingExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleIndexingExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- NegationExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::NegationExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::NegationExpressionContext::MINUS() {
  return getToken(RustParser::MINUS, 0);
}

tree::TerminalNode *RustParser::NegationExpressionContext::NOT() {
  return getToken(RustParser::NOT, 0);
}

RustParser::NegationExpressionContext::NegationExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::NegationExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNegationExpression(this);
}
void RustParser::NegationExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNegationExpression(this);
}

std::any
RustParser::NegationExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitNegationExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- CallExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *RustParser::CallExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::CallExpressionContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::CallExpressionContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::CallParamsContext *RustParser::CallExpressionContext::callParams() {
  return getRuleContext<RustParser::CallParamsContext>(0);
}

RustParser::CallExpressionContext::CallExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::CallExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCallExpression(this);
}
void RustParser::CallExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCallExpression(this);
}

std::any
RustParser::CallExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitCallExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LazyBooleanExpressionContext
//------------------------------------------------------------------

std::vector<RustParser::ExpressionContext *>
RustParser::LazyBooleanExpressionContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::LazyBooleanExpressionContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

tree::TerminalNode *RustParser::LazyBooleanExpressionContext::ANDAND() {
  return getToken(RustParser::ANDAND, 0);
}

tree::TerminalNode *RustParser::LazyBooleanExpressionContext::OROR() {
  return getToken(RustParser::OROR, 0);
}

RustParser::LazyBooleanExpressionContext::LazyBooleanExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::LazyBooleanExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLazyBooleanExpression(this);
}
void RustParser::LazyBooleanExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLazyBooleanExpression(this);
}

std::any RustParser::LazyBooleanExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLazyBooleanExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- DereferenceExpressionContext
//------------------------------------------------------------------

tree::TerminalNode *RustParser::DereferenceExpressionContext::STAR() {
  return getToken(RustParser::STAR, 0);
}

RustParser::ExpressionContext *
RustParser::DereferenceExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

RustParser::DereferenceExpressionContext::DereferenceExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::DereferenceExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDereferenceExpression(this);
}
void RustParser::DereferenceExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDereferenceExpression(this);
}

std::any RustParser::DereferenceExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitDereferenceExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ExpressionWithBlock_Context
//------------------------------------------------------------------

RustParser::ExpressionWithBlockContext *
RustParser::ExpressionWithBlock_Context::expressionWithBlock() {
  return getRuleContext<RustParser::ExpressionWithBlockContext>(0);
}

RustParser::ExpressionWithBlock_Context::ExpressionWithBlock_Context(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ExpressionWithBlock_Context::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressionWithBlock_(this);
}
void RustParser::ExpressionWithBlock_Context::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressionWithBlock_(this);
}

std::any RustParser::ExpressionWithBlock_Context::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitExpressionWithBlock_(this);
  else
    return visitor->visitChildren(this);
}
//----------------- GroupedExpressionContext
//------------------------------------------------------------------

tree::TerminalNode *RustParser::GroupedExpressionContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

RustParser::ExpressionContext *
RustParser::GroupedExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::GroupedExpressionContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::GroupedExpressionContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::GroupedExpressionContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

RustParser::GroupedExpressionContext::GroupedExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::GroupedExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGroupedExpression(this);
}
void RustParser::GroupedExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGroupedExpression(this);
}

std::any
RustParser::GroupedExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGroupedExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BreakExpressionContext
//------------------------------------------------------------------

tree::TerminalNode *RustParser::BreakExpressionContext::KW_BREAK() {
  return getToken(RustParser::KW_BREAK, 0);
}

tree::TerminalNode *RustParser::BreakExpressionContext::LIFETIME_OR_LABEL() {
  return getToken(RustParser::LIFETIME_OR_LABEL, 0);
}

RustParser::ExpressionContext *
RustParser::BreakExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

RustParser::BreakExpressionContext::BreakExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::BreakExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBreakExpression(this);
}
void RustParser::BreakExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBreakExpression(this);
}

std::any
RustParser::BreakExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitBreakExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ArithmeticOrLogicalExpressionContext
//------------------------------------------------------------------

std::vector<RustParser::ExpressionContext *>
RustParser::ArithmeticOrLogicalExpressionContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::ArithmeticOrLogicalExpressionContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

tree::TerminalNode *RustParser::ArithmeticOrLogicalExpressionContext::STAR() {
  return getToken(RustParser::STAR, 0);
}

tree::TerminalNode *RustParser::ArithmeticOrLogicalExpressionContext::SLASH() {
  return getToken(RustParser::SLASH, 0);
}

tree::TerminalNode *
RustParser::ArithmeticOrLogicalExpressionContext::PERCENT() {
  return getToken(RustParser::PERCENT, 0);
}

tree::TerminalNode *RustParser::ArithmeticOrLogicalExpressionContext::PLUS() {
  return getToken(RustParser::PLUS, 0);
}

tree::TerminalNode *RustParser::ArithmeticOrLogicalExpressionContext::MINUS() {
  return getToken(RustParser::MINUS, 0);
}

RustParser::ShlContext *
RustParser::ArithmeticOrLogicalExpressionContext::shl() {
  return getRuleContext<RustParser::ShlContext>(0);
}

RustParser::ShrContext *
RustParser::ArithmeticOrLogicalExpressionContext::shr() {
  return getRuleContext<RustParser::ShrContext>(0);
}

tree::TerminalNode *RustParser::ArithmeticOrLogicalExpressionContext::AND() {
  return getToken(RustParser::AND, 0);
}

tree::TerminalNode *RustParser::ArithmeticOrLogicalExpressionContext::CARET() {
  return getToken(RustParser::CARET, 0);
}

tree::TerminalNode *RustParser::ArithmeticOrLogicalExpressionContext::OR() {
  return getToken(RustParser::OR, 0);
}

RustParser::ArithmeticOrLogicalExpressionContext::
    ArithmeticOrLogicalExpressionContext(ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ArithmeticOrLogicalExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArithmeticOrLogicalExpression(this);
}
void RustParser::ArithmeticOrLogicalExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArithmeticOrLogicalExpression(this);
}

std::any RustParser::ArithmeticOrLogicalExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitArithmeticOrLogicalExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- FieldExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::FieldExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::FieldExpressionContext::DOT() {
  return getToken(RustParser::DOT, 0);
}

RustParser::IdentifierContext *
RustParser::FieldExpressionContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

RustParser::FieldExpressionContext::FieldExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::FieldExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFieldExpression(this);
}
void RustParser::FieldExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFieldExpression(this);
}

std::any
RustParser::FieldExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFieldExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- EnumerationVariantExpression_Context
//------------------------------------------------------------------

RustParser::EnumerationVariantExpressionContext *RustParser::
    EnumerationVariantExpression_Context::enumerationVariantExpression() {
  return getRuleContext<RustParser::EnumerationVariantExpressionContext>(0);
}

RustParser::EnumerationVariantExpression_Context::
    EnumerationVariantExpression_Context(ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::EnumerationVariantExpression_Context::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumerationVariantExpression_(this);
}
void RustParser::EnumerationVariantExpression_Context::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumerationVariantExpression_(this);
}

std::any RustParser::EnumerationVariantExpression_Context::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumerationVariantExpression_(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ComparisonExpressionContext
//------------------------------------------------------------------

std::vector<RustParser::ExpressionContext *>
RustParser::ComparisonExpressionContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::ComparisonExpressionContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

RustParser::ComparisonOperatorContext *
RustParser::ComparisonExpressionContext::comparisonOperator() {
  return getRuleContext<RustParser::ComparisonOperatorContext>(0);
}

RustParser::ComparisonExpressionContext::ComparisonExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ComparisonExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComparisonExpression(this);
}
void RustParser::ComparisonExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComparisonExpression(this);
}

std::any RustParser::ComparisonExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitComparisonExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AttributedExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::AttributedExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::AttributedExpressionContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::AttributedExpressionContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::AttributedExpressionContext::AttributedExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::AttributedExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAttributedExpression(this);
}
void RustParser::AttributedExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAttributedExpression(this);
}

std::any RustParser::AttributedExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAttributedExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BorrowExpressionContext
//------------------------------------------------------------------

RustParser::ExpressionContext *
RustParser::BorrowExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::BorrowExpressionContext::AND() {
  return getToken(RustParser::AND, 0);
}

tree::TerminalNode *RustParser::BorrowExpressionContext::ANDAND() {
  return getToken(RustParser::ANDAND, 0);
}

tree::TerminalNode *RustParser::BorrowExpressionContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

RustParser::BorrowExpressionContext::BorrowExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::BorrowExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBorrowExpression(this);
}
void RustParser::BorrowExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBorrowExpression(this);
}

std::any
RustParser::BorrowExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitBorrowExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- CompoundAssignmentExpressionContext
//------------------------------------------------------------------

std::vector<RustParser::ExpressionContext *>
RustParser::CompoundAssignmentExpressionContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::CompoundAssignmentExpressionContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

RustParser::CompoundAssignOperatorContext *
RustParser::CompoundAssignmentExpressionContext::compoundAssignOperator() {
  return getRuleContext<RustParser::CompoundAssignOperatorContext>(0);
}

RustParser::CompoundAssignmentExpressionContext::
    CompoundAssignmentExpressionContext(ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::CompoundAssignmentExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompoundAssignmentExpression(this);
}
void RustParser::CompoundAssignmentExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompoundAssignmentExpression(this);
}

std::any RustParser::CompoundAssignmentExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitCompoundAssignmentExpression(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ClosureExpression_Context
//------------------------------------------------------------------

RustParser::ClosureExpressionContext *
RustParser::ClosureExpression_Context::closureExpression() {
  return getRuleContext<RustParser::ClosureExpressionContext>(0);
}

RustParser::ClosureExpression_Context::ClosureExpression_Context(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ClosureExpression_Context::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClosureExpression_(this);
}
void RustParser::ClosureExpression_Context::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClosureExpression_(this);
}

std::any
RustParser::ClosureExpression_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitClosureExpression_(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ArrayExpressionContext
//------------------------------------------------------------------

tree::TerminalNode *RustParser::ArrayExpressionContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::ArrayExpressionContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::ArrayExpressionContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::ArrayExpressionContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

RustParser::ArrayElementsContext *
RustParser::ArrayExpressionContext::arrayElements() {
  return getRuleContext<RustParser::ArrayElementsContext>(0);
}

RustParser::ArrayExpressionContext::ArrayExpressionContext(
    ExpressionContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ArrayExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArrayExpression(this);
}
void RustParser::ArrayExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArrayExpression(this);
}

std::any
RustParser::ArrayExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitArrayExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ExpressionContext *RustParser::expression() {
  return expression(0);
}

RustParser::ExpressionContext *RustParser::expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  RustParser::ExpressionContext *_localctx =
      _tracker.createInstance<ExpressionContext>(_ctx, parentState);
  RustParser::ExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by
                         // generated code.
  size_t startState = 154;
  enterRecursionRule(_localctx, 154, RustParser::RuleExpression, precedence);

  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1366);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 174, _ctx)) {
    case 1: {
      _localctx =
          _tracker.createInstance<AttributedExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(1288);
      _errHandler->sync(this);
      alt = 1;
      do {
        switch (alt) {
        case 1: {
          setState(1287);
          outerAttribute();
          break;
        }

        default:
          throw NoViableAltException(this);
        }
        setState(1290);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 161, _ctx);
      } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
      setState(1292);
      expression(40);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<LiteralExpression_Context>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1294);
      literalExpression();
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<PathExpression_Context>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1295);
      pathExpression();
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<BorrowExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1296);
      _la = _input->LA(1);
      if (!(_la == RustParser::AND

            || _la == RustParser::ANDAND)) {
        _errHandler->recoverInline(this);
      } else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(1298);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::KW_MUT) {
        setState(1297);
        match(RustParser::KW_MUT);
      }
      setState(1300);
      expression(30);
      break;
    }

    case 5: {
      _localctx =
          _tracker.createInstance<DereferenceExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1301);
      match(RustParser::STAR);
      setState(1302);
      expression(29);
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<NegationExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1303);
      _la = _input->LA(1);
      if (!(_la == RustParser::MINUS

            || _la == RustParser::NOT)) {
        _errHandler->recoverInline(this);
      } else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(1304);
      expression(28);
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<RangeExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1305);
      match(RustParser::DOTDOT);
      setState(1307);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 163, _ctx)) {
      case 1: {
        setState(1306);
        expression(0);
        break;
      }

      default:
        break;
      }
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<RangeExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1309);
      match(RustParser::DOTDOTEQ);
      setState(1310);
      expression(15);
      break;
    }

    case 9: {
      _localctx = _tracker.createInstance<ContinueExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1311);
      match(RustParser::KW_CONTINUE);
      setState(1313);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 164, _ctx)) {
      case 1: {
        setState(1312);
        match(RustParser::LIFETIME_OR_LABEL);
        break;
      }

      default:
        break;
      }
      setState(1316);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 165, _ctx)) {
      case 1: {
        setState(1315);
        expression(0);
        break;
      }

      default:
        break;
      }
      break;
    }

    case 10: {
      _localctx = _tracker.createInstance<BreakExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1318);
      match(RustParser::KW_BREAK);
      setState(1320);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 166, _ctx)) {
      case 1: {
        setState(1319);
        match(RustParser::LIFETIME_OR_LABEL);
        break;
      }

      default:
        break;
      }
      setState(1323);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 167, _ctx)) {
      case 1: {
        setState(1322);
        expression(0);
        break;
      }

      default:
        break;
      }
      break;
    }

    case 11: {
      _localctx = _tracker.createInstance<ReturnExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1325);
      match(RustParser::KW_RETURN);
      setState(1327);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 168, _ctx)) {
      case 1: {
        setState(1326);
        expression(0);
        break;
      }

      default:
        break;
      }
      break;
    }

    case 12: {
      _localctx = _tracker.createInstance<GroupedExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1329);
      match(RustParser::LPAREN);
      setState(1333);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 169, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(1330);
          innerAttribute();
        }
        setState(1335);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 169, _ctx);
      }
      setState(1336);
      expression(0);
      setState(1337);
      match(RustParser::RPAREN);
      break;
    }

    case 13: {
      _localctx = _tracker.createInstance<ArrayExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1339);
      match(RustParser::LSQUAREBRACKET);
      setState(1343);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 170, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(1340);
          innerAttribute();
        }
        setState(1345);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 170, _ctx);
      }
      setState(1347);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 522417665550785076) != 0) ||
          ((((_la - 70) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 70)) & 1523430809782507647) != 0)) {
        setState(1346);
        arrayElements();
      }
      setState(1349);
      match(RustParser::RSQUAREBRACKET);
      break;
    }

    case 14: {
      _localctx = _tracker.createInstance<TupleExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1350);
      match(RustParser::LPAREN);
      setState(1354);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 172, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(1351);
          innerAttribute();
        }
        setState(1356);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 172, _ctx);
      }
      setState(1358);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 522417665550785076) != 0) ||
          ((((_la - 70) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 70)) & 1523430809782507647) != 0)) {
        setState(1357);
        tupleElements();
      }
      setState(1360);
      match(RustParser::RPAREN);
      break;
    }

    case 15: {
      _localctx = _tracker.createInstance<StructExpression_Context>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1361);
      structExpression();
      break;
    }

    case 16: {
      _localctx = _tracker.createInstance<EnumerationVariantExpression_Context>(
          _localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1362);
      enumerationVariantExpression();
      break;
    }

    case 17: {
      _localctx = _tracker.createInstance<ClosureExpression_Context>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1363);
      closureExpression();
      break;
    }

    case 18: {
      _localctx =
          _tracker.createInstance<ExpressionWithBlock_Context>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1364);
      expressionWithBlock();
      break;
    }

    case 19: {
      _localctx = _tracker.createInstance<MacroInvocationAsExpressionContext>(
          _localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(1365);
      macroInvocation();
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(1451);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     180, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(1449);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 179, _ctx)) {
        case 1: {
          auto newContext =
              _tracker.createInstance<ArithmeticOrLogicalExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1368);

          if (!(precpred(_ctx, 26)))
            throw FailedPredicateException(this, "precpred(_ctx, 26)");
          setState(1369);
          _la = _input->LA(1);
          if (!(((((_la - 85) & ~0x3fULL) == 0) &&
                 ((1ULL << (_la - 85)) & 7) != 0))) {
            _errHandler->recoverInline(this);
          } else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(1370);
          expression(27);
          break;
        }

        case 2: {
          auto newContext =
              _tracker.createInstance<ArithmeticOrLogicalExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1371);

          if (!(precpred(_ctx, 25)))
            throw FailedPredicateException(this, "precpred(_ctx, 25)");
          setState(1372);
          _la = _input->LA(1);
          if (!(_la == RustParser::PLUS

                || _la == RustParser::MINUS)) {
            _errHandler->recoverInline(this);
          } else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(1373);
          expression(26);
          break;
        }

        case 3: {
          auto newContext =
              _tracker.createInstance<ArithmeticOrLogicalExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1374);

          if (!(precpred(_ctx, 24)))
            throw FailedPredicateException(this, "precpred(_ctx, 24)");
          setState(1377);
          _errHandler->sync(this);
          switch (_input->LA(1)) {
          case RustParser::LT: {
            setState(1375);
            shl();
            break;
          }

          case RustParser::GT: {
            setState(1376);
            shr();
            break;
          }

          default:
            throw NoViableAltException(this);
          }
          setState(1379);
          expression(25);
          break;
        }

        case 4: {
          auto newContext =
              _tracker.createInstance<ArithmeticOrLogicalExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1381);

          if (!(precpred(_ctx, 23)))
            throw FailedPredicateException(this, "precpred(_ctx, 23)");
          setState(1382);
          match(RustParser::AND);
          setState(1383);
          expression(24);
          break;
        }

        case 5: {
          auto newContext =
              _tracker.createInstance<ArithmeticOrLogicalExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1384);

          if (!(precpred(_ctx, 22)))
            throw FailedPredicateException(this, "precpred(_ctx, 22)");
          setState(1385);
          match(RustParser::CARET);
          setState(1386);
          expression(23);
          break;
        }

        case 6: {
          auto newContext =
              _tracker.createInstance<ArithmeticOrLogicalExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1387);

          if (!(precpred(_ctx, 21)))
            throw FailedPredicateException(this, "precpred(_ctx, 21)");
          setState(1388);
          match(RustParser::OR);
          setState(1389);
          expression(22);
          break;
        }

        case 7: {
          auto newContext =
              _tracker.createInstance<ComparisonExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1390);

          if (!(precpred(_ctx, 20)))
            throw FailedPredicateException(this, "precpred(_ctx, 20)");
          setState(1391);
          comparisonOperator();
          setState(1392);
          expression(21);
          break;
        }

        case 8: {
          auto newContext =
              _tracker.createInstance<LazyBooleanExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1394);

          if (!(precpred(_ctx, 19)))
            throw FailedPredicateException(this, "precpred(_ctx, 19)");
          setState(1395);
          match(RustParser::ANDAND);
          setState(1396);
          expression(20);
          break;
        }

        case 9: {
          auto newContext =
              _tracker.createInstance<LazyBooleanExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1397);

          if (!(precpred(_ctx, 18)))
            throw FailedPredicateException(this, "precpred(_ctx, 18)");
          setState(1398);
          match(RustParser::OROR);
          setState(1399);
          expression(19);
          break;
        }

        case 10: {
          auto newContext = _tracker.createInstance<RangeExpressionContext>(
              _tracker.createInstance<ExpressionContext>(parentContext,
                                                         parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1400);

          if (!(precpred(_ctx, 14)))
            throw FailedPredicateException(this, "precpred(_ctx, 14)");
          setState(1401);
          match(RustParser::DOTDOTEQ);
          setState(1402);
          expression(15);
          break;
        }

        case 11: {
          auto newContext =
              _tracker.createInstance<AssignmentExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1403);

          if (!(precpred(_ctx, 13)))
            throw FailedPredicateException(this, "precpred(_ctx, 13)");
          setState(1404);
          match(RustParser::EQ);
          setState(1405);
          expression(14);
          break;
        }

        case 12: {
          auto newContext =
              _tracker.createInstance<CompoundAssignmentExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1406);

          if (!(precpred(_ctx, 12)))
            throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(1407);
          compoundAssignOperator();
          setState(1408);
          expression(13);
          break;
        }

        case 13: {
          auto newContext =
              _tracker.createInstance<MethodCallExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1410);

          if (!(precpred(_ctx, 37)))
            throw FailedPredicateException(this, "precpred(_ctx, 37)");
          setState(1411);
          match(RustParser::DOT);
          setState(1412);
          pathExprSegment();
          setState(1413);
          match(RustParser::LPAREN);
          setState(1415);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if ((((_la & ~0x3fULL) == 0) &&
               ((1ULL << _la) & 522417665550785076) != 0) ||
              ((((_la - 70) & ~0x3fULL) == 0) &&
               ((1ULL << (_la - 70)) & 1523430809782507647) != 0)) {
            setState(1414);
            callParams();
          }
          setState(1417);
          match(RustParser::RPAREN);
          break;
        }

        case 14: {
          auto newContext = _tracker.createInstance<FieldExpressionContext>(
              _tracker.createInstance<ExpressionContext>(parentContext,
                                                         parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1419);

          if (!(precpred(_ctx, 36)))
            throw FailedPredicateException(this, "precpred(_ctx, 36)");
          setState(1420);
          match(RustParser::DOT);
          setState(1421);
          identifier();
          break;
        }

        case 15: {
          auto newContext =
              _tracker.createInstance<TupleIndexingExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1422);

          if (!(precpred(_ctx, 35)))
            throw FailedPredicateException(this, "precpred(_ctx, 35)");
          setState(1423);
          match(RustParser::DOT);
          setState(1424);
          tupleIndex();
          break;
        }

        case 16: {
          auto newContext = _tracker.createInstance<AwaitExpressionContext>(
              _tracker.createInstance<ExpressionContext>(parentContext,
                                                         parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1425);

          if (!(precpred(_ctx, 34)))
            throw FailedPredicateException(this, "precpred(_ctx, 34)");
          setState(1426);
          match(RustParser::DOT);
          setState(1427);
          match(RustParser::KW_AWAIT);
          break;
        }

        case 17: {
          auto newContext = _tracker.createInstance<CallExpressionContext>(
              _tracker.createInstance<ExpressionContext>(parentContext,
                                                         parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1428);

          if (!(precpred(_ctx, 33)))
            throw FailedPredicateException(this, "precpred(_ctx, 33)");
          setState(1429);
          match(RustParser::LPAREN);
          setState(1431);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if ((((_la & ~0x3fULL) == 0) &&
               ((1ULL << _la) & 522417665550785076) != 0) ||
              ((((_la - 70) & ~0x3fULL) == 0) &&
               ((1ULL << (_la - 70)) & 1523430809782507647) != 0)) {
            setState(1430);
            callParams();
          }
          setState(1433);
          match(RustParser::RPAREN);
          break;
        }

        case 18: {
          auto newContext = _tracker.createInstance<IndexExpressionContext>(
              _tracker.createInstance<ExpressionContext>(parentContext,
                                                         parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1434);

          if (!(precpred(_ctx, 32)))
            throw FailedPredicateException(this, "precpred(_ctx, 32)");
          setState(1435);
          match(RustParser::LSQUAREBRACKET);
          setState(1436);
          expression(0);
          setState(1437);
          match(RustParser::RSQUAREBRACKET);
          break;
        }

        case 19: {
          auto newContext =
              _tracker.createInstance<ErrorPropagationExpressionContext>(
                  _tracker.createInstance<ExpressionContext>(parentContext,
                                                             parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1439);

          if (!(precpred(_ctx, 31)))
            throw FailedPredicateException(this, "precpred(_ctx, 31)");
          setState(1440);
          match(RustParser::QUESTION);
          break;
        }

        case 20: {
          auto newContext = _tracker.createInstance<TypeCastExpressionContext>(
              _tracker.createInstance<ExpressionContext>(parentContext,
                                                         parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1441);

          if (!(precpred(_ctx, 27)))
            throw FailedPredicateException(this, "precpred(_ctx, 27)");
          setState(1442);
          match(RustParser::KW_AS);
          setState(1443);
          typeNoBounds();
          break;
        }

        case 21: {
          auto newContext = _tracker.createInstance<RangeExpressionContext>(
              _tracker.createInstance<ExpressionContext>(parentContext,
                                                         parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(1444);

          if (!(precpred(_ctx, 17)))
            throw FailedPredicateException(this, "precpred(_ctx, 17)");
          setState(1445);
          match(RustParser::DOTDOT);
          setState(1447);
          _errHandler->sync(this);

          switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
              _input, 178, _ctx)) {
          case 1: {
            setState(1446);
            expression(0);
            break;
          }

          default:
            break;
          }
          break;
        }

        default:
          break;
        }
      }
      setState(1453);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 180, _ctx);
    }
  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ComparisonOperatorContext
//------------------------------------------------------------------

RustParser::ComparisonOperatorContext::ComparisonOperatorContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ComparisonOperatorContext::EQEQ() {
  return getToken(RustParser::EQEQ, 0);
}

tree::TerminalNode *RustParser::ComparisonOperatorContext::NE() {
  return getToken(RustParser::NE, 0);
}

tree::TerminalNode *RustParser::ComparisonOperatorContext::GT() {
  return getToken(RustParser::GT, 0);
}

tree::TerminalNode *RustParser::ComparisonOperatorContext::LT() {
  return getToken(RustParser::LT, 0);
}

tree::TerminalNode *RustParser::ComparisonOperatorContext::GE() {
  return getToken(RustParser::GE, 0);
}

tree::TerminalNode *RustParser::ComparisonOperatorContext::LE() {
  return getToken(RustParser::LE, 0);
}

size_t RustParser::ComparisonOperatorContext::getRuleIndex() const {
  return RustParser::RuleComparisonOperator;
}

void RustParser::ComparisonOperatorContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComparisonOperator(this);
}

void RustParser::ComparisonOperatorContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComparisonOperator(this);
}

std::any
RustParser::ComparisonOperatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitComparisonOperator(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ComparisonOperatorContext *RustParser::comparisonOperator() {
  ComparisonOperatorContext *_localctx =
      _tracker.createInstance<ComparisonOperatorContext>(_ctx, getState());
  enterRule(_localctx, 156, RustParser::RuleComparisonOperator);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1454);
    _la = _input->LA(1);
    if (!(((((_la - 105) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 105)) & 63) != 0))) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CompoundAssignOperatorContext
//------------------------------------------------------------------

RustParser::CompoundAssignOperatorContext::CompoundAssignOperatorContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::PLUSEQ() {
  return getToken(RustParser::PLUSEQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::MINUSEQ() {
  return getToken(RustParser::MINUSEQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::STAREQ() {
  return getToken(RustParser::STAREQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::SLASHEQ() {
  return getToken(RustParser::SLASHEQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::PERCENTEQ() {
  return getToken(RustParser::PERCENTEQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::ANDEQ() {
  return getToken(RustParser::ANDEQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::OREQ() {
  return getToken(RustParser::OREQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::CARETEQ() {
  return getToken(RustParser::CARETEQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::SHLEQ() {
  return getToken(RustParser::SHLEQ, 0);
}

tree::TerminalNode *RustParser::CompoundAssignOperatorContext::SHREQ() {
  return getToken(RustParser::SHREQ, 0);
}

size_t RustParser::CompoundAssignOperatorContext::getRuleIndex() const {
  return RustParser::RuleCompoundAssignOperator;
}

void RustParser::CompoundAssignOperatorContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompoundAssignOperator(this);
}

void RustParser::CompoundAssignOperatorContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompoundAssignOperator(this);
}

std::any RustParser::CompoundAssignOperatorContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitCompoundAssignOperator(this);
  else
    return visitor->visitChildren(this);
}

RustParser::CompoundAssignOperatorContext *
RustParser::compoundAssignOperator() {
  CompoundAssignOperatorContext *_localctx =
      _tracker.createInstance<CompoundAssignOperatorContext>(_ctx, getState());
  enterRule(_localctx, 158, RustParser::RuleCompoundAssignOperator);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1456);
    _la = _input->LA(1);
    if (!(((((_la - 94) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 94)) & 1023) != 0))) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionWithBlockContext
//------------------------------------------------------------------

RustParser::ExpressionWithBlockContext::ExpressionWithBlockContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::ExpressionWithBlockContext *
RustParser::ExpressionWithBlockContext::expressionWithBlock() {
  return getRuleContext<RustParser::ExpressionWithBlockContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::ExpressionWithBlockContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::ExpressionWithBlockContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::BlockExpressionContext *
RustParser::ExpressionWithBlockContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

RustParser::AsyncBlockExpressionContext *
RustParser::ExpressionWithBlockContext::asyncBlockExpression() {
  return getRuleContext<RustParser::AsyncBlockExpressionContext>(0);
}

RustParser::UnsafeBlockExpressionContext *
RustParser::ExpressionWithBlockContext::unsafeBlockExpression() {
  return getRuleContext<RustParser::UnsafeBlockExpressionContext>(0);
}

RustParser::LoopExpressionContext *
RustParser::ExpressionWithBlockContext::loopExpression() {
  return getRuleContext<RustParser::LoopExpressionContext>(0);
}

RustParser::IfExpressionContext *
RustParser::ExpressionWithBlockContext::ifExpression() {
  return getRuleContext<RustParser::IfExpressionContext>(0);
}

RustParser::IfLetExpressionContext *
RustParser::ExpressionWithBlockContext::ifLetExpression() {
  return getRuleContext<RustParser::IfLetExpressionContext>(0);
}

RustParser::MatchExpressionContext *
RustParser::ExpressionWithBlockContext::matchExpression() {
  return getRuleContext<RustParser::MatchExpressionContext>(0);
}

size_t RustParser::ExpressionWithBlockContext::getRuleIndex() const {
  return RustParser::RuleExpressionWithBlock;
}

void RustParser::ExpressionWithBlockContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressionWithBlock(this);
}

void RustParser::ExpressionWithBlockContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressionWithBlock(this);
}

std::any RustParser::ExpressionWithBlockContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitExpressionWithBlock(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ExpressionWithBlockContext *RustParser::expressionWithBlock() {
  ExpressionWithBlockContext *_localctx =
      _tracker.createInstance<ExpressionWithBlockContext>(_ctx, getState());
  enterRule(_localctx, 160, RustParser::RuleExpressionWithBlock);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(1472);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 182, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1459);
      _errHandler->sync(this);
      alt = 1;
      do {
        switch (alt) {
        case 1: {
          setState(1458);
          outerAttribute();
          break;
        }

        default:
          throw NoViableAltException(this);
        }
        setState(1461);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 181, _ctx);
      } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
      setState(1463);
      expressionWithBlock();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1465);
      blockExpression();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(1466);
      asyncBlockExpression();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(1467);
      unsafeBlockExpression();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(1468);
      loopExpression();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(1469);
      ifExpression();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(1470);
      ifLetExpression();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(1471);
      matchExpression();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LiteralExpressionContext
//------------------------------------------------------------------

RustParser::LiteralExpressionContext::LiteralExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::LiteralExpressionContext::CHAR_LITERAL() {
  return getToken(RustParser::CHAR_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralExpressionContext::STRING_LITERAL() {
  return getToken(RustParser::STRING_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralExpressionContext::RAW_STRING_LITERAL() {
  return getToken(RustParser::RAW_STRING_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralExpressionContext::BYTE_LITERAL() {
  return getToken(RustParser::BYTE_LITERAL, 0);
}

tree::TerminalNode *
RustParser::LiteralExpressionContext::BYTE_STRING_LITERAL() {
  return getToken(RustParser::BYTE_STRING_LITERAL, 0);
}

tree::TerminalNode *
RustParser::LiteralExpressionContext::RAW_BYTE_STRING_LITERAL() {
  return getToken(RustParser::RAW_BYTE_STRING_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralExpressionContext::INTEGER_LITERAL() {
  return getToken(RustParser::INTEGER_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralExpressionContext::FLOAT_LITERAL() {
  return getToken(RustParser::FLOAT_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralExpressionContext::KW_TRUE() {
  return getToken(RustParser::KW_TRUE, 0);
}

tree::TerminalNode *RustParser::LiteralExpressionContext::KW_FALSE() {
  return getToken(RustParser::KW_FALSE, 0);
}

size_t RustParser::LiteralExpressionContext::getRuleIndex() const {
  return RustParser::RuleLiteralExpression;
}

void RustParser::LiteralExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLiteralExpression(this);
}

void RustParser::LiteralExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLiteralExpression(this);
}

std::any
RustParser::LiteralExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLiteralExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LiteralExpressionContext *RustParser::literalExpression() {
  LiteralExpressionContext *_localctx =
      _tracker.createInstance<LiteralExpressionContext>(_ctx, getState());
  enterRule(_localctx, 162, RustParser::RuleLiteralExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1474);
    _la = _input->LA(1);
    if (!(_la == RustParser::KW_FALSE

          || _la == RustParser::KW_TRUE ||
          ((((_la - 70) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 70)) & 2175) != 0))) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PathExpressionContext
//------------------------------------------------------------------

RustParser::PathExpressionContext::PathExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::PathExpressionContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

RustParser::QualifiedPathInExpressionContext *
RustParser::PathExpressionContext::qualifiedPathInExpression() {
  return getRuleContext<RustParser::QualifiedPathInExpressionContext>(0);
}

size_t RustParser::PathExpressionContext::getRuleIndex() const {
  return RustParser::RulePathExpression;
}

void RustParser::PathExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPathExpression(this);
}

void RustParser::PathExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPathExpression(this);
}

std::any
RustParser::PathExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPathExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PathExpressionContext *RustParser::pathExpression() {
  PathExpressionContext *_localctx =
      _tracker.createInstance<PathExpressionContext>(_ctx, getState());
  enterRule(_localctx, 164, RustParser::RulePathExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1478);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_CRATE:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SELFTYPE:
    case RustParser::KW_SUPER:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::PATHSEP: {
      enterOuterAlt(_localctx, 1);
      setState(1476);
      pathInExpression();
      break;
    }

    case RustParser::LT: {
      enterOuterAlt(_localctx, 2);
      setState(1477);
      qualifiedPathInExpression();
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockExpressionContext
//------------------------------------------------------------------

RustParser::BlockExpressionContext::BlockExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::BlockExpressionContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::BlockExpressionContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::BlockExpressionContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::BlockExpressionContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

RustParser::StatementsContext *
RustParser::BlockExpressionContext::statements() {
  return getRuleContext<RustParser::StatementsContext>(0);
}

size_t RustParser::BlockExpressionContext::getRuleIndex() const {
  return RustParser::RuleBlockExpression;
}

void RustParser::BlockExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBlockExpression(this);
}

void RustParser::BlockExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBlockExpression(this);
}

std::any
RustParser::BlockExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitBlockExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::BlockExpressionContext *RustParser::blockExpression() {
  BlockExpressionContext *_localctx =
      _tracker.createInstance<BlockExpressionContext>(_ctx, getState());
  enterRule(_localctx, 166, RustParser::RuleBlockExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1480);
    match(RustParser::LCURLYBRACE);
    setState(1484);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     184, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1481);
        innerAttribute();
      }
      setState(1486);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 184, _ctx);
    }
    setState(1488);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 526921276656172988) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1523712284759218303) != 0)) {
      setState(1487);
      statements();
    }
    setState(1490);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementsContext
//------------------------------------------------------------------

RustParser::StatementsContext::StatementsContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::StatementContext *>
RustParser::StatementsContext::statement() {
  return getRuleContexts<RustParser::StatementContext>();
}

RustParser::StatementContext *
RustParser::StatementsContext::statement(size_t i) {
  return getRuleContext<RustParser::StatementContext>(i);
}

RustParser::ExpressionContext *RustParser::StatementsContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

size_t RustParser::StatementsContext::getRuleIndex() const {
  return RustParser::RuleStatements;
}

void RustParser::StatementsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatements(this);
}

void RustParser::StatementsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatements(this);
}

std::any
RustParser::StatementsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStatements(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StatementsContext *RustParser::statements() {
  StatementsContext *_localctx =
      _tracker.createInstance<StatementsContext>(_ctx, getState());
  enterRule(_localctx, 168, RustParser::RuleStatements);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(1501);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 188, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1493);
      _errHandler->sync(this);
      alt = 1;
      do {
        switch (alt) {
        case 1: {
          setState(1492);
          statement();
          break;
        }

        default:
          throw NoViableAltException(this);
        }
        setState(1495);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 186, _ctx);
      } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
      setState(1498);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 522417665550785076) != 0) ||
          ((((_la - 70) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 70)) & 1523430809782507647) != 0)) {
        setState(1497);
        expression(0);
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1500);
      expression(0);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AsyncBlockExpressionContext
//------------------------------------------------------------------

RustParser::AsyncBlockExpressionContext::AsyncBlockExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::AsyncBlockExpressionContext::KW_ASYNC() {
  return getToken(RustParser::KW_ASYNC, 0);
}

RustParser::BlockExpressionContext *
RustParser::AsyncBlockExpressionContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

tree::TerminalNode *RustParser::AsyncBlockExpressionContext::KW_MOVE() {
  return getToken(RustParser::KW_MOVE, 0);
}

size_t RustParser::AsyncBlockExpressionContext::getRuleIndex() const {
  return RustParser::RuleAsyncBlockExpression;
}

void RustParser::AsyncBlockExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAsyncBlockExpression(this);
}

void RustParser::AsyncBlockExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAsyncBlockExpression(this);
}

std::any RustParser::AsyncBlockExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitAsyncBlockExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::AsyncBlockExpressionContext *RustParser::asyncBlockExpression() {
  AsyncBlockExpressionContext *_localctx =
      _tracker.createInstance<AsyncBlockExpressionContext>(_ctx, getState());
  enterRule(_localctx, 170, RustParser::RuleAsyncBlockExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1503);
    match(RustParser::KW_ASYNC);
    setState(1505);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_MOVE) {
      setState(1504);
      match(RustParser::KW_MOVE);
    }
    setState(1507);
    blockExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnsafeBlockExpressionContext
//------------------------------------------------------------------

RustParser::UnsafeBlockExpressionContext::UnsafeBlockExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::UnsafeBlockExpressionContext::KW_UNSAFE() {
  return getToken(RustParser::KW_UNSAFE, 0);
}

RustParser::BlockExpressionContext *
RustParser::UnsafeBlockExpressionContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

size_t RustParser::UnsafeBlockExpressionContext::getRuleIndex() const {
  return RustParser::RuleUnsafeBlockExpression;
}

void RustParser::UnsafeBlockExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnsafeBlockExpression(this);
}

void RustParser::UnsafeBlockExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnsafeBlockExpression(this);
}

std::any RustParser::UnsafeBlockExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitUnsafeBlockExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::UnsafeBlockExpressionContext *RustParser::unsafeBlockExpression() {
  UnsafeBlockExpressionContext *_localctx =
      _tracker.createInstance<UnsafeBlockExpressionContext>(_ctx, getState());
  enterRule(_localctx, 172, RustParser::RuleUnsafeBlockExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1509);
    match(RustParser::KW_UNSAFE);
    setState(1510);
    blockExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArrayElementsContext
//------------------------------------------------------------------

RustParser::ArrayElementsContext::ArrayElementsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::ExpressionContext *>
RustParser::ArrayElementsContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::ArrayElementsContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::ArrayElementsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::ArrayElementsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

tree::TerminalNode *RustParser::ArrayElementsContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

size_t RustParser::ArrayElementsContext::getRuleIndex() const {
  return RustParser::RuleArrayElements;
}

void RustParser::ArrayElementsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArrayElements(this);
}

void RustParser::ArrayElementsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArrayElements(this);
}

std::any
RustParser::ArrayElementsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitArrayElements(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ArrayElementsContext *RustParser::arrayElements() {
  ArrayElementsContext *_localctx =
      _tracker.createInstance<ArrayElementsContext>(_ctx, getState());
  enterRule(_localctx, 174, RustParser::RuleArrayElements);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(1527);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 192, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1512);
      expression(0);
      setState(1517);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 190, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(1513);
          match(RustParser::COMMA);
          setState(1514);
          expression(0);
        }
        setState(1519);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 190, _ctx);
      }
      setState(1521);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(1520);
        match(RustParser::COMMA);
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1523);
      expression(0);
      setState(1524);
      match(RustParser::SEMI);
      setState(1525);
      expression(0);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TupleElementsContext
//------------------------------------------------------------------

RustParser::TupleElementsContext::TupleElementsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::ExpressionContext *>
RustParser::TupleElementsContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::TupleElementsContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::TupleElementsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::TupleElementsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::TupleElementsContext::getRuleIndex() const {
  return RustParser::RuleTupleElements;
}

void RustParser::TupleElementsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleElements(this);
}

void RustParser::TupleElementsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleElements(this);
}

std::any
RustParser::TupleElementsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleElements(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TupleElementsContext *RustParser::tupleElements() {
  TupleElementsContext *_localctx =
      _tracker.createInstance<TupleElementsContext>(_ctx, getState());
  enterRule(_localctx, 176, RustParser::RuleTupleElements);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1532);
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
      case 1: {
        setState(1529);
        expression(0);
        setState(1530);
        match(RustParser::COMMA);
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      setState(1534);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 193, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
    setState(1537);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 522417665550785076) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1523430809782507647) != 0)) {
      setState(1536);
      expression(0);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TupleIndexContext
//------------------------------------------------------------------

RustParser::TupleIndexContext::TupleIndexContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::TupleIndexContext::INTEGER_LITERAL() {
  return getToken(RustParser::INTEGER_LITERAL, 0);
}

size_t RustParser::TupleIndexContext::getRuleIndex() const {
  return RustParser::RuleTupleIndex;
}

void RustParser::TupleIndexContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleIndex(this);
}

void RustParser::TupleIndexContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleIndex(this);
}

std::any
RustParser::TupleIndexContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleIndex(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TupleIndexContext *RustParser::tupleIndex() {
  TupleIndexContext *_localctx =
      _tracker.createInstance<TupleIndexContext>(_ctx, getState());
  enterRule(_localctx, 178, RustParser::RuleTupleIndex);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1539);
    match(RustParser::INTEGER_LITERAL);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructExpressionContext
//------------------------------------------------------------------

RustParser::StructExpressionContext::StructExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::StructExprStructContext *
RustParser::StructExpressionContext::structExprStruct() {
  return getRuleContext<RustParser::StructExprStructContext>(0);
}

RustParser::StructExprTupleContext *
RustParser::StructExpressionContext::structExprTuple() {
  return getRuleContext<RustParser::StructExprTupleContext>(0);
}

RustParser::StructExprUnitContext *
RustParser::StructExpressionContext::structExprUnit() {
  return getRuleContext<RustParser::StructExprUnitContext>(0);
}

size_t RustParser::StructExpressionContext::getRuleIndex() const {
  return RustParser::RuleStructExpression;
}

void RustParser::StructExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructExpression(this);
}

void RustParser::StructExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructExpression(this);
}

std::any
RustParser::StructExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructExpressionContext *RustParser::structExpression() {
  StructExpressionContext *_localctx =
      _tracker.createInstance<StructExpressionContext>(_ctx, getState());
  enterRule(_localctx, 180, RustParser::RuleStructExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1544);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 195, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1541);
      structExprStruct();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1542);
      structExprTuple();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(1543);
      structExprUnit();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructExprStructContext
//------------------------------------------------------------------

RustParser::StructExprStructContext::StructExprStructContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::StructExprStructContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

tree::TerminalNode *RustParser::StructExprStructContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::StructExprStructContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::StructExprStructContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::StructExprStructContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

RustParser::StructExprFieldsContext *
RustParser::StructExprStructContext::structExprFields() {
  return getRuleContext<RustParser::StructExprFieldsContext>(0);
}

RustParser::StructBaseContext *
RustParser::StructExprStructContext::structBase() {
  return getRuleContext<RustParser::StructBaseContext>(0);
}

size_t RustParser::StructExprStructContext::getRuleIndex() const {
  return RustParser::RuleStructExprStruct;
}

void RustParser::StructExprStructContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructExprStruct(this);
}

void RustParser::StructExprStructContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructExprStruct(this);
}

std::any
RustParser::StructExprStructContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructExprStruct(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructExprStructContext *RustParser::structExprStruct() {
  StructExprStructContext *_localctx =
      _tracker.createInstance<StructExprStructContext>(_ctx, getState());
  enterRule(_localctx, 182, RustParser::RuleStructExprStruct);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1546);
    pathInExpression();
    setState(1547);
    match(RustParser::LCURLYBRACE);
    setState(1551);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     196, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1548);
        innerAttribute();
      }
      setState(1553);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 196, _ctx);
    }
    setState(1556);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_MACRORULES:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::INTEGER_LITERAL:
    case RustParser::POUND: {
      setState(1554);
      structExprFields();
      break;
    }

    case RustParser::DOTDOT: {
      setState(1555);
      structBase();
      break;
    }

    case RustParser::RCURLYBRACE: {
      break;
    }

    default:
      break;
    }
    setState(1558);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructExprFieldsContext
//------------------------------------------------------------------

RustParser::StructExprFieldsContext::StructExprFieldsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::StructExprFieldContext *>
RustParser::StructExprFieldsContext::structExprField() {
  return getRuleContexts<RustParser::StructExprFieldContext>();
}

RustParser::StructExprFieldContext *
RustParser::StructExprFieldsContext::structExprField(size_t i) {
  return getRuleContext<RustParser::StructExprFieldContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::StructExprFieldsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::StructExprFieldsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

RustParser::StructBaseContext *
RustParser::StructExprFieldsContext::structBase() {
  return getRuleContext<RustParser::StructBaseContext>(0);
}

size_t RustParser::StructExprFieldsContext::getRuleIndex() const {
  return RustParser::RuleStructExprFields;
}

void RustParser::StructExprFieldsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructExprFields(this);
}

void RustParser::StructExprFieldsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructExprFields(this);
}

std::any
RustParser::StructExprFieldsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructExprFields(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructExprFieldsContext *RustParser::structExprFields() {
  StructExprFieldsContext *_localctx =
      _tracker.createInstance<StructExprFieldsContext>(_ctx, getState());
  enterRule(_localctx, 184, RustParser::RuleStructExprFields);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1560);
    structExprField();
    setState(1565);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     198, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1561);
        match(RustParser::COMMA);
        setState(1562);
        structExprField();
      }
      setState(1567);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 198, _ctx);
    }
    setState(1573);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 200, _ctx)) {
    case 1: {
      setState(1568);
      match(RustParser::COMMA);
      setState(1569);
      structBase();
      break;
    }

    case 2: {
      setState(1571);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(1570);
        match(RustParser::COMMA);
      }
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructExprFieldContext
//------------------------------------------------------------------

RustParser::StructExprFieldContext::StructExprFieldContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *
RustParser::StructExprFieldContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::StructExprFieldContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::ExpressionContext *
RustParser::StructExprFieldContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::StructExprFieldContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::StructExprFieldContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::TupleIndexContext *
RustParser::StructExprFieldContext::tupleIndex() {
  return getRuleContext<RustParser::TupleIndexContext>(0);
}

size_t RustParser::StructExprFieldContext::getRuleIndex() const {
  return RustParser::RuleStructExprField;
}

void RustParser::StructExprFieldContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructExprField(this);
}

void RustParser::StructExprFieldContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructExprField(this);
}

std::any
RustParser::StructExprFieldContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructExprField(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructExprFieldContext *RustParser::structExprField() {
  StructExprFieldContext *_localctx =
      _tracker.createInstance<StructExprFieldContext>(_ctx, getState());
  enterRule(_localctx, 186, RustParser::RuleStructExprField);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1578);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(1575);
      outerAttribute();
      setState(1580);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1589);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 203, _ctx)) {
    case 1: {
      setState(1581);
      identifier();
      break;
    }

    case 2: {
      setState(1584);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
      case RustParser::KW_MACRORULES:
      case RustParser::NON_KEYWORD_IDENTIFIER:
      case RustParser::RAW_IDENTIFIER: {
        setState(1582);
        identifier();
        break;
      }

      case RustParser::INTEGER_LITERAL: {
        setState(1583);
        tupleIndex();
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      setState(1586);
      match(RustParser::COLON);
      setState(1587);
      expression(0);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructBaseContext
//------------------------------------------------------------------

RustParser::StructBaseContext::StructBaseContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::StructBaseContext::DOTDOT() {
  return getToken(RustParser::DOTDOT, 0);
}

RustParser::ExpressionContext *RustParser::StructBaseContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

size_t RustParser::StructBaseContext::getRuleIndex() const {
  return RustParser::RuleStructBase;
}

void RustParser::StructBaseContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructBase(this);
}

void RustParser::StructBaseContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructBase(this);
}

std::any
RustParser::StructBaseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructBase(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructBaseContext *RustParser::structBase() {
  StructBaseContext *_localctx =
      _tracker.createInstance<StructBaseContext>(_ctx, getState());
  enterRule(_localctx, 188, RustParser::RuleStructBase);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1591);
    match(RustParser::DOTDOT);
    setState(1592);
    expression(0);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructExprTupleContext
//------------------------------------------------------------------

RustParser::StructExprTupleContext::StructExprTupleContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::StructExprTupleContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

tree::TerminalNode *RustParser::StructExprTupleContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::StructExprTupleContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::StructExprTupleContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::StructExprTupleContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

std::vector<RustParser::ExpressionContext *>
RustParser::StructExprTupleContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::StructExprTupleContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::StructExprTupleContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::StructExprTupleContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::StructExprTupleContext::getRuleIndex() const {
  return RustParser::RuleStructExprTuple;
}

void RustParser::StructExprTupleContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructExprTuple(this);
}

void RustParser::StructExprTupleContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructExprTuple(this);
}

std::any
RustParser::StructExprTupleContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructExprTuple(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructExprTupleContext *RustParser::structExprTuple() {
  StructExprTupleContext *_localctx =
      _tracker.createInstance<StructExprTupleContext>(_ctx, getState());
  enterRule(_localctx, 190, RustParser::RuleStructExprTuple);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1594);
    pathInExpression();
    setState(1595);
    match(RustParser::LPAREN);
    setState(1599);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     204, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1596);
        innerAttribute();
      }
      setState(1601);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 204, _ctx);
    }
    setState(1613);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 522417665550785076) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1523430809782507647) != 0)) {
      setState(1602);
      expression(0);
      setState(1607);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 205, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(1603);
          match(RustParser::COMMA);
          setState(1604);
          expression(0);
        }
        setState(1609);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 205, _ctx);
      }
      setState(1611);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(1610);
        match(RustParser::COMMA);
      }
    }
    setState(1615);
    match(RustParser::RPAREN);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructExprUnitContext
//------------------------------------------------------------------

RustParser::StructExprUnitContext::StructExprUnitContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::StructExprUnitContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

size_t RustParser::StructExprUnitContext::getRuleIndex() const {
  return RustParser::RuleStructExprUnit;
}

void RustParser::StructExprUnitContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructExprUnit(this);
}

void RustParser::StructExprUnitContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructExprUnit(this);
}

std::any
RustParser::StructExprUnitContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructExprUnit(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructExprUnitContext *RustParser::structExprUnit() {
  StructExprUnitContext *_localctx =
      _tracker.createInstance<StructExprUnitContext>(_ctx, getState());
  enterRule(_localctx, 192, RustParser::RuleStructExprUnit);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1617);
    pathInExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumerationVariantExpressionContext
//------------------------------------------------------------------

RustParser::EnumerationVariantExpressionContext::
    EnumerationVariantExpressionContext(ParserRuleContext *parent,
                                        size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::EnumExprStructContext *
RustParser::EnumerationVariantExpressionContext::enumExprStruct() {
  return getRuleContext<RustParser::EnumExprStructContext>(0);
}

RustParser::EnumExprTupleContext *
RustParser::EnumerationVariantExpressionContext::enumExprTuple() {
  return getRuleContext<RustParser::EnumExprTupleContext>(0);
}

RustParser::EnumExprFieldlessContext *
RustParser::EnumerationVariantExpressionContext::enumExprFieldless() {
  return getRuleContext<RustParser::EnumExprFieldlessContext>(0);
}

size_t RustParser::EnumerationVariantExpressionContext::getRuleIndex() const {
  return RustParser::RuleEnumerationVariantExpression;
}

void RustParser::EnumerationVariantExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumerationVariantExpression(this);
}

void RustParser::EnumerationVariantExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumerationVariantExpression(this);
}

std::any RustParser::EnumerationVariantExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumerationVariantExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumerationVariantExpressionContext *
RustParser::enumerationVariantExpression() {
  EnumerationVariantExpressionContext *_localctx =
      _tracker.createInstance<EnumerationVariantExpressionContext>(_ctx,
                                                                   getState());
  enterRule(_localctx, 194, RustParser::RuleEnumerationVariantExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1622);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 208, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1619);
      enumExprStruct();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1620);
      enumExprTuple();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(1621);
      enumExprFieldless();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumExprStructContext
//------------------------------------------------------------------

RustParser::EnumExprStructContext::EnumExprStructContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::EnumExprStructContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

tree::TerminalNode *RustParser::EnumExprStructContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::EnumExprStructContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

RustParser::EnumExprFieldsContext *
RustParser::EnumExprStructContext::enumExprFields() {
  return getRuleContext<RustParser::EnumExprFieldsContext>(0);
}

size_t RustParser::EnumExprStructContext::getRuleIndex() const {
  return RustParser::RuleEnumExprStruct;
}

void RustParser::EnumExprStructContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumExprStruct(this);
}

void RustParser::EnumExprStructContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumExprStruct(this);
}

std::any
RustParser::EnumExprStructContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumExprStruct(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumExprStructContext *RustParser::enumExprStruct() {
  EnumExprStructContext *_localctx =
      _tracker.createInstance<EnumExprStructContext>(_ctx, getState());
  enterRule(_localctx, 196, RustParser::RuleEnumExprStruct);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1624);
    pathInExpression();
    setState(1625);
    match(RustParser::LCURLYBRACE);
    setState(1627);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 54) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 54)) & 4194329) != 0)) {
      setState(1626);
      enumExprFields();
    }
    setState(1629);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumExprFieldsContext
//------------------------------------------------------------------

RustParser::EnumExprFieldsContext::EnumExprFieldsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::EnumExprFieldContext *>
RustParser::EnumExprFieldsContext::enumExprField() {
  return getRuleContexts<RustParser::EnumExprFieldContext>();
}

RustParser::EnumExprFieldContext *
RustParser::EnumExprFieldsContext::enumExprField(size_t i) {
  return getRuleContext<RustParser::EnumExprFieldContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::EnumExprFieldsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::EnumExprFieldsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::EnumExprFieldsContext::getRuleIndex() const {
  return RustParser::RuleEnumExprFields;
}

void RustParser::EnumExprFieldsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumExprFields(this);
}

void RustParser::EnumExprFieldsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumExprFields(this);
}

std::any
RustParser::EnumExprFieldsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumExprFields(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumExprFieldsContext *RustParser::enumExprFields() {
  EnumExprFieldsContext *_localctx =
      _tracker.createInstance<EnumExprFieldsContext>(_ctx, getState());
  enterRule(_localctx, 198, RustParser::RuleEnumExprFields);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1631);
    enumExprField();
    setState(1636);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     210, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1632);
        match(RustParser::COMMA);
        setState(1633);
        enumExprField();
      }
      setState(1638);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 210, _ctx);
    }
    setState(1640);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(1639);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumExprFieldContext
//------------------------------------------------------------------

RustParser::EnumExprFieldContext::EnumExprFieldContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *RustParser::EnumExprFieldContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::EnumExprFieldContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::ExpressionContext *RustParser::EnumExprFieldContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

RustParser::TupleIndexContext *RustParser::EnumExprFieldContext::tupleIndex() {
  return getRuleContext<RustParser::TupleIndexContext>(0);
}

size_t RustParser::EnumExprFieldContext::getRuleIndex() const {
  return RustParser::RuleEnumExprField;
}

void RustParser::EnumExprFieldContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumExprField(this);
}

void RustParser::EnumExprFieldContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumExprField(this);
}

std::any
RustParser::EnumExprFieldContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumExprField(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumExprFieldContext *RustParser::enumExprField() {
  EnumExprFieldContext *_localctx =
      _tracker.createInstance<EnumExprFieldContext>(_ctx, getState());
  enterRule(_localctx, 200, RustParser::RuleEnumExprField);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1650);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 213, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1642);
      identifier();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1645);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
      case RustParser::KW_MACRORULES:
      case RustParser::NON_KEYWORD_IDENTIFIER:
      case RustParser::RAW_IDENTIFIER: {
        setState(1643);
        identifier();
        break;
      }

      case RustParser::INTEGER_LITERAL: {
        setState(1644);
        tupleIndex();
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      setState(1647);
      match(RustParser::COLON);
      setState(1648);
      expression(0);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumExprTupleContext
//------------------------------------------------------------------

RustParser::EnumExprTupleContext::EnumExprTupleContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::EnumExprTupleContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

tree::TerminalNode *RustParser::EnumExprTupleContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::EnumExprTupleContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

std::vector<RustParser::ExpressionContext *>
RustParser::EnumExprTupleContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::EnumExprTupleContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::EnumExprTupleContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::EnumExprTupleContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::EnumExprTupleContext::getRuleIndex() const {
  return RustParser::RuleEnumExprTuple;
}

void RustParser::EnumExprTupleContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumExprTuple(this);
}

void RustParser::EnumExprTupleContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumExprTuple(this);
}

std::any
RustParser::EnumExprTupleContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumExprTuple(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumExprTupleContext *RustParser::enumExprTuple() {
  EnumExprTupleContext *_localctx =
      _tracker.createInstance<EnumExprTupleContext>(_ctx, getState());
  enterRule(_localctx, 202, RustParser::RuleEnumExprTuple);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1652);
    pathInExpression();
    setState(1653);
    match(RustParser::LPAREN);
    setState(1665);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 522417665550785076) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1523430809782507647) != 0)) {
      setState(1654);
      expression(0);
      setState(1659);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 214, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(1655);
          match(RustParser::COMMA);
          setState(1656);
          expression(0);
        }
        setState(1661);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 214, _ctx);
      }
      setState(1663);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(1662);
        match(RustParser::COMMA);
      }
    }
    setState(1667);
    match(RustParser::RPAREN);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EnumExprFieldlessContext
//------------------------------------------------------------------

RustParser::EnumExprFieldlessContext::EnumExprFieldlessContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::EnumExprFieldlessContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

size_t RustParser::EnumExprFieldlessContext::getRuleIndex() const {
  return RustParser::RuleEnumExprFieldless;
}

void RustParser::EnumExprFieldlessContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEnumExprFieldless(this);
}

void RustParser::EnumExprFieldlessContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEnumExprFieldless(this);
}

std::any
RustParser::EnumExprFieldlessContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitEnumExprFieldless(this);
  else
    return visitor->visitChildren(this);
}

RustParser::EnumExprFieldlessContext *RustParser::enumExprFieldless() {
  EnumExprFieldlessContext *_localctx =
      _tracker.createInstance<EnumExprFieldlessContext>(_ctx, getState());
  enterRule(_localctx, 204, RustParser::RuleEnumExprFieldless);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1669);
    pathInExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CallParamsContext
//------------------------------------------------------------------

RustParser::CallParamsContext::CallParamsContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::ExpressionContext *>
RustParser::CallParamsContext::expression() {
  return getRuleContexts<RustParser::ExpressionContext>();
}

RustParser::ExpressionContext *
RustParser::CallParamsContext::expression(size_t i) {
  return getRuleContext<RustParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::CallParamsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::CallParamsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::CallParamsContext::getRuleIndex() const {
  return RustParser::RuleCallParams;
}

void RustParser::CallParamsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCallParams(this);
}

void RustParser::CallParamsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCallParams(this);
}

std::any
RustParser::CallParamsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitCallParams(this);
  else
    return visitor->visitChildren(this);
}

RustParser::CallParamsContext *RustParser::callParams() {
  CallParamsContext *_localctx =
      _tracker.createInstance<CallParamsContext>(_ctx, getState());
  enterRule(_localctx, 206, RustParser::RuleCallParams);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1671);
    expression(0);
    setState(1676);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     217, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1672);
        match(RustParser::COMMA);
        setState(1673);
        expression(0);
      }
      setState(1678);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 217, _ctx);
    }
    setState(1680);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(1679);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClosureExpressionContext
//------------------------------------------------------------------

RustParser::ClosureExpressionContext::ClosureExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ClosureExpressionContext::OROR() {
  return getToken(RustParser::OROR, 0);
}

std::vector<tree::TerminalNode *> RustParser::ClosureExpressionContext::OR() {
  return getTokens(RustParser::OR);
}

tree::TerminalNode *RustParser::ClosureExpressionContext::OR(size_t i) {
  return getToken(RustParser::OR, i);
}

RustParser::ExpressionContext *
RustParser::ClosureExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::ClosureExpressionContext::RARROW() {
  return getToken(RustParser::RARROW, 0);
}

RustParser::TypeNoBoundsContext *
RustParser::ClosureExpressionContext::typeNoBounds() {
  return getRuleContext<RustParser::TypeNoBoundsContext>(0);
}

RustParser::BlockExpressionContext *
RustParser::ClosureExpressionContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

tree::TerminalNode *RustParser::ClosureExpressionContext::KW_MOVE() {
  return getToken(RustParser::KW_MOVE, 0);
}

RustParser::ClosureParametersContext *
RustParser::ClosureExpressionContext::closureParameters() {
  return getRuleContext<RustParser::ClosureParametersContext>(0);
}

size_t RustParser::ClosureExpressionContext::getRuleIndex() const {
  return RustParser::RuleClosureExpression;
}

void RustParser::ClosureExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClosureExpression(this);
}

void RustParser::ClosureExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClosureExpression(this);
}

std::any
RustParser::ClosureExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitClosureExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ClosureExpressionContext *RustParser::closureExpression() {
  ClosureExpressionContext *_localctx =
      _tracker.createInstance<ClosureExpressionContext>(_ctx, getState());
  enterRule(_localctx, 208, RustParser::RuleClosureExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1683);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_MOVE) {
      setState(1682);
      match(RustParser::KW_MOVE);
    }
    setState(1691);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::OROR: {
      setState(1685);
      match(RustParser::OROR);
      break;
    }

    case RustParser::OR: {
      setState(1686);
      match(RustParser::OR);
      setState(1688);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 220, _ctx)) {
      case 1: {
        setState(1687);
        closureParameters();
        break;
      }

      default:
        break;
      }
      setState(1690);
      match(RustParser::OR);
      break;
    }

    default:
      throw NoViableAltException(this);
    }
    setState(1698);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_BREAK:
    case RustParser::KW_CONTINUE:
    case RustParser::KW_CRATE:
    case RustParser::KW_FALSE:
    case RustParser::KW_FOR:
    case RustParser::KW_IF:
    case RustParser::KW_LOOP:
    case RustParser::KW_MATCH:
    case RustParser::KW_MOVE:
    case RustParser::KW_RETURN:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SELFTYPE:
    case RustParser::KW_SUPER:
    case RustParser::KW_TRUE:
    case RustParser::KW_UNSAFE:
    case RustParser::KW_WHILE:
    case RustParser::KW_ASYNC:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::CHAR_LITERAL:
    case RustParser::STRING_LITERAL:
    case RustParser::RAW_STRING_LITERAL:
    case RustParser::BYTE_LITERAL:
    case RustParser::BYTE_STRING_LITERAL:
    case RustParser::RAW_BYTE_STRING_LITERAL:
    case RustParser::INTEGER_LITERAL:
    case RustParser::FLOAT_LITERAL:
    case RustParser::LIFETIME_OR_LABEL:
    case RustParser::MINUS:
    case RustParser::STAR:
    case RustParser::NOT:
    case RustParser::AND:
    case RustParser::OR:
    case RustParser::ANDAND:
    case RustParser::OROR:
    case RustParser::LT:
    case RustParser::DOTDOT:
    case RustParser::DOTDOTEQ:
    case RustParser::PATHSEP:
    case RustParser::POUND:
    case RustParser::LCURLYBRACE:
    case RustParser::LSQUAREBRACKET:
    case RustParser::LPAREN: {
      setState(1693);
      expression(0);
      break;
    }

    case RustParser::RARROW: {
      setState(1694);
      match(RustParser::RARROW);
      setState(1695);
      typeNoBounds();
      setState(1696);
      blockExpression();
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClosureParametersContext
//------------------------------------------------------------------

RustParser::ClosureParametersContext::ClosureParametersContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::ClosureParamContext *>
RustParser::ClosureParametersContext::closureParam() {
  return getRuleContexts<RustParser::ClosureParamContext>();
}

RustParser::ClosureParamContext *
RustParser::ClosureParametersContext::closureParam(size_t i) {
  return getRuleContext<RustParser::ClosureParamContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::ClosureParametersContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::ClosureParametersContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::ClosureParametersContext::getRuleIndex() const {
  return RustParser::RuleClosureParameters;
}

void RustParser::ClosureParametersContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClosureParameters(this);
}

void RustParser::ClosureParametersContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClosureParameters(this);
}

std::any
RustParser::ClosureParametersContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitClosureParameters(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ClosureParametersContext *RustParser::closureParameters() {
  ClosureParametersContext *_localctx =
      _tracker.createInstance<ClosureParametersContext>(_ctx, getState());
  enterRule(_localctx, 210, RustParser::RuleClosureParameters);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1700);
    closureParam();
    setState(1705);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     223, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1701);
        match(RustParser::COMMA);
        setState(1702);
        closureParam();
      }
      setState(1707);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 223, _ctx);
    }
    setState(1709);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(1708);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClosureParamContext
//------------------------------------------------------------------

RustParser::ClosureParamContext::ClosureParamContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PatternContext *RustParser::ClosureParamContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::ClosureParamContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::ClosureParamContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

tree::TerminalNode *RustParser::ClosureParamContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::Type_Context *RustParser::ClosureParamContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

size_t RustParser::ClosureParamContext::getRuleIndex() const {
  return RustParser::RuleClosureParam;
}

void RustParser::ClosureParamContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClosureParam(this);
}

void RustParser::ClosureParamContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClosureParam(this);
}

std::any
RustParser::ClosureParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitClosureParam(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ClosureParamContext *RustParser::closureParam() {
  ClosureParamContext *_localctx =
      _tracker.createInstance<ClosureParamContext>(_ctx, getState());
  enterRule(_localctx, 212, RustParser::RuleClosureParam);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1714);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(1711);
      outerAttribute();
      setState(1716);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1717);
    pattern();
    setState(1720);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COLON) {
      setState(1718);
      match(RustParser::COLON);
      setState(1719);
      type_();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LoopExpressionContext
//------------------------------------------------------------------

RustParser::LoopExpressionContext::LoopExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::InfiniteLoopExpressionContext *
RustParser::LoopExpressionContext::infiniteLoopExpression() {
  return getRuleContext<RustParser::InfiniteLoopExpressionContext>(0);
}

RustParser::PredicateLoopExpressionContext *
RustParser::LoopExpressionContext::predicateLoopExpression() {
  return getRuleContext<RustParser::PredicateLoopExpressionContext>(0);
}

RustParser::PredicatePatternLoopExpressionContext *
RustParser::LoopExpressionContext::predicatePatternLoopExpression() {
  return getRuleContext<RustParser::PredicatePatternLoopExpressionContext>(0);
}

RustParser::IteratorLoopExpressionContext *
RustParser::LoopExpressionContext::iteratorLoopExpression() {
  return getRuleContext<RustParser::IteratorLoopExpressionContext>(0);
}

RustParser::LoopLabelContext *RustParser::LoopExpressionContext::loopLabel() {
  return getRuleContext<RustParser::LoopLabelContext>(0);
}

size_t RustParser::LoopExpressionContext::getRuleIndex() const {
  return RustParser::RuleLoopExpression;
}

void RustParser::LoopExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoopExpression(this);
}

void RustParser::LoopExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoopExpression(this);
}

std::any
RustParser::LoopExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLoopExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LoopExpressionContext *RustParser::loopExpression() {
  LoopExpressionContext *_localctx =
      _tracker.createInstance<LoopExpressionContext>(_ctx, getState());
  enterRule(_localctx, 214, RustParser::RuleLoopExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1723);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::LIFETIME_OR_LABEL) {
      setState(1722);
      loopLabel();
    }
    setState(1729);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 228, _ctx)) {
    case 1: {
      setState(1725);
      infiniteLoopExpression();
      break;
    }

    case 2: {
      setState(1726);
      predicateLoopExpression();
      break;
    }

    case 3: {
      setState(1727);
      predicatePatternLoopExpression();
      break;
    }

    case 4: {
      setState(1728);
      iteratorLoopExpression();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- InfiniteLoopExpressionContext
//------------------------------------------------------------------

RustParser::InfiniteLoopExpressionContext::InfiniteLoopExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::InfiniteLoopExpressionContext::KW_LOOP() {
  return getToken(RustParser::KW_LOOP, 0);
}

RustParser::BlockExpressionContext *
RustParser::InfiniteLoopExpressionContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

size_t RustParser::InfiniteLoopExpressionContext::getRuleIndex() const {
  return RustParser::RuleInfiniteLoopExpression;
}

void RustParser::InfiniteLoopExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInfiniteLoopExpression(this);
}

void RustParser::InfiniteLoopExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInfiniteLoopExpression(this);
}

std::any RustParser::InfiniteLoopExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitInfiniteLoopExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::InfiniteLoopExpressionContext *
RustParser::infiniteLoopExpression() {
  InfiniteLoopExpressionContext *_localctx =
      _tracker.createInstance<InfiniteLoopExpressionContext>(_ctx, getState());
  enterRule(_localctx, 216, RustParser::RuleInfiniteLoopExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1731);
    match(RustParser::KW_LOOP);
    setState(1732);
    blockExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PredicateLoopExpressionContext
//------------------------------------------------------------------

RustParser::PredicateLoopExpressionContext::PredicateLoopExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::PredicateLoopExpressionContext::KW_WHILE() {
  return getToken(RustParser::KW_WHILE, 0);
}

RustParser::ExpressionContext *
RustParser::PredicateLoopExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

RustParser::BlockExpressionContext *
RustParser::PredicateLoopExpressionContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

size_t RustParser::PredicateLoopExpressionContext::getRuleIndex() const {
  return RustParser::RulePredicateLoopExpression;
}

void RustParser::PredicateLoopExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPredicateLoopExpression(this);
}

void RustParser::PredicateLoopExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPredicateLoopExpression(this);
}

std::any RustParser::PredicateLoopExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPredicateLoopExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PredicateLoopExpressionContext *
RustParser::predicateLoopExpression() {
  PredicateLoopExpressionContext *_localctx =
      _tracker.createInstance<PredicateLoopExpressionContext>(_ctx, getState());
  enterRule(_localctx, 218, RustParser::RulePredicateLoopExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1734);
    match(RustParser::KW_WHILE);
    setState(1735);
    expression(0);
    setState(1736);
    blockExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PredicatePatternLoopExpressionContext
//------------------------------------------------------------------

RustParser::PredicatePatternLoopExpressionContext::
    PredicatePatternLoopExpressionContext(ParserRuleContext *parent,
                                          size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *
RustParser::PredicatePatternLoopExpressionContext::KW_WHILE() {
  return getToken(RustParser::KW_WHILE, 0);
}

tree::TerminalNode *
RustParser::PredicatePatternLoopExpressionContext::KW_LET() {
  return getToken(RustParser::KW_LET, 0);
}

RustParser::PatternContext *
RustParser::PredicatePatternLoopExpressionContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

tree::TerminalNode *RustParser::PredicatePatternLoopExpressionContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::ExpressionContext *
RustParser::PredicatePatternLoopExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

RustParser::BlockExpressionContext *
RustParser::PredicatePatternLoopExpressionContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

size_t RustParser::PredicatePatternLoopExpressionContext::getRuleIndex() const {
  return RustParser::RulePredicatePatternLoopExpression;
}

void RustParser::PredicatePatternLoopExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPredicatePatternLoopExpression(this);
}

void RustParser::PredicatePatternLoopExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPredicatePatternLoopExpression(this);
}

std::any RustParser::PredicatePatternLoopExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPredicatePatternLoopExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PredicatePatternLoopExpressionContext *
RustParser::predicatePatternLoopExpression() {
  PredicatePatternLoopExpressionContext *_localctx =
      _tracker.createInstance<PredicatePatternLoopExpressionContext>(
          _ctx, getState());
  enterRule(_localctx, 220, RustParser::RulePredicatePatternLoopExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1738);
    match(RustParser::KW_WHILE);
    setState(1739);
    match(RustParser::KW_LET);
    setState(1740);
    pattern();
    setState(1741);
    match(RustParser::EQ);
    setState(1742);
    expression(0);
    setState(1743);
    blockExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IteratorLoopExpressionContext
//------------------------------------------------------------------

RustParser::IteratorLoopExpressionContext::IteratorLoopExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::IteratorLoopExpressionContext::KW_FOR() {
  return getToken(RustParser::KW_FOR, 0);
}

RustParser::PatternContext *
RustParser::IteratorLoopExpressionContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

tree::TerminalNode *RustParser::IteratorLoopExpressionContext::KW_IN() {
  return getToken(RustParser::KW_IN, 0);
}

RustParser::ExpressionContext *
RustParser::IteratorLoopExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

RustParser::BlockExpressionContext *
RustParser::IteratorLoopExpressionContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

size_t RustParser::IteratorLoopExpressionContext::getRuleIndex() const {
  return RustParser::RuleIteratorLoopExpression;
}

void RustParser::IteratorLoopExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIteratorLoopExpression(this);
}

void RustParser::IteratorLoopExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIteratorLoopExpression(this);
}

std::any RustParser::IteratorLoopExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitIteratorLoopExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::IteratorLoopExpressionContext *
RustParser::iteratorLoopExpression() {
  IteratorLoopExpressionContext *_localctx =
      _tracker.createInstance<IteratorLoopExpressionContext>(_ctx, getState());
  enterRule(_localctx, 222, RustParser::RuleIteratorLoopExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1745);
    match(RustParser::KW_FOR);
    setState(1746);
    pattern();
    setState(1747);
    match(RustParser::KW_IN);
    setState(1748);
    expression(0);
    setState(1749);
    blockExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LoopLabelContext
//------------------------------------------------------------------

RustParser::LoopLabelContext::LoopLabelContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::LoopLabelContext::LIFETIME_OR_LABEL() {
  return getToken(RustParser::LIFETIME_OR_LABEL, 0);
}

tree::TerminalNode *RustParser::LoopLabelContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

size_t RustParser::LoopLabelContext::getRuleIndex() const {
  return RustParser::RuleLoopLabel;
}

void RustParser::LoopLabelContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoopLabel(this);
}

void RustParser::LoopLabelContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoopLabel(this);
}

std::any RustParser::LoopLabelContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLoopLabel(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LoopLabelContext *RustParser::loopLabel() {
  LoopLabelContext *_localctx =
      _tracker.createInstance<LoopLabelContext>(_ctx, getState());
  enterRule(_localctx, 224, RustParser::RuleLoopLabel);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1751);
    match(RustParser::LIFETIME_OR_LABEL);
    setState(1752);
    match(RustParser::COLON);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IfExpressionContext
//------------------------------------------------------------------

RustParser::IfExpressionContext::IfExpressionContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::IfExpressionContext::KW_IF() {
  return getToken(RustParser::KW_IF, 0);
}

RustParser::ExpressionContext *RustParser::IfExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

std::vector<RustParser::BlockExpressionContext *>
RustParser::IfExpressionContext::blockExpression() {
  return getRuleContexts<RustParser::BlockExpressionContext>();
}

RustParser::BlockExpressionContext *
RustParser::IfExpressionContext::blockExpression(size_t i) {
  return getRuleContext<RustParser::BlockExpressionContext>(i);
}

tree::TerminalNode *RustParser::IfExpressionContext::KW_ELSE() {
  return getToken(RustParser::KW_ELSE, 0);
}

RustParser::IfExpressionContext *
RustParser::IfExpressionContext::ifExpression() {
  return getRuleContext<RustParser::IfExpressionContext>(0);
}

RustParser::IfLetExpressionContext *
RustParser::IfExpressionContext::ifLetExpression() {
  return getRuleContext<RustParser::IfLetExpressionContext>(0);
}

size_t RustParser::IfExpressionContext::getRuleIndex() const {
  return RustParser::RuleIfExpression;
}

void RustParser::IfExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfExpression(this);
}

void RustParser::IfExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfExpression(this);
}

std::any
RustParser::IfExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitIfExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::IfExpressionContext *RustParser::ifExpression() {
  IfExpressionContext *_localctx =
      _tracker.createInstance<IfExpressionContext>(_ctx, getState());
  enterRule(_localctx, 226, RustParser::RuleIfExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1754);
    match(RustParser::KW_IF);
    setState(1755);
    expression(0);
    setState(1756);
    blockExpression();
    setState(1763);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 230, _ctx)) {
    case 1: {
      setState(1757);
      match(RustParser::KW_ELSE);
      setState(1761);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 229, _ctx)) {
      case 1: {
        setState(1758);
        blockExpression();
        break;
      }

      case 2: {
        setState(1759);
        ifExpression();
        break;
      }

      case 3: {
        setState(1760);
        ifLetExpression();
        break;
      }

      default:
        break;
      }
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IfLetExpressionContext
//------------------------------------------------------------------

RustParser::IfLetExpressionContext::IfLetExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::IfLetExpressionContext::KW_IF() {
  return getToken(RustParser::KW_IF, 0);
}

tree::TerminalNode *RustParser::IfLetExpressionContext::KW_LET() {
  return getToken(RustParser::KW_LET, 0);
}

RustParser::PatternContext *RustParser::IfLetExpressionContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

tree::TerminalNode *RustParser::IfLetExpressionContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::ExpressionContext *
RustParser::IfLetExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

std::vector<RustParser::BlockExpressionContext *>
RustParser::IfLetExpressionContext::blockExpression() {
  return getRuleContexts<RustParser::BlockExpressionContext>();
}

RustParser::BlockExpressionContext *
RustParser::IfLetExpressionContext::blockExpression(size_t i) {
  return getRuleContext<RustParser::BlockExpressionContext>(i);
}

tree::TerminalNode *RustParser::IfLetExpressionContext::KW_ELSE() {
  return getToken(RustParser::KW_ELSE, 0);
}

RustParser::IfExpressionContext *
RustParser::IfLetExpressionContext::ifExpression() {
  return getRuleContext<RustParser::IfExpressionContext>(0);
}

RustParser::IfLetExpressionContext *
RustParser::IfLetExpressionContext::ifLetExpression() {
  return getRuleContext<RustParser::IfLetExpressionContext>(0);
}

size_t RustParser::IfLetExpressionContext::getRuleIndex() const {
  return RustParser::RuleIfLetExpression;
}

void RustParser::IfLetExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfLetExpression(this);
}

void RustParser::IfLetExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfLetExpression(this);
}

std::any
RustParser::IfLetExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitIfLetExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::IfLetExpressionContext *RustParser::ifLetExpression() {
  IfLetExpressionContext *_localctx =
      _tracker.createInstance<IfLetExpressionContext>(_ctx, getState());
  enterRule(_localctx, 228, RustParser::RuleIfLetExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1765);
    match(RustParser::KW_IF);
    setState(1766);
    match(RustParser::KW_LET);
    setState(1767);
    pattern();
    setState(1768);
    match(RustParser::EQ);
    setState(1769);
    expression(0);
    setState(1770);
    blockExpression();
    setState(1777);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 232, _ctx)) {
    case 1: {
      setState(1771);
      match(RustParser::KW_ELSE);
      setState(1775);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 231, _ctx)) {
      case 1: {
        setState(1772);
        blockExpression();
        break;
      }

      case 2: {
        setState(1773);
        ifExpression();
        break;
      }

      case 3: {
        setState(1774);
        ifLetExpression();
        break;
      }

      default:
        break;
      }
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MatchExpressionContext
//------------------------------------------------------------------

RustParser::MatchExpressionContext::MatchExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::MatchExpressionContext::KW_MATCH() {
  return getToken(RustParser::KW_MATCH, 0);
}

RustParser::ExpressionContext *
RustParser::MatchExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::MatchExpressionContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::MatchExpressionContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

std::vector<RustParser::InnerAttributeContext *>
RustParser::MatchExpressionContext::innerAttribute() {
  return getRuleContexts<RustParser::InnerAttributeContext>();
}

RustParser::InnerAttributeContext *
RustParser::MatchExpressionContext::innerAttribute(size_t i) {
  return getRuleContext<RustParser::InnerAttributeContext>(i);
}

RustParser::MatchArmsContext *RustParser::MatchExpressionContext::matchArms() {
  return getRuleContext<RustParser::MatchArmsContext>(0);
}

size_t RustParser::MatchExpressionContext::getRuleIndex() const {
  return RustParser::RuleMatchExpression;
}

void RustParser::MatchExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMatchExpression(this);
}

void RustParser::MatchExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMatchExpression(this);
}

std::any
RustParser::MatchExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMatchExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MatchExpressionContext *RustParser::matchExpression() {
  MatchExpressionContext *_localctx =
      _tracker.createInstance<MatchExpressionContext>(_ctx, getState());
  enterRule(_localctx, 230, RustParser::RuleMatchExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1779);
    match(RustParser::KW_MATCH);
    setState(1780);
    expression(0);
    setState(1781);
    match(RustParser::LCURLYBRACE);
    setState(1785);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     233, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1782);
        innerAttribute();
      }
      setState(1787);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 233, _ctx);
    }
    setState(1789);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 522417558172729888) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1451307245037963391) != 0)) {
      setState(1788);
      matchArms();
    }
    setState(1791);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MatchArmsContext
//------------------------------------------------------------------

RustParser::MatchArmsContext::MatchArmsContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::MatchArmContext *>
RustParser::MatchArmsContext::matchArm() {
  return getRuleContexts<RustParser::MatchArmContext>();
}

RustParser::MatchArmContext *RustParser::MatchArmsContext::matchArm(size_t i) {
  return getRuleContext<RustParser::MatchArmContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::MatchArmsContext::FATARROW() {
  return getTokens(RustParser::FATARROW);
}

tree::TerminalNode *RustParser::MatchArmsContext::FATARROW(size_t i) {
  return getToken(RustParser::FATARROW, i);
}

RustParser::ExpressionContext *RustParser::MatchArmsContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

std::vector<RustParser::MatchArmExpressionContext *>
RustParser::MatchArmsContext::matchArmExpression() {
  return getRuleContexts<RustParser::MatchArmExpressionContext>();
}

RustParser::MatchArmExpressionContext *
RustParser::MatchArmsContext::matchArmExpression(size_t i) {
  return getRuleContext<RustParser::MatchArmExpressionContext>(i);
}

tree::TerminalNode *RustParser::MatchArmsContext::COMMA() {
  return getToken(RustParser::COMMA, 0);
}

size_t RustParser::MatchArmsContext::getRuleIndex() const {
  return RustParser::RuleMatchArms;
}

void RustParser::MatchArmsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMatchArms(this);
}

void RustParser::MatchArmsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMatchArms(this);
}

std::any RustParser::MatchArmsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMatchArms(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MatchArmsContext *RustParser::matchArms() {
  MatchArmsContext *_localctx =
      _tracker.createInstance<MatchArmsContext>(_ctx, getState());
  enterRule(_localctx, 232, RustParser::RuleMatchArms);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1799);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     235, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1793);
        matchArm();
        setState(1794);
        match(RustParser::FATARROW);
        setState(1795);
        matchArmExpression();
      }
      setState(1801);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 235, _ctx);
    }
    setState(1802);
    matchArm();
    setState(1803);
    match(RustParser::FATARROW);
    setState(1804);
    expression(0);
    setState(1806);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(1805);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MatchArmExpressionContext
//------------------------------------------------------------------

RustParser::MatchArmExpressionContext::MatchArmExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::ExpressionContext *
RustParser::MatchArmExpressionContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::MatchArmExpressionContext::COMMA() {
  return getToken(RustParser::COMMA, 0);
}

RustParser::ExpressionWithBlockContext *
RustParser::MatchArmExpressionContext::expressionWithBlock() {
  return getRuleContext<RustParser::ExpressionWithBlockContext>(0);
}

size_t RustParser::MatchArmExpressionContext::getRuleIndex() const {
  return RustParser::RuleMatchArmExpression;
}

void RustParser::MatchArmExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMatchArmExpression(this);
}

void RustParser::MatchArmExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMatchArmExpression(this);
}

std::any
RustParser::MatchArmExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMatchArmExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MatchArmExpressionContext *RustParser::matchArmExpression() {
  MatchArmExpressionContext *_localctx =
      _tracker.createInstance<MatchArmExpressionContext>(_ctx, getState());
  enterRule(_localctx, 234, RustParser::RuleMatchArmExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1815);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 238, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1808);
      expression(0);
      setState(1809);
      match(RustParser::COMMA);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1811);
      expressionWithBlock();
      setState(1813);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(1812);
        match(RustParser::COMMA);
      }
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MatchArmContext
//------------------------------------------------------------------

RustParser::MatchArmContext::MatchArmContext(ParserRuleContext *parent,
                                             size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PatternContext *RustParser::MatchArmContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::MatchArmContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::MatchArmContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

RustParser::MatchArmGuardContext *RustParser::MatchArmContext::matchArmGuard() {
  return getRuleContext<RustParser::MatchArmGuardContext>(0);
}

size_t RustParser::MatchArmContext::getRuleIndex() const {
  return RustParser::RuleMatchArm;
}

void RustParser::MatchArmContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMatchArm(this);
}

void RustParser::MatchArmContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMatchArm(this);
}

std::any RustParser::MatchArmContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMatchArm(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MatchArmContext *RustParser::matchArm() {
  MatchArmContext *_localctx =
      _tracker.createInstance<MatchArmContext>(_ctx, getState());
  enterRule(_localctx, 236, RustParser::RuleMatchArm);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1820);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(1817);
      outerAttribute();
      setState(1822);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1823);
    pattern();
    setState(1825);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_IF) {
      setState(1824);
      matchArmGuard();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MatchArmGuardContext
//------------------------------------------------------------------

RustParser::MatchArmGuardContext::MatchArmGuardContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::MatchArmGuardContext::KW_IF() {
  return getToken(RustParser::KW_IF, 0);
}

RustParser::ExpressionContext *RustParser::MatchArmGuardContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

size_t RustParser::MatchArmGuardContext::getRuleIndex() const {
  return RustParser::RuleMatchArmGuard;
}

void RustParser::MatchArmGuardContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMatchArmGuard(this);
}

void RustParser::MatchArmGuardContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMatchArmGuard(this);
}

std::any
RustParser::MatchArmGuardContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMatchArmGuard(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MatchArmGuardContext *RustParser::matchArmGuard() {
  MatchArmGuardContext *_localctx =
      _tracker.createInstance<MatchArmGuardContext>(_ctx, getState());
  enterRule(_localctx, 238, RustParser::RuleMatchArmGuard);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1827);
    match(RustParser::KW_IF);
    setState(1828);
    expression(0);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PatternContext
//------------------------------------------------------------------

RustParser::PatternContext::PatternContext(ParserRuleContext *parent,
                                           size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::PatternNoTopAltContext *>
RustParser::PatternContext::patternNoTopAlt() {
  return getRuleContexts<RustParser::PatternNoTopAltContext>();
}

RustParser::PatternNoTopAltContext *
RustParser::PatternContext::patternNoTopAlt(size_t i) {
  return getRuleContext<RustParser::PatternNoTopAltContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::PatternContext::OR() {
  return getTokens(RustParser::OR);
}

tree::TerminalNode *RustParser::PatternContext::OR(size_t i) {
  return getToken(RustParser::OR, i);
}

size_t RustParser::PatternContext::getRuleIndex() const {
  return RustParser::RulePattern;
}

void RustParser::PatternContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPattern(this);
}

void RustParser::PatternContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPattern(this);
}

std::any RustParser::PatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PatternContext *RustParser::pattern() {
  PatternContext *_localctx =
      _tracker.createInstance<PatternContext>(_ctx, getState());
  enterRule(_localctx, 240, RustParser::RulePattern);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1831);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::OR) {
      setState(1830);
      match(RustParser::OR);
    }
    setState(1833);
    patternNoTopAlt();
    setState(1838);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     242, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1834);
        match(RustParser::OR);
        setState(1835);
        patternNoTopAlt();
      }
      setState(1840);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 242, _ctx);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PatternNoTopAltContext
//------------------------------------------------------------------

RustParser::PatternNoTopAltContext::PatternNoTopAltContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PatternWithoutRangeContext *
RustParser::PatternNoTopAltContext::patternWithoutRange() {
  return getRuleContext<RustParser::PatternWithoutRangeContext>(0);
}

RustParser::RangePatternContext *
RustParser::PatternNoTopAltContext::rangePattern() {
  return getRuleContext<RustParser::RangePatternContext>(0);
}

size_t RustParser::PatternNoTopAltContext::getRuleIndex() const {
  return RustParser::RulePatternNoTopAlt;
}

void RustParser::PatternNoTopAltContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPatternNoTopAlt(this);
}

void RustParser::PatternNoTopAltContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPatternNoTopAlt(this);
}

std::any
RustParser::PatternNoTopAltContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPatternNoTopAlt(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PatternNoTopAltContext *RustParser::patternNoTopAlt() {
  PatternNoTopAltContext *_localctx =
      _tracker.createInstance<PatternNoTopAltContext>(_ctx, getState());
  enterRule(_localctx, 242, RustParser::RulePatternNoTopAlt);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1843);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 243, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1841);
      patternWithoutRange();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1842);
      rangePattern();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PatternWithoutRangeContext
//------------------------------------------------------------------

RustParser::PatternWithoutRangeContext::PatternWithoutRangeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::LiteralPatternContext *
RustParser::PatternWithoutRangeContext::literalPattern() {
  return getRuleContext<RustParser::LiteralPatternContext>(0);
}

RustParser::IdentifierPatternContext *
RustParser::PatternWithoutRangeContext::identifierPattern() {
  return getRuleContext<RustParser::IdentifierPatternContext>(0);
}

RustParser::WildcardPatternContext *
RustParser::PatternWithoutRangeContext::wildcardPattern() {
  return getRuleContext<RustParser::WildcardPatternContext>(0);
}

RustParser::RestPatternContext *
RustParser::PatternWithoutRangeContext::restPattern() {
  return getRuleContext<RustParser::RestPatternContext>(0);
}

RustParser::ReferencePatternContext *
RustParser::PatternWithoutRangeContext::referencePattern() {
  return getRuleContext<RustParser::ReferencePatternContext>(0);
}

RustParser::StructPatternContext *
RustParser::PatternWithoutRangeContext::structPattern() {
  return getRuleContext<RustParser::StructPatternContext>(0);
}

RustParser::TupleStructPatternContext *
RustParser::PatternWithoutRangeContext::tupleStructPattern() {
  return getRuleContext<RustParser::TupleStructPatternContext>(0);
}

RustParser::TuplePatternContext *
RustParser::PatternWithoutRangeContext::tuplePattern() {
  return getRuleContext<RustParser::TuplePatternContext>(0);
}

RustParser::GroupedPatternContext *
RustParser::PatternWithoutRangeContext::groupedPattern() {
  return getRuleContext<RustParser::GroupedPatternContext>(0);
}

RustParser::SlicePatternContext *
RustParser::PatternWithoutRangeContext::slicePattern() {
  return getRuleContext<RustParser::SlicePatternContext>(0);
}

RustParser::PathPatternContext *
RustParser::PatternWithoutRangeContext::pathPattern() {
  return getRuleContext<RustParser::PathPatternContext>(0);
}

RustParser::MacroInvocationContext *
RustParser::PatternWithoutRangeContext::macroInvocation() {
  return getRuleContext<RustParser::MacroInvocationContext>(0);
}

size_t RustParser::PatternWithoutRangeContext::getRuleIndex() const {
  return RustParser::RulePatternWithoutRange;
}

void RustParser::PatternWithoutRangeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPatternWithoutRange(this);
}

void RustParser::PatternWithoutRangeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPatternWithoutRange(this);
}

std::any RustParser::PatternWithoutRangeContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPatternWithoutRange(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PatternWithoutRangeContext *RustParser::patternWithoutRange() {
  PatternWithoutRangeContext *_localctx =
      _tracker.createInstance<PatternWithoutRangeContext>(_ctx, getState());
  enterRule(_localctx, 244, RustParser::RulePatternWithoutRange);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1857);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 244, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1845);
      literalPattern();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1846);
      identifierPattern();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(1847);
      wildcardPattern();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(1848);
      restPattern();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(1849);
      referencePattern();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(1850);
      structPattern();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(1851);
      tupleStructPattern();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(1852);
      tuplePattern();
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(1853);
      groupedPattern();
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(1854);
      slicePattern();
      break;
    }

    case 11: {
      enterOuterAlt(_localctx, 11);
      setState(1855);
      pathPattern();
      break;
    }

    case 12: {
      enterOuterAlt(_localctx, 12);
      setState(1856);
      macroInvocation();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LiteralPatternContext
//------------------------------------------------------------------

RustParser::LiteralPatternContext::LiteralPatternContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::LiteralPatternContext::KW_TRUE() {
  return getToken(RustParser::KW_TRUE, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::KW_FALSE() {
  return getToken(RustParser::KW_FALSE, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::CHAR_LITERAL() {
  return getToken(RustParser::CHAR_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::BYTE_LITERAL() {
  return getToken(RustParser::BYTE_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::STRING_LITERAL() {
  return getToken(RustParser::STRING_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::RAW_STRING_LITERAL() {
  return getToken(RustParser::RAW_STRING_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::BYTE_STRING_LITERAL() {
  return getToken(RustParser::BYTE_STRING_LITERAL, 0);
}

tree::TerminalNode *
RustParser::LiteralPatternContext::RAW_BYTE_STRING_LITERAL() {
  return getToken(RustParser::RAW_BYTE_STRING_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::INTEGER_LITERAL() {
  return getToken(RustParser::INTEGER_LITERAL, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::MINUS() {
  return getToken(RustParser::MINUS, 0);
}

tree::TerminalNode *RustParser::LiteralPatternContext::FLOAT_LITERAL() {
  return getToken(RustParser::FLOAT_LITERAL, 0);
}

size_t RustParser::LiteralPatternContext::getRuleIndex() const {
  return RustParser::RuleLiteralPattern;
}

void RustParser::LiteralPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLiteralPattern(this);
}

void RustParser::LiteralPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLiteralPattern(this);
}

std::any
RustParser::LiteralPatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLiteralPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LiteralPatternContext *RustParser::literalPattern() {
  LiteralPatternContext *_localctx =
      _tracker.createInstance<LiteralPatternContext>(_ctx, getState());
  enterRule(_localctx, 246, RustParser::RuleLiteralPattern);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1875);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 247, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1859);
      match(RustParser::KW_TRUE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1860);
      match(RustParser::KW_FALSE);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(1861);
      match(RustParser::CHAR_LITERAL);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(1862);
      match(RustParser::BYTE_LITERAL);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(1863);
      match(RustParser::STRING_LITERAL);
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(1864);
      match(RustParser::RAW_STRING_LITERAL);
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(1865);
      match(RustParser::BYTE_STRING_LITERAL);
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(1866);
      match(RustParser::RAW_BYTE_STRING_LITERAL);
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(1868);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::MINUS) {
        setState(1867);
        match(RustParser::MINUS);
      }
      setState(1870);
      match(RustParser::INTEGER_LITERAL);
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(1872);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::MINUS) {
        setState(1871);
        match(RustParser::MINUS);
      }
      setState(1874);
      match(RustParser::FLOAT_LITERAL);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdentifierPatternContext
//------------------------------------------------------------------

RustParser::IdentifierPatternContext::IdentifierPatternContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *
RustParser::IdentifierPatternContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::IdentifierPatternContext::KW_REF() {
  return getToken(RustParser::KW_REF, 0);
}

tree::TerminalNode *RustParser::IdentifierPatternContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

tree::TerminalNode *RustParser::IdentifierPatternContext::AT() {
  return getToken(RustParser::AT, 0);
}

RustParser::PatternContext *RustParser::IdentifierPatternContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

size_t RustParser::IdentifierPatternContext::getRuleIndex() const {
  return RustParser::RuleIdentifierPattern;
}

void RustParser::IdentifierPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdentifierPattern(this);
}

void RustParser::IdentifierPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdentifierPattern(this);
}

std::any
RustParser::IdentifierPatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitIdentifierPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::IdentifierPatternContext *RustParser::identifierPattern() {
  IdentifierPatternContext *_localctx =
      _tracker.createInstance<IdentifierPatternContext>(_ctx, getState());
  enterRule(_localctx, 248, RustParser::RuleIdentifierPattern);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1878);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_REF) {
      setState(1877);
      match(RustParser::KW_REF);
    }
    setState(1881);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_MUT) {
      setState(1880);
      match(RustParser::KW_MUT);
    }
    setState(1883);
    identifier();
    setState(1886);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::AT) {
      setState(1884);
      match(RustParser::AT);
      setState(1885);
      pattern();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- WildcardPatternContext
//------------------------------------------------------------------

RustParser::WildcardPatternContext::WildcardPatternContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::WildcardPatternContext::UNDERSCORE() {
  return getToken(RustParser::UNDERSCORE, 0);
}

size_t RustParser::WildcardPatternContext::getRuleIndex() const {
  return RustParser::RuleWildcardPattern;
}

void RustParser::WildcardPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWildcardPattern(this);
}

void RustParser::WildcardPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWildcardPattern(this);
}

std::any
RustParser::WildcardPatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitWildcardPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::WildcardPatternContext *RustParser::wildcardPattern() {
  WildcardPatternContext *_localctx =
      _tracker.createInstance<WildcardPatternContext>(_ctx, getState());
  enterRule(_localctx, 250, RustParser::RuleWildcardPattern);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1888);
    match(RustParser::UNDERSCORE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RestPatternContext
//------------------------------------------------------------------

RustParser::RestPatternContext::RestPatternContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::RestPatternContext::DOTDOT() {
  return getToken(RustParser::DOTDOT, 0);
}

size_t RustParser::RestPatternContext::getRuleIndex() const {
  return RustParser::RuleRestPattern;
}

void RustParser::RestPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRestPattern(this);
}

void RustParser::RestPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRestPattern(this);
}

std::any
RustParser::RestPatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitRestPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::RestPatternContext *RustParser::restPattern() {
  RestPatternContext *_localctx =
      _tracker.createInstance<RestPatternContext>(_ctx, getState());
  enterRule(_localctx, 252, RustParser::RuleRestPattern);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1890);
    match(RustParser::DOTDOT);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RangePatternContext
//------------------------------------------------------------------

RustParser::RangePatternContext::RangePatternContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

size_t RustParser::RangePatternContext::getRuleIndex() const {
  return RustParser::RuleRangePattern;
}

void RustParser::RangePatternContext::copyFrom(RangePatternContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- InclusiveRangePatternContext
//------------------------------------------------------------------

std::vector<RustParser::RangePatternBoundContext *>
RustParser::InclusiveRangePatternContext::rangePatternBound() {
  return getRuleContexts<RustParser::RangePatternBoundContext>();
}

RustParser::RangePatternBoundContext *
RustParser::InclusiveRangePatternContext::rangePatternBound(size_t i) {
  return getRuleContext<RustParser::RangePatternBoundContext>(i);
}

tree::TerminalNode *RustParser::InclusiveRangePatternContext::DOTDOTEQ() {
  return getToken(RustParser::DOTDOTEQ, 0);
}

RustParser::InclusiveRangePatternContext::InclusiveRangePatternContext(
    RangePatternContext *ctx) {
  copyFrom(ctx);
}

void RustParser::InclusiveRangePatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInclusiveRangePattern(this);
}
void RustParser::InclusiveRangePatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInclusiveRangePattern(this);
}

std::any RustParser::InclusiveRangePatternContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitInclusiveRangePattern(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ObsoleteRangePatternContext
//------------------------------------------------------------------

std::vector<RustParser::RangePatternBoundContext *>
RustParser::ObsoleteRangePatternContext::rangePatternBound() {
  return getRuleContexts<RustParser::RangePatternBoundContext>();
}

RustParser::RangePatternBoundContext *
RustParser::ObsoleteRangePatternContext::rangePatternBound(size_t i) {
  return getRuleContext<RustParser::RangePatternBoundContext>(i);
}

tree::TerminalNode *RustParser::ObsoleteRangePatternContext::DOTDOTDOT() {
  return getToken(RustParser::DOTDOTDOT, 0);
}

RustParser::ObsoleteRangePatternContext::ObsoleteRangePatternContext(
    RangePatternContext *ctx) {
  copyFrom(ctx);
}

void RustParser::ObsoleteRangePatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterObsoleteRangePattern(this);
}
void RustParser::ObsoleteRangePatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitObsoleteRangePattern(this);
}

std::any RustParser::ObsoleteRangePatternContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitObsoleteRangePattern(this);
  else
    return visitor->visitChildren(this);
}
//----------------- HalfOpenRangePatternContext
//------------------------------------------------------------------

RustParser::RangePatternBoundContext *
RustParser::HalfOpenRangePatternContext::rangePatternBound() {
  return getRuleContext<RustParser::RangePatternBoundContext>(0);
}

tree::TerminalNode *RustParser::HalfOpenRangePatternContext::DOTDOT() {
  return getToken(RustParser::DOTDOT, 0);
}

RustParser::HalfOpenRangePatternContext::HalfOpenRangePatternContext(
    RangePatternContext *ctx) {
  copyFrom(ctx);
}

void RustParser::HalfOpenRangePatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterHalfOpenRangePattern(this);
}
void RustParser::HalfOpenRangePatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitHalfOpenRangePattern(this);
}

std::any RustParser::HalfOpenRangePatternContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitHalfOpenRangePattern(this);
  else
    return visitor->visitChildren(this);
}
RustParser::RangePatternContext *RustParser::rangePattern() {
  RangePatternContext *_localctx =
      _tracker.createInstance<RangePatternContext>(_ctx, getState());
  enterRule(_localctx, 254, RustParser::RuleRangePattern);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1903);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 251, _ctx)) {
    case 1: {
      _localctx =
          _tracker.createInstance<RustParser::InclusiveRangePatternContext>(
              _localctx);
      enterOuterAlt(_localctx, 1);
      setState(1892);
      rangePatternBound();
      setState(1893);
      match(RustParser::DOTDOTEQ);
      setState(1894);
      rangePatternBound();
      break;
    }

    case 2: {
      _localctx =
          _tracker.createInstance<RustParser::HalfOpenRangePatternContext>(
              _localctx);
      enterOuterAlt(_localctx, 2);
      setState(1896);
      rangePatternBound();
      setState(1897);
      match(RustParser::DOTDOT);
      break;
    }

    case 3: {
      _localctx =
          _tracker.createInstance<RustParser::ObsoleteRangePatternContext>(
              _localctx);
      enterOuterAlt(_localctx, 3);
      setState(1899);
      rangePatternBound();
      setState(1900);
      match(RustParser::DOTDOTDOT);
      setState(1901);
      rangePatternBound();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RangePatternBoundContext
//------------------------------------------------------------------

RustParser::RangePatternBoundContext::RangePatternBoundContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::RangePatternBoundContext::CHAR_LITERAL() {
  return getToken(RustParser::CHAR_LITERAL, 0);
}

tree::TerminalNode *RustParser::RangePatternBoundContext::BYTE_LITERAL() {
  return getToken(RustParser::BYTE_LITERAL, 0);
}

tree::TerminalNode *RustParser::RangePatternBoundContext::INTEGER_LITERAL() {
  return getToken(RustParser::INTEGER_LITERAL, 0);
}

tree::TerminalNode *RustParser::RangePatternBoundContext::MINUS() {
  return getToken(RustParser::MINUS, 0);
}

tree::TerminalNode *RustParser::RangePatternBoundContext::FLOAT_LITERAL() {
  return getToken(RustParser::FLOAT_LITERAL, 0);
}

RustParser::PathPatternContext *
RustParser::RangePatternBoundContext::pathPattern() {
  return getRuleContext<RustParser::PathPatternContext>(0);
}

size_t RustParser::RangePatternBoundContext::getRuleIndex() const {
  return RustParser::RuleRangePatternBound;
}

void RustParser::RangePatternBoundContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRangePatternBound(this);
}

void RustParser::RangePatternBoundContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRangePatternBound(this);
}

std::any
RustParser::RangePatternBoundContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitRangePatternBound(this);
  else
    return visitor->visitChildren(this);
}

RustParser::RangePatternBoundContext *RustParser::rangePatternBound() {
  RangePatternBoundContext *_localctx =
      _tracker.createInstance<RangePatternBoundContext>(_ctx, getState());
  enterRule(_localctx, 256, RustParser::RuleRangePatternBound);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1916);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 254, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1905);
      match(RustParser::CHAR_LITERAL);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1906);
      match(RustParser::BYTE_LITERAL);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(1908);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::MINUS) {
        setState(1907);
        match(RustParser::MINUS);
      }
      setState(1910);
      match(RustParser::INTEGER_LITERAL);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(1912);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::MINUS) {
        setState(1911);
        match(RustParser::MINUS);
      }
      setState(1914);
      match(RustParser::FLOAT_LITERAL);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(1915);
      pathPattern();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReferencePatternContext
//------------------------------------------------------------------

RustParser::ReferencePatternContext::ReferencePatternContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PatternWithoutRangeContext *
RustParser::ReferencePatternContext::patternWithoutRange() {
  return getRuleContext<RustParser::PatternWithoutRangeContext>(0);
}

tree::TerminalNode *RustParser::ReferencePatternContext::AND() {
  return getToken(RustParser::AND, 0);
}

tree::TerminalNode *RustParser::ReferencePatternContext::ANDAND() {
  return getToken(RustParser::ANDAND, 0);
}

tree::TerminalNode *RustParser::ReferencePatternContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

size_t RustParser::ReferencePatternContext::getRuleIndex() const {
  return RustParser::RuleReferencePattern;
}

void RustParser::ReferencePatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReferencePattern(this);
}

void RustParser::ReferencePatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReferencePattern(this);
}

std::any
RustParser::ReferencePatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitReferencePattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ReferencePatternContext *RustParser::referencePattern() {
  ReferencePatternContext *_localctx =
      _tracker.createInstance<ReferencePatternContext>(_ctx, getState());
  enterRule(_localctx, 258, RustParser::RuleReferencePattern);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1918);
    _la = _input->LA(1);
    if (!(_la == RustParser::AND

          || _la == RustParser::ANDAND)) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(1920);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 255, _ctx)) {
    case 1: {
      setState(1919);
      match(RustParser::KW_MUT);
      break;
    }

    default:
      break;
    }
    setState(1922);
    patternWithoutRange();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructPatternContext
//------------------------------------------------------------------

RustParser::StructPatternContext::StructPatternContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::StructPatternContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

tree::TerminalNode *RustParser::StructPatternContext::LCURLYBRACE() {
  return getToken(RustParser::LCURLYBRACE, 0);
}

tree::TerminalNode *RustParser::StructPatternContext::RCURLYBRACE() {
  return getToken(RustParser::RCURLYBRACE, 0);
}

RustParser::StructPatternElementsContext *
RustParser::StructPatternContext::structPatternElements() {
  return getRuleContext<RustParser::StructPatternElementsContext>(0);
}

size_t RustParser::StructPatternContext::getRuleIndex() const {
  return RustParser::RuleStructPattern;
}

void RustParser::StructPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructPattern(this);
}

void RustParser::StructPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructPattern(this);
}

std::any
RustParser::StructPatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructPatternContext *RustParser::structPattern() {
  StructPatternContext *_localctx =
      _tracker.createInstance<StructPatternContext>(_ctx, getState());
  enterRule(_localctx, 260, RustParser::RuleStructPattern);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1924);
    pathInExpression();
    setState(1925);
    match(RustParser::LCURLYBRACE);
    setState(1927);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 450359962742292480) != 0) ||
        ((((_la - 76) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 76)) & 141012366262273) != 0)) {
      setState(1926);
      structPatternElements();
    }
    setState(1929);
    match(RustParser::RCURLYBRACE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructPatternElementsContext
//------------------------------------------------------------------

RustParser::StructPatternElementsContext::StructPatternElementsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::StructPatternFieldsContext *
RustParser::StructPatternElementsContext::structPatternFields() {
  return getRuleContext<RustParser::StructPatternFieldsContext>(0);
}

tree::TerminalNode *RustParser::StructPatternElementsContext::COMMA() {
  return getToken(RustParser::COMMA, 0);
}

RustParser::StructPatternEtCeteraContext *
RustParser::StructPatternElementsContext::structPatternEtCetera() {
  return getRuleContext<RustParser::StructPatternEtCeteraContext>(0);
}

size_t RustParser::StructPatternElementsContext::getRuleIndex() const {
  return RustParser::RuleStructPatternElements;
}

void RustParser::StructPatternElementsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructPatternElements(this);
}

void RustParser::StructPatternElementsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructPatternElements(this);
}

std::any RustParser::StructPatternElementsContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructPatternElements(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructPatternElementsContext *RustParser::structPatternElements() {
  StructPatternElementsContext *_localctx =
      _tracker.createInstance<StructPatternElementsContext>(_ctx, getState());
  enterRule(_localctx, 262, RustParser::RuleStructPatternElements);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(1939);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 259, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(1931);
      structPatternFields();
      setState(1936);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(1932);
        match(RustParser::COMMA);
        setState(1934);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == RustParser::DOTDOT

            || _la == RustParser::POUND) {
          setState(1933);
          structPatternEtCetera();
        }
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(1938);
      structPatternEtCetera();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructPatternFieldsContext
//------------------------------------------------------------------

RustParser::StructPatternFieldsContext::StructPatternFieldsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::StructPatternFieldContext *>
RustParser::StructPatternFieldsContext::structPatternField() {
  return getRuleContexts<RustParser::StructPatternFieldContext>();
}

RustParser::StructPatternFieldContext *
RustParser::StructPatternFieldsContext::structPatternField(size_t i) {
  return getRuleContext<RustParser::StructPatternFieldContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::StructPatternFieldsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::StructPatternFieldsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::StructPatternFieldsContext::getRuleIndex() const {
  return RustParser::RuleStructPatternFields;
}

void RustParser::StructPatternFieldsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructPatternFields(this);
}

void RustParser::StructPatternFieldsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructPatternFields(this);
}

std::any RustParser::StructPatternFieldsContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructPatternFields(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructPatternFieldsContext *RustParser::structPatternFields() {
  StructPatternFieldsContext *_localctx =
      _tracker.createInstance<StructPatternFieldsContext>(_ctx, getState());
  enterRule(_localctx, 264, RustParser::RuleStructPatternFields);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1941);
    structPatternField();
    setState(1946);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     260, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1942);
        match(RustParser::COMMA);
        setState(1943);
        structPatternField();
      }
      setState(1948);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 260, _ctx);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructPatternFieldContext
//------------------------------------------------------------------

RustParser::StructPatternFieldContext::StructPatternFieldContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::TupleIndexContext *
RustParser::StructPatternFieldContext::tupleIndex() {
  return getRuleContext<RustParser::TupleIndexContext>(0);
}

tree::TerminalNode *RustParser::StructPatternFieldContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::PatternContext *RustParser::StructPatternFieldContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

RustParser::IdentifierContext *
RustParser::StructPatternFieldContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::StructPatternFieldContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::StructPatternFieldContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

tree::TerminalNode *RustParser::StructPatternFieldContext::KW_REF() {
  return getToken(RustParser::KW_REF, 0);
}

tree::TerminalNode *RustParser::StructPatternFieldContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

size_t RustParser::StructPatternFieldContext::getRuleIndex() const {
  return RustParser::RuleStructPatternField;
}

void RustParser::StructPatternFieldContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructPatternField(this);
}

void RustParser::StructPatternFieldContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructPatternField(this);
}

std::any
RustParser::StructPatternFieldContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructPatternField(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructPatternFieldContext *RustParser::structPatternField() {
  StructPatternFieldContext *_localctx =
      _tracker.createInstance<StructPatternFieldContext>(_ctx, getState());
  enterRule(_localctx, 266, RustParser::RuleStructPatternField);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1952);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(1949);
      outerAttribute();
      setState(1954);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1970);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 264, _ctx)) {
    case 1: {
      setState(1955);
      tupleIndex();
      setState(1956);
      match(RustParser::COLON);
      setState(1957);
      pattern();
      break;
    }

    case 2: {
      setState(1959);
      identifier();
      setState(1960);
      match(RustParser::COLON);
      setState(1961);
      pattern();
      break;
    }

    case 3: {
      setState(1964);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::KW_REF) {
        setState(1963);
        match(RustParser::KW_REF);
      }
      setState(1967);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::KW_MUT) {
        setState(1966);
        match(RustParser::KW_MUT);
      }
      setState(1969);
      identifier();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructPatternEtCeteraContext
//------------------------------------------------------------------

RustParser::StructPatternEtCeteraContext::StructPatternEtCeteraContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::StructPatternEtCeteraContext::DOTDOT() {
  return getToken(RustParser::DOTDOT, 0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::StructPatternEtCeteraContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::StructPatternEtCeteraContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

size_t RustParser::StructPatternEtCeteraContext::getRuleIndex() const {
  return RustParser::RuleStructPatternEtCetera;
}

void RustParser::StructPatternEtCeteraContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStructPatternEtCetera(this);
}

void RustParser::StructPatternEtCeteraContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStructPatternEtCetera(this);
}

std::any RustParser::StructPatternEtCeteraContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitStructPatternEtCetera(this);
  else
    return visitor->visitChildren(this);
}

RustParser::StructPatternEtCeteraContext *RustParser::structPatternEtCetera() {
  StructPatternEtCeteraContext *_localctx =
      _tracker.createInstance<StructPatternEtCeteraContext>(_ctx, getState());
  enterRule(_localctx, 268, RustParser::RuleStructPatternEtCetera);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1975);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(1972);
      outerAttribute();
      setState(1977);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1978);
    match(RustParser::DOTDOT);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TupleStructPatternContext
//------------------------------------------------------------------

RustParser::TupleStructPatternContext::TupleStructPatternContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::TupleStructPatternContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

tree::TerminalNode *RustParser::TupleStructPatternContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::TupleStructPatternContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::TupleStructItemsContext *
RustParser::TupleStructPatternContext::tupleStructItems() {
  return getRuleContext<RustParser::TupleStructItemsContext>(0);
}

size_t RustParser::TupleStructPatternContext::getRuleIndex() const {
  return RustParser::RuleTupleStructPattern;
}

void RustParser::TupleStructPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleStructPattern(this);
}

void RustParser::TupleStructPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleStructPattern(this);
}

std::any
RustParser::TupleStructPatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleStructPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TupleStructPatternContext *RustParser::tupleStructPattern() {
  TupleStructPatternContext *_localctx =
      _tracker.createInstance<TupleStructPatternContext>(_ctx, getState());
  enterRule(_localctx, 270, RustParser::RuleTupleStructPattern);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1980);
    pathInExpression();
    setState(1981);
    match(RustParser::LPAREN);
    setState(1983);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 522417558172729888) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1442300045783222399) != 0)) {
      setState(1982);
      tupleStructItems();
    }
    setState(1985);
    match(RustParser::RPAREN);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TupleStructItemsContext
//------------------------------------------------------------------

RustParser::TupleStructItemsContext::TupleStructItemsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::PatternContext *>
RustParser::TupleStructItemsContext::pattern() {
  return getRuleContexts<RustParser::PatternContext>();
}

RustParser::PatternContext *
RustParser::TupleStructItemsContext::pattern(size_t i) {
  return getRuleContext<RustParser::PatternContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::TupleStructItemsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::TupleStructItemsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::TupleStructItemsContext::getRuleIndex() const {
  return RustParser::RuleTupleStructItems;
}

void RustParser::TupleStructItemsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleStructItems(this);
}

void RustParser::TupleStructItemsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleStructItems(this);
}

std::any
RustParser::TupleStructItemsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleStructItems(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TupleStructItemsContext *RustParser::tupleStructItems() {
  TupleStructItemsContext *_localctx =
      _tracker.createInstance<TupleStructItemsContext>(_ctx, getState());
  enterRule(_localctx, 272, RustParser::RuleTupleStructItems);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(1987);
    pattern();
    setState(1992);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     267, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(1988);
        match(RustParser::COMMA);
        setState(1989);
        pattern();
      }
      setState(1994);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 267, _ctx);
    }
    setState(1996);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(1995);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TuplePatternContext
//------------------------------------------------------------------

RustParser::TuplePatternContext::TuplePatternContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::TuplePatternContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::TuplePatternContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::TuplePatternItemsContext *
RustParser::TuplePatternContext::tuplePatternItems() {
  return getRuleContext<RustParser::TuplePatternItemsContext>(0);
}

size_t RustParser::TuplePatternContext::getRuleIndex() const {
  return RustParser::RuleTuplePattern;
}

void RustParser::TuplePatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTuplePattern(this);
}

void RustParser::TuplePatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTuplePattern(this);
}

std::any
RustParser::TuplePatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTuplePattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TuplePatternContext *RustParser::tuplePattern() {
  TuplePatternContext *_localctx =
      _tracker.createInstance<TuplePatternContext>(_ctx, getState());
  enterRule(_localctx, 274, RustParser::RuleTuplePattern);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1998);
    match(RustParser::LPAREN);
    setState(2000);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 522417558172729888) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1442300045783222399) != 0)) {
      setState(1999);
      tuplePatternItems();
    }
    setState(2002);
    match(RustParser::RPAREN);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TuplePatternItemsContext
//------------------------------------------------------------------

RustParser::TuplePatternItemsContext::TuplePatternItemsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::PatternContext *>
RustParser::TuplePatternItemsContext::pattern() {
  return getRuleContexts<RustParser::PatternContext>();
}

RustParser::PatternContext *
RustParser::TuplePatternItemsContext::pattern(size_t i) {
  return getRuleContext<RustParser::PatternContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::TuplePatternItemsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::TuplePatternItemsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

RustParser::RestPatternContext *
RustParser::TuplePatternItemsContext::restPattern() {
  return getRuleContext<RustParser::RestPatternContext>(0);
}

size_t RustParser::TuplePatternItemsContext::getRuleIndex() const {
  return RustParser::RuleTuplePatternItems;
}

void RustParser::TuplePatternItemsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTuplePatternItems(this);
}

void RustParser::TuplePatternItemsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTuplePatternItems(this);
}

std::any
RustParser::TuplePatternItemsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTuplePatternItems(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TuplePatternItemsContext *RustParser::tuplePatternItems() {
  TuplePatternItemsContext *_localctx =
      _tracker.createInstance<TuplePatternItemsContext>(_ctx, getState());
  enterRule(_localctx, 276, RustParser::RuleTuplePatternItems);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(2018);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 272, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(2004);
      pattern();
      setState(2005);
      match(RustParser::COMMA);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(2007);
      restPattern();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(2008);
      pattern();
      setState(2011);
      _errHandler->sync(this);
      alt = 1;
      do {
        switch (alt) {
        case 1: {
          setState(2009);
          match(RustParser::COMMA);
          setState(2010);
          pattern();
          break;
        }

        default:
          throw NoViableAltException(this);
        }
        setState(2013);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 270, _ctx);
      } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
      setState(2016);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(2015);
        match(RustParser::COMMA);
      }
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GroupedPatternContext
//------------------------------------------------------------------

RustParser::GroupedPatternContext::GroupedPatternContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::GroupedPatternContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

RustParser::PatternContext *RustParser::GroupedPatternContext::pattern() {
  return getRuleContext<RustParser::PatternContext>(0);
}

tree::TerminalNode *RustParser::GroupedPatternContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

size_t RustParser::GroupedPatternContext::getRuleIndex() const {
  return RustParser::RuleGroupedPattern;
}

void RustParser::GroupedPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGroupedPattern(this);
}

void RustParser::GroupedPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGroupedPattern(this);
}

std::any
RustParser::GroupedPatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGroupedPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GroupedPatternContext *RustParser::groupedPattern() {
  GroupedPatternContext *_localctx =
      _tracker.createInstance<GroupedPatternContext>(_ctx, getState());
  enterRule(_localctx, 278, RustParser::RuleGroupedPattern);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2020);
    match(RustParser::LPAREN);
    setState(2021);
    pattern();
    setState(2022);
    match(RustParser::RPAREN);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SlicePatternContext
//------------------------------------------------------------------

RustParser::SlicePatternContext::SlicePatternContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::SlicePatternContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

tree::TerminalNode *RustParser::SlicePatternContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

RustParser::SlicePatternItemsContext *
RustParser::SlicePatternContext::slicePatternItems() {
  return getRuleContext<RustParser::SlicePatternItemsContext>(0);
}

size_t RustParser::SlicePatternContext::getRuleIndex() const {
  return RustParser::RuleSlicePattern;
}

void RustParser::SlicePatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSlicePattern(this);
}

void RustParser::SlicePatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSlicePattern(this);
}

std::any
RustParser::SlicePatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitSlicePattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::SlicePatternContext *RustParser::slicePattern() {
  SlicePatternContext *_localctx =
      _tracker.createInstance<SlicePatternContext>(_ctx, getState());
  enterRule(_localctx, 280, RustParser::RuleSlicePattern);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2024);
    match(RustParser::LSQUAREBRACKET);
    setState(2026);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 522417558172729888) != 0) ||
        ((((_la - 70) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 70)) & 1442300045783222399) != 0)) {
      setState(2025);
      slicePatternItems();
    }
    setState(2028);
    match(RustParser::RSQUAREBRACKET);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SlicePatternItemsContext
//------------------------------------------------------------------

RustParser::SlicePatternItemsContext::SlicePatternItemsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::PatternContext *>
RustParser::SlicePatternItemsContext::pattern() {
  return getRuleContexts<RustParser::PatternContext>();
}

RustParser::PatternContext *
RustParser::SlicePatternItemsContext::pattern(size_t i) {
  return getRuleContext<RustParser::PatternContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::SlicePatternItemsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::SlicePatternItemsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::SlicePatternItemsContext::getRuleIndex() const {
  return RustParser::RuleSlicePatternItems;
}

void RustParser::SlicePatternItemsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSlicePatternItems(this);
}

void RustParser::SlicePatternItemsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSlicePatternItems(this);
}

std::any
RustParser::SlicePatternItemsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitSlicePatternItems(this);
  else
    return visitor->visitChildren(this);
}

RustParser::SlicePatternItemsContext *RustParser::slicePatternItems() {
  SlicePatternItemsContext *_localctx =
      _tracker.createInstance<SlicePatternItemsContext>(_ctx, getState());
  enterRule(_localctx, 282, RustParser::RuleSlicePatternItems);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2030);
    pattern();
    setState(2035);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     274, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2031);
        match(RustParser::COMMA);
        setState(2032);
        pattern();
      }
      setState(2037);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 274, _ctx);
    }
    setState(2039);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(2038);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PathPatternContext
//------------------------------------------------------------------

RustParser::PathPatternContext::PathPatternContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathInExpressionContext *
RustParser::PathPatternContext::pathInExpression() {
  return getRuleContext<RustParser::PathInExpressionContext>(0);
}

RustParser::QualifiedPathInExpressionContext *
RustParser::PathPatternContext::qualifiedPathInExpression() {
  return getRuleContext<RustParser::QualifiedPathInExpressionContext>(0);
}

size_t RustParser::PathPatternContext::getRuleIndex() const {
  return RustParser::RulePathPattern;
}

void RustParser::PathPatternContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPathPattern(this);
}

void RustParser::PathPatternContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPathPattern(this);
}

std::any
RustParser::PathPatternContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPathPattern(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PathPatternContext *RustParser::pathPattern() {
  PathPatternContext *_localctx =
      _tracker.createInstance<PathPatternContext>(_ctx, getState());
  enterRule(_localctx, 284, RustParser::RulePathPattern);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2043);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_CRATE:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SELFTYPE:
    case RustParser::KW_SUPER:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::PATHSEP: {
      enterOuterAlt(_localctx, 1);
      setState(2041);
      pathInExpression();
      break;
    }

    case RustParser::LT: {
      enterOuterAlt(_localctx, 2);
      setState(2042);
      qualifiedPathInExpression();
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Type_Context
//------------------------------------------------------------------

RustParser::Type_Context::Type_Context(ParserRuleContext *parent,
                                       size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::TypeNoBoundsContext *RustParser::Type_Context::typeNoBounds() {
  return getRuleContext<RustParser::TypeNoBoundsContext>(0);
}

RustParser::ImplTraitTypeContext *RustParser::Type_Context::implTraitType() {
  return getRuleContext<RustParser::ImplTraitTypeContext>(0);
}

RustParser::TraitObjectTypeContext *
RustParser::Type_Context::traitObjectType() {
  return getRuleContext<RustParser::TraitObjectTypeContext>(0);
}

size_t RustParser::Type_Context::getRuleIndex() const {
  return RustParser::RuleType_;
}

void RustParser::Type_Context::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterType_(this);
}

void RustParser::Type_Context::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitType_(this);
}

std::any RustParser::Type_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitType_(this);
  else
    return visitor->visitChildren(this);
}

RustParser::Type_Context *RustParser::type_() {
  Type_Context *_localctx =
      _tracker.createInstance<Type_Context>(_ctx, getState());
  enterRule(_localctx, 286, RustParser::RuleType_);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2048);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 277, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(2045);
      typeNoBounds();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(2046);
      implTraitType();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(2047);
      traitObjectType();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeNoBoundsContext
//------------------------------------------------------------------

RustParser::TypeNoBoundsContext::TypeNoBoundsContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::ParenthesizedTypeContext *
RustParser::TypeNoBoundsContext::parenthesizedType() {
  return getRuleContext<RustParser::ParenthesizedTypeContext>(0);
}

RustParser::ImplTraitTypeOneBoundContext *
RustParser::TypeNoBoundsContext::implTraitTypeOneBound() {
  return getRuleContext<RustParser::ImplTraitTypeOneBoundContext>(0);
}

RustParser::TraitObjectTypeOneBoundContext *
RustParser::TypeNoBoundsContext::traitObjectTypeOneBound() {
  return getRuleContext<RustParser::TraitObjectTypeOneBoundContext>(0);
}

RustParser::TypePathContext *RustParser::TypeNoBoundsContext::typePath() {
  return getRuleContext<RustParser::TypePathContext>(0);
}

RustParser::TupleTypeContext *RustParser::TypeNoBoundsContext::tupleType() {
  return getRuleContext<RustParser::TupleTypeContext>(0);
}

RustParser::NeverTypeContext *RustParser::TypeNoBoundsContext::neverType() {
  return getRuleContext<RustParser::NeverTypeContext>(0);
}

RustParser::RawPointerTypeContext *
RustParser::TypeNoBoundsContext::rawPointerType() {
  return getRuleContext<RustParser::RawPointerTypeContext>(0);
}

RustParser::ReferenceTypeContext *
RustParser::TypeNoBoundsContext::referenceType() {
  return getRuleContext<RustParser::ReferenceTypeContext>(0);
}

RustParser::ArrayTypeContext *RustParser::TypeNoBoundsContext::arrayType() {
  return getRuleContext<RustParser::ArrayTypeContext>(0);
}

RustParser::SliceTypeContext *RustParser::TypeNoBoundsContext::sliceType() {
  return getRuleContext<RustParser::SliceTypeContext>(0);
}

RustParser::InferredTypeContext *
RustParser::TypeNoBoundsContext::inferredType() {
  return getRuleContext<RustParser::InferredTypeContext>(0);
}

RustParser::QualifiedPathInTypeContext *
RustParser::TypeNoBoundsContext::qualifiedPathInType() {
  return getRuleContext<RustParser::QualifiedPathInTypeContext>(0);
}

RustParser::BareFunctionTypeContext *
RustParser::TypeNoBoundsContext::bareFunctionType() {
  return getRuleContext<RustParser::BareFunctionTypeContext>(0);
}

RustParser::MacroInvocationContext *
RustParser::TypeNoBoundsContext::macroInvocation() {
  return getRuleContext<RustParser::MacroInvocationContext>(0);
}

size_t RustParser::TypeNoBoundsContext::getRuleIndex() const {
  return RustParser::RuleTypeNoBounds;
}

void RustParser::TypeNoBoundsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeNoBounds(this);
}

void RustParser::TypeNoBoundsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeNoBounds(this);
}

std::any
RustParser::TypeNoBoundsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypeNoBounds(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypeNoBoundsContext *RustParser::typeNoBounds() {
  TypeNoBoundsContext *_localctx =
      _tracker.createInstance<TypeNoBoundsContext>(_ctx, getState());
  enterRule(_localctx, 288, RustParser::RuleTypeNoBounds);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2064);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 278, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(2050);
      parenthesizedType();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(2051);
      implTraitTypeOneBound();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(2052);
      traitObjectTypeOneBound();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(2053);
      typePath();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(2054);
      tupleType();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(2055);
      neverType();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(2056);
      rawPointerType();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(2057);
      referenceType();
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(2058);
      arrayType();
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(2059);
      sliceType();
      break;
    }

    case 11: {
      enterOuterAlt(_localctx, 11);
      setState(2060);
      inferredType();
      break;
    }

    case 12: {
      enterOuterAlt(_localctx, 12);
      setState(2061);
      qualifiedPathInType();
      break;
    }

    case 13: {
      enterOuterAlt(_localctx, 13);
      setState(2062);
      bareFunctionType();
      break;
    }

    case 14: {
      enterOuterAlt(_localctx, 14);
      setState(2063);
      macroInvocation();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ParenthesizedTypeContext
//------------------------------------------------------------------

RustParser::ParenthesizedTypeContext::ParenthesizedTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ParenthesizedTypeContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

RustParser::Type_Context *RustParser::ParenthesizedTypeContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::ParenthesizedTypeContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

size_t RustParser::ParenthesizedTypeContext::getRuleIndex() const {
  return RustParser::RuleParenthesizedType;
}

void RustParser::ParenthesizedTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParenthesizedType(this);
}

void RustParser::ParenthesizedTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParenthesizedType(this);
}

std::any
RustParser::ParenthesizedTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitParenthesizedType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ParenthesizedTypeContext *RustParser::parenthesizedType() {
  ParenthesizedTypeContext *_localctx =
      _tracker.createInstance<ParenthesizedTypeContext>(_ctx, getState());
  enterRule(_localctx, 290, RustParser::RuleParenthesizedType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2066);
    match(RustParser::LPAREN);
    setState(2067);
    type_();
    setState(2068);
    match(RustParser::RPAREN);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NeverTypeContext
//------------------------------------------------------------------

RustParser::NeverTypeContext::NeverTypeContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::NeverTypeContext::NOT() {
  return getToken(RustParser::NOT, 0);
}

size_t RustParser::NeverTypeContext::getRuleIndex() const {
  return RustParser::RuleNeverType;
}

void RustParser::NeverTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNeverType(this);
}

void RustParser::NeverTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNeverType(this);
}

std::any RustParser::NeverTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitNeverType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::NeverTypeContext *RustParser::neverType() {
  NeverTypeContext *_localctx =
      _tracker.createInstance<NeverTypeContext>(_ctx, getState());
  enterRule(_localctx, 292, RustParser::RuleNeverType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2070);
    match(RustParser::NOT);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TupleTypeContext
//------------------------------------------------------------------

RustParser::TupleTypeContext::TupleTypeContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::TupleTypeContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::TupleTypeContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

std::vector<RustParser::Type_Context *> RustParser::TupleTypeContext::type_() {
  return getRuleContexts<RustParser::Type_Context>();
}

RustParser::Type_Context *RustParser::TupleTypeContext::type_(size_t i) {
  return getRuleContext<RustParser::Type_Context>(i);
}

std::vector<tree::TerminalNode *> RustParser::TupleTypeContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::TupleTypeContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::TupleTypeContext::getRuleIndex() const {
  return RustParser::RuleTupleType;
}

void RustParser::TupleTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTupleType(this);
}

void RustParser::TupleTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTupleType(this);
}

std::any RustParser::TupleTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTupleType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TupleTypeContext *RustParser::tupleType() {
  TupleTypeContext *_localctx =
      _tracker.createInstance<TupleTypeContext>(_ctx, getState());
  enterRule(_localctx, 294, RustParser::RuleTupleType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2072);
    match(RustParser::LPAREN);
    setState(2083);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 567453832540335392) != 0) ||
        ((((_la - 82) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 82)) & 360915832668553) != 0)) {
      setState(2076);
      _errHandler->sync(this);
      alt = 1;
      do {
        switch (alt) {
        case 1: {
          setState(2073);
          type_();
          setState(2074);
          match(RustParser::COMMA);
          break;
        }

        default:
          throw NoViableAltException(this);
        }
        setState(2078);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 279, _ctx);
      } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
      setState(2081);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 567453832540335392) != 0) ||
          ((((_la - 82) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 82)) & 360915832668553) != 0)) {
        setState(2080);
        type_();
      }
    }
    setState(2085);
    match(RustParser::RPAREN);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArrayTypeContext
//------------------------------------------------------------------

RustParser::ArrayTypeContext::ArrayTypeContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ArrayTypeContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

RustParser::Type_Context *RustParser::ArrayTypeContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::ArrayTypeContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

RustParser::ExpressionContext *RustParser::ArrayTypeContext::expression() {
  return getRuleContext<RustParser::ExpressionContext>(0);
}

tree::TerminalNode *RustParser::ArrayTypeContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

size_t RustParser::ArrayTypeContext::getRuleIndex() const {
  return RustParser::RuleArrayType;
}

void RustParser::ArrayTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArrayType(this);
}

void RustParser::ArrayTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArrayType(this);
}

std::any RustParser::ArrayTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitArrayType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ArrayTypeContext *RustParser::arrayType() {
  ArrayTypeContext *_localctx =
      _tracker.createInstance<ArrayTypeContext>(_ctx, getState());
  enterRule(_localctx, 296, RustParser::RuleArrayType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2087);
    match(RustParser::LSQUAREBRACKET);
    setState(2088);
    type_();
    setState(2089);
    match(RustParser::SEMI);
    setState(2090);
    expression(0);
    setState(2091);
    match(RustParser::RSQUAREBRACKET);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SliceTypeContext
//------------------------------------------------------------------

RustParser::SliceTypeContext::SliceTypeContext(ParserRuleContext *parent,
                                               size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::SliceTypeContext::LSQUAREBRACKET() {
  return getToken(RustParser::LSQUAREBRACKET, 0);
}

RustParser::Type_Context *RustParser::SliceTypeContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::SliceTypeContext::RSQUAREBRACKET() {
  return getToken(RustParser::RSQUAREBRACKET, 0);
}

size_t RustParser::SliceTypeContext::getRuleIndex() const {
  return RustParser::RuleSliceType;
}

void RustParser::SliceTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSliceType(this);
}

void RustParser::SliceTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSliceType(this);
}

std::any RustParser::SliceTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitSliceType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::SliceTypeContext *RustParser::sliceType() {
  SliceTypeContext *_localctx =
      _tracker.createInstance<SliceTypeContext>(_ctx, getState());
  enterRule(_localctx, 298, RustParser::RuleSliceType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2093);
    match(RustParser::LSQUAREBRACKET);
    setState(2094);
    type_();
    setState(2095);
    match(RustParser::RSQUAREBRACKET);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReferenceTypeContext
//------------------------------------------------------------------

RustParser::ReferenceTypeContext::ReferenceTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ReferenceTypeContext::AND() {
  return getToken(RustParser::AND, 0);
}

RustParser::TypeNoBoundsContext *
RustParser::ReferenceTypeContext::typeNoBounds() {
  return getRuleContext<RustParser::TypeNoBoundsContext>(0);
}

RustParser::LifetimeContext *RustParser::ReferenceTypeContext::lifetime() {
  return getRuleContext<RustParser::LifetimeContext>(0);
}

tree::TerminalNode *RustParser::ReferenceTypeContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

size_t RustParser::ReferenceTypeContext::getRuleIndex() const {
  return RustParser::RuleReferenceType;
}

void RustParser::ReferenceTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReferenceType(this);
}

void RustParser::ReferenceTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReferenceType(this);
}

std::any
RustParser::ReferenceTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitReferenceType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ReferenceTypeContext *RustParser::referenceType() {
  ReferenceTypeContext *_localctx =
      _tracker.createInstance<ReferenceTypeContext>(_ctx, getState());
  enterRule(_localctx, 300, RustParser::RuleReferenceType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2097);
    match(RustParser::AND);
    setState(2099);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 53) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 53)) & 536870917) != 0)) {
      setState(2098);
      lifetime();
    }
    setState(2102);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_MUT) {
      setState(2101);
      match(RustParser::KW_MUT);
    }
    setState(2104);
    typeNoBounds();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RawPointerTypeContext
//------------------------------------------------------------------

RustParser::RawPointerTypeContext::RawPointerTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::RawPointerTypeContext::STAR() {
  return getToken(RustParser::STAR, 0);
}

RustParser::TypeNoBoundsContext *
RustParser::RawPointerTypeContext::typeNoBounds() {
  return getRuleContext<RustParser::TypeNoBoundsContext>(0);
}

tree::TerminalNode *RustParser::RawPointerTypeContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

tree::TerminalNode *RustParser::RawPointerTypeContext::KW_CONST() {
  return getToken(RustParser::KW_CONST, 0);
}

size_t RustParser::RawPointerTypeContext::getRuleIndex() const {
  return RustParser::RuleRawPointerType;
}

void RustParser::RawPointerTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRawPointerType(this);
}

void RustParser::RawPointerTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRawPointerType(this);
}

std::any
RustParser::RawPointerTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitRawPointerType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::RawPointerTypeContext *RustParser::rawPointerType() {
  RawPointerTypeContext *_localctx =
      _tracker.createInstance<RawPointerTypeContext>(_ctx, getState());
  enterRule(_localctx, 302, RustParser::RuleRawPointerType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2106);
    match(RustParser::STAR);
    setState(2107);
    _la = _input->LA(1);
    if (!(_la == RustParser::KW_CONST

          || _la == RustParser::KW_MUT)) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(2108);
    typeNoBounds();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BareFunctionTypeContext
//------------------------------------------------------------------

RustParser::BareFunctionTypeContext::BareFunctionTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::FunctionTypeQualifiersContext *
RustParser::BareFunctionTypeContext::functionTypeQualifiers() {
  return getRuleContext<RustParser::FunctionTypeQualifiersContext>(0);
}

tree::TerminalNode *RustParser::BareFunctionTypeContext::KW_FN() {
  return getToken(RustParser::KW_FN, 0);
}

tree::TerminalNode *RustParser::BareFunctionTypeContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::BareFunctionTypeContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::ForLifetimesContext *
RustParser::BareFunctionTypeContext::forLifetimes() {
  return getRuleContext<RustParser::ForLifetimesContext>(0);
}

RustParser::FunctionParametersMaybeNamedVariadicContext *
RustParser::BareFunctionTypeContext::functionParametersMaybeNamedVariadic() {
  return getRuleContext<
      RustParser::FunctionParametersMaybeNamedVariadicContext>(0);
}

RustParser::BareFunctionReturnTypeContext *
RustParser::BareFunctionTypeContext::bareFunctionReturnType() {
  return getRuleContext<RustParser::BareFunctionReturnTypeContext>(0);
}

size_t RustParser::BareFunctionTypeContext::getRuleIndex() const {
  return RustParser::RuleBareFunctionType;
}

void RustParser::BareFunctionTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBareFunctionType(this);
}

void RustParser::BareFunctionTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBareFunctionType(this);
}

std::any
RustParser::BareFunctionTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitBareFunctionType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::BareFunctionTypeContext *RustParser::bareFunctionType() {
  BareFunctionTypeContext *_localctx =
      _tracker.createInstance<BareFunctionTypeContext>(_ctx, getState());
  enterRule(_localctx, 304, RustParser::RuleBareFunctionType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2111);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_FOR) {
      setState(2110);
      forLifetimes();
    }
    setState(2113);
    functionTypeQualifiers();
    setState(2114);
    match(RustParser::KW_FN);
    setState(2115);
    match(RustParser::LPAREN);
    setState(2117);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 567453832540335392) != 0) ||
        ((((_la - 82) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 82)) & 363114855924105) != 0)) {
      setState(2116);
      functionParametersMaybeNamedVariadic();
    }
    setState(2119);
    match(RustParser::RPAREN);
    setState(2121);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 286, _ctx)) {
    case 1: {
      setState(2120);
      bareFunctionReturnType();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionTypeQualifiersContext
//------------------------------------------------------------------

RustParser::FunctionTypeQualifiersContext::FunctionTypeQualifiersContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::FunctionTypeQualifiersContext::KW_UNSAFE() {
  return getToken(RustParser::KW_UNSAFE, 0);
}

tree::TerminalNode *RustParser::FunctionTypeQualifiersContext::KW_EXTERN() {
  return getToken(RustParser::KW_EXTERN, 0);
}

RustParser::AbiContext *RustParser::FunctionTypeQualifiersContext::abi() {
  return getRuleContext<RustParser::AbiContext>(0);
}

size_t RustParser::FunctionTypeQualifiersContext::getRuleIndex() const {
  return RustParser::RuleFunctionTypeQualifiers;
}

void RustParser::FunctionTypeQualifiersContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionTypeQualifiers(this);
}

void RustParser::FunctionTypeQualifiersContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionTypeQualifiers(this);
}

std::any RustParser::FunctionTypeQualifiersContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFunctionTypeQualifiers(this);
  else
    return visitor->visitChildren(this);
}

RustParser::FunctionTypeQualifiersContext *
RustParser::functionTypeQualifiers() {
  FunctionTypeQualifiersContext *_localctx =
      _tracker.createInstance<FunctionTypeQualifiersContext>(_ctx, getState());
  enterRule(_localctx, 306, RustParser::RuleFunctionTypeQualifiers);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2124);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_UNSAFE) {
      setState(2123);
      match(RustParser::KW_UNSAFE);
    }
    setState(2130);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_EXTERN) {
      setState(2126);
      match(RustParser::KW_EXTERN);
      setState(2128);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::STRING_LITERAL

          || _la == RustParser::RAW_STRING_LITERAL) {
        setState(2127);
        abi();
      }
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BareFunctionReturnTypeContext
//------------------------------------------------------------------

RustParser::BareFunctionReturnTypeContext::BareFunctionReturnTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::BareFunctionReturnTypeContext::RARROW() {
  return getToken(RustParser::RARROW, 0);
}

RustParser::TypeNoBoundsContext *
RustParser::BareFunctionReturnTypeContext::typeNoBounds() {
  return getRuleContext<RustParser::TypeNoBoundsContext>(0);
}

size_t RustParser::BareFunctionReturnTypeContext::getRuleIndex() const {
  return RustParser::RuleBareFunctionReturnType;
}

void RustParser::BareFunctionReturnTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBareFunctionReturnType(this);
}

void RustParser::BareFunctionReturnTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBareFunctionReturnType(this);
}

std::any RustParser::BareFunctionReturnTypeContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitBareFunctionReturnType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::BareFunctionReturnTypeContext *
RustParser::bareFunctionReturnType() {
  BareFunctionReturnTypeContext *_localctx =
      _tracker.createInstance<BareFunctionReturnTypeContext>(_ctx, getState());
  enterRule(_localctx, 308, RustParser::RuleBareFunctionReturnType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2132);
    match(RustParser::RARROW);
    setState(2133);
    typeNoBounds();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionParametersMaybeNamedVariadicContext
//------------------------------------------------------------------

RustParser::FunctionParametersMaybeNamedVariadicContext::
    FunctionParametersMaybeNamedVariadicContext(ParserRuleContext *parent,
                                                size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::MaybeNamedFunctionParametersContext *
RustParser::FunctionParametersMaybeNamedVariadicContext::
    maybeNamedFunctionParameters() {
  return getRuleContext<RustParser::MaybeNamedFunctionParametersContext>(0);
}

RustParser::MaybeNamedFunctionParametersVariadicContext *
RustParser::FunctionParametersMaybeNamedVariadicContext::
    maybeNamedFunctionParametersVariadic() {
  return getRuleContext<
      RustParser::MaybeNamedFunctionParametersVariadicContext>(0);
}

size_t
RustParser::FunctionParametersMaybeNamedVariadicContext::getRuleIndex() const {
  return RustParser::RuleFunctionParametersMaybeNamedVariadic;
}

void RustParser::FunctionParametersMaybeNamedVariadicContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionParametersMaybeNamedVariadic(this);
}

void RustParser::FunctionParametersMaybeNamedVariadicContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionParametersMaybeNamedVariadic(this);
}

std::any RustParser::FunctionParametersMaybeNamedVariadicContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitFunctionParametersMaybeNamedVariadic(this);
  else
    return visitor->visitChildren(this);
}

RustParser::FunctionParametersMaybeNamedVariadicContext *
RustParser::functionParametersMaybeNamedVariadic() {
  FunctionParametersMaybeNamedVariadicContext *_localctx =
      _tracker.createInstance<FunctionParametersMaybeNamedVariadicContext>(
          _ctx, getState());
  enterRule(_localctx, 310,
            RustParser::RuleFunctionParametersMaybeNamedVariadic);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2137);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 290, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(2135);
      maybeNamedFunctionParameters();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(2136);
      maybeNamedFunctionParametersVariadic();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MaybeNamedFunctionParametersContext
//------------------------------------------------------------------

RustParser::MaybeNamedFunctionParametersContext::
    MaybeNamedFunctionParametersContext(ParserRuleContext *parent,
                                        size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::MaybeNamedParamContext *>
RustParser::MaybeNamedFunctionParametersContext::maybeNamedParam() {
  return getRuleContexts<RustParser::MaybeNamedParamContext>();
}

RustParser::MaybeNamedParamContext *
RustParser::MaybeNamedFunctionParametersContext::maybeNamedParam(size_t i) {
  return getRuleContext<RustParser::MaybeNamedParamContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::MaybeNamedFunctionParametersContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *
RustParser::MaybeNamedFunctionParametersContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::MaybeNamedFunctionParametersContext::getRuleIndex() const {
  return RustParser::RuleMaybeNamedFunctionParameters;
}

void RustParser::MaybeNamedFunctionParametersContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMaybeNamedFunctionParameters(this);
}

void RustParser::MaybeNamedFunctionParametersContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMaybeNamedFunctionParameters(this);
}

std::any RustParser::MaybeNamedFunctionParametersContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMaybeNamedFunctionParameters(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MaybeNamedFunctionParametersContext *
RustParser::maybeNamedFunctionParameters() {
  MaybeNamedFunctionParametersContext *_localctx =
      _tracker.createInstance<MaybeNamedFunctionParametersContext>(_ctx,
                                                                   getState());
  enterRule(_localctx, 312, RustParser::RuleMaybeNamedFunctionParameters);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2139);
    maybeNamedParam();
    setState(2144);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     291, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2140);
        match(RustParser::COMMA);
        setState(2141);
        maybeNamedParam();
      }
      setState(2146);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 291, _ctx);
    }
    setState(2148);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(2147);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MaybeNamedParamContext
//------------------------------------------------------------------

RustParser::MaybeNamedParamContext::MaybeNamedParamContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::Type_Context *RustParser::MaybeNamedParamContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::MaybeNamedParamContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::MaybeNamedParamContext::outerAttribute(size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

tree::TerminalNode *RustParser::MaybeNamedParamContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

RustParser::IdentifierContext *
RustParser::MaybeNamedParamContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::MaybeNamedParamContext::UNDERSCORE() {
  return getToken(RustParser::UNDERSCORE, 0);
}

size_t RustParser::MaybeNamedParamContext::getRuleIndex() const {
  return RustParser::RuleMaybeNamedParam;
}

void RustParser::MaybeNamedParamContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMaybeNamedParam(this);
}

void RustParser::MaybeNamedParamContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMaybeNamedParam(this);
}

std::any
RustParser::MaybeNamedParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMaybeNamedParam(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MaybeNamedParamContext *RustParser::maybeNamedParam() {
  MaybeNamedParamContext *_localctx =
      _tracker.createInstance<MaybeNamedParamContext>(_ctx, getState());
  enterRule(_localctx, 314, RustParser::RuleMaybeNamedParam);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2153);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(2150);
      outerAttribute();
      setState(2155);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(2161);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 295, _ctx)) {
    case 1: {
      setState(2158);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
      case RustParser::KW_MACRORULES:
      case RustParser::NON_KEYWORD_IDENTIFIER:
      case RustParser::RAW_IDENTIFIER: {
        setState(2156);
        identifier();
        break;
      }

      case RustParser::UNDERSCORE: {
        setState(2157);
        match(RustParser::UNDERSCORE);
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      setState(2160);
      match(RustParser::COLON);
      break;
    }

    default:
      break;
    }
    setState(2163);
    type_();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MaybeNamedFunctionParametersVariadicContext
//------------------------------------------------------------------

RustParser::MaybeNamedFunctionParametersVariadicContext::
    MaybeNamedFunctionParametersVariadicContext(ParserRuleContext *parent,
                                                size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::MaybeNamedParamContext *>
RustParser::MaybeNamedFunctionParametersVariadicContext::maybeNamedParam() {
  return getRuleContexts<RustParser::MaybeNamedParamContext>();
}

RustParser::MaybeNamedParamContext *
RustParser::MaybeNamedFunctionParametersVariadicContext::maybeNamedParam(
    size_t i) {
  return getRuleContext<RustParser::MaybeNamedParamContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::MaybeNamedFunctionParametersVariadicContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *
RustParser::MaybeNamedFunctionParametersVariadicContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

tree::TerminalNode *
RustParser::MaybeNamedFunctionParametersVariadicContext::DOTDOTDOT() {
  return getToken(RustParser::DOTDOTDOT, 0);
}

std::vector<RustParser::OuterAttributeContext *>
RustParser::MaybeNamedFunctionParametersVariadicContext::outerAttribute() {
  return getRuleContexts<RustParser::OuterAttributeContext>();
}

RustParser::OuterAttributeContext *
RustParser::MaybeNamedFunctionParametersVariadicContext::outerAttribute(
    size_t i) {
  return getRuleContext<RustParser::OuterAttributeContext>(i);
}

size_t
RustParser::MaybeNamedFunctionParametersVariadicContext::getRuleIndex() const {
  return RustParser::RuleMaybeNamedFunctionParametersVariadic;
}

void RustParser::MaybeNamedFunctionParametersVariadicContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMaybeNamedFunctionParametersVariadic(this);
}

void RustParser::MaybeNamedFunctionParametersVariadicContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMaybeNamedFunctionParametersVariadic(this);
}

std::any RustParser::MaybeNamedFunctionParametersVariadicContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMaybeNamedFunctionParametersVariadic(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MaybeNamedFunctionParametersVariadicContext *
RustParser::maybeNamedFunctionParametersVariadic() {
  MaybeNamedFunctionParametersVariadicContext *_localctx =
      _tracker.createInstance<MaybeNamedFunctionParametersVariadicContext>(
          _ctx, getState());
  enterRule(_localctx, 316,
            RustParser::RuleMaybeNamedFunctionParametersVariadic);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2170);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     296, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2165);
        maybeNamedParam();
        setState(2166);
        match(RustParser::COMMA);
      }
      setState(2172);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 296, _ctx);
    }
    setState(2173);
    maybeNamedParam();
    setState(2174);
    match(RustParser::COMMA);
    setState(2178);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == RustParser::POUND) {
      setState(2175);
      outerAttribute();
      setState(2180);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(2181);
    match(RustParser::DOTDOTDOT);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TraitObjectTypeContext
//------------------------------------------------------------------

RustParser::TraitObjectTypeContext::TraitObjectTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::TypeParamBoundsContext *
RustParser::TraitObjectTypeContext::typeParamBounds() {
  return getRuleContext<RustParser::TypeParamBoundsContext>(0);
}

tree::TerminalNode *RustParser::TraitObjectTypeContext::KW_DYN() {
  return getToken(RustParser::KW_DYN, 0);
}

size_t RustParser::TraitObjectTypeContext::getRuleIndex() const {
  return RustParser::RuleTraitObjectType;
}

void RustParser::TraitObjectTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTraitObjectType(this);
}

void RustParser::TraitObjectTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTraitObjectType(this);
}

std::any
RustParser::TraitObjectTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTraitObjectType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TraitObjectTypeContext *RustParser::traitObjectType() {
  TraitObjectTypeContext *_localctx =
      _tracker.createInstance<TraitObjectTypeContext>(_ctx, getState());
  enterRule(_localctx, 318, RustParser::RuleTraitObjectType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2184);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_DYN) {
      setState(2183);
      match(RustParser::KW_DYN);
    }
    setState(2186);
    typeParamBounds();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TraitObjectTypeOneBoundContext
//------------------------------------------------------------------

RustParser::TraitObjectTypeOneBoundContext::TraitObjectTypeOneBoundContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::TraitBoundContext *
RustParser::TraitObjectTypeOneBoundContext::traitBound() {
  return getRuleContext<RustParser::TraitBoundContext>(0);
}

tree::TerminalNode *RustParser::TraitObjectTypeOneBoundContext::KW_DYN() {
  return getToken(RustParser::KW_DYN, 0);
}

size_t RustParser::TraitObjectTypeOneBoundContext::getRuleIndex() const {
  return RustParser::RuleTraitObjectTypeOneBound;
}

void RustParser::TraitObjectTypeOneBoundContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTraitObjectTypeOneBound(this);
}

void RustParser::TraitObjectTypeOneBoundContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTraitObjectTypeOneBound(this);
}

std::any RustParser::TraitObjectTypeOneBoundContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTraitObjectTypeOneBound(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TraitObjectTypeOneBoundContext *
RustParser::traitObjectTypeOneBound() {
  TraitObjectTypeOneBoundContext *_localctx =
      _tracker.createInstance<TraitObjectTypeOneBoundContext>(_ctx, getState());
  enterRule(_localctx, 320, RustParser::RuleTraitObjectTypeOneBound);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2189);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_DYN) {
      setState(2188);
      match(RustParser::KW_DYN);
    }
    setState(2191);
    traitBound();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ImplTraitTypeContext
//------------------------------------------------------------------

RustParser::ImplTraitTypeContext::ImplTraitTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ImplTraitTypeContext::KW_IMPL() {
  return getToken(RustParser::KW_IMPL, 0);
}

RustParser::TypeParamBoundsContext *
RustParser::ImplTraitTypeContext::typeParamBounds() {
  return getRuleContext<RustParser::TypeParamBoundsContext>(0);
}

size_t RustParser::ImplTraitTypeContext::getRuleIndex() const {
  return RustParser::RuleImplTraitType;
}

void RustParser::ImplTraitTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterImplTraitType(this);
}

void RustParser::ImplTraitTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitImplTraitType(this);
}

std::any
RustParser::ImplTraitTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitImplTraitType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ImplTraitTypeContext *RustParser::implTraitType() {
  ImplTraitTypeContext *_localctx =
      _tracker.createInstance<ImplTraitTypeContext>(_ctx, getState());
  enterRule(_localctx, 322, RustParser::RuleImplTraitType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2193);
    match(RustParser::KW_IMPL);
    setState(2194);
    typeParamBounds();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ImplTraitTypeOneBoundContext
//------------------------------------------------------------------

RustParser::ImplTraitTypeOneBoundContext::ImplTraitTypeOneBoundContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::ImplTraitTypeOneBoundContext::KW_IMPL() {
  return getToken(RustParser::KW_IMPL, 0);
}

RustParser::TraitBoundContext *
RustParser::ImplTraitTypeOneBoundContext::traitBound() {
  return getRuleContext<RustParser::TraitBoundContext>(0);
}

size_t RustParser::ImplTraitTypeOneBoundContext::getRuleIndex() const {
  return RustParser::RuleImplTraitTypeOneBound;
}

void RustParser::ImplTraitTypeOneBoundContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterImplTraitTypeOneBound(this);
}

void RustParser::ImplTraitTypeOneBoundContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitImplTraitTypeOneBound(this);
}

std::any RustParser::ImplTraitTypeOneBoundContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitImplTraitTypeOneBound(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ImplTraitTypeOneBoundContext *RustParser::implTraitTypeOneBound() {
  ImplTraitTypeOneBoundContext *_localctx =
      _tracker.createInstance<ImplTraitTypeOneBoundContext>(_ctx, getState());
  enterRule(_localctx, 324, RustParser::RuleImplTraitTypeOneBound);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2196);
    match(RustParser::KW_IMPL);
    setState(2197);
    traitBound();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- InferredTypeContext
//------------------------------------------------------------------

RustParser::InferredTypeContext::InferredTypeContext(ParserRuleContext *parent,
                                                     size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::InferredTypeContext::UNDERSCORE() {
  return getToken(RustParser::UNDERSCORE, 0);
}

size_t RustParser::InferredTypeContext::getRuleIndex() const {
  return RustParser::RuleInferredType;
}

void RustParser::InferredTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInferredType(this);
}

void RustParser::InferredTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInferredType(this);
}

std::any
RustParser::InferredTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitInferredType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::InferredTypeContext *RustParser::inferredType() {
  InferredTypeContext *_localctx =
      _tracker.createInstance<InferredTypeContext>(_ctx, getState());
  enterRule(_localctx, 326, RustParser::RuleInferredType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2199);
    match(RustParser::UNDERSCORE);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeParamBoundsContext
//------------------------------------------------------------------

RustParser::TypeParamBoundsContext::TypeParamBoundsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::TypeParamBoundContext *>
RustParser::TypeParamBoundsContext::typeParamBound() {
  return getRuleContexts<RustParser::TypeParamBoundContext>();
}

RustParser::TypeParamBoundContext *
RustParser::TypeParamBoundsContext::typeParamBound(size_t i) {
  return getRuleContext<RustParser::TypeParamBoundContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::TypeParamBoundsContext::PLUS() {
  return getTokens(RustParser::PLUS);
}

tree::TerminalNode *RustParser::TypeParamBoundsContext::PLUS(size_t i) {
  return getToken(RustParser::PLUS, i);
}

size_t RustParser::TypeParamBoundsContext::getRuleIndex() const {
  return RustParser::RuleTypeParamBounds;
}

void RustParser::TypeParamBoundsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeParamBounds(this);
}

void RustParser::TypeParamBoundsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeParamBounds(this);
}

std::any
RustParser::TypeParamBoundsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypeParamBounds(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypeParamBoundsContext *RustParser::typeParamBounds() {
  TypeParamBoundsContext *_localctx =
      _tracker.createInstance<TypeParamBoundsContext>(_ctx, getState());
  enterRule(_localctx, 328, RustParser::RuleTypeParamBounds);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2201);
    typeParamBound();
    setState(2206);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     300, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2202);
        match(RustParser::PLUS);
        setState(2203);
        typeParamBound();
      }
      setState(2208);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 300, _ctx);
    }
    setState(2210);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 301, _ctx)) {
    case 1: {
      setState(2209);
      match(RustParser::PLUS);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeParamBoundContext
//------------------------------------------------------------------

RustParser::TypeParamBoundContext::TypeParamBoundContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::LifetimeContext *RustParser::TypeParamBoundContext::lifetime() {
  return getRuleContext<RustParser::LifetimeContext>(0);
}

RustParser::TraitBoundContext *RustParser::TypeParamBoundContext::traitBound() {
  return getRuleContext<RustParser::TraitBoundContext>(0);
}

size_t RustParser::TypeParamBoundContext::getRuleIndex() const {
  return RustParser::RuleTypeParamBound;
}

void RustParser::TypeParamBoundContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeParamBound(this);
}

void RustParser::TypeParamBoundContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeParamBound(this);
}

std::any
RustParser::TypeParamBoundContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypeParamBound(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypeParamBoundContext *RustParser::typeParamBound() {
  TypeParamBoundContext *_localctx =
      _tracker.createInstance<TypeParamBoundContext>(_ctx, getState());
  enterRule(_localctx, 330, RustParser::RuleTypeParamBound);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2214);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_STATICLIFETIME:
    case RustParser::KW_UNDERLINELIFETIME:
    case RustParser::LIFETIME_OR_LABEL: {
      enterOuterAlt(_localctx, 1);
      setState(2212);
      lifetime();
      break;
    }

    case RustParser::KW_CRATE:
    case RustParser::KW_FOR:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SELFTYPE:
    case RustParser::KW_SUPER:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::PATHSEP:
    case RustParser::QUESTION:
    case RustParser::LPAREN: {
      enterOuterAlt(_localctx, 2);
      setState(2213);
      traitBound();
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TraitBoundContext
//------------------------------------------------------------------

RustParser::TraitBoundContext::TraitBoundContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::TypePathContext *RustParser::TraitBoundContext::typePath() {
  return getRuleContext<RustParser::TypePathContext>(0);
}

tree::TerminalNode *RustParser::TraitBoundContext::QUESTION() {
  return getToken(RustParser::QUESTION, 0);
}

RustParser::ForLifetimesContext *RustParser::TraitBoundContext::forLifetimes() {
  return getRuleContext<RustParser::ForLifetimesContext>(0);
}

tree::TerminalNode *RustParser::TraitBoundContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::TraitBoundContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

size_t RustParser::TraitBoundContext::getRuleIndex() const {
  return RustParser::RuleTraitBound;
}

void RustParser::TraitBoundContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTraitBound(this);
}

void RustParser::TraitBoundContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTraitBound(this);
}

std::any
RustParser::TraitBoundContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTraitBound(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TraitBoundContext *RustParser::traitBound() {
  TraitBoundContext *_localctx =
      _tracker.createInstance<TraitBoundContext>(_ctx, getState());
  enterRule(_localctx, 332, RustParser::RuleTraitBound);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2233);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_CRATE:
    case RustParser::KW_FOR:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SELFTYPE:
    case RustParser::KW_SUPER:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER:
    case RustParser::PATHSEP:
    case RustParser::QUESTION: {
      enterOuterAlt(_localctx, 1);
      setState(2217);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::QUESTION) {
        setState(2216);
        match(RustParser::QUESTION);
      }
      setState(2220);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::KW_FOR) {
        setState(2219);
        forLifetimes();
      }
      setState(2222);
      typePath();
      break;
    }

    case RustParser::LPAREN: {
      enterOuterAlt(_localctx, 2);
      setState(2223);
      match(RustParser::LPAREN);
      setState(2225);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::QUESTION) {
        setState(2224);
        match(RustParser::QUESTION);
      }
      setState(2228);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::KW_FOR) {
        setState(2227);
        forLifetimes();
      }
      setState(2230);
      typePath();
      setState(2231);
      match(RustParser::RPAREN);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LifetimeBoundsContext
//------------------------------------------------------------------

RustParser::LifetimeBoundsContext::LifetimeBoundsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::LifetimeContext *>
RustParser::LifetimeBoundsContext::lifetime() {
  return getRuleContexts<RustParser::LifetimeContext>();
}

RustParser::LifetimeContext *
RustParser::LifetimeBoundsContext::lifetime(size_t i) {
  return getRuleContext<RustParser::LifetimeContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::LifetimeBoundsContext::PLUS() {
  return getTokens(RustParser::PLUS);
}

tree::TerminalNode *RustParser::LifetimeBoundsContext::PLUS(size_t i) {
  return getToken(RustParser::PLUS, i);
}

size_t RustParser::LifetimeBoundsContext::getRuleIndex() const {
  return RustParser::RuleLifetimeBounds;
}

void RustParser::LifetimeBoundsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLifetimeBounds(this);
}

void RustParser::LifetimeBoundsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLifetimeBounds(this);
}

std::any
RustParser::LifetimeBoundsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLifetimeBounds(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LifetimeBoundsContext *RustParser::lifetimeBounds() {
  LifetimeBoundsContext *_localctx =
      _tracker.createInstance<LifetimeBoundsContext>(_ctx, getState());
  enterRule(_localctx, 334, RustParser::RuleLifetimeBounds);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2240);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     308, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2235);
        lifetime();
        setState(2236);
        match(RustParser::PLUS);
      }
      setState(2242);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 308, _ctx);
    }
    setState(2244);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 53) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 53)) & 536870917) != 0)) {
      setState(2243);
      lifetime();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LifetimeContext
//------------------------------------------------------------------

RustParser::LifetimeContext::LifetimeContext(ParserRuleContext *parent,
                                             size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::LifetimeContext::LIFETIME_OR_LABEL() {
  return getToken(RustParser::LIFETIME_OR_LABEL, 0);
}

tree::TerminalNode *RustParser::LifetimeContext::KW_STATICLIFETIME() {
  return getToken(RustParser::KW_STATICLIFETIME, 0);
}

tree::TerminalNode *RustParser::LifetimeContext::KW_UNDERLINELIFETIME() {
  return getToken(RustParser::KW_UNDERLINELIFETIME, 0);
}

size_t RustParser::LifetimeContext::getRuleIndex() const {
  return RustParser::RuleLifetime;
}

void RustParser::LifetimeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLifetime(this);
}

void RustParser::LifetimeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLifetime(this);
}

std::any RustParser::LifetimeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitLifetime(this);
  else
    return visitor->visitChildren(this);
}

RustParser::LifetimeContext *RustParser::lifetime() {
  LifetimeContext *_localctx =
      _tracker.createInstance<LifetimeContext>(_ctx, getState());
  enterRule(_localctx, 336, RustParser::RuleLifetime);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2246);
    _la = _input->LA(1);
    if (!(((((_la - 53) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 53)) & 536870917) != 0))) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SimplePathContext
//------------------------------------------------------------------

RustParser::SimplePathContext::SimplePathContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::SimplePathSegmentContext *>
RustParser::SimplePathContext::simplePathSegment() {
  return getRuleContexts<RustParser::SimplePathSegmentContext>();
}

RustParser::SimplePathSegmentContext *
RustParser::SimplePathContext::simplePathSegment(size_t i) {
  return getRuleContext<RustParser::SimplePathSegmentContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::SimplePathContext::PATHSEP() {
  return getTokens(RustParser::PATHSEP);
}

tree::TerminalNode *RustParser::SimplePathContext::PATHSEP(size_t i) {
  return getToken(RustParser::PATHSEP, i);
}

size_t RustParser::SimplePathContext::getRuleIndex() const {
  return RustParser::RuleSimplePath;
}

void RustParser::SimplePathContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSimplePath(this);
}

void RustParser::SimplePathContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSimplePath(this);
}

std::any
RustParser::SimplePathContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitSimplePath(this);
  else
    return visitor->visitChildren(this);
}

RustParser::SimplePathContext *RustParser::simplePath() {
  SimplePathContext *_localctx =
      _tracker.createInstance<SimplePathContext>(_ctx, getState());
  enterRule(_localctx, 338, RustParser::RuleSimplePath);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2249);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::PATHSEP) {
      setState(2248);
      match(RustParser::PATHSEP);
    }
    setState(2251);
    simplePathSegment();
    setState(2256);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     311, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2252);
        match(RustParser::PATHSEP);
        setState(2253);
        simplePathSegment();
      }
      setState(2258);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 311, _ctx);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SimplePathSegmentContext
//------------------------------------------------------------------

RustParser::SimplePathSegmentContext::SimplePathSegmentContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *
RustParser::SimplePathSegmentContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::SimplePathSegmentContext::KW_SUPER() {
  return getToken(RustParser::KW_SUPER, 0);
}

tree::TerminalNode *RustParser::SimplePathSegmentContext::KW_SELFVALUE() {
  return getToken(RustParser::KW_SELFVALUE, 0);
}

tree::TerminalNode *RustParser::SimplePathSegmentContext::KW_CRATE() {
  return getToken(RustParser::KW_CRATE, 0);
}

tree::TerminalNode *RustParser::SimplePathSegmentContext::KW_DOLLARCRATE() {
  return getToken(RustParser::KW_DOLLARCRATE, 0);
}

size_t RustParser::SimplePathSegmentContext::getRuleIndex() const {
  return RustParser::RuleSimplePathSegment;
}

void RustParser::SimplePathSegmentContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSimplePathSegment(this);
}

void RustParser::SimplePathSegmentContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSimplePathSegment(this);
}

std::any
RustParser::SimplePathSegmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitSimplePathSegment(this);
  else
    return visitor->visitChildren(this);
}

RustParser::SimplePathSegmentContext *RustParser::simplePathSegment() {
  SimplePathSegmentContext *_localctx =
      _tracker.createInstance<SimplePathSegmentContext>(_ctx, getState());
  enterRule(_localctx, 340, RustParser::RuleSimplePathSegment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2264);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_MACRORULES:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER: {
      enterOuterAlt(_localctx, 1);
      setState(2259);
      identifier();
      break;
    }

    case RustParser::KW_SUPER: {
      enterOuterAlt(_localctx, 2);
      setState(2260);
      match(RustParser::KW_SUPER);
      break;
    }

    case RustParser::KW_SELFVALUE: {
      enterOuterAlt(_localctx, 3);
      setState(2261);
      match(RustParser::KW_SELFVALUE);
      break;
    }

    case RustParser::KW_CRATE: {
      enterOuterAlt(_localctx, 4);
      setState(2262);
      match(RustParser::KW_CRATE);
      break;
    }

    case RustParser::KW_DOLLARCRATE: {
      enterOuterAlt(_localctx, 5);
      setState(2263);
      match(RustParser::KW_DOLLARCRATE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PathInExpressionContext
//------------------------------------------------------------------

RustParser::PathInExpressionContext::PathInExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::PathExprSegmentContext *>
RustParser::PathInExpressionContext::pathExprSegment() {
  return getRuleContexts<RustParser::PathExprSegmentContext>();
}

RustParser::PathExprSegmentContext *
RustParser::PathInExpressionContext::pathExprSegment(size_t i) {
  return getRuleContext<RustParser::PathExprSegmentContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::PathInExpressionContext::PATHSEP() {
  return getTokens(RustParser::PATHSEP);
}

tree::TerminalNode *RustParser::PathInExpressionContext::PATHSEP(size_t i) {
  return getToken(RustParser::PATHSEP, i);
}

size_t RustParser::PathInExpressionContext::getRuleIndex() const {
  return RustParser::RulePathInExpression;
}

void RustParser::PathInExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPathInExpression(this);
}

void RustParser::PathInExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPathInExpression(this);
}

std::any
RustParser::PathInExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPathInExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PathInExpressionContext *RustParser::pathInExpression() {
  PathInExpressionContext *_localctx =
      _tracker.createInstance<PathInExpressionContext>(_ctx, getState());
  enterRule(_localctx, 342, RustParser::RulePathInExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2267);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::PATHSEP) {
      setState(2266);
      match(RustParser::PATHSEP);
    }
    setState(2269);
    pathExprSegment();
    setState(2274);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     314, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2270);
        match(RustParser::PATHSEP);
        setState(2271);
        pathExprSegment();
      }
      setState(2276);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 314, _ctx);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PathExprSegmentContext
//------------------------------------------------------------------

RustParser::PathExprSegmentContext::PathExprSegmentContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathIdentSegmentContext *
RustParser::PathExprSegmentContext::pathIdentSegment() {
  return getRuleContext<RustParser::PathIdentSegmentContext>(0);
}

tree::TerminalNode *RustParser::PathExprSegmentContext::PATHSEP() {
  return getToken(RustParser::PATHSEP, 0);
}

RustParser::GenericArgsContext *
RustParser::PathExprSegmentContext::genericArgs() {
  return getRuleContext<RustParser::GenericArgsContext>(0);
}

size_t RustParser::PathExprSegmentContext::getRuleIndex() const {
  return RustParser::RulePathExprSegment;
}

void RustParser::PathExprSegmentContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPathExprSegment(this);
}

void RustParser::PathExprSegmentContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPathExprSegment(this);
}

std::any
RustParser::PathExprSegmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPathExprSegment(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PathExprSegmentContext *RustParser::pathExprSegment() {
  PathExprSegmentContext *_localctx =
      _tracker.createInstance<PathExprSegmentContext>(_ctx, getState());
  enterRule(_localctx, 344, RustParser::RulePathExprSegment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2277);
    pathIdentSegment();
    setState(2280);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 315, _ctx)) {
    case 1: {
      setState(2278);
      match(RustParser::PATHSEP);
      setState(2279);
      genericArgs();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PathIdentSegmentContext
//------------------------------------------------------------------

RustParser::PathIdentSegmentContext::PathIdentSegmentContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *
RustParser::PathIdentSegmentContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::PathIdentSegmentContext::KW_SUPER() {
  return getToken(RustParser::KW_SUPER, 0);
}

tree::TerminalNode *RustParser::PathIdentSegmentContext::KW_SELFVALUE() {
  return getToken(RustParser::KW_SELFVALUE, 0);
}

tree::TerminalNode *RustParser::PathIdentSegmentContext::KW_SELFTYPE() {
  return getToken(RustParser::KW_SELFTYPE, 0);
}

tree::TerminalNode *RustParser::PathIdentSegmentContext::KW_CRATE() {
  return getToken(RustParser::KW_CRATE, 0);
}

tree::TerminalNode *RustParser::PathIdentSegmentContext::KW_DOLLARCRATE() {
  return getToken(RustParser::KW_DOLLARCRATE, 0);
}

size_t RustParser::PathIdentSegmentContext::getRuleIndex() const {
  return RustParser::RulePathIdentSegment;
}

void RustParser::PathIdentSegmentContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPathIdentSegment(this);
}

void RustParser::PathIdentSegmentContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPathIdentSegment(this);
}

std::any
RustParser::PathIdentSegmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitPathIdentSegment(this);
  else
    return visitor->visitChildren(this);
}

RustParser::PathIdentSegmentContext *RustParser::pathIdentSegment() {
  PathIdentSegmentContext *_localctx =
      _tracker.createInstance<PathIdentSegmentContext>(_ctx, getState());
  enterRule(_localctx, 346, RustParser::RulePathIdentSegment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2288);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::KW_MACRORULES:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER: {
      enterOuterAlt(_localctx, 1);
      setState(2282);
      identifier();
      break;
    }

    case RustParser::KW_SUPER: {
      enterOuterAlt(_localctx, 2);
      setState(2283);
      match(RustParser::KW_SUPER);
      break;
    }

    case RustParser::KW_SELFVALUE: {
      enterOuterAlt(_localctx, 3);
      setState(2284);
      match(RustParser::KW_SELFVALUE);
      break;
    }

    case RustParser::KW_SELFTYPE: {
      enterOuterAlt(_localctx, 4);
      setState(2285);
      match(RustParser::KW_SELFTYPE);
      break;
    }

    case RustParser::KW_CRATE: {
      enterOuterAlt(_localctx, 5);
      setState(2286);
      match(RustParser::KW_CRATE);
      break;
    }

    case RustParser::KW_DOLLARCRATE: {
      enterOuterAlt(_localctx, 6);
      setState(2287);
      match(RustParser::KW_DOLLARCRATE);
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericArgsContext
//------------------------------------------------------------------

RustParser::GenericArgsContext::GenericArgsContext(ParserRuleContext *parent,
                                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::GenericArgsContext::LT() {
  return getToken(RustParser::LT, 0);
}

tree::TerminalNode *RustParser::GenericArgsContext::GT() {
  return getToken(RustParser::GT, 0);
}

RustParser::GenericArgsLifetimesContext *
RustParser::GenericArgsContext::genericArgsLifetimes() {
  return getRuleContext<RustParser::GenericArgsLifetimesContext>(0);
}

std::vector<tree::TerminalNode *> RustParser::GenericArgsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::GenericArgsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

RustParser::GenericArgsTypesContext *
RustParser::GenericArgsContext::genericArgsTypes() {
  return getRuleContext<RustParser::GenericArgsTypesContext>(0);
}

RustParser::GenericArgsBindingsContext *
RustParser::GenericArgsContext::genericArgsBindings() {
  return getRuleContext<RustParser::GenericArgsBindingsContext>(0);
}

std::vector<RustParser::GenericArgContext *>
RustParser::GenericArgsContext::genericArg() {
  return getRuleContexts<RustParser::GenericArgContext>();
}

RustParser::GenericArgContext *
RustParser::GenericArgsContext::genericArg(size_t i) {
  return getRuleContext<RustParser::GenericArgContext>(i);
}

size_t RustParser::GenericArgsContext::getRuleIndex() const {
  return RustParser::RuleGenericArgs;
}

void RustParser::GenericArgsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericArgs(this);
}

void RustParser::GenericArgsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericArgs(this);
}

std::any
RustParser::GenericArgsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericArgs(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericArgsContext *RustParser::genericArgs() {
  GenericArgsContext *_localctx =
      _tracker.createInstance<GenericArgsContext>(_ctx, getState());
  enterRule(_localctx, 348, RustParser::RuleGenericArgs);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(2333);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 324, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(2290);
      match(RustParser::LT);
      setState(2291);
      match(RustParser::GT);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(2292);
      match(RustParser::LT);
      setState(2293);
      genericArgsLifetimes();
      setState(2296);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 317, _ctx)) {
      case 1: {
        setState(2294);
        match(RustParser::COMMA);
        setState(2295);
        genericArgsTypes();
        break;
      }

      default:
        break;
      }
      setState(2300);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 318, _ctx)) {
      case 1: {
        setState(2298);
        match(RustParser::COMMA);
        setState(2299);
        genericArgsBindings();
        break;
      }

      default:
        break;
      }
      setState(2303);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(2302);
        match(RustParser::COMMA);
      }
      setState(2305);
      match(RustParser::GT);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(2307);
      match(RustParser::LT);
      setState(2308);
      genericArgsTypes();
      setState(2311);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 320, _ctx)) {
      case 1: {
        setState(2309);
        match(RustParser::COMMA);
        setState(2310);
        genericArgsBindings();
        break;
      }

      default:
        break;
      }
      setState(2314);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(2313);
        match(RustParser::COMMA);
      }
      setState(2316);
      match(RustParser::GT);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(2318);
      match(RustParser::LT);
      setState(2324);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 322, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(2319);
          genericArg();
          setState(2320);
          match(RustParser::COMMA);
        }
        setState(2326);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
            _input, 322, _ctx);
      }
      setState(2327);
      genericArg();
      setState(2329);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::COMMA) {
        setState(2328);
        match(RustParser::COMMA);
      }
      setState(2331);
      match(RustParser::GT);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericArgContext
//------------------------------------------------------------------

RustParser::GenericArgContext::GenericArgContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::LifetimeContext *RustParser::GenericArgContext::lifetime() {
  return getRuleContext<RustParser::LifetimeContext>(0);
}

RustParser::Type_Context *RustParser::GenericArgContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

RustParser::GenericArgsConstContext *
RustParser::GenericArgContext::genericArgsConst() {
  return getRuleContext<RustParser::GenericArgsConstContext>(0);
}

RustParser::GenericArgsBindingContext *
RustParser::GenericArgContext::genericArgsBinding() {
  return getRuleContext<RustParser::GenericArgsBindingContext>(0);
}

size_t RustParser::GenericArgContext::getRuleIndex() const {
  return RustParser::RuleGenericArg;
}

void RustParser::GenericArgContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericArg(this);
}

void RustParser::GenericArgContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericArg(this);
}

std::any
RustParser::GenericArgContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericArg(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericArgContext *RustParser::genericArg() {
  GenericArgContext *_localctx =
      _tracker.createInstance<GenericArgContext>(_ctx, getState());
  enterRule(_localctx, 350, RustParser::RuleGenericArg);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2339);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 325, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(2335);
      lifetime();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(2336);
      type_();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(2337);
      genericArgsConst();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(2338);
      genericArgsBinding();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericArgsConstContext
//------------------------------------------------------------------

RustParser::GenericArgsConstContext::GenericArgsConstContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::BlockExpressionContext *
RustParser::GenericArgsConstContext::blockExpression() {
  return getRuleContext<RustParser::BlockExpressionContext>(0);
}

RustParser::LiteralExpressionContext *
RustParser::GenericArgsConstContext::literalExpression() {
  return getRuleContext<RustParser::LiteralExpressionContext>(0);
}

tree::TerminalNode *RustParser::GenericArgsConstContext::MINUS() {
  return getToken(RustParser::MINUS, 0);
}

RustParser::SimplePathSegmentContext *
RustParser::GenericArgsConstContext::simplePathSegment() {
  return getRuleContext<RustParser::SimplePathSegmentContext>(0);
}

size_t RustParser::GenericArgsConstContext::getRuleIndex() const {
  return RustParser::RuleGenericArgsConst;
}

void RustParser::GenericArgsConstContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericArgsConst(this);
}

void RustParser::GenericArgsConstContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericArgsConst(this);
}

std::any
RustParser::GenericArgsConstContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericArgsConst(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericArgsConstContext *RustParser::genericArgsConst() {
  GenericArgsConstContext *_localctx =
      _tracker.createInstance<GenericArgsConstContext>(_ctx, getState());
  enterRule(_localctx, 352, RustParser::RuleGenericArgsConst);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2347);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
    case RustParser::LCURLYBRACE: {
      enterOuterAlt(_localctx, 1);
      setState(2341);
      blockExpression();
      break;
    }

    case RustParser::KW_FALSE:
    case RustParser::KW_TRUE:
    case RustParser::CHAR_LITERAL:
    case RustParser::STRING_LITERAL:
    case RustParser::RAW_STRING_LITERAL:
    case RustParser::BYTE_LITERAL:
    case RustParser::BYTE_STRING_LITERAL:
    case RustParser::RAW_BYTE_STRING_LITERAL:
    case RustParser::INTEGER_LITERAL:
    case RustParser::FLOAT_LITERAL:
    case RustParser::MINUS: {
      enterOuterAlt(_localctx, 2);
      setState(2343);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == RustParser::MINUS) {
        setState(2342);
        match(RustParser::MINUS);
      }
      setState(2345);
      literalExpression();
      break;
    }

    case RustParser::KW_CRATE:
    case RustParser::KW_SELFVALUE:
    case RustParser::KW_SUPER:
    case RustParser::KW_MACRORULES:
    case RustParser::KW_DOLLARCRATE:
    case RustParser::NON_KEYWORD_IDENTIFIER:
    case RustParser::RAW_IDENTIFIER: {
      enterOuterAlt(_localctx, 3);
      setState(2346);
      simplePathSegment();
      break;
    }

    default:
      throw NoViableAltException(this);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericArgsLifetimesContext
//------------------------------------------------------------------

RustParser::GenericArgsLifetimesContext::GenericArgsLifetimesContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::LifetimeContext *>
RustParser::GenericArgsLifetimesContext::lifetime() {
  return getRuleContexts<RustParser::LifetimeContext>();
}

RustParser::LifetimeContext *
RustParser::GenericArgsLifetimesContext::lifetime(size_t i) {
  return getRuleContext<RustParser::LifetimeContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::GenericArgsLifetimesContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::GenericArgsLifetimesContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::GenericArgsLifetimesContext::getRuleIndex() const {
  return RustParser::RuleGenericArgsLifetimes;
}

void RustParser::GenericArgsLifetimesContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericArgsLifetimes(this);
}

void RustParser::GenericArgsLifetimesContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericArgsLifetimes(this);
}

std::any RustParser::GenericArgsLifetimesContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericArgsLifetimes(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericArgsLifetimesContext *RustParser::genericArgsLifetimes() {
  GenericArgsLifetimesContext *_localctx =
      _tracker.createInstance<GenericArgsLifetimesContext>(_ctx, getState());
  enterRule(_localctx, 354, RustParser::RuleGenericArgsLifetimes);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2349);
    lifetime();
    setState(2354);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     328, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2350);
        match(RustParser::COMMA);
        setState(2351);
        lifetime();
      }
      setState(2356);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 328, _ctx);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericArgsTypesContext
//------------------------------------------------------------------

RustParser::GenericArgsTypesContext::GenericArgsTypesContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::Type_Context *>
RustParser::GenericArgsTypesContext::type_() {
  return getRuleContexts<RustParser::Type_Context>();
}

RustParser::Type_Context *RustParser::GenericArgsTypesContext::type_(size_t i) {
  return getRuleContext<RustParser::Type_Context>(i);
}

std::vector<tree::TerminalNode *> RustParser::GenericArgsTypesContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::GenericArgsTypesContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::GenericArgsTypesContext::getRuleIndex() const {
  return RustParser::RuleGenericArgsTypes;
}

void RustParser::GenericArgsTypesContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericArgsTypes(this);
}

void RustParser::GenericArgsTypesContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericArgsTypes(this);
}

std::any
RustParser::GenericArgsTypesContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericArgsTypes(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericArgsTypesContext *RustParser::genericArgsTypes() {
  GenericArgsTypesContext *_localctx =
      _tracker.createInstance<GenericArgsTypesContext>(_ctx, getState());
  enterRule(_localctx, 356, RustParser::RuleGenericArgsTypes);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2357);
    type_();
    setState(2362);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     329, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2358);
        match(RustParser::COMMA);
        setState(2359);
        type_();
      }
      setState(2364);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 329, _ctx);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericArgsBindingsContext
//------------------------------------------------------------------

RustParser::GenericArgsBindingsContext::GenericArgsBindingsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::GenericArgsBindingContext *>
RustParser::GenericArgsBindingsContext::genericArgsBinding() {
  return getRuleContexts<RustParser::GenericArgsBindingContext>();
}

RustParser::GenericArgsBindingContext *
RustParser::GenericArgsBindingsContext::genericArgsBinding(size_t i) {
  return getRuleContext<RustParser::GenericArgsBindingContext>(i);
}

std::vector<tree::TerminalNode *>
RustParser::GenericArgsBindingsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::GenericArgsBindingsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::GenericArgsBindingsContext::getRuleIndex() const {
  return RustParser::RuleGenericArgsBindings;
}

void RustParser::GenericArgsBindingsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericArgsBindings(this);
}

void RustParser::GenericArgsBindingsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericArgsBindings(this);
}

std::any RustParser::GenericArgsBindingsContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericArgsBindings(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericArgsBindingsContext *RustParser::genericArgsBindings() {
  GenericArgsBindingsContext *_localctx =
      _tracker.createInstance<GenericArgsBindingsContext>(_ctx, getState());
  enterRule(_localctx, 358, RustParser::RuleGenericArgsBindings);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2365);
    genericArgsBinding();
    setState(2370);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     330, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2366);
        match(RustParser::COMMA);
        setState(2367);
        genericArgsBinding();
      }
      setState(2372);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 330, _ctx);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GenericArgsBindingContext
//------------------------------------------------------------------

RustParser::GenericArgsBindingContext::GenericArgsBindingContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::IdentifierContext *
RustParser::GenericArgsBindingContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *RustParser::GenericArgsBindingContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

RustParser::Type_Context *RustParser::GenericArgsBindingContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

size_t RustParser::GenericArgsBindingContext::getRuleIndex() const {
  return RustParser::RuleGenericArgsBinding;
}

void RustParser::GenericArgsBindingContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGenericArgsBinding(this);
}

void RustParser::GenericArgsBindingContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGenericArgsBinding(this);
}

std::any
RustParser::GenericArgsBindingContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitGenericArgsBinding(this);
  else
    return visitor->visitChildren(this);
}

RustParser::GenericArgsBindingContext *RustParser::genericArgsBinding() {
  GenericArgsBindingContext *_localctx =
      _tracker.createInstance<GenericArgsBindingContext>(_ctx, getState());
  enterRule(_localctx, 360, RustParser::RuleGenericArgsBinding);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2373);
    identifier();
    setState(2374);
    match(RustParser::EQ);
    setState(2375);
    type_();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QualifiedPathInExpressionContext
//------------------------------------------------------------------

RustParser::QualifiedPathInExpressionContext::QualifiedPathInExpressionContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::QualifiedPathTypeContext *
RustParser::QualifiedPathInExpressionContext::qualifiedPathType() {
  return getRuleContext<RustParser::QualifiedPathTypeContext>(0);
}

std::vector<tree::TerminalNode *>
RustParser::QualifiedPathInExpressionContext::PATHSEP() {
  return getTokens(RustParser::PATHSEP);
}

tree::TerminalNode *
RustParser::QualifiedPathInExpressionContext::PATHSEP(size_t i) {
  return getToken(RustParser::PATHSEP, i);
}

std::vector<RustParser::PathExprSegmentContext *>
RustParser::QualifiedPathInExpressionContext::pathExprSegment() {
  return getRuleContexts<RustParser::PathExprSegmentContext>();
}

RustParser::PathExprSegmentContext *
RustParser::QualifiedPathInExpressionContext::pathExprSegment(size_t i) {
  return getRuleContext<RustParser::PathExprSegmentContext>(i);
}

size_t RustParser::QualifiedPathInExpressionContext::getRuleIndex() const {
  return RustParser::RuleQualifiedPathInExpression;
}

void RustParser::QualifiedPathInExpressionContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQualifiedPathInExpression(this);
}

void RustParser::QualifiedPathInExpressionContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQualifiedPathInExpression(this);
}

std::any RustParser::QualifiedPathInExpressionContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitQualifiedPathInExpression(this);
  else
    return visitor->visitChildren(this);
}

RustParser::QualifiedPathInExpressionContext *
RustParser::qualifiedPathInExpression() {
  QualifiedPathInExpressionContext *_localctx =
      _tracker.createInstance<QualifiedPathInExpressionContext>(_ctx,
                                                                getState());
  enterRule(_localctx, 362, RustParser::RuleQualifiedPathInExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2377);
    qualifiedPathType();
    setState(2380);
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
      case 1: {
        setState(2378);
        match(RustParser::PATHSEP);
        setState(2379);
        pathExprSegment();
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      setState(2382);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 331, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QualifiedPathTypeContext
//------------------------------------------------------------------

RustParser::QualifiedPathTypeContext::QualifiedPathTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::QualifiedPathTypeContext::LT() {
  return getToken(RustParser::LT, 0);
}

RustParser::Type_Context *RustParser::QualifiedPathTypeContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

tree::TerminalNode *RustParser::QualifiedPathTypeContext::GT() {
  return getToken(RustParser::GT, 0);
}

tree::TerminalNode *RustParser::QualifiedPathTypeContext::KW_AS() {
  return getToken(RustParser::KW_AS, 0);
}

RustParser::TypePathContext *RustParser::QualifiedPathTypeContext::typePath() {
  return getRuleContext<RustParser::TypePathContext>(0);
}

size_t RustParser::QualifiedPathTypeContext::getRuleIndex() const {
  return RustParser::RuleQualifiedPathType;
}

void RustParser::QualifiedPathTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQualifiedPathType(this);
}

void RustParser::QualifiedPathTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQualifiedPathType(this);
}

std::any
RustParser::QualifiedPathTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitQualifiedPathType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::QualifiedPathTypeContext *RustParser::qualifiedPathType() {
  QualifiedPathTypeContext *_localctx =
      _tracker.createInstance<QualifiedPathTypeContext>(_ctx, getState());
  enterRule(_localctx, 364, RustParser::RuleQualifiedPathType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2384);
    match(RustParser::LT);
    setState(2385);
    type_();
    setState(2388);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::KW_AS) {
      setState(2386);
      match(RustParser::KW_AS);
      setState(2387);
      typePath();
    }
    setState(2390);
    match(RustParser::GT);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QualifiedPathInTypeContext
//------------------------------------------------------------------

RustParser::QualifiedPathInTypeContext::QualifiedPathInTypeContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::QualifiedPathTypeContext *
RustParser::QualifiedPathInTypeContext::qualifiedPathType() {
  return getRuleContext<RustParser::QualifiedPathTypeContext>(0);
}

std::vector<tree::TerminalNode *>
RustParser::QualifiedPathInTypeContext::PATHSEP() {
  return getTokens(RustParser::PATHSEP);
}

tree::TerminalNode *RustParser::QualifiedPathInTypeContext::PATHSEP(size_t i) {
  return getToken(RustParser::PATHSEP, i);
}

std::vector<RustParser::TypePathSegmentContext *>
RustParser::QualifiedPathInTypeContext::typePathSegment() {
  return getRuleContexts<RustParser::TypePathSegmentContext>();
}

RustParser::TypePathSegmentContext *
RustParser::QualifiedPathInTypeContext::typePathSegment(size_t i) {
  return getRuleContext<RustParser::TypePathSegmentContext>(i);
}

size_t RustParser::QualifiedPathInTypeContext::getRuleIndex() const {
  return RustParser::RuleQualifiedPathInType;
}

void RustParser::QualifiedPathInTypeContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQualifiedPathInType(this);
}

void RustParser::QualifiedPathInTypeContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQualifiedPathInType(this);
}

std::any RustParser::QualifiedPathInTypeContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitQualifiedPathInType(this);
  else
    return visitor->visitChildren(this);
}

RustParser::QualifiedPathInTypeContext *RustParser::qualifiedPathInType() {
  QualifiedPathInTypeContext *_localctx =
      _tracker.createInstance<QualifiedPathInTypeContext>(_ctx, getState());
  enterRule(_localctx, 366, RustParser::RuleQualifiedPathInType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2392);
    qualifiedPathType();
    setState(2395);
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
      case 1: {
        setState(2393);
        match(RustParser::PATHSEP);
        setState(2394);
        typePathSegment();
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      setState(2397);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 333, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypePathContext
//------------------------------------------------------------------

RustParser::TypePathContext::TypePathContext(ParserRuleContext *parent,
                                             size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::TypePathSegmentContext *>
RustParser::TypePathContext::typePathSegment() {
  return getRuleContexts<RustParser::TypePathSegmentContext>();
}

RustParser::TypePathSegmentContext *
RustParser::TypePathContext::typePathSegment(size_t i) {
  return getRuleContext<RustParser::TypePathSegmentContext>(i);
}

std::vector<tree::TerminalNode *> RustParser::TypePathContext::PATHSEP() {
  return getTokens(RustParser::PATHSEP);
}

tree::TerminalNode *RustParser::TypePathContext::PATHSEP(size_t i) {
  return getToken(RustParser::PATHSEP, i);
}

size_t RustParser::TypePathContext::getRuleIndex() const {
  return RustParser::RuleTypePath;
}

void RustParser::TypePathContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypePath(this);
}

void RustParser::TypePathContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypePath(this);
}

std::any RustParser::TypePathContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypePath(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypePathContext *RustParser::typePath() {
  TypePathContext *_localctx =
      _tracker.createInstance<TypePathContext>(_ctx, getState());
  enterRule(_localctx, 368, RustParser::RuleTypePath);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2400);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::PATHSEP) {
      setState(2399);
      match(RustParser::PATHSEP);
    }
    setState(2402);
    typePathSegment();
    setState(2407);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     335, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2403);
        match(RustParser::PATHSEP);
        setState(2404);
        typePathSegment();
      }
      setState(2409);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 335, _ctx);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypePathSegmentContext
//------------------------------------------------------------------

RustParser::TypePathSegmentContext::TypePathSegmentContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::PathIdentSegmentContext *
RustParser::TypePathSegmentContext::pathIdentSegment() {
  return getRuleContext<RustParser::PathIdentSegmentContext>(0);
}

tree::TerminalNode *RustParser::TypePathSegmentContext::PATHSEP() {
  return getToken(RustParser::PATHSEP, 0);
}

RustParser::GenericArgsContext *
RustParser::TypePathSegmentContext::genericArgs() {
  return getRuleContext<RustParser::GenericArgsContext>(0);
}

RustParser::TypePathFnContext *
RustParser::TypePathSegmentContext::typePathFn() {
  return getRuleContext<RustParser::TypePathFnContext>(0);
}

size_t RustParser::TypePathSegmentContext::getRuleIndex() const {
  return RustParser::RuleTypePathSegment;
}

void RustParser::TypePathSegmentContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypePathSegment(this);
}

void RustParser::TypePathSegmentContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypePathSegment(this);
}

std::any
RustParser::TypePathSegmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypePathSegment(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypePathSegmentContext *RustParser::typePathSegment() {
  TypePathSegmentContext *_localctx =
      _tracker.createInstance<TypePathSegmentContext>(_ctx, getState());
  enterRule(_localctx, 370, RustParser::RuleTypePathSegment);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2410);
    pathIdentSegment();
    setState(2412);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 336, _ctx)) {
    case 1: {
      setState(2411);
      match(RustParser::PATHSEP);
      break;
    }

    default:
      break;
    }
    setState(2416);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 337, _ctx)) {
    case 1: {
      setState(2414);
      genericArgs();
      break;
    }

    case 2: {
      setState(2415);
      typePathFn();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypePathFnContext
//------------------------------------------------------------------

RustParser::TypePathFnContext::TypePathFnContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::TypePathFnContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::TypePathFnContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

RustParser::TypePathInputsContext *
RustParser::TypePathFnContext::typePathInputs() {
  return getRuleContext<RustParser::TypePathInputsContext>(0);
}

tree::TerminalNode *RustParser::TypePathFnContext::RARROW() {
  return getToken(RustParser::RARROW, 0);
}

RustParser::Type_Context *RustParser::TypePathFnContext::type_() {
  return getRuleContext<RustParser::Type_Context>(0);
}

size_t RustParser::TypePathFnContext::getRuleIndex() const {
  return RustParser::RuleTypePathFn;
}

void RustParser::TypePathFnContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypePathFn(this);
}

void RustParser::TypePathFnContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypePathFn(this);
}

std::any
RustParser::TypePathFnContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypePathFn(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypePathFnContext *RustParser::typePathFn() {
  TypePathFnContext *_localctx =
      _tracker.createInstance<TypePathFnContext>(_ctx, getState());
  enterRule(_localctx, 372, RustParser::RuleTypePathFn);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2418);
    match(RustParser::LPAREN);
    setState(2420);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~0x3fULL) == 0) &&
         ((1ULL << _la) & 567453832540335392) != 0) ||
        ((((_la - 82) & ~0x3fULL) == 0) &&
         ((1ULL << (_la - 82)) & 360915832668553) != 0)) {
      setState(2419);
      typePathInputs();
    }
    setState(2422);
    match(RustParser::RPAREN);
    setState(2425);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 339, _ctx)) {
    case 1: {
      setState(2423);
      match(RustParser::RARROW);
      setState(2424);
      type_();
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypePathInputsContext
//------------------------------------------------------------------

RustParser::TypePathInputsContext::TypePathInputsContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<RustParser::Type_Context *>
RustParser::TypePathInputsContext::type_() {
  return getRuleContexts<RustParser::Type_Context>();
}

RustParser::Type_Context *RustParser::TypePathInputsContext::type_(size_t i) {
  return getRuleContext<RustParser::Type_Context>(i);
}

std::vector<tree::TerminalNode *> RustParser::TypePathInputsContext::COMMA() {
  return getTokens(RustParser::COMMA);
}

tree::TerminalNode *RustParser::TypePathInputsContext::COMMA(size_t i) {
  return getToken(RustParser::COMMA, i);
}

size_t RustParser::TypePathInputsContext::getRuleIndex() const {
  return RustParser::RuleTypePathInputs;
}

void RustParser::TypePathInputsContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypePathInputs(this);
}

void RustParser::TypePathInputsContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypePathInputs(this);
}

std::any
RustParser::TypePathInputsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitTypePathInputs(this);
  else
    return visitor->visitChildren(this);
}

RustParser::TypePathInputsContext *RustParser::typePathInputs() {
  TypePathInputsContext *_localctx =
      _tracker.createInstance<TypePathInputsContext>(_ctx, getState());
  enterRule(_localctx, 374, RustParser::RuleTypePathInputs);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(2427);
    type_();
    setState(2432);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input,
                                                                     340, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(2428);
        match(RustParser::COMMA);
        setState(2429);
        type_();
      }
      setState(2434);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
          _input, 340, _ctx);
    }
    setState(2436);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == RustParser::COMMA) {
      setState(2435);
      match(RustParser::COMMA);
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VisibilityContext
//------------------------------------------------------------------

RustParser::VisibilityContext::VisibilityContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::VisibilityContext::KW_PUB() {
  return getToken(RustParser::KW_PUB, 0);
}

tree::TerminalNode *RustParser::VisibilityContext::LPAREN() {
  return getToken(RustParser::LPAREN, 0);
}

tree::TerminalNode *RustParser::VisibilityContext::RPAREN() {
  return getToken(RustParser::RPAREN, 0);
}

tree::TerminalNode *RustParser::VisibilityContext::KW_CRATE() {
  return getToken(RustParser::KW_CRATE, 0);
}

tree::TerminalNode *RustParser::VisibilityContext::KW_SELFVALUE() {
  return getToken(RustParser::KW_SELFVALUE, 0);
}

tree::TerminalNode *RustParser::VisibilityContext::KW_SUPER() {
  return getToken(RustParser::KW_SUPER, 0);
}

tree::TerminalNode *RustParser::VisibilityContext::KW_IN() {
  return getToken(RustParser::KW_IN, 0);
}

RustParser::SimplePathContext *RustParser::VisibilityContext::simplePath() {
  return getRuleContext<RustParser::SimplePathContext>(0);
}

size_t RustParser::VisibilityContext::getRuleIndex() const {
  return RustParser::RuleVisibility;
}

void RustParser::VisibilityContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVisibility(this);
}

void RustParser::VisibilityContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVisibility(this);
}

std::any
RustParser::VisibilityContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitVisibility(this);
  else
    return visitor->visitChildren(this);
}

RustParser::VisibilityContext *RustParser::visibility() {
  VisibilityContext *_localctx =
      _tracker.createInstance<VisibilityContext>(_ctx, getState());
  enterRule(_localctx, 376, RustParser::RuleVisibility);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2438);
    match(RustParser::KW_PUB);
    setState(2448);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 343, _ctx)) {
    case 1: {
      setState(2439);
      match(RustParser::LPAREN);
      setState(2445);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
      case RustParser::KW_CRATE: {
        setState(2440);
        match(RustParser::KW_CRATE);
        break;
      }

      case RustParser::KW_SELFVALUE: {
        setState(2441);
        match(RustParser::KW_SELFVALUE);
        break;
      }

      case RustParser::KW_SUPER: {
        setState(2442);
        match(RustParser::KW_SUPER);
        break;
      }

      case RustParser::KW_IN: {
        setState(2443);
        match(RustParser::KW_IN);
        setState(2444);
        simplePath();
        break;
      }

      default:
        throw NoViableAltException(this);
      }
      setState(2447);
      match(RustParser::RPAREN);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdentifierContext
//------------------------------------------------------------------

RustParser::IdentifierContext::IdentifierContext(ParserRuleContext *parent,
                                                 size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::IdentifierContext::NON_KEYWORD_IDENTIFIER() {
  return getToken(RustParser::NON_KEYWORD_IDENTIFIER, 0);
}

tree::TerminalNode *RustParser::IdentifierContext::RAW_IDENTIFIER() {
  return getToken(RustParser::RAW_IDENTIFIER, 0);
}

tree::TerminalNode *RustParser::IdentifierContext::KW_MACRORULES() {
  return getToken(RustParser::KW_MACRORULES, 0);
}

size_t RustParser::IdentifierContext::getRuleIndex() const {
  return RustParser::RuleIdentifier;
}

void RustParser::IdentifierContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdentifier(this);
}

void RustParser::IdentifierContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdentifier(this);
}

std::any
RustParser::IdentifierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitIdentifier(this);
  else
    return visitor->visitChildren(this);
}

RustParser::IdentifierContext *RustParser::identifier() {
  IdentifierContext *_localctx =
      _tracker.createInstance<IdentifierContext>(_ctx, getState());
  enterRule(_localctx, 378, RustParser::RuleIdentifier);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2450);
    _la = _input->LA(1);
    if (!((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 450359962737049600) != 0))) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- KeywordContext
//------------------------------------------------------------------

RustParser::KeywordContext::KeywordContext(ParserRuleContext *parent,
                                           size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::KeywordContext::KW_AS() {
  return getToken(RustParser::KW_AS, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_BREAK() {
  return getToken(RustParser::KW_BREAK, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_CONST() {
  return getToken(RustParser::KW_CONST, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_CONTINUE() {
  return getToken(RustParser::KW_CONTINUE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_CRATE() {
  return getToken(RustParser::KW_CRATE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_ELSE() {
  return getToken(RustParser::KW_ELSE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_ENUM() {
  return getToken(RustParser::KW_ENUM, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_EXTERN() {
  return getToken(RustParser::KW_EXTERN, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_FALSE() {
  return getToken(RustParser::KW_FALSE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_FN() {
  return getToken(RustParser::KW_FN, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_FOR() {
  return getToken(RustParser::KW_FOR, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_IF() {
  return getToken(RustParser::KW_IF, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_IMPL() {
  return getToken(RustParser::KW_IMPL, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_IN() {
  return getToken(RustParser::KW_IN, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_LET() {
  return getToken(RustParser::KW_LET, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_LOOP() {
  return getToken(RustParser::KW_LOOP, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_MATCH() {
  return getToken(RustParser::KW_MATCH, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_MOD() {
  return getToken(RustParser::KW_MOD, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_MOVE() {
  return getToken(RustParser::KW_MOVE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_MUT() {
  return getToken(RustParser::KW_MUT, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_PUB() {
  return getToken(RustParser::KW_PUB, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_REF() {
  return getToken(RustParser::KW_REF, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_RETURN() {
  return getToken(RustParser::KW_RETURN, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_SELFVALUE() {
  return getToken(RustParser::KW_SELFVALUE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_SELFTYPE() {
  return getToken(RustParser::KW_SELFTYPE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_STATIC() {
  return getToken(RustParser::KW_STATIC, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_STRUCT() {
  return getToken(RustParser::KW_STRUCT, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_SUPER() {
  return getToken(RustParser::KW_SUPER, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_TRAIT() {
  return getToken(RustParser::KW_TRAIT, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_TRUE() {
  return getToken(RustParser::KW_TRUE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_TYPE() {
  return getToken(RustParser::KW_TYPE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_UNSAFE() {
  return getToken(RustParser::KW_UNSAFE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_USE() {
  return getToken(RustParser::KW_USE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_WHERE() {
  return getToken(RustParser::KW_WHERE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_WHILE() {
  return getToken(RustParser::KW_WHILE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_ASYNC() {
  return getToken(RustParser::KW_ASYNC, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_AWAIT() {
  return getToken(RustParser::KW_AWAIT, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_DYN() {
  return getToken(RustParser::KW_DYN, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_ABSTRACT() {
  return getToken(RustParser::KW_ABSTRACT, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_BECOME() {
  return getToken(RustParser::KW_BECOME, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_BOX() {
  return getToken(RustParser::KW_BOX, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_DO() {
  return getToken(RustParser::KW_DO, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_FINAL() {
  return getToken(RustParser::KW_FINAL, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_MACRO() {
  return getToken(RustParser::KW_MACRO, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_OVERRIDE() {
  return getToken(RustParser::KW_OVERRIDE, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_PRIV() {
  return getToken(RustParser::KW_PRIV, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_TYPEOF() {
  return getToken(RustParser::KW_TYPEOF, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_UNSIZED() {
  return getToken(RustParser::KW_UNSIZED, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_VIRTUAL() {
  return getToken(RustParser::KW_VIRTUAL, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_YIELD() {
  return getToken(RustParser::KW_YIELD, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_TRY() {
  return getToken(RustParser::KW_TRY, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_UNION() {
  return getToken(RustParser::KW_UNION, 0);
}

tree::TerminalNode *RustParser::KeywordContext::KW_STATICLIFETIME() {
  return getToken(RustParser::KW_STATICLIFETIME, 0);
}

size_t RustParser::KeywordContext::getRuleIndex() const {
  return RustParser::RuleKeyword;
}

void RustParser::KeywordContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterKeyword(this);
}

void RustParser::KeywordContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitKeyword(this);
}

std::any RustParser::KeywordContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitKeyword(this);
  else
    return visitor->visitChildren(this);
}

RustParser::KeywordContext *RustParser::keyword() {
  KeywordContext *_localctx =
      _tracker.createInstance<KeywordContext>(_ctx, getState());
  enterRule(_localctx, 380, RustParser::RuleKeyword);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2452);
    _la = _input->LA(1);
    if (!((((_la & ~0x3fULL) == 0) &&
           ((1ULL << _la) & 18014398509481982) != 0))) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroIdentifierLikeTokenContext
//------------------------------------------------------------------

RustParser::MacroIdentifierLikeTokenContext::MacroIdentifierLikeTokenContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::KeywordContext *
RustParser::MacroIdentifierLikeTokenContext::keyword() {
  return getRuleContext<RustParser::KeywordContext>(0);
}

RustParser::IdentifierContext *
RustParser::MacroIdentifierLikeTokenContext::identifier() {
  return getRuleContext<RustParser::IdentifierContext>(0);
}

tree::TerminalNode *
RustParser::MacroIdentifierLikeTokenContext::KW_MACRORULES() {
  return getToken(RustParser::KW_MACRORULES, 0);
}

tree::TerminalNode *
RustParser::MacroIdentifierLikeTokenContext::KW_UNDERLINELIFETIME() {
  return getToken(RustParser::KW_UNDERLINELIFETIME, 0);
}

tree::TerminalNode *
RustParser::MacroIdentifierLikeTokenContext::KW_DOLLARCRATE() {
  return getToken(RustParser::KW_DOLLARCRATE, 0);
}

tree::TerminalNode *
RustParser::MacroIdentifierLikeTokenContext::LIFETIME_OR_LABEL() {
  return getToken(RustParser::LIFETIME_OR_LABEL, 0);
}

size_t RustParser::MacroIdentifierLikeTokenContext::getRuleIndex() const {
  return RustParser::RuleMacroIdentifierLikeToken;
}

void RustParser::MacroIdentifierLikeTokenContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroIdentifierLikeToken(this);
}

void RustParser::MacroIdentifierLikeTokenContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroIdentifierLikeToken(this);
}

std::any RustParser::MacroIdentifierLikeTokenContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroIdentifierLikeToken(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroIdentifierLikeTokenContext *
RustParser::macroIdentifierLikeToken() {
  MacroIdentifierLikeTokenContext *_localctx =
      _tracker.createInstance<MacroIdentifierLikeTokenContext>(_ctx,
                                                               getState());
  enterRule(_localctx, 382, RustParser::RuleMacroIdentifierLikeToken);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(2460);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(
        _input, 344, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(2454);
      keyword();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(2455);
      identifier();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(2456);
      match(RustParser::KW_MACRORULES);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(2457);
      match(RustParser::KW_UNDERLINELIFETIME);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(2458);
      match(RustParser::KW_DOLLARCRATE);
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(2459);
      match(RustParser::LIFETIME_OR_LABEL);
      break;
    }

    default:
      break;
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroLiteralTokenContext
//------------------------------------------------------------------

RustParser::MacroLiteralTokenContext::MacroLiteralTokenContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

RustParser::LiteralExpressionContext *
RustParser::MacroLiteralTokenContext::literalExpression() {
  return getRuleContext<RustParser::LiteralExpressionContext>(0);
}

size_t RustParser::MacroLiteralTokenContext::getRuleIndex() const {
  return RustParser::RuleMacroLiteralToken;
}

void RustParser::MacroLiteralTokenContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroLiteralToken(this);
}

void RustParser::MacroLiteralTokenContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroLiteralToken(this);
}

std::any
RustParser::MacroLiteralTokenContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroLiteralToken(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroLiteralTokenContext *RustParser::macroLiteralToken() {
  MacroLiteralTokenContext *_localctx =
      _tracker.createInstance<MacroLiteralTokenContext>(_ctx, getState());
  enterRule(_localctx, 384, RustParser::RuleMacroLiteralToken);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2462);
    literalExpression();

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MacroPunctuationTokenContext
//------------------------------------------------------------------

RustParser::MacroPunctuationTokenContext::MacroPunctuationTokenContext(
    ParserRuleContext *parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::MINUS() {
  return getToken(RustParser::MINUS, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::SLASH() {
  return getToken(RustParser::SLASH, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::PERCENT() {
  return getToken(RustParser::PERCENT, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::CARET() {
  return getToken(RustParser::CARET, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::NOT() {
  return getToken(RustParser::NOT, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::AND() {
  return getToken(RustParser::AND, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::OR() {
  return getToken(RustParser::OR, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::ANDAND() {
  return getToken(RustParser::ANDAND, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::OROR() {
  return getToken(RustParser::OROR, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::PLUSEQ() {
  return getToken(RustParser::PLUSEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::MINUSEQ() {
  return getToken(RustParser::MINUSEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::STAREQ() {
  return getToken(RustParser::STAREQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::SLASHEQ() {
  return getToken(RustParser::SLASHEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::PERCENTEQ() {
  return getToken(RustParser::PERCENTEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::CARETEQ() {
  return getToken(RustParser::CARETEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::ANDEQ() {
  return getToken(RustParser::ANDEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::OREQ() {
  return getToken(RustParser::OREQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::SHLEQ() {
  return getToken(RustParser::SHLEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::SHREQ() {
  return getToken(RustParser::SHREQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::EQ() {
  return getToken(RustParser::EQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::EQEQ() {
  return getToken(RustParser::EQEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::NE() {
  return getToken(RustParser::NE, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::GT() {
  return getToken(RustParser::GT, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::LT() {
  return getToken(RustParser::LT, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::GE() {
  return getToken(RustParser::GE, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::LE() {
  return getToken(RustParser::LE, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::AT() {
  return getToken(RustParser::AT, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::UNDERSCORE() {
  return getToken(RustParser::UNDERSCORE, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::DOT() {
  return getToken(RustParser::DOT, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::DOTDOT() {
  return getToken(RustParser::DOTDOT, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::DOTDOTDOT() {
  return getToken(RustParser::DOTDOTDOT, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::DOTDOTEQ() {
  return getToken(RustParser::DOTDOTEQ, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::COMMA() {
  return getToken(RustParser::COMMA, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::SEMI() {
  return getToken(RustParser::SEMI, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::COLON() {
  return getToken(RustParser::COLON, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::PATHSEP() {
  return getToken(RustParser::PATHSEP, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::RARROW() {
  return getToken(RustParser::RARROW, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::FATARROW() {
  return getToken(RustParser::FATARROW, 0);
}

tree::TerminalNode *RustParser::MacroPunctuationTokenContext::POUND() {
  return getToken(RustParser::POUND, 0);
}

size_t RustParser::MacroPunctuationTokenContext::getRuleIndex() const {
  return RustParser::RuleMacroPunctuationToken;
}

void RustParser::MacroPunctuationTokenContext::enterRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMacroPunctuationToken(this);
}

void RustParser::MacroPunctuationTokenContext::exitRule(
    tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMacroPunctuationToken(this);
}

std::any RustParser::MacroPunctuationTokenContext::accept(
    tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitMacroPunctuationToken(this);
  else
    return visitor->visitChildren(this);
}

RustParser::MacroPunctuationTokenContext *RustParser::macroPunctuationToken() {
  MacroPunctuationTokenContext *_localctx =
      _tracker.createInstance<MacroPunctuationTokenContext>(_ctx, getState());
  enterRule(_localctx, 386, RustParser::RuleMacroPunctuationToken);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2464);
    _la = _input->LA(1);
    if (!(((((_la - 84) & ~0x3fULL) == 0) &&
           ((1ULL << (_la - 84)) & 1099511627773) != 0))) {
      _errHandler->recoverInline(this);
    } else {
      _errHandler->reportMatch(this);
      consume();
    }

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShlContext
//------------------------------------------------------------------

RustParser::ShlContext::ShlContext(ParserRuleContext *parent,
                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<tree::TerminalNode *> RustParser::ShlContext::LT() {
  return getTokens(RustParser::LT);
}

tree::TerminalNode *RustParser::ShlContext::LT(size_t i) {
  return getToken(RustParser::LT, i);
}

size_t RustParser::ShlContext::getRuleIndex() const {
  return RustParser::RuleShl;
}

void RustParser::ShlContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShl(this);
}

void RustParser::ShlContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShl(this);
}

std::any RustParser::ShlContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitShl(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ShlContext *RustParser::shl() {
  ShlContext *_localctx = _tracker.createInstance<ShlContext>(_ctx, getState());
  enterRule(_localctx, 388, RustParser::RuleShl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2466);
    match(RustParser::LT);
    setState(2467);

    if (!(this->next('<')))
      throw FailedPredicateException(this, "this->next('<')");
    setState(2468);
    match(RustParser::LT);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShrContext
//------------------------------------------------------------------

RustParser::ShrContext::ShrContext(ParserRuleContext *parent,
                                   size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<tree::TerminalNode *> RustParser::ShrContext::GT() {
  return getTokens(RustParser::GT);
}

tree::TerminalNode *RustParser::ShrContext::GT(size_t i) {
  return getToken(RustParser::GT, i);
}

size_t RustParser::ShrContext::getRuleIndex() const {
  return RustParser::RuleShr;
}

void RustParser::ShrContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShr(this);
}

void RustParser::ShrContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<RustParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShr(this);
}

std::any RustParser::ShrContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<RustParserVisitor *>(visitor))
    return parserVisitor->visitShr(this);
  else
    return visitor->visitChildren(this);
}

RustParser::ShrContext *RustParser::shr() {
  ShrContext *_localctx = _tracker.createInstance<ShrContext>(_ctx, getState());
  enterRule(_localctx, 390, RustParser::RuleShr);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2470);
    match(RustParser::GT);
    setState(2471);

    if (!(this->next('>')))
      throw FailedPredicateException(this, "this->next('>')");
    setState(2472);
    match(RustParser::GT);

  } catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool RustParser::sempred(RuleContext *context, size_t ruleIndex,
                         size_t predicateIndex) {
  switch (ruleIndex) {
  case 77:
    return expressionSempred(antlrcpp::downCast<ExpressionContext *>(context),
                             predicateIndex);
  case 194:
    return shlSempred(antlrcpp::downCast<ShlContext *>(context),
                      predicateIndex);
  case 195:
    return shrSempred(antlrcpp::downCast<ShrContext *>(context),
                      predicateIndex);

  default:
    break;
  }
  return true;
}

bool RustParser::expressionSempred(ExpressionContext *_localctx,
                                   size_t predicateIndex) {
  switch (predicateIndex) {
  case 0:
    return precpred(_ctx, 26);
  case 1:
    return precpred(_ctx, 25);
  case 2:
    return precpred(_ctx, 24);
  case 3:
    return precpred(_ctx, 23);
  case 4:
    return precpred(_ctx, 22);
  case 5:
    return precpred(_ctx, 21);
  case 6:
    return precpred(_ctx, 20);
  case 7:
    return precpred(_ctx, 19);
  case 8:
    return precpred(_ctx, 18);
  case 9:
    return precpred(_ctx, 14);
  case 10:
    return precpred(_ctx, 13);
  case 11:
    return precpred(_ctx, 12);
  case 12:
    return precpred(_ctx, 37);
  case 13:
    return precpred(_ctx, 36);
  case 14:
    return precpred(_ctx, 35);
  case 15:
    return precpred(_ctx, 34);
  case 16:
    return precpred(_ctx, 33);
  case 17:
    return precpred(_ctx, 32);
  case 18:
    return precpred(_ctx, 31);
  case 19:
    return precpred(_ctx, 27);
  case 20:
    return precpred(_ctx, 17);

  default:
    break;
  }
  return true;
}

bool RustParser::shlSempred(ShlContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
  case 21:
    return this->next('<');

  default:
    break;
  }
  return true;
}

bool RustParser::shrSempred(ShrContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
  case 22:
    return this->next('>');

  default:
    break;
  }
  return true;
}

void RustParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  rustparserParserInitialize();
#else
  ::antlr4::internal::call_once(rustparserParserOnceFlag,
                                rustparserParserInitialize);
#endif
}
