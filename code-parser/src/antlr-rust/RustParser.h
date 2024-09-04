
// Generated from RustParser.g4 by ANTLR 4.13.1

#pragma once

#include "antlr4-runtime.h"

class RustParser : public antlr4::Parser {
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

  enum {
    RuleCrate = 0,
    RuleMacroInvocation = 1,
    RuleDelimTokenTree = 2,
    RuleTokenTree = 3,
    RuleTokenTreeToken = 4,
    RuleMacroInvocationSemi = 5,
    RuleMacroRulesDefinition = 6,
    RuleMacroRulesDef = 7,
    RuleMacroRules = 8,
    RuleMacroRule = 9,
    RuleMacroMatcher = 10,
    RuleMacroMatch = 11,
    RuleMacroMatchToken = 12,
    RuleMacroFragSpec = 13,
    RuleMacroRepSep = 14,
    RuleMacroRepOp = 15,
    RuleMacroTranscriber = 16,
    RuleItem = 17,
    RuleVisItem = 18,
    RuleMacroItem = 19,
    RuleModule = 20,
    RuleExternCrate = 21,
    RuleCrateRef = 22,
    RuleAsClause = 23,
    RuleUseDeclaration = 24,
    RuleUseTree = 25,
    RuleFunction_ = 26,
    RuleFunctionQualifiers = 27,
    RuleAbi = 28,
    RuleFunctionParameters = 29,
    RuleSelfParam = 30,
    RuleShorthandSelf = 31,
    RuleTypedSelf = 32,
    RuleFunctionParam = 33,
    RuleFunctionParamPattern = 34,
    RuleFunctionReturnType = 35,
    RuleTypeAlias = 36,
    RuleStruct_ = 37,
    RuleStructStruct = 38,
    RuleTupleStruct = 39,
    RuleStructFields = 40,
    RuleStructField = 41,
    RuleTupleFields = 42,
    RuleTupleField = 43,
    RuleEnumeration = 44,
    RuleEnumItems = 45,
    RuleEnumItem = 46,
    RuleEnumItemTuple = 47,
    RuleEnumItemStruct = 48,
    RuleEnumItemDiscriminant = 49,
    RuleUnion_ = 50,
    RuleConstantItem = 51,
    RuleStaticItem = 52,
    RuleTrait_ = 53,
    RuleImplementation = 54,
    RuleInherentImpl = 55,
    RuleTraitImpl = 56,
    RuleExternBlock = 57,
    RuleExternalItem = 58,
    RuleGenericParams = 59,
    RuleGenericParam = 60,
    RuleLifetimeParam = 61,
    RuleTypeParam = 62,
    RuleConstParam = 63,
    RuleWhereClause = 64,
    RuleWhereClauseItem = 65,
    RuleLifetimeWhereClauseItem = 66,
    RuleTypeBoundWhereClauseItem = 67,
    RuleForLifetimes = 68,
    RuleAssociatedItem = 69,
    RuleInnerAttribute = 70,
    RuleOuterAttribute = 71,
    RuleAttr = 72,
    RuleAttrInput = 73,
    RuleStatement = 74,
    RuleLetStatement = 75,
    RuleExpressionStatement = 76,
    RuleExpression = 77,
    RuleComparisonOperator = 78,
    RuleCompoundAssignOperator = 79,
    RuleExpressionWithBlock = 80,
    RuleLiteralExpression = 81,
    RulePathExpression = 82,
    RuleBlockExpression = 83,
    RuleStatements = 84,
    RuleAsyncBlockExpression = 85,
    RuleUnsafeBlockExpression = 86,
    RuleArrayElements = 87,
    RuleTupleElements = 88,
    RuleTupleIndex = 89,
    RuleStructExpression = 90,
    RuleStructExprStruct = 91,
    RuleStructExprFields = 92,
    RuleStructExprField = 93,
    RuleStructBase = 94,
    RuleStructExprTuple = 95,
    RuleStructExprUnit = 96,
    RuleEnumerationVariantExpression = 97,
    RuleEnumExprStruct = 98,
    RuleEnumExprFields = 99,
    RuleEnumExprField = 100,
    RuleEnumExprTuple = 101,
    RuleEnumExprFieldless = 102,
    RuleCallParams = 103,
    RuleClosureExpression = 104,
    RuleClosureParameters = 105,
    RuleClosureParam = 106,
    RuleLoopExpression = 107,
    RuleInfiniteLoopExpression = 108,
    RulePredicateLoopExpression = 109,
    RulePredicatePatternLoopExpression = 110,
    RuleIteratorLoopExpression = 111,
    RuleLoopLabel = 112,
    RuleIfExpression = 113,
    RuleIfLetExpression = 114,
    RuleMatchExpression = 115,
    RuleMatchArms = 116,
    RuleMatchArmExpression = 117,
    RuleMatchArm = 118,
    RuleMatchArmGuard = 119,
    RulePattern = 120,
    RulePatternNoTopAlt = 121,
    RulePatternWithoutRange = 122,
    RuleLiteralPattern = 123,
    RuleIdentifierPattern = 124,
    RuleWildcardPattern = 125,
    RuleRestPattern = 126,
    RuleRangePattern = 127,
    RuleRangePatternBound = 128,
    RuleReferencePattern = 129,
    RuleStructPattern = 130,
    RuleStructPatternElements = 131,
    RuleStructPatternFields = 132,
    RuleStructPatternField = 133,
    RuleStructPatternEtCetera = 134,
    RuleTupleStructPattern = 135,
    RuleTupleStructItems = 136,
    RuleTuplePattern = 137,
    RuleTuplePatternItems = 138,
    RuleGroupedPattern = 139,
    RuleSlicePattern = 140,
    RuleSlicePatternItems = 141,
    RulePathPattern = 142,
    RuleType_ = 143,
    RuleTypeNoBounds = 144,
    RuleParenthesizedType = 145,
    RuleNeverType = 146,
    RuleTupleType = 147,
    RuleArrayType = 148,
    RuleSliceType = 149,
    RuleReferenceType = 150,
    RuleRawPointerType = 151,
    RuleBareFunctionType = 152,
    RuleFunctionTypeQualifiers = 153,
    RuleBareFunctionReturnType = 154,
    RuleFunctionParametersMaybeNamedVariadic = 155,
    RuleMaybeNamedFunctionParameters = 156,
    RuleMaybeNamedParam = 157,
    RuleMaybeNamedFunctionParametersVariadic = 158,
    RuleTraitObjectType = 159,
    RuleTraitObjectTypeOneBound = 160,
    RuleImplTraitType = 161,
    RuleImplTraitTypeOneBound = 162,
    RuleInferredType = 163,
    RuleTypeParamBounds = 164,
    RuleTypeParamBound = 165,
    RuleTraitBound = 166,
    RuleLifetimeBounds = 167,
    RuleLifetime = 168,
    RuleSimplePath = 169,
    RuleSimplePathSegment = 170,
    RulePathInExpression = 171,
    RulePathExprSegment = 172,
    RulePathIdentSegment = 173,
    RuleGenericArgs = 174,
    RuleGenericArg = 175,
    RuleGenericArgsConst = 176,
    RuleGenericArgsLifetimes = 177,
    RuleGenericArgsTypes = 178,
    RuleGenericArgsBindings = 179,
    RuleGenericArgsBinding = 180,
    RuleQualifiedPathInExpression = 181,
    RuleQualifiedPathType = 182,
    RuleQualifiedPathInType = 183,
    RuleTypePath = 184,
    RuleTypePathSegment = 185,
    RuleTypePathFn = 186,
    RuleTypePathInputs = 187,
    RuleVisibility = 188,
    RuleIdentifier = 189,
    RuleKeyword = 190,
    RuleMacroIdentifierLikeToken = 191,
    RuleMacroLiteralToken = 192,
    RuleMacroPunctuationToken = 193,
    RuleShl = 194,
    RuleShr = 195
  };

  bool next(char expect) {
    return _input->LA(1) == static_cast<size_t>(expect);
  }

  explicit RustParser(antlr4::TokenStream *input);

  RustParser(antlr4::TokenStream *input,
             const antlr4::atn::ParserATNSimulatorOptions &options);

  ~RustParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN &getATN() const override;

  const std::vector<std::string> &getRuleNames() const override;

  const antlr4::dfa::Vocabulary &getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  class CrateContext;
  class MacroInvocationContext;
  class DelimTokenTreeContext;
  class TokenTreeContext;
  class TokenTreeTokenContext;
  class MacroInvocationSemiContext;
  class MacroRulesDefinitionContext;
  class MacroRulesDefContext;
  class MacroRulesContext;
  class MacroRuleContext;
  class MacroMatcherContext;
  class MacroMatchContext;
  class MacroMatchTokenContext;
  class MacroFragSpecContext;
  class MacroRepSepContext;
  class MacroRepOpContext;
  class MacroTranscriberContext;
  class ItemContext;
  class VisItemContext;
  class MacroItemContext;
  class ModuleContext;
  class ExternCrateContext;
  class CrateRefContext;
  class AsClauseContext;
  class UseDeclarationContext;
  class UseTreeContext;
  class Function_Context;
  class FunctionQualifiersContext;
  class AbiContext;
  class FunctionParametersContext;
  class SelfParamContext;
  class ShorthandSelfContext;
  class TypedSelfContext;
  class FunctionParamContext;
  class FunctionParamPatternContext;
  class FunctionReturnTypeContext;
  class TypeAliasContext;
  class Struct_Context;
  class StructStructContext;
  class TupleStructContext;
  class StructFieldsContext;
  class StructFieldContext;
  class TupleFieldsContext;
  class TupleFieldContext;
  class EnumerationContext;
  class EnumItemsContext;
  class EnumItemContext;
  class EnumItemTupleContext;
  class EnumItemStructContext;
  class EnumItemDiscriminantContext;
  class Union_Context;
  class ConstantItemContext;
  class StaticItemContext;
  class Trait_Context;
  class ImplementationContext;
  class InherentImplContext;
  class TraitImplContext;
  class ExternBlockContext;
  class ExternalItemContext;
  class GenericParamsContext;
  class GenericParamContext;
  class LifetimeParamContext;
  class TypeParamContext;
  class ConstParamContext;
  class WhereClauseContext;
  class WhereClauseItemContext;
  class LifetimeWhereClauseItemContext;
  class TypeBoundWhereClauseItemContext;
  class ForLifetimesContext;
  class AssociatedItemContext;
  class InnerAttributeContext;
  class OuterAttributeContext;
  class AttrContext;
  class AttrInputContext;
  class StatementContext;
  class LetStatementContext;
  class ExpressionStatementContext;
  class ExpressionContext;
  class ComparisonOperatorContext;
  class CompoundAssignOperatorContext;
  class ExpressionWithBlockContext;
  class LiteralExpressionContext;
  class PathExpressionContext;
  class BlockExpressionContext;
  class StatementsContext;
  class AsyncBlockExpressionContext;
  class UnsafeBlockExpressionContext;
  class ArrayElementsContext;
  class TupleElementsContext;
  class TupleIndexContext;
  class StructExpressionContext;
  class StructExprStructContext;
  class StructExprFieldsContext;
  class StructExprFieldContext;
  class StructBaseContext;
  class StructExprTupleContext;
  class StructExprUnitContext;
  class EnumerationVariantExpressionContext;
  class EnumExprStructContext;
  class EnumExprFieldsContext;
  class EnumExprFieldContext;
  class EnumExprTupleContext;
  class EnumExprFieldlessContext;
  class CallParamsContext;
  class ClosureExpressionContext;
  class ClosureParametersContext;
  class ClosureParamContext;
  class LoopExpressionContext;
  class InfiniteLoopExpressionContext;
  class PredicateLoopExpressionContext;
  class PredicatePatternLoopExpressionContext;
  class IteratorLoopExpressionContext;
  class LoopLabelContext;
  class IfExpressionContext;
  class IfLetExpressionContext;
  class MatchExpressionContext;
  class MatchArmsContext;
  class MatchArmExpressionContext;
  class MatchArmContext;
  class MatchArmGuardContext;
  class PatternContext;
  class PatternNoTopAltContext;
  class PatternWithoutRangeContext;
  class LiteralPatternContext;
  class IdentifierPatternContext;
  class WildcardPatternContext;
  class RestPatternContext;
  class RangePatternContext;
  class RangePatternBoundContext;
  class ReferencePatternContext;
  class StructPatternContext;
  class StructPatternElementsContext;
  class StructPatternFieldsContext;
  class StructPatternFieldContext;
  class StructPatternEtCeteraContext;
  class TupleStructPatternContext;
  class TupleStructItemsContext;
  class TuplePatternContext;
  class TuplePatternItemsContext;
  class GroupedPatternContext;
  class SlicePatternContext;
  class SlicePatternItemsContext;
  class PathPatternContext;
  class Type_Context;
  class TypeNoBoundsContext;
  class ParenthesizedTypeContext;
  class NeverTypeContext;
  class TupleTypeContext;
  class ArrayTypeContext;
  class SliceTypeContext;
  class ReferenceTypeContext;
  class RawPointerTypeContext;
  class BareFunctionTypeContext;
  class FunctionTypeQualifiersContext;
  class BareFunctionReturnTypeContext;
  class FunctionParametersMaybeNamedVariadicContext;
  class MaybeNamedFunctionParametersContext;
  class MaybeNamedParamContext;
  class MaybeNamedFunctionParametersVariadicContext;
  class TraitObjectTypeContext;
  class TraitObjectTypeOneBoundContext;
  class ImplTraitTypeContext;
  class ImplTraitTypeOneBoundContext;
  class InferredTypeContext;
  class TypeParamBoundsContext;
  class TypeParamBoundContext;
  class TraitBoundContext;
  class LifetimeBoundsContext;
  class LifetimeContext;
  class SimplePathContext;
  class SimplePathSegmentContext;
  class PathInExpressionContext;
  class PathExprSegmentContext;
  class PathIdentSegmentContext;
  class GenericArgsContext;
  class GenericArgContext;
  class GenericArgsConstContext;
  class GenericArgsLifetimesContext;
  class GenericArgsTypesContext;
  class GenericArgsBindingsContext;
  class GenericArgsBindingContext;
  class QualifiedPathInExpressionContext;
  class QualifiedPathTypeContext;
  class QualifiedPathInTypeContext;
  class TypePathContext;
  class TypePathSegmentContext;
  class TypePathFnContext;
  class TypePathInputsContext;
  class VisibilityContext;
  class IdentifierContext;
  class KeywordContext;
  class MacroIdentifierLikeTokenContext;
  class MacroLiteralTokenContext;
  class MacroPunctuationTokenContext;
  class ShlContext;
  class ShrContext;

  class CrateContext : public antlr4::ParserRuleContext {
  public:
    CrateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EOF();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    std::vector<ItemContext *> item();
    ItemContext *item(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  CrateContext *crate();

  class MacroInvocationContext : public antlr4::ParserRuleContext {
  public:
    MacroInvocationContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SimplePathContext *simplePath();
    antlr4::tree::TerminalNode *NOT();
    DelimTokenTreeContext *delimTokenTree();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroInvocationContext *macroInvocation();

  class DelimTokenTreeContext : public antlr4::ParserRuleContext {
  public:
    DelimTokenTreeContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<TokenTreeContext *> tokenTree();
    TokenTreeContext *tokenTree(size_t i);
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  DelimTokenTreeContext *delimTokenTree();

  class TokenTreeContext : public antlr4::ParserRuleContext {
  public:
    TokenTreeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TokenTreeTokenContext *> tokenTreeToken();
    TokenTreeTokenContext *tokenTreeToken(size_t i);
    DelimTokenTreeContext *delimTokenTree();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TokenTreeContext *tokenTree();

  class TokenTreeTokenContext : public antlr4::ParserRuleContext {
  public:
    TokenTreeTokenContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MacroIdentifierLikeTokenContext *macroIdentifierLikeToken();
    MacroLiteralTokenContext *macroLiteralToken();
    MacroPunctuationTokenContext *macroPunctuationToken();
    MacroRepOpContext *macroRepOp();
    antlr4::tree::TerminalNode *DOLLAR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TokenTreeTokenContext *tokenTreeToken();

  class MacroInvocationSemiContext : public antlr4::ParserRuleContext {
  public:
    MacroInvocationSemiContext(antlr4::ParserRuleContext *parent,
                               size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SimplePathContext *simplePath();
    antlr4::tree::TerminalNode *NOT();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *SEMI();
    std::vector<TokenTreeContext *> tokenTree();
    TokenTreeContext *tokenTree(size_t i);
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroInvocationSemiContext *macroInvocationSemi();

  class MacroRulesDefinitionContext : public antlr4::ParserRuleContext {
  public:
    MacroRulesDefinitionContext(antlr4::ParserRuleContext *parent,
                                size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_MACRORULES();
    antlr4::tree::TerminalNode *NOT();
    IdentifierContext *identifier();
    MacroRulesDefContext *macroRulesDef();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroRulesDefinitionContext *macroRulesDefinition();

  class MacroRulesDefContext : public antlr4::ParserRuleContext {
  public:
    MacroRulesDefContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    MacroRulesContext *macroRules();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *SEMI();
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroRulesDefContext *macroRulesDef();

  class MacroRulesContext : public antlr4::ParserRuleContext {
  public:
    MacroRulesContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<MacroRuleContext *> macroRule();
    MacroRuleContext *macroRule(size_t i);
    std::vector<antlr4::tree::TerminalNode *> SEMI();
    antlr4::tree::TerminalNode *SEMI(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroRulesContext *macroRules();

  class MacroRuleContext : public antlr4::ParserRuleContext {
  public:
    MacroRuleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MacroMatcherContext *macroMatcher();
    antlr4::tree::TerminalNode *FATARROW();
    MacroTranscriberContext *macroTranscriber();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroRuleContext *macroRule();

  class MacroMatcherContext : public antlr4::ParserRuleContext {
  public:
    MacroMatcherContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<MacroMatchContext *> macroMatch();
    MacroMatchContext *macroMatch(size_t i);
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroMatcherContext *macroMatcher();

  class MacroMatchContext : public antlr4::ParserRuleContext {
  public:
    MacroMatchContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<MacroMatchTokenContext *> macroMatchToken();
    MacroMatchTokenContext *macroMatchToken(size_t i);
    MacroMatcherContext *macroMatcher();
    antlr4::tree::TerminalNode *DOLLAR();
    antlr4::tree::TerminalNode *COLON();
    MacroFragSpecContext *macroFragSpec();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *KW_SELFVALUE();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    MacroRepOpContext *macroRepOp();
    std::vector<MacroMatchContext *> macroMatch();
    MacroMatchContext *macroMatch(size_t i);
    MacroRepSepContext *macroRepSep();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroMatchContext *macroMatch();

  class MacroMatchTokenContext : public antlr4::ParserRuleContext {
  public:
    MacroMatchTokenContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MacroIdentifierLikeTokenContext *macroIdentifierLikeToken();
    MacroLiteralTokenContext *macroLiteralToken();
    MacroPunctuationTokenContext *macroPunctuationToken();
    MacroRepOpContext *macroRepOp();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroMatchTokenContext *macroMatchToken();

  class MacroFragSpecContext : public antlr4::ParserRuleContext {
  public:
    MacroFragSpecContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroFragSpecContext *macroFragSpec();

  class MacroRepSepContext : public antlr4::ParserRuleContext {
  public:
    MacroRepSepContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MacroIdentifierLikeTokenContext *macroIdentifierLikeToken();
    MacroLiteralTokenContext *macroLiteralToken();
    MacroPunctuationTokenContext *macroPunctuationToken();
    antlr4::tree::TerminalNode *DOLLAR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroRepSepContext *macroRepSep();

  class MacroRepOpContext : public antlr4::ParserRuleContext {
  public:
    MacroRepOpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STAR();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *QUESTION();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroRepOpContext *macroRepOp();

  class MacroTranscriberContext : public antlr4::ParserRuleContext {
  public:
    MacroTranscriberContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DelimTokenTreeContext *delimTokenTree();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroTranscriberContext *macroTranscriber();

  class ItemContext : public antlr4::ParserRuleContext {
  public:
    ItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VisItemContext *visItem();
    MacroItemContext *macroItem();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ItemContext *item();

  class VisItemContext : public antlr4::ParserRuleContext {
  public:
    VisItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ModuleContext *module();
    ExternCrateContext *externCrate();
    UseDeclarationContext *useDeclaration();
    Function_Context *function_();
    TypeAliasContext *typeAlias();
    Struct_Context *struct_();
    EnumerationContext *enumeration();
    Union_Context *union_();
    ConstantItemContext *constantItem();
    StaticItemContext *staticItem();
    Trait_Context *trait_();
    ImplementationContext *implementation();
    ExternBlockContext *externBlock();
    VisibilityContext *visibility();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  VisItemContext *visItem();

  class MacroItemContext : public antlr4::ParserRuleContext {
  public:
    MacroItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MacroInvocationSemiContext *macroInvocationSemi();
    MacroRulesDefinitionContext *macroRulesDefinition();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroItemContext *macroItem();

  class ModuleContext : public antlr4::ParserRuleContext {
  public:
    ModuleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_MOD();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *SEMI();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    antlr4::tree::TerminalNode *KW_UNSAFE();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    std::vector<ItemContext *> item();
    ItemContext *item(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ModuleContext *module();

  class ExternCrateContext : public antlr4::ParserRuleContext {
  public:
    ExternCrateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_EXTERN();
    antlr4::tree::TerminalNode *KW_CRATE();
    CrateRefContext *crateRef();
    antlr4::tree::TerminalNode *SEMI();
    AsClauseContext *asClause();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExternCrateContext *externCrate();

  class CrateRefContext : public antlr4::ParserRuleContext {
  public:
    CrateRefContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *KW_SELFVALUE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  CrateRefContext *crateRef();

  class AsClauseContext : public antlr4::ParserRuleContext {
  public:
    AsClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_AS();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *UNDERSCORE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AsClauseContext *asClause();

  class UseDeclarationContext : public antlr4::ParserRuleContext {
  public:
    UseDeclarationContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_USE();
    UseTreeContext *useTree();
    antlr4::tree::TerminalNode *SEMI();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  UseDeclarationContext *useDeclaration();

  class UseTreeContext : public antlr4::ParserRuleContext {
  public:
    UseTreeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STAR();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    antlr4::tree::TerminalNode *PATHSEP();
    std::vector<UseTreeContext *> useTree();
    UseTreeContext *useTree(size_t i);
    SimplePathContext *simplePath();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);
    antlr4::tree::TerminalNode *KW_AS();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *UNDERSCORE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  UseTreeContext *useTree();

  class Function_Context : public antlr4::ParserRuleContext {
  public:
    Function_Context(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FunctionQualifiersContext *functionQualifiers();
    antlr4::tree::TerminalNode *KW_FN();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    BlockExpressionContext *blockExpression();
    antlr4::tree::TerminalNode *SEMI();
    GenericParamsContext *genericParams();
    FunctionParametersContext *functionParameters();
    FunctionReturnTypeContext *functionReturnType();
    WhereClauseContext *whereClause();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Function_Context *function_();

  class FunctionQualifiersContext : public antlr4::ParserRuleContext {
  public:
    FunctionQualifiersContext(antlr4::ParserRuleContext *parent,
                              size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_CONST();
    antlr4::tree::TerminalNode *KW_ASYNC();
    antlr4::tree::TerminalNode *KW_UNSAFE();
    antlr4::tree::TerminalNode *KW_EXTERN();
    AbiContext *abi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionQualifiersContext *functionQualifiers();

  class AbiContext : public antlr4::ParserRuleContext {
  public:
    AbiContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STRING_LITERAL();
    antlr4::tree::TerminalNode *RAW_STRING_LITERAL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AbiContext *abi();

  class FunctionParametersContext : public antlr4::ParserRuleContext {
  public:
    FunctionParametersContext(antlr4::ParserRuleContext *parent,
                              size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SelfParamContext *selfParam();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);
    std::vector<FunctionParamContext *> functionParam();
    FunctionParamContext *functionParam(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionParametersContext *functionParameters();

  class SelfParamContext : public antlr4::ParserRuleContext {
  public:
    SelfParamContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ShorthandSelfContext *shorthandSelf();
    TypedSelfContext *typedSelf();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  SelfParamContext *selfParam();

  class ShorthandSelfContext : public antlr4::ParserRuleContext {
  public:
    ShorthandSelfContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_SELFVALUE();
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *KW_MUT();
    LifetimeContext *lifetime();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ShorthandSelfContext *shorthandSelf();

  class TypedSelfContext : public antlr4::ParserRuleContext {
  public:
    TypedSelfContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_SELFVALUE();
    antlr4::tree::TerminalNode *COLON();
    Type_Context *type_();
    antlr4::tree::TerminalNode *KW_MUT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypedSelfContext *typedSelf();

  class FunctionParamContext : public antlr4::ParserRuleContext {
  public:
    FunctionParamContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FunctionParamPatternContext *functionParamPattern();
    antlr4::tree::TerminalNode *DOTDOTDOT();
    Type_Context *type_();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionParamContext *functionParam();

  class FunctionParamPatternContext : public antlr4::ParserRuleContext {
  public:
    FunctionParamPatternContext(antlr4::ParserRuleContext *parent,
                                size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PatternContext *pattern();
    antlr4::tree::TerminalNode *COLON();
    Type_Context *type_();
    antlr4::tree::TerminalNode *DOTDOTDOT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionParamPatternContext *functionParamPattern();

  class FunctionReturnTypeContext : public antlr4::ParserRuleContext {
  public:
    FunctionReturnTypeContext(antlr4::ParserRuleContext *parent,
                              size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RARROW();
    Type_Context *type_();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionReturnTypeContext *functionReturnType();

  class TypeAliasContext : public antlr4::ParserRuleContext {
  public:
    TypeAliasContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_TYPE();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *SEMI();
    GenericParamsContext *genericParams();
    WhereClauseContext *whereClause();
    antlr4::tree::TerminalNode *EQ();
    Type_Context *type_();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypeAliasContext *typeAlias();

  class Struct_Context : public antlr4::ParserRuleContext {
  public:
    Struct_Context(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StructStructContext *structStruct();
    TupleStructContext *tupleStruct();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Struct_Context *struct_();

  class StructStructContext : public antlr4::ParserRuleContext {
  public:
    StructStructContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_STRUCT();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    antlr4::tree::TerminalNode *SEMI();
    GenericParamsContext *genericParams();
    WhereClauseContext *whereClause();
    StructFieldsContext *structFields();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructStructContext *structStruct();

  class TupleStructContext : public antlr4::ParserRuleContext {
  public:
    TupleStructContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_STRUCT();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *SEMI();
    GenericParamsContext *genericParams();
    TupleFieldsContext *tupleFields();
    WhereClauseContext *whereClause();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TupleStructContext *tupleStruct();

  class StructFieldsContext : public antlr4::ParserRuleContext {
  public:
    StructFieldsContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<StructFieldContext *> structField();
    StructFieldContext *structField(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructFieldsContext *structFields();

  class StructFieldContext : public antlr4::ParserRuleContext {
  public:
    StructFieldContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *COLON();
    Type_Context *type_();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    VisibilityContext *visibility();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructFieldContext *structField();

  class TupleFieldsContext : public antlr4::ParserRuleContext {
  public:
    TupleFieldsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TupleFieldContext *> tupleField();
    TupleFieldContext *tupleField(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TupleFieldsContext *tupleFields();

  class TupleFieldContext : public antlr4::ParserRuleContext {
  public:
    TupleFieldContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Type_Context *type_();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    VisibilityContext *visibility();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TupleFieldContext *tupleField();

  class EnumerationContext : public antlr4::ParserRuleContext {
  public:
    EnumerationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_ENUM();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    GenericParamsContext *genericParams();
    WhereClauseContext *whereClause();
    EnumItemsContext *enumItems();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumerationContext *enumeration();

  class EnumItemsContext : public antlr4::ParserRuleContext {
  public:
    EnumItemsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<EnumItemContext *> enumItem();
    EnumItemContext *enumItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumItemsContext *enumItems();

  class EnumItemContext : public antlr4::ParserRuleContext {
  public:
    EnumItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    VisibilityContext *visibility();
    EnumItemTupleContext *enumItemTuple();
    EnumItemStructContext *enumItemStruct();
    EnumItemDiscriminantContext *enumItemDiscriminant();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumItemContext *enumItem();

  class EnumItemTupleContext : public antlr4::ParserRuleContext {
  public:
    EnumItemTupleContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    TupleFieldsContext *tupleFields();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumItemTupleContext *enumItemTuple();

  class EnumItemStructContext : public antlr4::ParserRuleContext {
  public:
    EnumItemStructContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    StructFieldsContext *structFields();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumItemStructContext *enumItemStruct();

  class EnumItemDiscriminantContext : public antlr4::ParserRuleContext {
  public:
    EnumItemDiscriminantContext(antlr4::ParserRuleContext *parent,
                                size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EQ();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumItemDiscriminantContext *enumItemDiscriminant();

  class Union_Context : public antlr4::ParserRuleContext {
  public:
    Union_Context(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_UNION();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    StructFieldsContext *structFields();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    GenericParamsContext *genericParams();
    WhereClauseContext *whereClause();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Union_Context *union_();

  class ConstantItemContext : public antlr4::ParserRuleContext {
  public:
    ConstantItemContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_CONST();
    antlr4::tree::TerminalNode *COLON();
    Type_Context *type_();
    antlr4::tree::TerminalNode *SEMI();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *UNDERSCORE();
    antlr4::tree::TerminalNode *EQ();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ConstantItemContext *constantItem();

  class StaticItemContext : public antlr4::ParserRuleContext {
  public:
    StaticItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_STATIC();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *COLON();
    Type_Context *type_();
    antlr4::tree::TerminalNode *SEMI();
    antlr4::tree::TerminalNode *KW_MUT();
    antlr4::tree::TerminalNode *EQ();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StaticItemContext *staticItem();

  class Trait_Context : public antlr4::ParserRuleContext {
  public:
    Trait_Context(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_TRAIT();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    antlr4::tree::TerminalNode *KW_UNSAFE();
    GenericParamsContext *genericParams();
    antlr4::tree::TerminalNode *COLON();
    WhereClauseContext *whereClause();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    std::vector<AssociatedItemContext *> associatedItem();
    AssociatedItemContext *associatedItem(size_t i);
    TypeParamBoundsContext *typeParamBounds();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Trait_Context *trait_();

  class ImplementationContext : public antlr4::ParserRuleContext {
  public:
    ImplementationContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    InherentImplContext *inherentImpl();
    TraitImplContext *traitImpl();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ImplementationContext *implementation();

  class InherentImplContext : public antlr4::ParserRuleContext {
  public:
    InherentImplContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_IMPL();
    Type_Context *type_();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    GenericParamsContext *genericParams();
    WhereClauseContext *whereClause();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    std::vector<AssociatedItemContext *> associatedItem();
    AssociatedItemContext *associatedItem(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  InherentImplContext *inherentImpl();

  class TraitImplContext : public antlr4::ParserRuleContext {
  public:
    TraitImplContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_IMPL();
    TypePathContext *typePath();
    antlr4::tree::TerminalNode *KW_FOR();
    Type_Context *type_();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    antlr4::tree::TerminalNode *KW_UNSAFE();
    GenericParamsContext *genericParams();
    antlr4::tree::TerminalNode *NOT();
    WhereClauseContext *whereClause();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    std::vector<AssociatedItemContext *> associatedItem();
    AssociatedItemContext *associatedItem(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TraitImplContext *traitImpl();

  class ExternBlockContext : public antlr4::ParserRuleContext {
  public:
    ExternBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_EXTERN();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    antlr4::tree::TerminalNode *KW_UNSAFE();
    AbiContext *abi();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    std::vector<ExternalItemContext *> externalItem();
    ExternalItemContext *externalItem(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExternBlockContext *externBlock();

  class ExternalItemContext : public antlr4::ParserRuleContext {
  public:
    ExternalItemContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MacroInvocationSemiContext *macroInvocationSemi();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    StaticItemContext *staticItem();
    Function_Context *function_();
    VisibilityContext *visibility();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExternalItemContext *externalItem();

  class GenericParamsContext : public antlr4::ParserRuleContext {
  public:
    GenericParamsContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GT();
    std::vector<GenericParamContext *> genericParam();
    GenericParamContext *genericParam(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericParamsContext *genericParams();

  class GenericParamContext : public antlr4::ParserRuleContext {
  public:
    GenericParamContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LifetimeParamContext *lifetimeParam();
    TypeParamContext *typeParam();
    ConstParamContext *constParam();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericParamContext *genericParam();

  class LifetimeParamContext : public antlr4::ParserRuleContext {
  public:
    LifetimeParamContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LIFETIME_OR_LABEL();
    OuterAttributeContext *outerAttribute();
    antlr4::tree::TerminalNode *COLON();
    LifetimeBoundsContext *lifetimeBounds();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LifetimeParamContext *lifetimeParam();

  class TypeParamContext : public antlr4::ParserRuleContext {
  public:
    TypeParamContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    OuterAttributeContext *outerAttribute();
    antlr4::tree::TerminalNode *COLON();
    antlr4::tree::TerminalNode *EQ();
    Type_Context *type_();
    TypeParamBoundsContext *typeParamBounds();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypeParamContext *typeParam();

  class ConstParamContext : public antlr4::ParserRuleContext {
  public:
    ConstParamContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_CONST();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *COLON();
    Type_Context *type_();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ConstParamContext *constParam();

  class WhereClauseContext : public antlr4::ParserRuleContext {
  public:
    WhereClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_WHERE();
    std::vector<WhereClauseItemContext *> whereClauseItem();
    WhereClauseItemContext *whereClauseItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  WhereClauseContext *whereClause();

  class WhereClauseItemContext : public antlr4::ParserRuleContext {
  public:
    WhereClauseItemContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LifetimeWhereClauseItemContext *lifetimeWhereClauseItem();
    TypeBoundWhereClauseItemContext *typeBoundWhereClauseItem();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  WhereClauseItemContext *whereClauseItem();

  class LifetimeWhereClauseItemContext : public antlr4::ParserRuleContext {
  public:
    LifetimeWhereClauseItemContext(antlr4::ParserRuleContext *parent,
                                   size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LifetimeContext *lifetime();
    antlr4::tree::TerminalNode *COLON();
    LifetimeBoundsContext *lifetimeBounds();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LifetimeWhereClauseItemContext *lifetimeWhereClauseItem();

  class TypeBoundWhereClauseItemContext : public antlr4::ParserRuleContext {
  public:
    TypeBoundWhereClauseItemContext(antlr4::ParserRuleContext *parent,
                                    size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Type_Context *type_();
    antlr4::tree::TerminalNode *COLON();
    ForLifetimesContext *forLifetimes();
    TypeParamBoundsContext *typeParamBounds();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypeBoundWhereClauseItemContext *typeBoundWhereClauseItem();

  class ForLifetimesContext : public antlr4::ParserRuleContext {
  public:
    ForLifetimesContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_FOR();
    GenericParamsContext *genericParams();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ForLifetimesContext *forLifetimes();

  class AssociatedItemContext : public antlr4::ParserRuleContext {
  public:
    AssociatedItemContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MacroInvocationSemiContext *macroInvocationSemi();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    TypeAliasContext *typeAlias();
    ConstantItemContext *constantItem();
    Function_Context *function_();
    VisibilityContext *visibility();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AssociatedItemContext *associatedItem();

  class InnerAttributeContext : public antlr4::ParserRuleContext {
  public:
    InnerAttributeContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *POUND();
    antlr4::tree::TerminalNode *NOT();
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    AttrContext *attr();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  InnerAttributeContext *innerAttribute();

  class OuterAttributeContext : public antlr4::ParserRuleContext {
  public:
    OuterAttributeContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *POUND();
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    AttrContext *attr();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  OuterAttributeContext *outerAttribute();

  class AttrContext : public antlr4::ParserRuleContext {
  public:
    AttrContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SimplePathContext *simplePath();
    AttrInputContext *attrInput();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AttrContext *attr();

  class AttrInputContext : public antlr4::ParserRuleContext {
  public:
    AttrInputContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DelimTokenTreeContext *delimTokenTree();
    antlr4::tree::TerminalNode *EQ();
    LiteralExpressionContext *literalExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AttrInputContext *attrInput();

  class StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *SEMI();
    ItemContext *item();
    LetStatementContext *letStatement();
    ExpressionStatementContext *expressionStatement();
    MacroInvocationSemiContext *macroInvocationSemi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StatementContext *statement();

  class LetStatementContext : public antlr4::ParserRuleContext {
  public:
    LetStatementContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_LET();
    PatternNoTopAltContext *patternNoTopAlt();
    antlr4::tree::TerminalNode *SEMI();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    antlr4::tree::TerminalNode *COLON();
    Type_Context *type_();
    antlr4::tree::TerminalNode *EQ();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LetStatementContext *letStatement();

  class ExpressionStatementContext : public antlr4::ParserRuleContext {
  public:
    ExpressionStatementContext(antlr4::ParserRuleContext *parent,
                               size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *SEMI();
    ExpressionWithBlockContext *expressionWithBlock();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExpressionStatementContext *expressionStatement();

  class ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);

    ExpressionContext() = default;
    void copyFrom(ExpressionContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;
  };

  class TypeCastExpressionContext : public ExpressionContext {
  public:
    TypeCastExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *KW_AS();
    TypeNoBoundsContext *typeNoBounds();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class PathExpression_Context : public ExpressionContext {
  public:
    PathExpression_Context(ExpressionContext *ctx);

    PathExpressionContext *pathExpression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class TupleExpressionContext : public ExpressionContext {
  public:
    TupleExpressionContext(ExpressionContext *ctx);

    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    TupleElementsContext *tupleElements();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class IndexExpressionContext : public ExpressionContext {
  public:
    IndexExpressionContext(ExpressionContext *ctx);

    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class RangeExpressionContext : public ExpressionContext {
  public:
    RangeExpressionContext(ExpressionContext *ctx);

    antlr4::tree::TerminalNode *DOTDOT();
    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    antlr4::tree::TerminalNode *DOTDOTEQ();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class MacroInvocationAsExpressionContext : public ExpressionContext {
  public:
    MacroInvocationAsExpressionContext(ExpressionContext *ctx);

    MacroInvocationContext *macroInvocation();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ReturnExpressionContext : public ExpressionContext {
  public:
    ReturnExpressionContext(ExpressionContext *ctx);

    antlr4::tree::TerminalNode *KW_RETURN();
    ExpressionContext *expression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class AwaitExpressionContext : public ExpressionContext {
  public:
    AwaitExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *DOT();
    antlr4::tree::TerminalNode *KW_AWAIT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ErrorPropagationExpressionContext : public ExpressionContext {
  public:
    ErrorPropagationExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *QUESTION();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ContinueExpressionContext : public ExpressionContext {
  public:
    ContinueExpressionContext(ExpressionContext *ctx);

    antlr4::tree::TerminalNode *KW_CONTINUE();
    antlr4::tree::TerminalNode *LIFETIME_OR_LABEL();
    ExpressionContext *expression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class AssignmentExpressionContext : public ExpressionContext {
  public:
    AssignmentExpressionContext(ExpressionContext *ctx);

    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    antlr4::tree::TerminalNode *EQ();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class MethodCallExpressionContext : public ExpressionContext {
  public:
    MethodCallExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *DOT();
    PathExprSegmentContext *pathExprSegment();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    CallParamsContext *callParams();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class LiteralExpression_Context : public ExpressionContext {
  public:
    LiteralExpression_Context(ExpressionContext *ctx);

    LiteralExpressionContext *literalExpression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class StructExpression_Context : public ExpressionContext {
  public:
    StructExpression_Context(ExpressionContext *ctx);

    StructExpressionContext *structExpression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class TupleIndexingExpressionContext : public ExpressionContext {
  public:
    TupleIndexingExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *DOT();
    TupleIndexContext *tupleIndex();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class NegationExpressionContext : public ExpressionContext {
  public:
    NegationExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *NOT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class CallExpressionContext : public ExpressionContext {
  public:
    CallExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    CallParamsContext *callParams();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class LazyBooleanExpressionContext : public ExpressionContext {
  public:
    LazyBooleanExpressionContext(ExpressionContext *ctx);

    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    antlr4::tree::TerminalNode *ANDAND();
    antlr4::tree::TerminalNode *OROR();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class DereferenceExpressionContext : public ExpressionContext {
  public:
    DereferenceExpressionContext(ExpressionContext *ctx);

    antlr4::tree::TerminalNode *STAR();
    ExpressionContext *expression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ExpressionWithBlock_Context : public ExpressionContext {
  public:
    ExpressionWithBlock_Context(ExpressionContext *ctx);

    ExpressionWithBlockContext *expressionWithBlock();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class GroupedExpressionContext : public ExpressionContext {
  public:
    GroupedExpressionContext(ExpressionContext *ctx);

    antlr4::tree::TerminalNode *LPAREN();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class BreakExpressionContext : public ExpressionContext {
  public:
    BreakExpressionContext(ExpressionContext *ctx);

    antlr4::tree::TerminalNode *KW_BREAK();
    antlr4::tree::TerminalNode *LIFETIME_OR_LABEL();
    ExpressionContext *expression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ArithmeticOrLogicalExpressionContext : public ExpressionContext {
  public:
    ArithmeticOrLogicalExpressionContext(ExpressionContext *ctx);

    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    antlr4::tree::TerminalNode *STAR();
    antlr4::tree::TerminalNode *SLASH();
    antlr4::tree::TerminalNode *PERCENT();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();
    ShlContext *shl();
    ShrContext *shr();
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *CARET();
    antlr4::tree::TerminalNode *OR();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class FieldExpressionContext : public ExpressionContext {
  public:
    FieldExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *DOT();
    IdentifierContext *identifier();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class EnumerationVariantExpression_Context : public ExpressionContext {
  public:
    EnumerationVariantExpression_Context(ExpressionContext *ctx);

    EnumerationVariantExpressionContext *enumerationVariantExpression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ComparisonExpressionContext : public ExpressionContext {
  public:
    ComparisonExpressionContext(ExpressionContext *ctx);

    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    ComparisonOperatorContext *comparisonOperator();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class AttributedExpressionContext : public ExpressionContext {
  public:
    AttributedExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class BorrowExpressionContext : public ExpressionContext {
  public:
    BorrowExpressionContext(ExpressionContext *ctx);

    ExpressionContext *expression();
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *ANDAND();
    antlr4::tree::TerminalNode *KW_MUT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class CompoundAssignmentExpressionContext : public ExpressionContext {
  public:
    CompoundAssignmentExpressionContext(ExpressionContext *ctx);

    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    CompoundAssignOperatorContext *compoundAssignOperator();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ClosureExpression_Context : public ExpressionContext {
  public:
    ClosureExpression_Context(ExpressionContext *ctx);

    ClosureExpressionContext *closureExpression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ArrayExpressionContext : public ExpressionContext {
  public:
    ArrayExpressionContext(ExpressionContext *ctx);

    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    ArrayElementsContext *arrayElements();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExpressionContext *expression();
  ExpressionContext *expression(int precedence);
  class ComparisonOperatorContext : public antlr4::ParserRuleContext {
  public:
    ComparisonOperatorContext(antlr4::ParserRuleContext *parent,
                              size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EQEQ();
    antlr4::tree::TerminalNode *NE();
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GE();
    antlr4::tree::TerminalNode *LE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ComparisonOperatorContext *comparisonOperator();

  class CompoundAssignOperatorContext : public antlr4::ParserRuleContext {
  public:
    CompoundAssignOperatorContext(antlr4::ParserRuleContext *parent,
                                  size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *PLUSEQ();
    antlr4::tree::TerminalNode *MINUSEQ();
    antlr4::tree::TerminalNode *STAREQ();
    antlr4::tree::TerminalNode *SLASHEQ();
    antlr4::tree::TerminalNode *PERCENTEQ();
    antlr4::tree::TerminalNode *ANDEQ();
    antlr4::tree::TerminalNode *OREQ();
    antlr4::tree::TerminalNode *CARETEQ();
    antlr4::tree::TerminalNode *SHLEQ();
    antlr4::tree::TerminalNode *SHREQ();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  CompoundAssignOperatorContext *compoundAssignOperator();

  class ExpressionWithBlockContext : public antlr4::ParserRuleContext {
  public:
    ExpressionWithBlockContext(antlr4::ParserRuleContext *parent,
                               size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionWithBlockContext *expressionWithBlock();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    BlockExpressionContext *blockExpression();
    AsyncBlockExpressionContext *asyncBlockExpression();
    UnsafeBlockExpressionContext *unsafeBlockExpression();
    LoopExpressionContext *loopExpression();
    IfExpressionContext *ifExpression();
    IfLetExpressionContext *ifLetExpression();
    MatchExpressionContext *matchExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExpressionWithBlockContext *expressionWithBlock();

  class LiteralExpressionContext : public antlr4::ParserRuleContext {
  public:
    LiteralExpressionContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CHAR_LITERAL();
    antlr4::tree::TerminalNode *STRING_LITERAL();
    antlr4::tree::TerminalNode *RAW_STRING_LITERAL();
    antlr4::tree::TerminalNode *BYTE_LITERAL();
    antlr4::tree::TerminalNode *BYTE_STRING_LITERAL();
    antlr4::tree::TerminalNode *RAW_BYTE_STRING_LITERAL();
    antlr4::tree::TerminalNode *INTEGER_LITERAL();
    antlr4::tree::TerminalNode *FLOAT_LITERAL();
    antlr4::tree::TerminalNode *KW_TRUE();
    antlr4::tree::TerminalNode *KW_FALSE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LiteralExpressionContext *literalExpression();

  class PathExpressionContext : public antlr4::ParserRuleContext {
  public:
    PathExpressionContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();
    QualifiedPathInExpressionContext *qualifiedPathInExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PathExpressionContext *pathExpression();

  class BlockExpressionContext : public antlr4::ParserRuleContext {
  public:
    BlockExpressionContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    StatementsContext *statements();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  BlockExpressionContext *blockExpression();

  class StatementsContext : public antlr4::ParserRuleContext {
  public:
    StatementsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<StatementContext *> statement();
    StatementContext *statement(size_t i);
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StatementsContext *statements();

  class AsyncBlockExpressionContext : public antlr4::ParserRuleContext {
  public:
    AsyncBlockExpressionContext(antlr4::ParserRuleContext *parent,
                                size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_ASYNC();
    BlockExpressionContext *blockExpression();
    antlr4::tree::TerminalNode *KW_MOVE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AsyncBlockExpressionContext *asyncBlockExpression();

  class UnsafeBlockExpressionContext : public antlr4::ParserRuleContext {
  public:
    UnsafeBlockExpressionContext(antlr4::ParserRuleContext *parent,
                                 size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_UNSAFE();
    BlockExpressionContext *blockExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  UnsafeBlockExpressionContext *unsafeBlockExpression();

  class ArrayElementsContext : public antlr4::ParserRuleContext {
  public:
    ArrayElementsContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);
    antlr4::tree::TerminalNode *SEMI();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ArrayElementsContext *arrayElements();

  class TupleElementsContext : public antlr4::ParserRuleContext {
  public:
    TupleElementsContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TupleElementsContext *tupleElements();

  class TupleIndexContext : public antlr4::ParserRuleContext {
  public:
    TupleIndexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INTEGER_LITERAL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TupleIndexContext *tupleIndex();

  class StructExpressionContext : public antlr4::ParserRuleContext {
  public:
    StructExpressionContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StructExprStructContext *structExprStruct();
    StructExprTupleContext *structExprTuple();
    StructExprUnitContext *structExprUnit();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructExpressionContext *structExpression();

  class StructExprStructContext : public antlr4::ParserRuleContext {
  public:
    StructExprStructContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    StructExprFieldsContext *structExprFields();
    StructBaseContext *structBase();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructExprStructContext *structExprStruct();

  class StructExprFieldsContext : public antlr4::ParserRuleContext {
  public:
    StructExprFieldsContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<StructExprFieldContext *> structExprField();
    StructExprFieldContext *structExprField(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);
    StructBaseContext *structBase();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructExprFieldsContext *structExprFields();

  class StructExprFieldContext : public antlr4::ParserRuleContext {
  public:
    StructExprFieldContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *COLON();
    ExpressionContext *expression();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    TupleIndexContext *tupleIndex();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructExprFieldContext *structExprField();

  class StructBaseContext : public antlr4::ParserRuleContext {
  public:
    StructBaseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DOTDOT();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructBaseContext *structBase();

  class StructExprTupleContext : public antlr4::ParserRuleContext {
  public:
    StructExprTupleContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructExprTupleContext *structExprTuple();

  class StructExprUnitContext : public antlr4::ParserRuleContext {
  public:
    StructExprUnitContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructExprUnitContext *structExprUnit();

  class EnumerationVariantExpressionContext : public antlr4::ParserRuleContext {
  public:
    EnumerationVariantExpressionContext(antlr4::ParserRuleContext *parent,
                                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    EnumExprStructContext *enumExprStruct();
    EnumExprTupleContext *enumExprTuple();
    EnumExprFieldlessContext *enumExprFieldless();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumerationVariantExpressionContext *enumerationVariantExpression();

  class EnumExprStructContext : public antlr4::ParserRuleContext {
  public:
    EnumExprStructContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    EnumExprFieldsContext *enumExprFields();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumExprStructContext *enumExprStruct();

  class EnumExprFieldsContext : public antlr4::ParserRuleContext {
  public:
    EnumExprFieldsContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<EnumExprFieldContext *> enumExprField();
    EnumExprFieldContext *enumExprField(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumExprFieldsContext *enumExprFields();

  class EnumExprFieldContext : public antlr4::ParserRuleContext {
  public:
    EnumExprFieldContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *COLON();
    ExpressionContext *expression();
    TupleIndexContext *tupleIndex();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumExprFieldContext *enumExprField();

  class EnumExprTupleContext : public antlr4::ParserRuleContext {
  public:
    EnumExprTupleContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumExprTupleContext *enumExprTuple();

  class EnumExprFieldlessContext : public antlr4::ParserRuleContext {
  public:
    EnumExprFieldlessContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  EnumExprFieldlessContext *enumExprFieldless();

  class CallParamsContext : public antlr4::ParserRuleContext {
  public:
    CallParamsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext *expression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  CallParamsContext *callParams();

  class ClosureExpressionContext : public antlr4::ParserRuleContext {
  public:
    ClosureExpressionContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *OROR();
    std::vector<antlr4::tree::TerminalNode *> OR();
    antlr4::tree::TerminalNode *OR(size_t i);
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RARROW();
    TypeNoBoundsContext *typeNoBounds();
    BlockExpressionContext *blockExpression();
    antlr4::tree::TerminalNode *KW_MOVE();
    ClosureParametersContext *closureParameters();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ClosureExpressionContext *closureExpression();

  class ClosureParametersContext : public antlr4::ParserRuleContext {
  public:
    ClosureParametersContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ClosureParamContext *> closureParam();
    ClosureParamContext *closureParam(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ClosureParametersContext *closureParameters();

  class ClosureParamContext : public antlr4::ParserRuleContext {
  public:
    ClosureParamContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PatternContext *pattern();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    antlr4::tree::TerminalNode *COLON();
    Type_Context *type_();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ClosureParamContext *closureParam();

  class LoopExpressionContext : public antlr4::ParserRuleContext {
  public:
    LoopExpressionContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    InfiniteLoopExpressionContext *infiniteLoopExpression();
    PredicateLoopExpressionContext *predicateLoopExpression();
    PredicatePatternLoopExpressionContext *predicatePatternLoopExpression();
    IteratorLoopExpressionContext *iteratorLoopExpression();
    LoopLabelContext *loopLabel();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LoopExpressionContext *loopExpression();

  class InfiniteLoopExpressionContext : public antlr4::ParserRuleContext {
  public:
    InfiniteLoopExpressionContext(antlr4::ParserRuleContext *parent,
                                  size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_LOOP();
    BlockExpressionContext *blockExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  InfiniteLoopExpressionContext *infiniteLoopExpression();

  class PredicateLoopExpressionContext : public antlr4::ParserRuleContext {
  public:
    PredicateLoopExpressionContext(antlr4::ParserRuleContext *parent,
                                   size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_WHILE();
    ExpressionContext *expression();
    BlockExpressionContext *blockExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PredicateLoopExpressionContext *predicateLoopExpression();

  class PredicatePatternLoopExpressionContext
      : public antlr4::ParserRuleContext {
  public:
    PredicatePatternLoopExpressionContext(antlr4::ParserRuleContext *parent,
                                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_WHILE();
    antlr4::tree::TerminalNode *KW_LET();
    PatternContext *pattern();
    antlr4::tree::TerminalNode *EQ();
    ExpressionContext *expression();
    BlockExpressionContext *blockExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PredicatePatternLoopExpressionContext *predicatePatternLoopExpression();

  class IteratorLoopExpressionContext : public antlr4::ParserRuleContext {
  public:
    IteratorLoopExpressionContext(antlr4::ParserRuleContext *parent,
                                  size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_FOR();
    PatternContext *pattern();
    antlr4::tree::TerminalNode *KW_IN();
    ExpressionContext *expression();
    BlockExpressionContext *blockExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  IteratorLoopExpressionContext *iteratorLoopExpression();

  class LoopLabelContext : public antlr4::ParserRuleContext {
  public:
    LoopLabelContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LIFETIME_OR_LABEL();
    antlr4::tree::TerminalNode *COLON();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LoopLabelContext *loopLabel();

  class IfExpressionContext : public antlr4::ParserRuleContext {
  public:
    IfExpressionContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_IF();
    ExpressionContext *expression();
    std::vector<BlockExpressionContext *> blockExpression();
    BlockExpressionContext *blockExpression(size_t i);
    antlr4::tree::TerminalNode *KW_ELSE();
    IfExpressionContext *ifExpression();
    IfLetExpressionContext *ifLetExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  IfExpressionContext *ifExpression();

  class IfLetExpressionContext : public antlr4::ParserRuleContext {
  public:
    IfLetExpressionContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_IF();
    antlr4::tree::TerminalNode *KW_LET();
    PatternContext *pattern();
    antlr4::tree::TerminalNode *EQ();
    ExpressionContext *expression();
    std::vector<BlockExpressionContext *> blockExpression();
    BlockExpressionContext *blockExpression(size_t i);
    antlr4::tree::TerminalNode *KW_ELSE();
    IfExpressionContext *ifExpression();
    IfLetExpressionContext *ifLetExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  IfLetExpressionContext *ifLetExpression();

  class MatchExpressionContext : public antlr4::ParserRuleContext {
  public:
    MatchExpressionContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_MATCH();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    std::vector<InnerAttributeContext *> innerAttribute();
    InnerAttributeContext *innerAttribute(size_t i);
    MatchArmsContext *matchArms();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MatchExpressionContext *matchExpression();

  class MatchArmsContext : public antlr4::ParserRuleContext {
  public:
    MatchArmsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<MatchArmContext *> matchArm();
    MatchArmContext *matchArm(size_t i);
    std::vector<antlr4::tree::TerminalNode *> FATARROW();
    antlr4::tree::TerminalNode *FATARROW(size_t i);
    ExpressionContext *expression();
    std::vector<MatchArmExpressionContext *> matchArmExpression();
    MatchArmExpressionContext *matchArmExpression(size_t i);
    antlr4::tree::TerminalNode *COMMA();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MatchArmsContext *matchArms();

  class MatchArmExpressionContext : public antlr4::ParserRuleContext {
  public:
    MatchArmExpressionContext(antlr4::ParserRuleContext *parent,
                              size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *COMMA();
    ExpressionWithBlockContext *expressionWithBlock();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MatchArmExpressionContext *matchArmExpression();

  class MatchArmContext : public antlr4::ParserRuleContext {
  public:
    MatchArmContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PatternContext *pattern();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    MatchArmGuardContext *matchArmGuard();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MatchArmContext *matchArm();

  class MatchArmGuardContext : public antlr4::ParserRuleContext {
  public:
    MatchArmGuardContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_IF();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MatchArmGuardContext *matchArmGuard();

  class PatternContext : public antlr4::ParserRuleContext {
  public:
    PatternContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PatternNoTopAltContext *> patternNoTopAlt();
    PatternNoTopAltContext *patternNoTopAlt(size_t i);
    std::vector<antlr4::tree::TerminalNode *> OR();
    antlr4::tree::TerminalNode *OR(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PatternContext *pattern();

  class PatternNoTopAltContext : public antlr4::ParserRuleContext {
  public:
    PatternNoTopAltContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PatternWithoutRangeContext *patternWithoutRange();
    RangePatternContext *rangePattern();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PatternNoTopAltContext *patternNoTopAlt();

  class PatternWithoutRangeContext : public antlr4::ParserRuleContext {
  public:
    PatternWithoutRangeContext(antlr4::ParserRuleContext *parent,
                               size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LiteralPatternContext *literalPattern();
    IdentifierPatternContext *identifierPattern();
    WildcardPatternContext *wildcardPattern();
    RestPatternContext *restPattern();
    ReferencePatternContext *referencePattern();
    StructPatternContext *structPattern();
    TupleStructPatternContext *tupleStructPattern();
    TuplePatternContext *tuplePattern();
    GroupedPatternContext *groupedPattern();
    SlicePatternContext *slicePattern();
    PathPatternContext *pathPattern();
    MacroInvocationContext *macroInvocation();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PatternWithoutRangeContext *patternWithoutRange();

  class LiteralPatternContext : public antlr4::ParserRuleContext {
  public:
    LiteralPatternContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_TRUE();
    antlr4::tree::TerminalNode *KW_FALSE();
    antlr4::tree::TerminalNode *CHAR_LITERAL();
    antlr4::tree::TerminalNode *BYTE_LITERAL();
    antlr4::tree::TerminalNode *STRING_LITERAL();
    antlr4::tree::TerminalNode *RAW_STRING_LITERAL();
    antlr4::tree::TerminalNode *BYTE_STRING_LITERAL();
    antlr4::tree::TerminalNode *RAW_BYTE_STRING_LITERAL();
    antlr4::tree::TerminalNode *INTEGER_LITERAL();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *FLOAT_LITERAL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LiteralPatternContext *literalPattern();

  class IdentifierPatternContext : public antlr4::ParserRuleContext {
  public:
    IdentifierPatternContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *KW_REF();
    antlr4::tree::TerminalNode *KW_MUT();
    antlr4::tree::TerminalNode *AT();
    PatternContext *pattern();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  IdentifierPatternContext *identifierPattern();

  class WildcardPatternContext : public antlr4::ParserRuleContext {
  public:
    WildcardPatternContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *UNDERSCORE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  WildcardPatternContext *wildcardPattern();

  class RestPatternContext : public antlr4::ParserRuleContext {
  public:
    RestPatternContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DOTDOT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  RestPatternContext *restPattern();

  class RangePatternContext : public antlr4::ParserRuleContext {
  public:
    RangePatternContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);

    RangePatternContext() = default;
    void copyFrom(RangePatternContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;
  };

  class InclusiveRangePatternContext : public RangePatternContext {
  public:
    InclusiveRangePatternContext(RangePatternContext *ctx);

    std::vector<RangePatternBoundContext *> rangePatternBound();
    RangePatternBoundContext *rangePatternBound(size_t i);
    antlr4::tree::TerminalNode *DOTDOTEQ();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class ObsoleteRangePatternContext : public RangePatternContext {
  public:
    ObsoleteRangePatternContext(RangePatternContext *ctx);

    std::vector<RangePatternBoundContext *> rangePatternBound();
    RangePatternBoundContext *rangePatternBound(size_t i);
    antlr4::tree::TerminalNode *DOTDOTDOT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class HalfOpenRangePatternContext : public RangePatternContext {
  public:
    HalfOpenRangePatternContext(RangePatternContext *ctx);

    RangePatternBoundContext *rangePatternBound();
    antlr4::tree::TerminalNode *DOTDOT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  RangePatternContext *rangePattern();

  class RangePatternBoundContext : public antlr4::ParserRuleContext {
  public:
    RangePatternBoundContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CHAR_LITERAL();
    antlr4::tree::TerminalNode *BYTE_LITERAL();
    antlr4::tree::TerminalNode *INTEGER_LITERAL();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *FLOAT_LITERAL();
    PathPatternContext *pathPattern();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  RangePatternBoundContext *rangePatternBound();

  class ReferencePatternContext : public antlr4::ParserRuleContext {
  public:
    ReferencePatternContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PatternWithoutRangeContext *patternWithoutRange();
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *ANDAND();
    antlr4::tree::TerminalNode *KW_MUT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ReferencePatternContext *referencePattern();

  class StructPatternContext : public antlr4::ParserRuleContext {
  public:
    StructPatternContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();
    antlr4::tree::TerminalNode *LCURLYBRACE();
    antlr4::tree::TerminalNode *RCURLYBRACE();
    StructPatternElementsContext *structPatternElements();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructPatternContext *structPattern();

  class StructPatternElementsContext : public antlr4::ParserRuleContext {
  public:
    StructPatternElementsContext(antlr4::ParserRuleContext *parent,
                                 size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StructPatternFieldsContext *structPatternFields();
    antlr4::tree::TerminalNode *COMMA();
    StructPatternEtCeteraContext *structPatternEtCetera();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructPatternElementsContext *structPatternElements();

  class StructPatternFieldsContext : public antlr4::ParserRuleContext {
  public:
    StructPatternFieldsContext(antlr4::ParserRuleContext *parent,
                               size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<StructPatternFieldContext *> structPatternField();
    StructPatternFieldContext *structPatternField(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructPatternFieldsContext *structPatternFields();

  class StructPatternFieldContext : public antlr4::ParserRuleContext {
  public:
    StructPatternFieldContext(antlr4::ParserRuleContext *parent,
                              size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TupleIndexContext *tupleIndex();
    antlr4::tree::TerminalNode *COLON();
    PatternContext *pattern();
    IdentifierContext *identifier();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    antlr4::tree::TerminalNode *KW_REF();
    antlr4::tree::TerminalNode *KW_MUT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructPatternFieldContext *structPatternField();

  class StructPatternEtCeteraContext : public antlr4::ParserRuleContext {
  public:
    StructPatternEtCeteraContext(antlr4::ParserRuleContext *parent,
                                 size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DOTDOT();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StructPatternEtCeteraContext *structPatternEtCetera();

  class TupleStructPatternContext : public antlr4::ParserRuleContext {
  public:
    TupleStructPatternContext(antlr4::ParserRuleContext *parent,
                              size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    TupleStructItemsContext *tupleStructItems();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TupleStructPatternContext *tupleStructPattern();

  class TupleStructItemsContext : public antlr4::ParserRuleContext {
  public:
    TupleStructItemsContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PatternContext *> pattern();
    PatternContext *pattern(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TupleStructItemsContext *tupleStructItems();

  class TuplePatternContext : public antlr4::ParserRuleContext {
  public:
    TuplePatternContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    TuplePatternItemsContext *tuplePatternItems();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TuplePatternContext *tuplePattern();

  class TuplePatternItemsContext : public antlr4::ParserRuleContext {
  public:
    TuplePatternItemsContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PatternContext *> pattern();
    PatternContext *pattern(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);
    RestPatternContext *restPattern();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TuplePatternItemsContext *tuplePatternItems();

  class GroupedPatternContext : public antlr4::ParserRuleContext {
  public:
    GroupedPatternContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    PatternContext *pattern();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GroupedPatternContext *groupedPattern();

  class SlicePatternContext : public antlr4::ParserRuleContext {
  public:
    SlicePatternContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();
    SlicePatternItemsContext *slicePatternItems();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  SlicePatternContext *slicePattern();

  class SlicePatternItemsContext : public antlr4::ParserRuleContext {
  public:
    SlicePatternItemsContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PatternContext *> pattern();
    PatternContext *pattern(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  SlicePatternItemsContext *slicePatternItems();

  class PathPatternContext : public antlr4::ParserRuleContext {
  public:
    PathPatternContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathInExpressionContext *pathInExpression();
    QualifiedPathInExpressionContext *qualifiedPathInExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PathPatternContext *pathPattern();

  class Type_Context : public antlr4::ParserRuleContext {
  public:
    Type_Context(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TypeNoBoundsContext *typeNoBounds();
    ImplTraitTypeContext *implTraitType();
    TraitObjectTypeContext *traitObjectType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Type_Context *type_();

  class TypeNoBoundsContext : public antlr4::ParserRuleContext {
  public:
    TypeNoBoundsContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ParenthesizedTypeContext *parenthesizedType();
    ImplTraitTypeOneBoundContext *implTraitTypeOneBound();
    TraitObjectTypeOneBoundContext *traitObjectTypeOneBound();
    TypePathContext *typePath();
    TupleTypeContext *tupleType();
    NeverTypeContext *neverType();
    RawPointerTypeContext *rawPointerType();
    ReferenceTypeContext *referenceType();
    ArrayTypeContext *arrayType();
    SliceTypeContext *sliceType();
    InferredTypeContext *inferredType();
    QualifiedPathInTypeContext *qualifiedPathInType();
    BareFunctionTypeContext *bareFunctionType();
    MacroInvocationContext *macroInvocation();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypeNoBoundsContext *typeNoBounds();

  class ParenthesizedTypeContext : public antlr4::ParserRuleContext {
  public:
    ParenthesizedTypeContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    Type_Context *type_();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ParenthesizedTypeContext *parenthesizedType();

  class NeverTypeContext : public antlr4::ParserRuleContext {
  public:
    NeverTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *NOT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  NeverTypeContext *neverType();

  class TupleTypeContext : public antlr4::ParserRuleContext {
  public:
    TupleTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<Type_Context *> type_();
    Type_Context *type_(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TupleTypeContext *tupleType();

  class ArrayTypeContext : public antlr4::ParserRuleContext {
  public:
    ArrayTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    Type_Context *type_();
    antlr4::tree::TerminalNode *SEMI();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ArrayTypeContext *arrayType();

  class SliceTypeContext : public antlr4::ParserRuleContext {
  public:
    SliceTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LSQUAREBRACKET();
    Type_Context *type_();
    antlr4::tree::TerminalNode *RSQUAREBRACKET();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  SliceTypeContext *sliceType();

  class ReferenceTypeContext : public antlr4::ParserRuleContext {
  public:
    ReferenceTypeContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *AND();
    TypeNoBoundsContext *typeNoBounds();
    LifetimeContext *lifetime();
    antlr4::tree::TerminalNode *KW_MUT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ReferenceTypeContext *referenceType();

  class RawPointerTypeContext : public antlr4::ParserRuleContext {
  public:
    RawPointerTypeContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STAR();
    TypeNoBoundsContext *typeNoBounds();
    antlr4::tree::TerminalNode *KW_MUT();
    antlr4::tree::TerminalNode *KW_CONST();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  RawPointerTypeContext *rawPointerType();

  class BareFunctionTypeContext : public antlr4::ParserRuleContext {
  public:
    BareFunctionTypeContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FunctionTypeQualifiersContext *functionTypeQualifiers();
    antlr4::tree::TerminalNode *KW_FN();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ForLifetimesContext *forLifetimes();
    FunctionParametersMaybeNamedVariadicContext *
    functionParametersMaybeNamedVariadic();
    BareFunctionReturnTypeContext *bareFunctionReturnType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  BareFunctionTypeContext *bareFunctionType();

  class FunctionTypeQualifiersContext : public antlr4::ParserRuleContext {
  public:
    FunctionTypeQualifiersContext(antlr4::ParserRuleContext *parent,
                                  size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_UNSAFE();
    antlr4::tree::TerminalNode *KW_EXTERN();
    AbiContext *abi();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionTypeQualifiersContext *functionTypeQualifiers();

  class BareFunctionReturnTypeContext : public antlr4::ParserRuleContext {
  public:
    BareFunctionReturnTypeContext(antlr4::ParserRuleContext *parent,
                                  size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RARROW();
    TypeNoBoundsContext *typeNoBounds();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  BareFunctionReturnTypeContext *bareFunctionReturnType();

  class FunctionParametersMaybeNamedVariadicContext
      : public antlr4::ParserRuleContext {
  public:
    FunctionParametersMaybeNamedVariadicContext(
        antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    MaybeNamedFunctionParametersContext *maybeNamedFunctionParameters();
    MaybeNamedFunctionParametersVariadicContext *
    maybeNamedFunctionParametersVariadic();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionParametersMaybeNamedVariadicContext *
  functionParametersMaybeNamedVariadic();

  class MaybeNamedFunctionParametersContext : public antlr4::ParserRuleContext {
  public:
    MaybeNamedFunctionParametersContext(antlr4::ParserRuleContext *parent,
                                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<MaybeNamedParamContext *> maybeNamedParam();
    MaybeNamedParamContext *maybeNamedParam(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MaybeNamedFunctionParametersContext *maybeNamedFunctionParameters();

  class MaybeNamedParamContext : public antlr4::ParserRuleContext {
  public:
    MaybeNamedParamContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Type_Context *type_();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);
    antlr4::tree::TerminalNode *COLON();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *UNDERSCORE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MaybeNamedParamContext *maybeNamedParam();

  class MaybeNamedFunctionParametersVariadicContext
      : public antlr4::ParserRuleContext {
  public:
    MaybeNamedFunctionParametersVariadicContext(
        antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<MaybeNamedParamContext *> maybeNamedParam();
    MaybeNamedParamContext *maybeNamedParam(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);
    antlr4::tree::TerminalNode *DOTDOTDOT();
    std::vector<OuterAttributeContext *> outerAttribute();
    OuterAttributeContext *outerAttribute(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MaybeNamedFunctionParametersVariadicContext *
  maybeNamedFunctionParametersVariadic();

  class TraitObjectTypeContext : public antlr4::ParserRuleContext {
  public:
    TraitObjectTypeContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TypeParamBoundsContext *typeParamBounds();
    antlr4::tree::TerminalNode *KW_DYN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TraitObjectTypeContext *traitObjectType();

  class TraitObjectTypeOneBoundContext : public antlr4::ParserRuleContext {
  public:
    TraitObjectTypeOneBoundContext(antlr4::ParserRuleContext *parent,
                                   size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TraitBoundContext *traitBound();
    antlr4::tree::TerminalNode *KW_DYN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TraitObjectTypeOneBoundContext *traitObjectTypeOneBound();

  class ImplTraitTypeContext : public antlr4::ParserRuleContext {
  public:
    ImplTraitTypeContext(antlr4::ParserRuleContext *parent,
                         size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_IMPL();
    TypeParamBoundsContext *typeParamBounds();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ImplTraitTypeContext *implTraitType();

  class ImplTraitTypeOneBoundContext : public antlr4::ParserRuleContext {
  public:
    ImplTraitTypeOneBoundContext(antlr4::ParserRuleContext *parent,
                                 size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_IMPL();
    TraitBoundContext *traitBound();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ImplTraitTypeOneBoundContext *implTraitTypeOneBound();

  class InferredTypeContext : public antlr4::ParserRuleContext {
  public:
    InferredTypeContext(antlr4::ParserRuleContext *parent,
                        size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *UNDERSCORE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  InferredTypeContext *inferredType();

  class TypeParamBoundsContext : public antlr4::ParserRuleContext {
  public:
    TypeParamBoundsContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TypeParamBoundContext *> typeParamBound();
    TypeParamBoundContext *typeParamBound(size_t i);
    std::vector<antlr4::tree::TerminalNode *> PLUS();
    antlr4::tree::TerminalNode *PLUS(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypeParamBoundsContext *typeParamBounds();

  class TypeParamBoundContext : public antlr4::ParserRuleContext {
  public:
    TypeParamBoundContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LifetimeContext *lifetime();
    TraitBoundContext *traitBound();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypeParamBoundContext *typeParamBound();

  class TraitBoundContext : public antlr4::ParserRuleContext {
  public:
    TraitBoundContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TypePathContext *typePath();
    antlr4::tree::TerminalNode *QUESTION();
    ForLifetimesContext *forLifetimes();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TraitBoundContext *traitBound();

  class LifetimeBoundsContext : public antlr4::ParserRuleContext {
  public:
    LifetimeBoundsContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<LifetimeContext *> lifetime();
    LifetimeContext *lifetime(size_t i);
    std::vector<antlr4::tree::TerminalNode *> PLUS();
    antlr4::tree::TerminalNode *PLUS(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LifetimeBoundsContext *lifetimeBounds();

  class LifetimeContext : public antlr4::ParserRuleContext {
  public:
    LifetimeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LIFETIME_OR_LABEL();
    antlr4::tree::TerminalNode *KW_STATICLIFETIME();
    antlr4::tree::TerminalNode *KW_UNDERLINELIFETIME();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  LifetimeContext *lifetime();

  class SimplePathContext : public antlr4::ParserRuleContext {
  public:
    SimplePathContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<SimplePathSegmentContext *> simplePathSegment();
    SimplePathSegmentContext *simplePathSegment(size_t i);
    std::vector<antlr4::tree::TerminalNode *> PATHSEP();
    antlr4::tree::TerminalNode *PATHSEP(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  SimplePathContext *simplePath();

  class SimplePathSegmentContext : public antlr4::ParserRuleContext {
  public:
    SimplePathSegmentContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *KW_SUPER();
    antlr4::tree::TerminalNode *KW_SELFVALUE();
    antlr4::tree::TerminalNode *KW_CRATE();
    antlr4::tree::TerminalNode *KW_DOLLARCRATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  SimplePathSegmentContext *simplePathSegment();

  class PathInExpressionContext : public antlr4::ParserRuleContext {
  public:
    PathInExpressionContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PathExprSegmentContext *> pathExprSegment();
    PathExprSegmentContext *pathExprSegment(size_t i);
    std::vector<antlr4::tree::TerminalNode *> PATHSEP();
    antlr4::tree::TerminalNode *PATHSEP(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PathInExpressionContext *pathInExpression();

  class PathExprSegmentContext : public antlr4::ParserRuleContext {
  public:
    PathExprSegmentContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathIdentSegmentContext *pathIdentSegment();
    antlr4::tree::TerminalNode *PATHSEP();
    GenericArgsContext *genericArgs();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PathExprSegmentContext *pathExprSegment();

  class PathIdentSegmentContext : public antlr4::ParserRuleContext {
  public:
    PathIdentSegmentContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *KW_SUPER();
    antlr4::tree::TerminalNode *KW_SELFVALUE();
    antlr4::tree::TerminalNode *KW_SELFTYPE();
    antlr4::tree::TerminalNode *KW_CRATE();
    antlr4::tree::TerminalNode *KW_DOLLARCRATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  PathIdentSegmentContext *pathIdentSegment();

  class GenericArgsContext : public antlr4::ParserRuleContext {
  public:
    GenericArgsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GT();
    GenericArgsLifetimesContext *genericArgsLifetimes();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);
    GenericArgsTypesContext *genericArgsTypes();
    GenericArgsBindingsContext *genericArgsBindings();
    std::vector<GenericArgContext *> genericArg();
    GenericArgContext *genericArg(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericArgsContext *genericArgs();

  class GenericArgContext : public antlr4::ParserRuleContext {
  public:
    GenericArgContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LifetimeContext *lifetime();
    Type_Context *type_();
    GenericArgsConstContext *genericArgsConst();
    GenericArgsBindingContext *genericArgsBinding();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericArgContext *genericArg();

  class GenericArgsConstContext : public antlr4::ParserRuleContext {
  public:
    GenericArgsConstContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BlockExpressionContext *blockExpression();
    LiteralExpressionContext *literalExpression();
    antlr4::tree::TerminalNode *MINUS();
    SimplePathSegmentContext *simplePathSegment();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericArgsConstContext *genericArgsConst();

  class GenericArgsLifetimesContext : public antlr4::ParserRuleContext {
  public:
    GenericArgsLifetimesContext(antlr4::ParserRuleContext *parent,
                                size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<LifetimeContext *> lifetime();
    LifetimeContext *lifetime(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericArgsLifetimesContext *genericArgsLifetimes();

  class GenericArgsTypesContext : public antlr4::ParserRuleContext {
  public:
    GenericArgsTypesContext(antlr4::ParserRuleContext *parent,
                            size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<Type_Context *> type_();
    Type_Context *type_(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericArgsTypesContext *genericArgsTypes();

  class GenericArgsBindingsContext : public antlr4::ParserRuleContext {
  public:
    GenericArgsBindingsContext(antlr4::ParserRuleContext *parent,
                               size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<GenericArgsBindingContext *> genericArgsBinding();
    GenericArgsBindingContext *genericArgsBinding(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericArgsBindingsContext *genericArgsBindings();

  class GenericArgsBindingContext : public antlr4::ParserRuleContext {
  public:
    GenericArgsBindingContext(antlr4::ParserRuleContext *parent,
                              size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *EQ();
    Type_Context *type_();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericArgsBindingContext *genericArgsBinding();

  class QualifiedPathInExpressionContext : public antlr4::ParserRuleContext {
  public:
    QualifiedPathInExpressionContext(antlr4::ParserRuleContext *parent,
                                     size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QualifiedPathTypeContext *qualifiedPathType();
    std::vector<antlr4::tree::TerminalNode *> PATHSEP();
    antlr4::tree::TerminalNode *PATHSEP(size_t i);
    std::vector<PathExprSegmentContext *> pathExprSegment();
    PathExprSegmentContext *pathExprSegment(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  QualifiedPathInExpressionContext *qualifiedPathInExpression();

  class QualifiedPathTypeContext : public antlr4::ParserRuleContext {
  public:
    QualifiedPathTypeContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LT();
    Type_Context *type_();
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *KW_AS();
    TypePathContext *typePath();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  QualifiedPathTypeContext *qualifiedPathType();

  class QualifiedPathInTypeContext : public antlr4::ParserRuleContext {
  public:
    QualifiedPathInTypeContext(antlr4::ParserRuleContext *parent,
                               size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QualifiedPathTypeContext *qualifiedPathType();
    std::vector<antlr4::tree::TerminalNode *> PATHSEP();
    antlr4::tree::TerminalNode *PATHSEP(size_t i);
    std::vector<TypePathSegmentContext *> typePathSegment();
    TypePathSegmentContext *typePathSegment(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  QualifiedPathInTypeContext *qualifiedPathInType();

  class TypePathContext : public antlr4::ParserRuleContext {
  public:
    TypePathContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TypePathSegmentContext *> typePathSegment();
    TypePathSegmentContext *typePathSegment(size_t i);
    std::vector<antlr4::tree::TerminalNode *> PATHSEP();
    antlr4::tree::TerminalNode *PATHSEP(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypePathContext *typePath();

  class TypePathSegmentContext : public antlr4::ParserRuleContext {
  public:
    TypePathSegmentContext(antlr4::ParserRuleContext *parent,
                           size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PathIdentSegmentContext *pathIdentSegment();
    antlr4::tree::TerminalNode *PATHSEP();
    GenericArgsContext *genericArgs();
    TypePathFnContext *typePathFn();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypePathSegmentContext *typePathSegment();

  class TypePathFnContext : public antlr4::ParserRuleContext {
  public:
    TypePathFnContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    TypePathInputsContext *typePathInputs();
    antlr4::tree::TerminalNode *RARROW();
    Type_Context *type_();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypePathFnContext *typePathFn();

  class TypePathInputsContext : public antlr4::ParserRuleContext {
  public:
    TypePathInputsContext(antlr4::ParserRuleContext *parent,
                          size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<Type_Context *> type_();
    Type_Context *type_(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode *COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TypePathInputsContext *typePathInputs();

  class VisibilityContext : public antlr4::ParserRuleContext {
  public:
    VisibilityContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_PUB();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *KW_CRATE();
    antlr4::tree::TerminalNode *KW_SELFVALUE();
    antlr4::tree::TerminalNode *KW_SUPER();
    antlr4::tree::TerminalNode *KW_IN();
    SimplePathContext *simplePath();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  VisibilityContext *visibility();

  class IdentifierContext : public antlr4::ParserRuleContext {
  public:
    IdentifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *NON_KEYWORD_IDENTIFIER();
    antlr4::tree::TerminalNode *RAW_IDENTIFIER();
    antlr4::tree::TerminalNode *KW_MACRORULES();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  IdentifierContext *identifier();

  class KeywordContext : public antlr4::ParserRuleContext {
  public:
    KeywordContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *KW_AS();
    antlr4::tree::TerminalNode *KW_BREAK();
    antlr4::tree::TerminalNode *KW_CONST();
    antlr4::tree::TerminalNode *KW_CONTINUE();
    antlr4::tree::TerminalNode *KW_CRATE();
    antlr4::tree::TerminalNode *KW_ELSE();
    antlr4::tree::TerminalNode *KW_ENUM();
    antlr4::tree::TerminalNode *KW_EXTERN();
    antlr4::tree::TerminalNode *KW_FALSE();
    antlr4::tree::TerminalNode *KW_FN();
    antlr4::tree::TerminalNode *KW_FOR();
    antlr4::tree::TerminalNode *KW_IF();
    antlr4::tree::TerminalNode *KW_IMPL();
    antlr4::tree::TerminalNode *KW_IN();
    antlr4::tree::TerminalNode *KW_LET();
    antlr4::tree::TerminalNode *KW_LOOP();
    antlr4::tree::TerminalNode *KW_MATCH();
    antlr4::tree::TerminalNode *KW_MOD();
    antlr4::tree::TerminalNode *KW_MOVE();
    antlr4::tree::TerminalNode *KW_MUT();
    antlr4::tree::TerminalNode *KW_PUB();
    antlr4::tree::TerminalNode *KW_REF();
    antlr4::tree::TerminalNode *KW_RETURN();
    antlr4::tree::TerminalNode *KW_SELFVALUE();
    antlr4::tree::TerminalNode *KW_SELFTYPE();
    antlr4::tree::TerminalNode *KW_STATIC();
    antlr4::tree::TerminalNode *KW_STRUCT();
    antlr4::tree::TerminalNode *KW_SUPER();
    antlr4::tree::TerminalNode *KW_TRAIT();
    antlr4::tree::TerminalNode *KW_TRUE();
    antlr4::tree::TerminalNode *KW_TYPE();
    antlr4::tree::TerminalNode *KW_UNSAFE();
    antlr4::tree::TerminalNode *KW_USE();
    antlr4::tree::TerminalNode *KW_WHERE();
    antlr4::tree::TerminalNode *KW_WHILE();
    antlr4::tree::TerminalNode *KW_ASYNC();
    antlr4::tree::TerminalNode *KW_AWAIT();
    antlr4::tree::TerminalNode *KW_DYN();
    antlr4::tree::TerminalNode *KW_ABSTRACT();
    antlr4::tree::TerminalNode *KW_BECOME();
    antlr4::tree::TerminalNode *KW_BOX();
    antlr4::tree::TerminalNode *KW_DO();
    antlr4::tree::TerminalNode *KW_FINAL();
    antlr4::tree::TerminalNode *KW_MACRO();
    antlr4::tree::TerminalNode *KW_OVERRIDE();
    antlr4::tree::TerminalNode *KW_PRIV();
    antlr4::tree::TerminalNode *KW_TYPEOF();
    antlr4::tree::TerminalNode *KW_UNSIZED();
    antlr4::tree::TerminalNode *KW_VIRTUAL();
    antlr4::tree::TerminalNode *KW_YIELD();
    antlr4::tree::TerminalNode *KW_TRY();
    antlr4::tree::TerminalNode *KW_UNION();
    antlr4::tree::TerminalNode *KW_STATICLIFETIME();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  KeywordContext *keyword();

  class MacroIdentifierLikeTokenContext : public antlr4::ParserRuleContext {
  public:
    MacroIdentifierLikeTokenContext(antlr4::ParserRuleContext *parent,
                                    size_t invokingState);
    virtual size_t getRuleIndex() const override;
    KeywordContext *keyword();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *KW_MACRORULES();
    antlr4::tree::TerminalNode *KW_UNDERLINELIFETIME();
    antlr4::tree::TerminalNode *KW_DOLLARCRATE();
    antlr4::tree::TerminalNode *LIFETIME_OR_LABEL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroIdentifierLikeTokenContext *macroIdentifierLikeToken();

  class MacroLiteralTokenContext : public antlr4::ParserRuleContext {
  public:
    MacroLiteralTokenContext(antlr4::ParserRuleContext *parent,
                             size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LiteralExpressionContext *literalExpression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroLiteralTokenContext *macroLiteralToken();

  class MacroPunctuationTokenContext : public antlr4::ParserRuleContext {
  public:
    MacroPunctuationTokenContext(antlr4::ParserRuleContext *parent,
                                 size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *SLASH();
    antlr4::tree::TerminalNode *PERCENT();
    antlr4::tree::TerminalNode *CARET();
    antlr4::tree::TerminalNode *NOT();
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *OR();
    antlr4::tree::TerminalNode *ANDAND();
    antlr4::tree::TerminalNode *OROR();
    antlr4::tree::TerminalNode *PLUSEQ();
    antlr4::tree::TerminalNode *MINUSEQ();
    antlr4::tree::TerminalNode *STAREQ();
    antlr4::tree::TerminalNode *SLASHEQ();
    antlr4::tree::TerminalNode *PERCENTEQ();
    antlr4::tree::TerminalNode *CARETEQ();
    antlr4::tree::TerminalNode *ANDEQ();
    antlr4::tree::TerminalNode *OREQ();
    antlr4::tree::TerminalNode *SHLEQ();
    antlr4::tree::TerminalNode *SHREQ();
    antlr4::tree::TerminalNode *EQ();
    antlr4::tree::TerminalNode *EQEQ();
    antlr4::tree::TerminalNode *NE();
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GE();
    antlr4::tree::TerminalNode *LE();
    antlr4::tree::TerminalNode *AT();
    antlr4::tree::TerminalNode *UNDERSCORE();
    antlr4::tree::TerminalNode *DOT();
    antlr4::tree::TerminalNode *DOTDOT();
    antlr4::tree::TerminalNode *DOTDOTDOT();
    antlr4::tree::TerminalNode *DOTDOTEQ();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *SEMI();
    antlr4::tree::TerminalNode *COLON();
    antlr4::tree::TerminalNode *PATHSEP();
    antlr4::tree::TerminalNode *RARROW();
    antlr4::tree::TerminalNode *FATARROW();
    antlr4::tree::TerminalNode *POUND();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  MacroPunctuationTokenContext *macroPunctuationToken();

  class ShlContext : public antlr4::ParserRuleContext {
  public:
    ShlContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> LT();
    antlr4::tree::TerminalNode *LT(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ShlContext *shl();

  class ShrContext : public antlr4::ParserRuleContext {
  public:
    ShrContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> GT();
    antlr4::tree::TerminalNode *GT(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ShrContext *shr();

  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex,
               size_t predicateIndex) override;

  bool expressionSempred(ExpressionContext *_localctx, size_t predicateIndex);
  bool shlSempred(ShlContext *_localctx, size_t predicateIndex);
  bool shrSempred(ShrContext *_localctx, size_t predicateIndex);

  // By default the static state used to implement the parser is lazily
  // initialized during the first call to the constructor. You can call this
  // function if you wish to initialize the static state ahead of time.
  static void initialize();

private:
};
