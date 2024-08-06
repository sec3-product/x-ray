
// Generated from RustParser.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "RustParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by RustParser.
 */
class  RustParserVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by RustParser.
   */
    virtual std::any visitCrate(RustParser::CrateContext *context) = 0;

    virtual std::any visitMacroInvocation(RustParser::MacroInvocationContext *context) = 0;

    virtual std::any visitDelimTokenTree(RustParser::DelimTokenTreeContext *context) = 0;

    virtual std::any visitTokenTree(RustParser::TokenTreeContext *context) = 0;

    virtual std::any visitTokenTreeToken(RustParser::TokenTreeTokenContext *context) = 0;

    virtual std::any visitMacroInvocationSemi(RustParser::MacroInvocationSemiContext *context) = 0;

    virtual std::any visitMacroRulesDefinition(RustParser::MacroRulesDefinitionContext *context) = 0;

    virtual std::any visitMacroRulesDef(RustParser::MacroRulesDefContext *context) = 0;

    virtual std::any visitMacroRules(RustParser::MacroRulesContext *context) = 0;

    virtual std::any visitMacroRule(RustParser::MacroRuleContext *context) = 0;

    virtual std::any visitMacroMatcher(RustParser::MacroMatcherContext *context) = 0;

    virtual std::any visitMacroMatch(RustParser::MacroMatchContext *context) = 0;

    virtual std::any visitMacroMatchToken(RustParser::MacroMatchTokenContext *context) = 0;

    virtual std::any visitMacroFragSpec(RustParser::MacroFragSpecContext *context) = 0;

    virtual std::any visitMacroRepSep(RustParser::MacroRepSepContext *context) = 0;

    virtual std::any visitMacroRepOp(RustParser::MacroRepOpContext *context) = 0;

    virtual std::any visitMacroTranscriber(RustParser::MacroTranscriberContext *context) = 0;

    virtual std::any visitItem(RustParser::ItemContext *context) = 0;

    virtual std::any visitVisItem(RustParser::VisItemContext *context) = 0;

    virtual std::any visitMacroItem(RustParser::MacroItemContext *context) = 0;

    virtual std::any visitModule(RustParser::ModuleContext *context) = 0;

    virtual std::any visitExternCrate(RustParser::ExternCrateContext *context) = 0;

    virtual std::any visitCrateRef(RustParser::CrateRefContext *context) = 0;

    virtual std::any visitAsClause(RustParser::AsClauseContext *context) = 0;

    virtual std::any visitUseDeclaration(RustParser::UseDeclarationContext *context) = 0;

    virtual std::any visitUseTree(RustParser::UseTreeContext *context) = 0;

    virtual std::any visitFunction_(RustParser::Function_Context *context) = 0;

    virtual std::any visitFunctionQualifiers(RustParser::FunctionQualifiersContext *context) = 0;

    virtual std::any visitAbi(RustParser::AbiContext *context) = 0;

    virtual std::any visitFunctionParameters(RustParser::FunctionParametersContext *context) = 0;

    virtual std::any visitSelfParam(RustParser::SelfParamContext *context) = 0;

    virtual std::any visitShorthandSelf(RustParser::ShorthandSelfContext *context) = 0;

    virtual std::any visitTypedSelf(RustParser::TypedSelfContext *context) = 0;

    virtual std::any visitFunctionParam(RustParser::FunctionParamContext *context) = 0;

    virtual std::any visitFunctionParamPattern(RustParser::FunctionParamPatternContext *context) = 0;

    virtual std::any visitFunctionReturnType(RustParser::FunctionReturnTypeContext *context) = 0;

    virtual std::any visitTypeAlias(RustParser::TypeAliasContext *context) = 0;

    virtual std::any visitStruct_(RustParser::Struct_Context *context) = 0;

    virtual std::any visitStructStruct(RustParser::StructStructContext *context) = 0;

    virtual std::any visitTupleStruct(RustParser::TupleStructContext *context) = 0;

    virtual std::any visitStructFields(RustParser::StructFieldsContext *context) = 0;

    virtual std::any visitStructField(RustParser::StructFieldContext *context) = 0;

    virtual std::any visitTupleFields(RustParser::TupleFieldsContext *context) = 0;

    virtual std::any visitTupleField(RustParser::TupleFieldContext *context) = 0;

    virtual std::any visitEnumeration(RustParser::EnumerationContext *context) = 0;

    virtual std::any visitEnumItems(RustParser::EnumItemsContext *context) = 0;

    virtual std::any visitEnumItem(RustParser::EnumItemContext *context) = 0;

    virtual std::any visitEnumItemTuple(RustParser::EnumItemTupleContext *context) = 0;

    virtual std::any visitEnumItemStruct(RustParser::EnumItemStructContext *context) = 0;

    virtual std::any visitEnumItemDiscriminant(RustParser::EnumItemDiscriminantContext *context) = 0;

    virtual std::any visitUnion_(RustParser::Union_Context *context) = 0;

    virtual std::any visitConstantItem(RustParser::ConstantItemContext *context) = 0;

    virtual std::any visitStaticItem(RustParser::StaticItemContext *context) = 0;

    virtual std::any visitTrait_(RustParser::Trait_Context *context) = 0;

    virtual std::any visitImplementation(RustParser::ImplementationContext *context) = 0;

    virtual std::any visitInherentImpl(RustParser::InherentImplContext *context) = 0;

    virtual std::any visitTraitImpl(RustParser::TraitImplContext *context) = 0;

    virtual std::any visitExternBlock(RustParser::ExternBlockContext *context) = 0;

    virtual std::any visitExternalItem(RustParser::ExternalItemContext *context) = 0;

    virtual std::any visitGenericParams(RustParser::GenericParamsContext *context) = 0;

    virtual std::any visitGenericParam(RustParser::GenericParamContext *context) = 0;

    virtual std::any visitLifetimeParam(RustParser::LifetimeParamContext *context) = 0;

    virtual std::any visitTypeParam(RustParser::TypeParamContext *context) = 0;

    virtual std::any visitConstParam(RustParser::ConstParamContext *context) = 0;

    virtual std::any visitWhereClause(RustParser::WhereClauseContext *context) = 0;

    virtual std::any visitWhereClauseItem(RustParser::WhereClauseItemContext *context) = 0;

    virtual std::any visitLifetimeWhereClauseItem(RustParser::LifetimeWhereClauseItemContext *context) = 0;

    virtual std::any visitTypeBoundWhereClauseItem(RustParser::TypeBoundWhereClauseItemContext *context) = 0;

    virtual std::any visitForLifetimes(RustParser::ForLifetimesContext *context) = 0;

    virtual std::any visitAssociatedItem(RustParser::AssociatedItemContext *context) = 0;

    virtual std::any visitInnerAttribute(RustParser::InnerAttributeContext *context) = 0;

    virtual std::any visitOuterAttribute(RustParser::OuterAttributeContext *context) = 0;

    virtual std::any visitAttr(RustParser::AttrContext *context) = 0;

    virtual std::any visitAttrInput(RustParser::AttrInputContext *context) = 0;

    virtual std::any visitStatement(RustParser::StatementContext *context) = 0;

    virtual std::any visitLetStatement(RustParser::LetStatementContext *context) = 0;

    virtual std::any visitExpressionStatement(RustParser::ExpressionStatementContext *context) = 0;

    virtual std::any visitTypeCastExpression(RustParser::TypeCastExpressionContext *context) = 0;

    virtual std::any visitPathExpression_(RustParser::PathExpression_Context *context) = 0;

    virtual std::any visitTupleExpression(RustParser::TupleExpressionContext *context) = 0;

    virtual std::any visitIndexExpression(RustParser::IndexExpressionContext *context) = 0;

    virtual std::any visitRangeExpression(RustParser::RangeExpressionContext *context) = 0;

    virtual std::any visitMacroInvocationAsExpression(RustParser::MacroInvocationAsExpressionContext *context) = 0;

    virtual std::any visitReturnExpression(RustParser::ReturnExpressionContext *context) = 0;

    virtual std::any visitAwaitExpression(RustParser::AwaitExpressionContext *context) = 0;

    virtual std::any visitErrorPropagationExpression(RustParser::ErrorPropagationExpressionContext *context) = 0;

    virtual std::any visitContinueExpression(RustParser::ContinueExpressionContext *context) = 0;

    virtual std::any visitAssignmentExpression(RustParser::AssignmentExpressionContext *context) = 0;

    virtual std::any visitMethodCallExpression(RustParser::MethodCallExpressionContext *context) = 0;

    virtual std::any visitLiteralExpression_(RustParser::LiteralExpression_Context *context) = 0;

    virtual std::any visitStructExpression_(RustParser::StructExpression_Context *context) = 0;

    virtual std::any visitTupleIndexingExpression(RustParser::TupleIndexingExpressionContext *context) = 0;

    virtual std::any visitNegationExpression(RustParser::NegationExpressionContext *context) = 0;

    virtual std::any visitCallExpression(RustParser::CallExpressionContext *context) = 0;

    virtual std::any visitLazyBooleanExpression(RustParser::LazyBooleanExpressionContext *context) = 0;

    virtual std::any visitDereferenceExpression(RustParser::DereferenceExpressionContext *context) = 0;

    virtual std::any visitExpressionWithBlock_(RustParser::ExpressionWithBlock_Context *context) = 0;

    virtual std::any visitGroupedExpression(RustParser::GroupedExpressionContext *context) = 0;

    virtual std::any visitBreakExpression(RustParser::BreakExpressionContext *context) = 0;

    virtual std::any visitArithmeticOrLogicalExpression(RustParser::ArithmeticOrLogicalExpressionContext *context) = 0;

    virtual std::any visitFieldExpression(RustParser::FieldExpressionContext *context) = 0;

    virtual std::any visitEnumerationVariantExpression_(RustParser::EnumerationVariantExpression_Context *context) = 0;

    virtual std::any visitComparisonExpression(RustParser::ComparisonExpressionContext *context) = 0;

    virtual std::any visitAttributedExpression(RustParser::AttributedExpressionContext *context) = 0;

    virtual std::any visitBorrowExpression(RustParser::BorrowExpressionContext *context) = 0;

    virtual std::any visitCompoundAssignmentExpression(RustParser::CompoundAssignmentExpressionContext *context) = 0;

    virtual std::any visitClosureExpression_(RustParser::ClosureExpression_Context *context) = 0;

    virtual std::any visitArrayExpression(RustParser::ArrayExpressionContext *context) = 0;

    virtual std::any visitComparisonOperator(RustParser::ComparisonOperatorContext *context) = 0;

    virtual std::any visitCompoundAssignOperator(RustParser::CompoundAssignOperatorContext *context) = 0;

    virtual std::any visitExpressionWithBlock(RustParser::ExpressionWithBlockContext *context) = 0;

    virtual std::any visitLiteralExpression(RustParser::LiteralExpressionContext *context) = 0;

    virtual std::any visitPathExpression(RustParser::PathExpressionContext *context) = 0;

    virtual std::any visitBlockExpression(RustParser::BlockExpressionContext *context) = 0;

    virtual std::any visitStatements(RustParser::StatementsContext *context) = 0;

    virtual std::any visitAsyncBlockExpression(RustParser::AsyncBlockExpressionContext *context) = 0;

    virtual std::any visitUnsafeBlockExpression(RustParser::UnsafeBlockExpressionContext *context) = 0;

    virtual std::any visitArrayElements(RustParser::ArrayElementsContext *context) = 0;

    virtual std::any visitTupleElements(RustParser::TupleElementsContext *context) = 0;

    virtual std::any visitTupleIndex(RustParser::TupleIndexContext *context) = 0;

    virtual std::any visitStructExpression(RustParser::StructExpressionContext *context) = 0;

    virtual std::any visitStructExprStruct(RustParser::StructExprStructContext *context) = 0;

    virtual std::any visitStructExprFields(RustParser::StructExprFieldsContext *context) = 0;

    virtual std::any visitStructExprField(RustParser::StructExprFieldContext *context) = 0;

    virtual std::any visitStructBase(RustParser::StructBaseContext *context) = 0;

    virtual std::any visitStructExprTuple(RustParser::StructExprTupleContext *context) = 0;

    virtual std::any visitStructExprUnit(RustParser::StructExprUnitContext *context) = 0;

    virtual std::any visitEnumerationVariantExpression(RustParser::EnumerationVariantExpressionContext *context) = 0;

    virtual std::any visitEnumExprStruct(RustParser::EnumExprStructContext *context) = 0;

    virtual std::any visitEnumExprFields(RustParser::EnumExprFieldsContext *context) = 0;

    virtual std::any visitEnumExprField(RustParser::EnumExprFieldContext *context) = 0;

    virtual std::any visitEnumExprTuple(RustParser::EnumExprTupleContext *context) = 0;

    virtual std::any visitEnumExprFieldless(RustParser::EnumExprFieldlessContext *context) = 0;

    virtual std::any visitCallParams(RustParser::CallParamsContext *context) = 0;

    virtual std::any visitClosureExpression(RustParser::ClosureExpressionContext *context) = 0;

    virtual std::any visitClosureParameters(RustParser::ClosureParametersContext *context) = 0;

    virtual std::any visitClosureParam(RustParser::ClosureParamContext *context) = 0;

    virtual std::any visitLoopExpression(RustParser::LoopExpressionContext *context) = 0;

    virtual std::any visitInfiniteLoopExpression(RustParser::InfiniteLoopExpressionContext *context) = 0;

    virtual std::any visitPredicateLoopExpression(RustParser::PredicateLoopExpressionContext *context) = 0;

    virtual std::any visitPredicatePatternLoopExpression(RustParser::PredicatePatternLoopExpressionContext *context) = 0;

    virtual std::any visitIteratorLoopExpression(RustParser::IteratorLoopExpressionContext *context) = 0;

    virtual std::any visitLoopLabel(RustParser::LoopLabelContext *context) = 0;

    virtual std::any visitIfExpression(RustParser::IfExpressionContext *context) = 0;

    virtual std::any visitIfLetExpression(RustParser::IfLetExpressionContext *context) = 0;

    virtual std::any visitMatchExpression(RustParser::MatchExpressionContext *context) = 0;

    virtual std::any visitMatchArms(RustParser::MatchArmsContext *context) = 0;

    virtual std::any visitMatchArmExpression(RustParser::MatchArmExpressionContext *context) = 0;

    virtual std::any visitMatchArm(RustParser::MatchArmContext *context) = 0;

    virtual std::any visitMatchArmGuard(RustParser::MatchArmGuardContext *context) = 0;

    virtual std::any visitPattern(RustParser::PatternContext *context) = 0;

    virtual std::any visitPatternNoTopAlt(RustParser::PatternNoTopAltContext *context) = 0;

    virtual std::any visitPatternWithoutRange(RustParser::PatternWithoutRangeContext *context) = 0;

    virtual std::any visitLiteralPattern(RustParser::LiteralPatternContext *context) = 0;

    virtual std::any visitIdentifierPattern(RustParser::IdentifierPatternContext *context) = 0;

    virtual std::any visitWildcardPattern(RustParser::WildcardPatternContext *context) = 0;

    virtual std::any visitRestPattern(RustParser::RestPatternContext *context) = 0;

    virtual std::any visitInclusiveRangePattern(RustParser::InclusiveRangePatternContext *context) = 0;

    virtual std::any visitHalfOpenRangePattern(RustParser::HalfOpenRangePatternContext *context) = 0;

    virtual std::any visitObsoleteRangePattern(RustParser::ObsoleteRangePatternContext *context) = 0;

    virtual std::any visitRangePatternBound(RustParser::RangePatternBoundContext *context) = 0;

    virtual std::any visitReferencePattern(RustParser::ReferencePatternContext *context) = 0;

    virtual std::any visitStructPattern(RustParser::StructPatternContext *context) = 0;

    virtual std::any visitStructPatternElements(RustParser::StructPatternElementsContext *context) = 0;

    virtual std::any visitStructPatternFields(RustParser::StructPatternFieldsContext *context) = 0;

    virtual std::any visitStructPatternField(RustParser::StructPatternFieldContext *context) = 0;

    virtual std::any visitStructPatternEtCetera(RustParser::StructPatternEtCeteraContext *context) = 0;

    virtual std::any visitTupleStructPattern(RustParser::TupleStructPatternContext *context) = 0;

    virtual std::any visitTupleStructItems(RustParser::TupleStructItemsContext *context) = 0;

    virtual std::any visitTuplePattern(RustParser::TuplePatternContext *context) = 0;

    virtual std::any visitTuplePatternItems(RustParser::TuplePatternItemsContext *context) = 0;

    virtual std::any visitGroupedPattern(RustParser::GroupedPatternContext *context) = 0;

    virtual std::any visitSlicePattern(RustParser::SlicePatternContext *context) = 0;

    virtual std::any visitSlicePatternItems(RustParser::SlicePatternItemsContext *context) = 0;

    virtual std::any visitPathPattern(RustParser::PathPatternContext *context) = 0;

    virtual std::any visitType_(RustParser::Type_Context *context) = 0;

    virtual std::any visitTypeNoBounds(RustParser::TypeNoBoundsContext *context) = 0;

    virtual std::any visitParenthesizedType(RustParser::ParenthesizedTypeContext *context) = 0;

    virtual std::any visitNeverType(RustParser::NeverTypeContext *context) = 0;

    virtual std::any visitTupleType(RustParser::TupleTypeContext *context) = 0;

    virtual std::any visitArrayType(RustParser::ArrayTypeContext *context) = 0;

    virtual std::any visitSliceType(RustParser::SliceTypeContext *context) = 0;

    virtual std::any visitReferenceType(RustParser::ReferenceTypeContext *context) = 0;

    virtual std::any visitRawPointerType(RustParser::RawPointerTypeContext *context) = 0;

    virtual std::any visitBareFunctionType(RustParser::BareFunctionTypeContext *context) = 0;

    virtual std::any visitFunctionTypeQualifiers(RustParser::FunctionTypeQualifiersContext *context) = 0;

    virtual std::any visitBareFunctionReturnType(RustParser::BareFunctionReturnTypeContext *context) = 0;

    virtual std::any visitFunctionParametersMaybeNamedVariadic(RustParser::FunctionParametersMaybeNamedVariadicContext *context) = 0;

    virtual std::any visitMaybeNamedFunctionParameters(RustParser::MaybeNamedFunctionParametersContext *context) = 0;

    virtual std::any visitMaybeNamedParam(RustParser::MaybeNamedParamContext *context) = 0;

    virtual std::any visitMaybeNamedFunctionParametersVariadic(RustParser::MaybeNamedFunctionParametersVariadicContext *context) = 0;

    virtual std::any visitTraitObjectType(RustParser::TraitObjectTypeContext *context) = 0;

    virtual std::any visitTraitObjectTypeOneBound(RustParser::TraitObjectTypeOneBoundContext *context) = 0;

    virtual std::any visitImplTraitType(RustParser::ImplTraitTypeContext *context) = 0;

    virtual std::any visitImplTraitTypeOneBound(RustParser::ImplTraitTypeOneBoundContext *context) = 0;

    virtual std::any visitInferredType(RustParser::InferredTypeContext *context) = 0;

    virtual std::any visitTypeParamBounds(RustParser::TypeParamBoundsContext *context) = 0;

    virtual std::any visitTypeParamBound(RustParser::TypeParamBoundContext *context) = 0;

    virtual std::any visitTraitBound(RustParser::TraitBoundContext *context) = 0;

    virtual std::any visitLifetimeBounds(RustParser::LifetimeBoundsContext *context) = 0;

    virtual std::any visitLifetime(RustParser::LifetimeContext *context) = 0;

    virtual std::any visitSimplePath(RustParser::SimplePathContext *context) = 0;

    virtual std::any visitSimplePathSegment(RustParser::SimplePathSegmentContext *context) = 0;

    virtual std::any visitPathInExpression(RustParser::PathInExpressionContext *context) = 0;

    virtual std::any visitPathExprSegment(RustParser::PathExprSegmentContext *context) = 0;

    virtual std::any visitPathIdentSegment(RustParser::PathIdentSegmentContext *context) = 0;

    virtual std::any visitGenericArgs(RustParser::GenericArgsContext *context) = 0;

    virtual std::any visitGenericArg(RustParser::GenericArgContext *context) = 0;

    virtual std::any visitGenericArgsConst(RustParser::GenericArgsConstContext *context) = 0;

    virtual std::any visitGenericArgsLifetimes(RustParser::GenericArgsLifetimesContext *context) = 0;

    virtual std::any visitGenericArgsTypes(RustParser::GenericArgsTypesContext *context) = 0;

    virtual std::any visitGenericArgsBindings(RustParser::GenericArgsBindingsContext *context) = 0;

    virtual std::any visitGenericArgsBinding(RustParser::GenericArgsBindingContext *context) = 0;

    virtual std::any visitQualifiedPathInExpression(RustParser::QualifiedPathInExpressionContext *context) = 0;

    virtual std::any visitQualifiedPathType(RustParser::QualifiedPathTypeContext *context) = 0;

    virtual std::any visitQualifiedPathInType(RustParser::QualifiedPathInTypeContext *context) = 0;

    virtual std::any visitTypePath(RustParser::TypePathContext *context) = 0;

    virtual std::any visitTypePathSegment(RustParser::TypePathSegmentContext *context) = 0;

    virtual std::any visitTypePathFn(RustParser::TypePathFnContext *context) = 0;

    virtual std::any visitTypePathInputs(RustParser::TypePathInputsContext *context) = 0;

    virtual std::any visitVisibility(RustParser::VisibilityContext *context) = 0;

    virtual std::any visitIdentifier(RustParser::IdentifierContext *context) = 0;

    virtual std::any visitKeyword(RustParser::KeywordContext *context) = 0;

    virtual std::any visitMacroIdentifierLikeToken(RustParser::MacroIdentifierLikeTokenContext *context) = 0;

    virtual std::any visitMacroLiteralToken(RustParser::MacroLiteralTokenContext *context) = 0;

    virtual std::any visitMacroPunctuationToken(RustParser::MacroPunctuationTokenContext *context) = 0;

    virtual std::any visitShl(RustParser::ShlContext *context) = 0;

    virtual std::any visitShr(RustParser::ShrContext *context) = 0;


};

