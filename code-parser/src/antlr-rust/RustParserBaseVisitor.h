
// Generated from RustParser.g4 by ANTLR 4.13.1

#pragma once

#include "RustParserVisitor.h"
#include "antlr4-runtime.h"

/**
 * This class provides an empty implementation of RustParserVisitor, which can
 * be extended to create a visitor which only needs to handle a subset of the
 * available methods.
 */
class RustParserBaseVisitor : public RustParserVisitor {
public:
  virtual std::any visitCrate(RustParser::CrateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroInvocation(RustParser::MacroInvocationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitDelimTokenTree(RustParser::DelimTokenTreeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTokenTree(RustParser::TokenTreeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTokenTreeToken(RustParser::TokenTreeTokenContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMacroInvocationSemi(
      RustParser::MacroInvocationSemiContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMacroRulesDefinition(
      RustParser::MacroRulesDefinitionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroRulesDef(RustParser::MacroRulesDefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroRules(RustParser::MacroRulesContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMacroRule(RustParser::MacroRuleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroMatcher(RustParser::MacroMatcherContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroMatch(RustParser::MacroMatchContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroMatchToken(RustParser::MacroMatchTokenContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroFragSpec(RustParser::MacroFragSpecContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroRepSep(RustParser::MacroRepSepContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroRepOp(RustParser::MacroRepOpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroTranscriber(RustParser::MacroTranscriberContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitItem(RustParser::ItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVisItem(RustParser::VisItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMacroItem(RustParser::MacroItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitModule(RustParser::ModuleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitExternCrate(RustParser::ExternCrateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCrateRef(RustParser::CrateRefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAsClause(RustParser::AsClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitUseDeclaration(RustParser::UseDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUseTree(RustParser::UseTreeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunction_(RustParser::Function_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitFunctionQualifiers(RustParser::FunctionQualifiersContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAbi(RustParser::AbiContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitFunctionParameters(RustParser::FunctionParametersContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSelfParam(RustParser::SelfParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitShorthandSelf(RustParser::ShorthandSelfContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTypedSelf(RustParser::TypedSelfContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitFunctionParam(RustParser::FunctionParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionParamPattern(
      RustParser::FunctionParamPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitFunctionReturnType(RustParser::FunctionReturnTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTypeAlias(RustParser::TypeAliasContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStruct_(RustParser::Struct_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructStruct(RustParser::StructStructContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTupleStruct(RustParser::TupleStructContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructFields(RustParser::StructFieldsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructField(RustParser::StructFieldContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTupleFields(RustParser::TupleFieldsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTupleField(RustParser::TupleFieldContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEnumeration(RustParser::EnumerationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEnumItems(RustParser::EnumItemsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEnumItem(RustParser::EnumItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEnumItemTuple(RustParser::EnumItemTupleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEnumItemStruct(RustParser::EnumItemStructContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEnumItemDiscriminant(
      RustParser::EnumItemDiscriminantContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnion_(RustParser::Union_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitConstantItem(RustParser::ConstantItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStaticItem(RustParser::StaticItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTrait_(RustParser::Trait_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitImplementation(RustParser::ImplementationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitInherentImpl(RustParser::InherentImplContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTraitImpl(RustParser::TraitImplContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitExternBlock(RustParser::ExternBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitExternalItem(RustParser::ExternalItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGenericParams(RustParser::GenericParamsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGenericParam(RustParser::GenericParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitLifetimeParam(RustParser::LifetimeParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTypeParam(RustParser::TypeParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitConstParam(RustParser::ConstParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitWhereClause(RustParser::WhereClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitWhereClauseItem(RustParser::WhereClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLifetimeWhereClauseItem(
      RustParser::LifetimeWhereClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTypeBoundWhereClauseItem(
      RustParser::TypeBoundWhereClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitForLifetimes(RustParser::ForLifetimesContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitAssociatedItem(RustParser::AssociatedItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitInnerAttribute(RustParser::InnerAttributeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitOuterAttribute(RustParser::OuterAttributeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAttr(RustParser::AttrContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAttrInput(RustParser::AttrInputContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStatement(RustParser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitLetStatement(RustParser::LetStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpressionStatement(
      RustParser::ExpressionStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTypeCastExpression(RustParser::TypeCastExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitPathExpression_(RustParser::PathExpression_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTupleExpression(RustParser::TupleExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIndexExpression(RustParser::IndexExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitRangeExpression(RustParser::RangeExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMacroInvocationAsExpression(
      RustParser::MacroInvocationAsExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitReturnExpression(RustParser::ReturnExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitAwaitExpression(RustParser::AwaitExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitErrorPropagationExpression(
      RustParser::ErrorPropagationExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitContinueExpression(RustParser::ContinueExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAssignmentExpression(
      RustParser::AssignmentExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMethodCallExpression(
      RustParser::MethodCallExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitLiteralExpression_(RustParser::LiteralExpression_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructExpression_(RustParser::StructExpression_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTupleIndexingExpression(
      RustParser::TupleIndexingExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitNegationExpression(RustParser::NegationExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitCallExpression(RustParser::CallExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLazyBooleanExpression(
      RustParser::LazyBooleanExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDereferenceExpression(
      RustParser::DereferenceExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpressionWithBlock_(
      RustParser::ExpressionWithBlock_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGroupedExpression(RustParser::GroupedExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitBreakExpression(RustParser::BreakExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticOrLogicalExpression(
      RustParser::ArithmeticOrLogicalExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitFieldExpression(RustParser::FieldExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEnumerationVariantExpression_(
      RustParser::EnumerationVariantExpression_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonExpression(
      RustParser::ComparisonExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAttributedExpression(
      RustParser::AttributedExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitBorrowExpression(RustParser::BorrowExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCompoundAssignmentExpression(
      RustParser::CompoundAssignmentExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitClosureExpression_(RustParser::ClosureExpression_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitArrayExpression(RustParser::ArrayExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitComparisonOperator(RustParser::ComparisonOperatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCompoundAssignOperator(
      RustParser::CompoundAssignOperatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpressionWithBlock(
      RustParser::ExpressionWithBlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitLiteralExpression(RustParser::LiteralExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitPathExpression(RustParser::PathExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitBlockExpression(RustParser::BlockExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStatements(RustParser::StatementsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAsyncBlockExpression(
      RustParser::AsyncBlockExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnsafeBlockExpression(
      RustParser::UnsafeBlockExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitArrayElements(RustParser::ArrayElementsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTupleElements(RustParser::TupleElementsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTupleIndex(RustParser::TupleIndexContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructExpression(RustParser::StructExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructExprStruct(RustParser::StructExprStructContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructExprFields(RustParser::StructExprFieldsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructExprField(RustParser::StructExprFieldContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructBase(RustParser::StructBaseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructExprTuple(RustParser::StructExprTupleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructExprUnit(RustParser::StructExprUnitContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEnumerationVariantExpression(
      RustParser::EnumerationVariantExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEnumExprStruct(RustParser::EnumExprStructContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEnumExprFields(RustParser::EnumExprFieldsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEnumExprField(RustParser::EnumExprFieldContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEnumExprTuple(RustParser::EnumExprTupleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEnumExprFieldless(RustParser::EnumExprFieldlessContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitCallParams(RustParser::CallParamsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitClosureExpression(RustParser::ClosureExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitClosureParameters(RustParser::ClosureParametersContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitClosureParam(RustParser::ClosureParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitLoopExpression(RustParser::LoopExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInfiniteLoopExpression(
      RustParser::InfiniteLoopExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPredicateLoopExpression(
      RustParser::PredicateLoopExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPredicatePatternLoopExpression(
      RustParser::PredicatePatternLoopExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIteratorLoopExpression(
      RustParser::IteratorLoopExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLoopLabel(RustParser::LoopLabelContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIfExpression(RustParser::IfExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIfLetExpression(RustParser::IfLetExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMatchExpression(RustParser::MatchExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMatchArms(RustParser::MatchArmsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMatchArmExpression(RustParser::MatchArmExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMatchArm(RustParser::MatchArmContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMatchArmGuard(RustParser::MatchArmGuardContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPattern(RustParser::PatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitPatternNoTopAlt(RustParser::PatternNoTopAltContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPatternWithoutRange(
      RustParser::PatternWithoutRangeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitLiteralPattern(RustParser::LiteralPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIdentifierPattern(RustParser::IdentifierPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitWildcardPattern(RustParser::WildcardPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitRestPattern(RustParser::RestPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInclusiveRangePattern(
      RustParser::InclusiveRangePatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHalfOpenRangePattern(
      RustParser::HalfOpenRangePatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitObsoleteRangePattern(
      RustParser::ObsoleteRangePatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitRangePatternBound(RustParser::RangePatternBoundContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitReferencePattern(RustParser::ReferencePatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructPattern(RustParser::StructPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStructPatternElements(
      RustParser::StructPatternElementsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStructPatternFields(
      RustParser::StructPatternFieldsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStructPatternField(RustParser::StructPatternFieldContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStructPatternEtCetera(
      RustParser::StructPatternEtCeteraContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTupleStructPattern(RustParser::TupleStructPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTupleStructItems(RustParser::TupleStructItemsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTuplePattern(RustParser::TuplePatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTuplePatternItems(RustParser::TuplePatternItemsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGroupedPattern(RustParser::GroupedPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitSlicePattern(RustParser::SlicePatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitSlicePatternItems(RustParser::SlicePatternItemsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitPathPattern(RustParser::PathPatternContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitType_(RustParser::Type_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTypeNoBounds(RustParser::TypeNoBoundsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitParenthesizedType(RustParser::ParenthesizedTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNeverType(RustParser::NeverTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTupleType(RustParser::TupleTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArrayType(RustParser::ArrayTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSliceType(RustParser::SliceTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitReferenceType(RustParser::ReferenceTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitRawPointerType(RustParser::RawPointerTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitBareFunctionType(RustParser::BareFunctionTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionTypeQualifiers(
      RustParser::FunctionTypeQualifiersContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBareFunctionReturnType(
      RustParser::BareFunctionReturnTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionParametersMaybeNamedVariadic(
      RustParser::FunctionParametersMaybeNamedVariadicContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMaybeNamedFunctionParameters(
      RustParser::MaybeNamedFunctionParametersContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMaybeNamedParam(RustParser::MaybeNamedParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMaybeNamedFunctionParametersVariadic(
      RustParser::MaybeNamedFunctionParametersVariadicContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTraitObjectType(RustParser::TraitObjectTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTraitObjectTypeOneBound(
      RustParser::TraitObjectTypeOneBoundContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitImplTraitType(RustParser::ImplTraitTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitImplTraitTypeOneBound(
      RustParser::ImplTraitTypeOneBoundContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitInferredType(RustParser::InferredTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTypeParamBounds(RustParser::TypeParamBoundsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTypeParamBound(RustParser::TypeParamBoundContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTraitBound(RustParser::TraitBoundContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitLifetimeBounds(RustParser::LifetimeBoundsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLifetime(RustParser::LifetimeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitSimplePath(RustParser::SimplePathContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitSimplePathSegment(RustParser::SimplePathSegmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitPathInExpression(RustParser::PathInExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitPathExprSegment(RustParser::PathExprSegmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitPathIdentSegment(RustParser::PathIdentSegmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGenericArgs(RustParser::GenericArgsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGenericArg(RustParser::GenericArgContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGenericArgsConst(RustParser::GenericArgsConstContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGenericArgsLifetimes(
      RustParser::GenericArgsLifetimesContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGenericArgsTypes(RustParser::GenericArgsTypesContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGenericArgsBindings(
      RustParser::GenericArgsBindingsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGenericArgsBinding(RustParser::GenericArgsBindingContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitQualifiedPathInExpression(
      RustParser::QualifiedPathInExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitQualifiedPathType(RustParser::QualifiedPathTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitQualifiedPathInType(
      RustParser::QualifiedPathInTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTypePath(RustParser::TypePathContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTypePathSegment(RustParser::TypePathSegmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTypePathFn(RustParser::TypePathFnContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitTypePathInputs(RustParser::TypePathInputsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitVisibility(RustParser::VisibilityContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIdentifier(RustParser::IdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitKeyword(RustParser::KeywordContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMacroIdentifierLikeToken(
      RustParser::MacroIdentifierLikeTokenContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMacroLiteralToken(RustParser::MacroLiteralTokenContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMacroPunctuationToken(
      RustParser::MacroPunctuationTokenContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitShl(RustParser::ShlContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitShr(RustParser::ShrContext *ctx) override {
    return visitChildren(ctx);
  }
};
