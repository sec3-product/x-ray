
// Generated from RustParser.g4 by ANTLR 4.13.1

#pragma once

#include "RustParser.h"
#include "antlr4-runtime.h"

/**
 * This interface defines an abstract listener for a parse tree produced by
 * RustParser.
 */
class RustParserListener : public antlr4::tree::ParseTreeListener {
public:
  virtual void enterCrate(RustParser::CrateContext *ctx) = 0;
  virtual void exitCrate(RustParser::CrateContext *ctx) = 0;

  virtual void
  enterMacroInvocation(RustParser::MacroInvocationContext *ctx) = 0;
  virtual void exitMacroInvocation(RustParser::MacroInvocationContext *ctx) = 0;

  virtual void enterDelimTokenTree(RustParser::DelimTokenTreeContext *ctx) = 0;
  virtual void exitDelimTokenTree(RustParser::DelimTokenTreeContext *ctx) = 0;

  virtual void enterTokenTree(RustParser::TokenTreeContext *ctx) = 0;
  virtual void exitTokenTree(RustParser::TokenTreeContext *ctx) = 0;

  virtual void enterTokenTreeToken(RustParser::TokenTreeTokenContext *ctx) = 0;
  virtual void exitTokenTreeToken(RustParser::TokenTreeTokenContext *ctx) = 0;

  virtual void
  enterMacroInvocationSemi(RustParser::MacroInvocationSemiContext *ctx) = 0;
  virtual void
  exitMacroInvocationSemi(RustParser::MacroInvocationSemiContext *ctx) = 0;

  virtual void
  enterMacroRulesDefinition(RustParser::MacroRulesDefinitionContext *ctx) = 0;
  virtual void
  exitMacroRulesDefinition(RustParser::MacroRulesDefinitionContext *ctx) = 0;

  virtual void enterMacroRulesDef(RustParser::MacroRulesDefContext *ctx) = 0;
  virtual void exitMacroRulesDef(RustParser::MacroRulesDefContext *ctx) = 0;

  virtual void enterMacroRules(RustParser::MacroRulesContext *ctx) = 0;
  virtual void exitMacroRules(RustParser::MacroRulesContext *ctx) = 0;

  virtual void enterMacroRule(RustParser::MacroRuleContext *ctx) = 0;
  virtual void exitMacroRule(RustParser::MacroRuleContext *ctx) = 0;

  virtual void enterMacroMatcher(RustParser::MacroMatcherContext *ctx) = 0;
  virtual void exitMacroMatcher(RustParser::MacroMatcherContext *ctx) = 0;

  virtual void enterMacroMatch(RustParser::MacroMatchContext *ctx) = 0;
  virtual void exitMacroMatch(RustParser::MacroMatchContext *ctx) = 0;

  virtual void
  enterMacroMatchToken(RustParser::MacroMatchTokenContext *ctx) = 0;
  virtual void exitMacroMatchToken(RustParser::MacroMatchTokenContext *ctx) = 0;

  virtual void enterMacroFragSpec(RustParser::MacroFragSpecContext *ctx) = 0;
  virtual void exitMacroFragSpec(RustParser::MacroFragSpecContext *ctx) = 0;

  virtual void enterMacroRepSep(RustParser::MacroRepSepContext *ctx) = 0;
  virtual void exitMacroRepSep(RustParser::MacroRepSepContext *ctx) = 0;

  virtual void enterMacroRepOp(RustParser::MacroRepOpContext *ctx) = 0;
  virtual void exitMacroRepOp(RustParser::MacroRepOpContext *ctx) = 0;

  virtual void
  enterMacroTranscriber(RustParser::MacroTranscriberContext *ctx) = 0;
  virtual void
  exitMacroTranscriber(RustParser::MacroTranscriberContext *ctx) = 0;

  virtual void enterItem(RustParser::ItemContext *ctx) = 0;
  virtual void exitItem(RustParser::ItemContext *ctx) = 0;

  virtual void enterVisItem(RustParser::VisItemContext *ctx) = 0;
  virtual void exitVisItem(RustParser::VisItemContext *ctx) = 0;

  virtual void enterMacroItem(RustParser::MacroItemContext *ctx) = 0;
  virtual void exitMacroItem(RustParser::MacroItemContext *ctx) = 0;

  virtual void enterModule(RustParser::ModuleContext *ctx) = 0;
  virtual void exitModule(RustParser::ModuleContext *ctx) = 0;

  virtual void enterExternCrate(RustParser::ExternCrateContext *ctx) = 0;
  virtual void exitExternCrate(RustParser::ExternCrateContext *ctx) = 0;

  virtual void enterCrateRef(RustParser::CrateRefContext *ctx) = 0;
  virtual void exitCrateRef(RustParser::CrateRefContext *ctx) = 0;

  virtual void enterAsClause(RustParser::AsClauseContext *ctx) = 0;
  virtual void exitAsClause(RustParser::AsClauseContext *ctx) = 0;

  virtual void enterUseDeclaration(RustParser::UseDeclarationContext *ctx) = 0;
  virtual void exitUseDeclaration(RustParser::UseDeclarationContext *ctx) = 0;

  virtual void enterUseTree(RustParser::UseTreeContext *ctx) = 0;
  virtual void exitUseTree(RustParser::UseTreeContext *ctx) = 0;

  virtual void enterFunction_(RustParser::Function_Context *ctx) = 0;
  virtual void exitFunction_(RustParser::Function_Context *ctx) = 0;

  virtual void
  enterFunctionQualifiers(RustParser::FunctionQualifiersContext *ctx) = 0;
  virtual void
  exitFunctionQualifiers(RustParser::FunctionQualifiersContext *ctx) = 0;

  virtual void enterAbi(RustParser::AbiContext *ctx) = 0;
  virtual void exitAbi(RustParser::AbiContext *ctx) = 0;

  virtual void
  enterFunctionParameters(RustParser::FunctionParametersContext *ctx) = 0;
  virtual void
  exitFunctionParameters(RustParser::FunctionParametersContext *ctx) = 0;

  virtual void enterSelfParam(RustParser::SelfParamContext *ctx) = 0;
  virtual void exitSelfParam(RustParser::SelfParamContext *ctx) = 0;

  virtual void enterShorthandSelf(RustParser::ShorthandSelfContext *ctx) = 0;
  virtual void exitShorthandSelf(RustParser::ShorthandSelfContext *ctx) = 0;

  virtual void enterTypedSelf(RustParser::TypedSelfContext *ctx) = 0;
  virtual void exitTypedSelf(RustParser::TypedSelfContext *ctx) = 0;

  virtual void enterFunctionParam(RustParser::FunctionParamContext *ctx) = 0;
  virtual void exitFunctionParam(RustParser::FunctionParamContext *ctx) = 0;

  virtual void
  enterFunctionParamPattern(RustParser::FunctionParamPatternContext *ctx) = 0;
  virtual void
  exitFunctionParamPattern(RustParser::FunctionParamPatternContext *ctx) = 0;

  virtual void
  enterFunctionReturnType(RustParser::FunctionReturnTypeContext *ctx) = 0;
  virtual void
  exitFunctionReturnType(RustParser::FunctionReturnTypeContext *ctx) = 0;

  virtual void enterTypeAlias(RustParser::TypeAliasContext *ctx) = 0;
  virtual void exitTypeAlias(RustParser::TypeAliasContext *ctx) = 0;

  virtual void enterStruct_(RustParser::Struct_Context *ctx) = 0;
  virtual void exitStruct_(RustParser::Struct_Context *ctx) = 0;

  virtual void enterStructStruct(RustParser::StructStructContext *ctx) = 0;
  virtual void exitStructStruct(RustParser::StructStructContext *ctx) = 0;

  virtual void enterTupleStruct(RustParser::TupleStructContext *ctx) = 0;
  virtual void exitTupleStruct(RustParser::TupleStructContext *ctx) = 0;

  virtual void enterStructFields(RustParser::StructFieldsContext *ctx) = 0;
  virtual void exitStructFields(RustParser::StructFieldsContext *ctx) = 0;

  virtual void enterStructField(RustParser::StructFieldContext *ctx) = 0;
  virtual void exitStructField(RustParser::StructFieldContext *ctx) = 0;

  virtual void enterTupleFields(RustParser::TupleFieldsContext *ctx) = 0;
  virtual void exitTupleFields(RustParser::TupleFieldsContext *ctx) = 0;

  virtual void enterTupleField(RustParser::TupleFieldContext *ctx) = 0;
  virtual void exitTupleField(RustParser::TupleFieldContext *ctx) = 0;

  virtual void enterEnumeration(RustParser::EnumerationContext *ctx) = 0;
  virtual void exitEnumeration(RustParser::EnumerationContext *ctx) = 0;

  virtual void enterEnumItems(RustParser::EnumItemsContext *ctx) = 0;
  virtual void exitEnumItems(RustParser::EnumItemsContext *ctx) = 0;

  virtual void enterEnumItem(RustParser::EnumItemContext *ctx) = 0;
  virtual void exitEnumItem(RustParser::EnumItemContext *ctx) = 0;

  virtual void enterEnumItemTuple(RustParser::EnumItemTupleContext *ctx) = 0;
  virtual void exitEnumItemTuple(RustParser::EnumItemTupleContext *ctx) = 0;

  virtual void enterEnumItemStruct(RustParser::EnumItemStructContext *ctx) = 0;
  virtual void exitEnumItemStruct(RustParser::EnumItemStructContext *ctx) = 0;

  virtual void
  enterEnumItemDiscriminant(RustParser::EnumItemDiscriminantContext *ctx) = 0;
  virtual void
  exitEnumItemDiscriminant(RustParser::EnumItemDiscriminantContext *ctx) = 0;

  virtual void enterUnion_(RustParser::Union_Context *ctx) = 0;
  virtual void exitUnion_(RustParser::Union_Context *ctx) = 0;

  virtual void enterConstantItem(RustParser::ConstantItemContext *ctx) = 0;
  virtual void exitConstantItem(RustParser::ConstantItemContext *ctx) = 0;

  virtual void enterStaticItem(RustParser::StaticItemContext *ctx) = 0;
  virtual void exitStaticItem(RustParser::StaticItemContext *ctx) = 0;

  virtual void enterTrait_(RustParser::Trait_Context *ctx) = 0;
  virtual void exitTrait_(RustParser::Trait_Context *ctx) = 0;

  virtual void enterImplementation(RustParser::ImplementationContext *ctx) = 0;
  virtual void exitImplementation(RustParser::ImplementationContext *ctx) = 0;

  virtual void enterInherentImpl(RustParser::InherentImplContext *ctx) = 0;
  virtual void exitInherentImpl(RustParser::InherentImplContext *ctx) = 0;

  virtual void enterTraitImpl(RustParser::TraitImplContext *ctx) = 0;
  virtual void exitTraitImpl(RustParser::TraitImplContext *ctx) = 0;

  virtual void enterExternBlock(RustParser::ExternBlockContext *ctx) = 0;
  virtual void exitExternBlock(RustParser::ExternBlockContext *ctx) = 0;

  virtual void enterExternalItem(RustParser::ExternalItemContext *ctx) = 0;
  virtual void exitExternalItem(RustParser::ExternalItemContext *ctx) = 0;

  virtual void enterGenericParams(RustParser::GenericParamsContext *ctx) = 0;
  virtual void exitGenericParams(RustParser::GenericParamsContext *ctx) = 0;

  virtual void enterGenericParam(RustParser::GenericParamContext *ctx) = 0;
  virtual void exitGenericParam(RustParser::GenericParamContext *ctx) = 0;

  virtual void enterLifetimeParam(RustParser::LifetimeParamContext *ctx) = 0;
  virtual void exitLifetimeParam(RustParser::LifetimeParamContext *ctx) = 0;

  virtual void enterTypeParam(RustParser::TypeParamContext *ctx) = 0;
  virtual void exitTypeParam(RustParser::TypeParamContext *ctx) = 0;

  virtual void enterConstParam(RustParser::ConstParamContext *ctx) = 0;
  virtual void exitConstParam(RustParser::ConstParamContext *ctx) = 0;

  virtual void enterWhereClause(RustParser::WhereClauseContext *ctx) = 0;
  virtual void exitWhereClause(RustParser::WhereClauseContext *ctx) = 0;

  virtual void
  enterWhereClauseItem(RustParser::WhereClauseItemContext *ctx) = 0;
  virtual void exitWhereClauseItem(RustParser::WhereClauseItemContext *ctx) = 0;

  virtual void enterLifetimeWhereClauseItem(
      RustParser::LifetimeWhereClauseItemContext *ctx) = 0;
  virtual void exitLifetimeWhereClauseItem(
      RustParser::LifetimeWhereClauseItemContext *ctx) = 0;

  virtual void enterTypeBoundWhereClauseItem(
      RustParser::TypeBoundWhereClauseItemContext *ctx) = 0;
  virtual void exitTypeBoundWhereClauseItem(
      RustParser::TypeBoundWhereClauseItemContext *ctx) = 0;

  virtual void enterForLifetimes(RustParser::ForLifetimesContext *ctx) = 0;
  virtual void exitForLifetimes(RustParser::ForLifetimesContext *ctx) = 0;

  virtual void enterAssociatedItem(RustParser::AssociatedItemContext *ctx) = 0;
  virtual void exitAssociatedItem(RustParser::AssociatedItemContext *ctx) = 0;

  virtual void enterInnerAttribute(RustParser::InnerAttributeContext *ctx) = 0;
  virtual void exitInnerAttribute(RustParser::InnerAttributeContext *ctx) = 0;

  virtual void enterOuterAttribute(RustParser::OuterAttributeContext *ctx) = 0;
  virtual void exitOuterAttribute(RustParser::OuterAttributeContext *ctx) = 0;

  virtual void enterAttr(RustParser::AttrContext *ctx) = 0;
  virtual void exitAttr(RustParser::AttrContext *ctx) = 0;

  virtual void enterAttrInput(RustParser::AttrInputContext *ctx) = 0;
  virtual void exitAttrInput(RustParser::AttrInputContext *ctx) = 0;

  virtual void enterStatement(RustParser::StatementContext *ctx) = 0;
  virtual void exitStatement(RustParser::StatementContext *ctx) = 0;

  virtual void enterLetStatement(RustParser::LetStatementContext *ctx) = 0;
  virtual void exitLetStatement(RustParser::LetStatementContext *ctx) = 0;

  virtual void
  enterExpressionStatement(RustParser::ExpressionStatementContext *ctx) = 0;
  virtual void
  exitExpressionStatement(RustParser::ExpressionStatementContext *ctx) = 0;

  virtual void
  enterTypeCastExpression(RustParser::TypeCastExpressionContext *ctx) = 0;
  virtual void
  exitTypeCastExpression(RustParser::TypeCastExpressionContext *ctx) = 0;

  virtual void
  enterPathExpression_(RustParser::PathExpression_Context *ctx) = 0;
  virtual void exitPathExpression_(RustParser::PathExpression_Context *ctx) = 0;

  virtual void
  enterTupleExpression(RustParser::TupleExpressionContext *ctx) = 0;
  virtual void exitTupleExpression(RustParser::TupleExpressionContext *ctx) = 0;

  virtual void
  enterIndexExpression(RustParser::IndexExpressionContext *ctx) = 0;
  virtual void exitIndexExpression(RustParser::IndexExpressionContext *ctx) = 0;

  virtual void
  enterRangeExpression(RustParser::RangeExpressionContext *ctx) = 0;
  virtual void exitRangeExpression(RustParser::RangeExpressionContext *ctx) = 0;

  virtual void enterMacroInvocationAsExpression(
      RustParser::MacroInvocationAsExpressionContext *ctx) = 0;
  virtual void exitMacroInvocationAsExpression(
      RustParser::MacroInvocationAsExpressionContext *ctx) = 0;

  virtual void
  enterReturnExpression(RustParser::ReturnExpressionContext *ctx) = 0;
  virtual void
  exitReturnExpression(RustParser::ReturnExpressionContext *ctx) = 0;

  virtual void
  enterAwaitExpression(RustParser::AwaitExpressionContext *ctx) = 0;
  virtual void exitAwaitExpression(RustParser::AwaitExpressionContext *ctx) = 0;

  virtual void enterErrorPropagationExpression(
      RustParser::ErrorPropagationExpressionContext *ctx) = 0;
  virtual void exitErrorPropagationExpression(
      RustParser::ErrorPropagationExpressionContext *ctx) = 0;

  virtual void
  enterContinueExpression(RustParser::ContinueExpressionContext *ctx) = 0;
  virtual void
  exitContinueExpression(RustParser::ContinueExpressionContext *ctx) = 0;

  virtual void
  enterAssignmentExpression(RustParser::AssignmentExpressionContext *ctx) = 0;
  virtual void
  exitAssignmentExpression(RustParser::AssignmentExpressionContext *ctx) = 0;

  virtual void
  enterMethodCallExpression(RustParser::MethodCallExpressionContext *ctx) = 0;
  virtual void
  exitMethodCallExpression(RustParser::MethodCallExpressionContext *ctx) = 0;

  virtual void
  enterLiteralExpression_(RustParser::LiteralExpression_Context *ctx) = 0;
  virtual void
  exitLiteralExpression_(RustParser::LiteralExpression_Context *ctx) = 0;

  virtual void
  enterStructExpression_(RustParser::StructExpression_Context *ctx) = 0;
  virtual void
  exitStructExpression_(RustParser::StructExpression_Context *ctx) = 0;

  virtual void enterTupleIndexingExpression(
      RustParser::TupleIndexingExpressionContext *ctx) = 0;
  virtual void exitTupleIndexingExpression(
      RustParser::TupleIndexingExpressionContext *ctx) = 0;

  virtual void
  enterNegationExpression(RustParser::NegationExpressionContext *ctx) = 0;
  virtual void
  exitNegationExpression(RustParser::NegationExpressionContext *ctx) = 0;

  virtual void enterCallExpression(RustParser::CallExpressionContext *ctx) = 0;
  virtual void exitCallExpression(RustParser::CallExpressionContext *ctx) = 0;

  virtual void
  enterLazyBooleanExpression(RustParser::LazyBooleanExpressionContext *ctx) = 0;
  virtual void
  exitLazyBooleanExpression(RustParser::LazyBooleanExpressionContext *ctx) = 0;

  virtual void
  enterDereferenceExpression(RustParser::DereferenceExpressionContext *ctx) = 0;
  virtual void
  exitDereferenceExpression(RustParser::DereferenceExpressionContext *ctx) = 0;

  virtual void
  enterExpressionWithBlock_(RustParser::ExpressionWithBlock_Context *ctx) = 0;
  virtual void
  exitExpressionWithBlock_(RustParser::ExpressionWithBlock_Context *ctx) = 0;

  virtual void
  enterGroupedExpression(RustParser::GroupedExpressionContext *ctx) = 0;
  virtual void
  exitGroupedExpression(RustParser::GroupedExpressionContext *ctx) = 0;

  virtual void
  enterBreakExpression(RustParser::BreakExpressionContext *ctx) = 0;
  virtual void exitBreakExpression(RustParser::BreakExpressionContext *ctx) = 0;

  virtual void enterArithmeticOrLogicalExpression(
      RustParser::ArithmeticOrLogicalExpressionContext *ctx) = 0;
  virtual void exitArithmeticOrLogicalExpression(
      RustParser::ArithmeticOrLogicalExpressionContext *ctx) = 0;

  virtual void
  enterFieldExpression(RustParser::FieldExpressionContext *ctx) = 0;
  virtual void exitFieldExpression(RustParser::FieldExpressionContext *ctx) = 0;

  virtual void enterEnumerationVariantExpression_(
      RustParser::EnumerationVariantExpression_Context *ctx) = 0;
  virtual void exitEnumerationVariantExpression_(
      RustParser::EnumerationVariantExpression_Context *ctx) = 0;

  virtual void
  enterComparisonExpression(RustParser::ComparisonExpressionContext *ctx) = 0;
  virtual void
  exitComparisonExpression(RustParser::ComparisonExpressionContext *ctx) = 0;

  virtual void
  enterAttributedExpression(RustParser::AttributedExpressionContext *ctx) = 0;
  virtual void
  exitAttributedExpression(RustParser::AttributedExpressionContext *ctx) = 0;

  virtual void
  enterBorrowExpression(RustParser::BorrowExpressionContext *ctx) = 0;
  virtual void
  exitBorrowExpression(RustParser::BorrowExpressionContext *ctx) = 0;

  virtual void enterCompoundAssignmentExpression(
      RustParser::CompoundAssignmentExpressionContext *ctx) = 0;
  virtual void exitCompoundAssignmentExpression(
      RustParser::CompoundAssignmentExpressionContext *ctx) = 0;

  virtual void
  enterClosureExpression_(RustParser::ClosureExpression_Context *ctx) = 0;
  virtual void
  exitClosureExpression_(RustParser::ClosureExpression_Context *ctx) = 0;

  virtual void
  enterArrayExpression(RustParser::ArrayExpressionContext *ctx) = 0;
  virtual void exitArrayExpression(RustParser::ArrayExpressionContext *ctx) = 0;

  virtual void
  enterComparisonOperator(RustParser::ComparisonOperatorContext *ctx) = 0;
  virtual void
  exitComparisonOperator(RustParser::ComparisonOperatorContext *ctx) = 0;

  virtual void enterCompoundAssignOperator(
      RustParser::CompoundAssignOperatorContext *ctx) = 0;
  virtual void exitCompoundAssignOperator(
      RustParser::CompoundAssignOperatorContext *ctx) = 0;

  virtual void
  enterExpressionWithBlock(RustParser::ExpressionWithBlockContext *ctx) = 0;
  virtual void
  exitExpressionWithBlock(RustParser::ExpressionWithBlockContext *ctx) = 0;

  virtual void
  enterLiteralExpression(RustParser::LiteralExpressionContext *ctx) = 0;
  virtual void
  exitLiteralExpression(RustParser::LiteralExpressionContext *ctx) = 0;

  virtual void enterPathExpression(RustParser::PathExpressionContext *ctx) = 0;
  virtual void exitPathExpression(RustParser::PathExpressionContext *ctx) = 0;

  virtual void
  enterBlockExpression(RustParser::BlockExpressionContext *ctx) = 0;
  virtual void exitBlockExpression(RustParser::BlockExpressionContext *ctx) = 0;

  virtual void enterStatements(RustParser::StatementsContext *ctx) = 0;
  virtual void exitStatements(RustParser::StatementsContext *ctx) = 0;

  virtual void
  enterAsyncBlockExpression(RustParser::AsyncBlockExpressionContext *ctx) = 0;
  virtual void
  exitAsyncBlockExpression(RustParser::AsyncBlockExpressionContext *ctx) = 0;

  virtual void
  enterUnsafeBlockExpression(RustParser::UnsafeBlockExpressionContext *ctx) = 0;
  virtual void
  exitUnsafeBlockExpression(RustParser::UnsafeBlockExpressionContext *ctx) = 0;

  virtual void enterArrayElements(RustParser::ArrayElementsContext *ctx) = 0;
  virtual void exitArrayElements(RustParser::ArrayElementsContext *ctx) = 0;

  virtual void enterTupleElements(RustParser::TupleElementsContext *ctx) = 0;
  virtual void exitTupleElements(RustParser::TupleElementsContext *ctx) = 0;

  virtual void enterTupleIndex(RustParser::TupleIndexContext *ctx) = 0;
  virtual void exitTupleIndex(RustParser::TupleIndexContext *ctx) = 0;

  virtual void
  enterStructExpression(RustParser::StructExpressionContext *ctx) = 0;
  virtual void
  exitStructExpression(RustParser::StructExpressionContext *ctx) = 0;

  virtual void
  enterStructExprStruct(RustParser::StructExprStructContext *ctx) = 0;
  virtual void
  exitStructExprStruct(RustParser::StructExprStructContext *ctx) = 0;

  virtual void
  enterStructExprFields(RustParser::StructExprFieldsContext *ctx) = 0;
  virtual void
  exitStructExprFields(RustParser::StructExprFieldsContext *ctx) = 0;

  virtual void
  enterStructExprField(RustParser::StructExprFieldContext *ctx) = 0;
  virtual void exitStructExprField(RustParser::StructExprFieldContext *ctx) = 0;

  virtual void enterStructBase(RustParser::StructBaseContext *ctx) = 0;
  virtual void exitStructBase(RustParser::StructBaseContext *ctx) = 0;

  virtual void
  enterStructExprTuple(RustParser::StructExprTupleContext *ctx) = 0;
  virtual void exitStructExprTuple(RustParser::StructExprTupleContext *ctx) = 0;

  virtual void enterStructExprUnit(RustParser::StructExprUnitContext *ctx) = 0;
  virtual void exitStructExprUnit(RustParser::StructExprUnitContext *ctx) = 0;

  virtual void enterEnumerationVariantExpression(
      RustParser::EnumerationVariantExpressionContext *ctx) = 0;
  virtual void exitEnumerationVariantExpression(
      RustParser::EnumerationVariantExpressionContext *ctx) = 0;

  virtual void enterEnumExprStruct(RustParser::EnumExprStructContext *ctx) = 0;
  virtual void exitEnumExprStruct(RustParser::EnumExprStructContext *ctx) = 0;

  virtual void enterEnumExprFields(RustParser::EnumExprFieldsContext *ctx) = 0;
  virtual void exitEnumExprFields(RustParser::EnumExprFieldsContext *ctx) = 0;

  virtual void enterEnumExprField(RustParser::EnumExprFieldContext *ctx) = 0;
  virtual void exitEnumExprField(RustParser::EnumExprFieldContext *ctx) = 0;

  virtual void enterEnumExprTuple(RustParser::EnumExprTupleContext *ctx) = 0;
  virtual void exitEnumExprTuple(RustParser::EnumExprTupleContext *ctx) = 0;

  virtual void
  enterEnumExprFieldless(RustParser::EnumExprFieldlessContext *ctx) = 0;
  virtual void
  exitEnumExprFieldless(RustParser::EnumExprFieldlessContext *ctx) = 0;

  virtual void enterCallParams(RustParser::CallParamsContext *ctx) = 0;
  virtual void exitCallParams(RustParser::CallParamsContext *ctx) = 0;

  virtual void
  enterClosureExpression(RustParser::ClosureExpressionContext *ctx) = 0;
  virtual void
  exitClosureExpression(RustParser::ClosureExpressionContext *ctx) = 0;

  virtual void
  enterClosureParameters(RustParser::ClosureParametersContext *ctx) = 0;
  virtual void
  exitClosureParameters(RustParser::ClosureParametersContext *ctx) = 0;

  virtual void enterClosureParam(RustParser::ClosureParamContext *ctx) = 0;
  virtual void exitClosureParam(RustParser::ClosureParamContext *ctx) = 0;

  virtual void enterLoopExpression(RustParser::LoopExpressionContext *ctx) = 0;
  virtual void exitLoopExpression(RustParser::LoopExpressionContext *ctx) = 0;

  virtual void enterInfiniteLoopExpression(
      RustParser::InfiniteLoopExpressionContext *ctx) = 0;
  virtual void exitInfiniteLoopExpression(
      RustParser::InfiniteLoopExpressionContext *ctx) = 0;

  virtual void enterPredicateLoopExpression(
      RustParser::PredicateLoopExpressionContext *ctx) = 0;
  virtual void exitPredicateLoopExpression(
      RustParser::PredicateLoopExpressionContext *ctx) = 0;

  virtual void enterPredicatePatternLoopExpression(
      RustParser::PredicatePatternLoopExpressionContext *ctx) = 0;
  virtual void exitPredicatePatternLoopExpression(
      RustParser::PredicatePatternLoopExpressionContext *ctx) = 0;

  virtual void enterIteratorLoopExpression(
      RustParser::IteratorLoopExpressionContext *ctx) = 0;
  virtual void exitIteratorLoopExpression(
      RustParser::IteratorLoopExpressionContext *ctx) = 0;

  virtual void enterLoopLabel(RustParser::LoopLabelContext *ctx) = 0;
  virtual void exitLoopLabel(RustParser::LoopLabelContext *ctx) = 0;

  virtual void enterIfExpression(RustParser::IfExpressionContext *ctx) = 0;
  virtual void exitIfExpression(RustParser::IfExpressionContext *ctx) = 0;

  virtual void
  enterIfLetExpression(RustParser::IfLetExpressionContext *ctx) = 0;
  virtual void exitIfLetExpression(RustParser::IfLetExpressionContext *ctx) = 0;

  virtual void
  enterMatchExpression(RustParser::MatchExpressionContext *ctx) = 0;
  virtual void exitMatchExpression(RustParser::MatchExpressionContext *ctx) = 0;

  virtual void enterMatchArms(RustParser::MatchArmsContext *ctx) = 0;
  virtual void exitMatchArms(RustParser::MatchArmsContext *ctx) = 0;

  virtual void
  enterMatchArmExpression(RustParser::MatchArmExpressionContext *ctx) = 0;
  virtual void
  exitMatchArmExpression(RustParser::MatchArmExpressionContext *ctx) = 0;

  virtual void enterMatchArm(RustParser::MatchArmContext *ctx) = 0;
  virtual void exitMatchArm(RustParser::MatchArmContext *ctx) = 0;

  virtual void enterMatchArmGuard(RustParser::MatchArmGuardContext *ctx) = 0;
  virtual void exitMatchArmGuard(RustParser::MatchArmGuardContext *ctx) = 0;

  virtual void enterPattern(RustParser::PatternContext *ctx) = 0;
  virtual void exitPattern(RustParser::PatternContext *ctx) = 0;

  virtual void
  enterPatternNoTopAlt(RustParser::PatternNoTopAltContext *ctx) = 0;
  virtual void exitPatternNoTopAlt(RustParser::PatternNoTopAltContext *ctx) = 0;

  virtual void
  enterPatternWithoutRange(RustParser::PatternWithoutRangeContext *ctx) = 0;
  virtual void
  exitPatternWithoutRange(RustParser::PatternWithoutRangeContext *ctx) = 0;

  virtual void enterLiteralPattern(RustParser::LiteralPatternContext *ctx) = 0;
  virtual void exitLiteralPattern(RustParser::LiteralPatternContext *ctx) = 0;

  virtual void
  enterIdentifierPattern(RustParser::IdentifierPatternContext *ctx) = 0;
  virtual void
  exitIdentifierPattern(RustParser::IdentifierPatternContext *ctx) = 0;

  virtual void
  enterWildcardPattern(RustParser::WildcardPatternContext *ctx) = 0;
  virtual void exitWildcardPattern(RustParser::WildcardPatternContext *ctx) = 0;

  virtual void enterRestPattern(RustParser::RestPatternContext *ctx) = 0;
  virtual void exitRestPattern(RustParser::RestPatternContext *ctx) = 0;

  virtual void
  enterInclusiveRangePattern(RustParser::InclusiveRangePatternContext *ctx) = 0;
  virtual void
  exitInclusiveRangePattern(RustParser::InclusiveRangePatternContext *ctx) = 0;

  virtual void
  enterHalfOpenRangePattern(RustParser::HalfOpenRangePatternContext *ctx) = 0;
  virtual void
  exitHalfOpenRangePattern(RustParser::HalfOpenRangePatternContext *ctx) = 0;

  virtual void
  enterObsoleteRangePattern(RustParser::ObsoleteRangePatternContext *ctx) = 0;
  virtual void
  exitObsoleteRangePattern(RustParser::ObsoleteRangePatternContext *ctx) = 0;

  virtual void
  enterRangePatternBound(RustParser::RangePatternBoundContext *ctx) = 0;
  virtual void
  exitRangePatternBound(RustParser::RangePatternBoundContext *ctx) = 0;

  virtual void
  enterReferencePattern(RustParser::ReferencePatternContext *ctx) = 0;
  virtual void
  exitReferencePattern(RustParser::ReferencePatternContext *ctx) = 0;

  virtual void enterStructPattern(RustParser::StructPatternContext *ctx) = 0;
  virtual void exitStructPattern(RustParser::StructPatternContext *ctx) = 0;

  virtual void
  enterStructPatternElements(RustParser::StructPatternElementsContext *ctx) = 0;
  virtual void
  exitStructPatternElements(RustParser::StructPatternElementsContext *ctx) = 0;

  virtual void
  enterStructPatternFields(RustParser::StructPatternFieldsContext *ctx) = 0;
  virtual void
  exitStructPatternFields(RustParser::StructPatternFieldsContext *ctx) = 0;

  virtual void
  enterStructPatternField(RustParser::StructPatternFieldContext *ctx) = 0;
  virtual void
  exitStructPatternField(RustParser::StructPatternFieldContext *ctx) = 0;

  virtual void
  enterStructPatternEtCetera(RustParser::StructPatternEtCeteraContext *ctx) = 0;
  virtual void
  exitStructPatternEtCetera(RustParser::StructPatternEtCeteraContext *ctx) = 0;

  virtual void
  enterTupleStructPattern(RustParser::TupleStructPatternContext *ctx) = 0;
  virtual void
  exitTupleStructPattern(RustParser::TupleStructPatternContext *ctx) = 0;

  virtual void
  enterTupleStructItems(RustParser::TupleStructItemsContext *ctx) = 0;
  virtual void
  exitTupleStructItems(RustParser::TupleStructItemsContext *ctx) = 0;

  virtual void enterTuplePattern(RustParser::TuplePatternContext *ctx) = 0;
  virtual void exitTuplePattern(RustParser::TuplePatternContext *ctx) = 0;

  virtual void
  enterTuplePatternItems(RustParser::TuplePatternItemsContext *ctx) = 0;
  virtual void
  exitTuplePatternItems(RustParser::TuplePatternItemsContext *ctx) = 0;

  virtual void enterGroupedPattern(RustParser::GroupedPatternContext *ctx) = 0;
  virtual void exitGroupedPattern(RustParser::GroupedPatternContext *ctx) = 0;

  virtual void enterSlicePattern(RustParser::SlicePatternContext *ctx) = 0;
  virtual void exitSlicePattern(RustParser::SlicePatternContext *ctx) = 0;

  virtual void
  enterSlicePatternItems(RustParser::SlicePatternItemsContext *ctx) = 0;
  virtual void
  exitSlicePatternItems(RustParser::SlicePatternItemsContext *ctx) = 0;

  virtual void enterPathPattern(RustParser::PathPatternContext *ctx) = 0;
  virtual void exitPathPattern(RustParser::PathPatternContext *ctx) = 0;

  virtual void enterType_(RustParser::Type_Context *ctx) = 0;
  virtual void exitType_(RustParser::Type_Context *ctx) = 0;

  virtual void enterTypeNoBounds(RustParser::TypeNoBoundsContext *ctx) = 0;
  virtual void exitTypeNoBounds(RustParser::TypeNoBoundsContext *ctx) = 0;

  virtual void
  enterParenthesizedType(RustParser::ParenthesizedTypeContext *ctx) = 0;
  virtual void
  exitParenthesizedType(RustParser::ParenthesizedTypeContext *ctx) = 0;

  virtual void enterNeverType(RustParser::NeverTypeContext *ctx) = 0;
  virtual void exitNeverType(RustParser::NeverTypeContext *ctx) = 0;

  virtual void enterTupleType(RustParser::TupleTypeContext *ctx) = 0;
  virtual void exitTupleType(RustParser::TupleTypeContext *ctx) = 0;

  virtual void enterArrayType(RustParser::ArrayTypeContext *ctx) = 0;
  virtual void exitArrayType(RustParser::ArrayTypeContext *ctx) = 0;

  virtual void enterSliceType(RustParser::SliceTypeContext *ctx) = 0;
  virtual void exitSliceType(RustParser::SliceTypeContext *ctx) = 0;

  virtual void enterReferenceType(RustParser::ReferenceTypeContext *ctx) = 0;
  virtual void exitReferenceType(RustParser::ReferenceTypeContext *ctx) = 0;

  virtual void enterRawPointerType(RustParser::RawPointerTypeContext *ctx) = 0;
  virtual void exitRawPointerType(RustParser::RawPointerTypeContext *ctx) = 0;

  virtual void
  enterBareFunctionType(RustParser::BareFunctionTypeContext *ctx) = 0;
  virtual void
  exitBareFunctionType(RustParser::BareFunctionTypeContext *ctx) = 0;

  virtual void enterFunctionTypeQualifiers(
      RustParser::FunctionTypeQualifiersContext *ctx) = 0;
  virtual void exitFunctionTypeQualifiers(
      RustParser::FunctionTypeQualifiersContext *ctx) = 0;

  virtual void enterBareFunctionReturnType(
      RustParser::BareFunctionReturnTypeContext *ctx) = 0;
  virtual void exitBareFunctionReturnType(
      RustParser::BareFunctionReturnTypeContext *ctx) = 0;

  virtual void enterFunctionParametersMaybeNamedVariadic(
      RustParser::FunctionParametersMaybeNamedVariadicContext *ctx) = 0;
  virtual void exitFunctionParametersMaybeNamedVariadic(
      RustParser::FunctionParametersMaybeNamedVariadicContext *ctx) = 0;

  virtual void enterMaybeNamedFunctionParameters(
      RustParser::MaybeNamedFunctionParametersContext *ctx) = 0;
  virtual void exitMaybeNamedFunctionParameters(
      RustParser::MaybeNamedFunctionParametersContext *ctx) = 0;

  virtual void
  enterMaybeNamedParam(RustParser::MaybeNamedParamContext *ctx) = 0;
  virtual void exitMaybeNamedParam(RustParser::MaybeNamedParamContext *ctx) = 0;

  virtual void enterMaybeNamedFunctionParametersVariadic(
      RustParser::MaybeNamedFunctionParametersVariadicContext *ctx) = 0;
  virtual void exitMaybeNamedFunctionParametersVariadic(
      RustParser::MaybeNamedFunctionParametersVariadicContext *ctx) = 0;

  virtual void
  enterTraitObjectType(RustParser::TraitObjectTypeContext *ctx) = 0;
  virtual void exitTraitObjectType(RustParser::TraitObjectTypeContext *ctx) = 0;

  virtual void enterTraitObjectTypeOneBound(
      RustParser::TraitObjectTypeOneBoundContext *ctx) = 0;
  virtual void exitTraitObjectTypeOneBound(
      RustParser::TraitObjectTypeOneBoundContext *ctx) = 0;

  virtual void enterImplTraitType(RustParser::ImplTraitTypeContext *ctx) = 0;
  virtual void exitImplTraitType(RustParser::ImplTraitTypeContext *ctx) = 0;

  virtual void
  enterImplTraitTypeOneBound(RustParser::ImplTraitTypeOneBoundContext *ctx) = 0;
  virtual void
  exitImplTraitTypeOneBound(RustParser::ImplTraitTypeOneBoundContext *ctx) = 0;

  virtual void enterInferredType(RustParser::InferredTypeContext *ctx) = 0;
  virtual void exitInferredType(RustParser::InferredTypeContext *ctx) = 0;

  virtual void
  enterTypeParamBounds(RustParser::TypeParamBoundsContext *ctx) = 0;
  virtual void exitTypeParamBounds(RustParser::TypeParamBoundsContext *ctx) = 0;

  virtual void enterTypeParamBound(RustParser::TypeParamBoundContext *ctx) = 0;
  virtual void exitTypeParamBound(RustParser::TypeParamBoundContext *ctx) = 0;

  virtual void enterTraitBound(RustParser::TraitBoundContext *ctx) = 0;
  virtual void exitTraitBound(RustParser::TraitBoundContext *ctx) = 0;

  virtual void enterLifetimeBounds(RustParser::LifetimeBoundsContext *ctx) = 0;
  virtual void exitLifetimeBounds(RustParser::LifetimeBoundsContext *ctx) = 0;

  virtual void enterLifetime(RustParser::LifetimeContext *ctx) = 0;
  virtual void exitLifetime(RustParser::LifetimeContext *ctx) = 0;

  virtual void enterSimplePath(RustParser::SimplePathContext *ctx) = 0;
  virtual void exitSimplePath(RustParser::SimplePathContext *ctx) = 0;

  virtual void
  enterSimplePathSegment(RustParser::SimplePathSegmentContext *ctx) = 0;
  virtual void
  exitSimplePathSegment(RustParser::SimplePathSegmentContext *ctx) = 0;

  virtual void
  enterPathInExpression(RustParser::PathInExpressionContext *ctx) = 0;
  virtual void
  exitPathInExpression(RustParser::PathInExpressionContext *ctx) = 0;

  virtual void
  enterPathExprSegment(RustParser::PathExprSegmentContext *ctx) = 0;
  virtual void exitPathExprSegment(RustParser::PathExprSegmentContext *ctx) = 0;

  virtual void
  enterPathIdentSegment(RustParser::PathIdentSegmentContext *ctx) = 0;
  virtual void
  exitPathIdentSegment(RustParser::PathIdentSegmentContext *ctx) = 0;

  virtual void enterGenericArgs(RustParser::GenericArgsContext *ctx) = 0;
  virtual void exitGenericArgs(RustParser::GenericArgsContext *ctx) = 0;

  virtual void enterGenericArg(RustParser::GenericArgContext *ctx) = 0;
  virtual void exitGenericArg(RustParser::GenericArgContext *ctx) = 0;

  virtual void
  enterGenericArgsConst(RustParser::GenericArgsConstContext *ctx) = 0;
  virtual void
  exitGenericArgsConst(RustParser::GenericArgsConstContext *ctx) = 0;

  virtual void
  enterGenericArgsLifetimes(RustParser::GenericArgsLifetimesContext *ctx) = 0;
  virtual void
  exitGenericArgsLifetimes(RustParser::GenericArgsLifetimesContext *ctx) = 0;

  virtual void
  enterGenericArgsTypes(RustParser::GenericArgsTypesContext *ctx) = 0;
  virtual void
  exitGenericArgsTypes(RustParser::GenericArgsTypesContext *ctx) = 0;

  virtual void
  enterGenericArgsBindings(RustParser::GenericArgsBindingsContext *ctx) = 0;
  virtual void
  exitGenericArgsBindings(RustParser::GenericArgsBindingsContext *ctx) = 0;

  virtual void
  enterGenericArgsBinding(RustParser::GenericArgsBindingContext *ctx) = 0;
  virtual void
  exitGenericArgsBinding(RustParser::GenericArgsBindingContext *ctx) = 0;

  virtual void enterQualifiedPathInExpression(
      RustParser::QualifiedPathInExpressionContext *ctx) = 0;
  virtual void exitQualifiedPathInExpression(
      RustParser::QualifiedPathInExpressionContext *ctx) = 0;

  virtual void
  enterQualifiedPathType(RustParser::QualifiedPathTypeContext *ctx) = 0;
  virtual void
  exitQualifiedPathType(RustParser::QualifiedPathTypeContext *ctx) = 0;

  virtual void
  enterQualifiedPathInType(RustParser::QualifiedPathInTypeContext *ctx) = 0;
  virtual void
  exitQualifiedPathInType(RustParser::QualifiedPathInTypeContext *ctx) = 0;

  virtual void enterTypePath(RustParser::TypePathContext *ctx) = 0;
  virtual void exitTypePath(RustParser::TypePathContext *ctx) = 0;

  virtual void
  enterTypePathSegment(RustParser::TypePathSegmentContext *ctx) = 0;
  virtual void exitTypePathSegment(RustParser::TypePathSegmentContext *ctx) = 0;

  virtual void enterTypePathFn(RustParser::TypePathFnContext *ctx) = 0;
  virtual void exitTypePathFn(RustParser::TypePathFnContext *ctx) = 0;

  virtual void enterTypePathInputs(RustParser::TypePathInputsContext *ctx) = 0;
  virtual void exitTypePathInputs(RustParser::TypePathInputsContext *ctx) = 0;

  virtual void enterVisibility(RustParser::VisibilityContext *ctx) = 0;
  virtual void exitVisibility(RustParser::VisibilityContext *ctx) = 0;

  virtual void enterIdentifier(RustParser::IdentifierContext *ctx) = 0;
  virtual void exitIdentifier(RustParser::IdentifierContext *ctx) = 0;

  virtual void enterKeyword(RustParser::KeywordContext *ctx) = 0;
  virtual void exitKeyword(RustParser::KeywordContext *ctx) = 0;

  virtual void enterMacroIdentifierLikeToken(
      RustParser::MacroIdentifierLikeTokenContext *ctx) = 0;
  virtual void exitMacroIdentifierLikeToken(
      RustParser::MacroIdentifierLikeTokenContext *ctx) = 0;

  virtual void
  enterMacroLiteralToken(RustParser::MacroLiteralTokenContext *ctx) = 0;
  virtual void
  exitMacroLiteralToken(RustParser::MacroLiteralTokenContext *ctx) = 0;

  virtual void
  enterMacroPunctuationToken(RustParser::MacroPunctuationTokenContext *ctx) = 0;
  virtual void
  exitMacroPunctuationToken(RustParser::MacroPunctuationTokenContext *ctx) = 0;

  virtual void enterShl(RustParser::ShlContext *ctx) = 0;
  virtual void exitShl(RustParser::ShlContext *ctx) = 0;

  virtual void enterShr(RustParser::ShrContext *ctx) = 0;
  virtual void exitShr(RustParser::ShrContext *ctx) = 0;
};
