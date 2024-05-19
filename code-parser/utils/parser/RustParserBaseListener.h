
// Generated from RustParser.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "RustParserListener.h"


/**
 * This class provides an empty implementation of RustParserListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  RustParserBaseListener : public RustParserListener {
public:

  virtual void enterCrate(RustParser::CrateContext * /*ctx*/) override { }
  virtual void exitCrate(RustParser::CrateContext * /*ctx*/) override { }

  virtual void enterMacroInvocation(RustParser::MacroInvocationContext * /*ctx*/) override { }
  virtual void exitMacroInvocation(RustParser::MacroInvocationContext * /*ctx*/) override { }

  virtual void enterDelimTokenTree(RustParser::DelimTokenTreeContext * /*ctx*/) override { }
  virtual void exitDelimTokenTree(RustParser::DelimTokenTreeContext * /*ctx*/) override { }

  virtual void enterTokenTree(RustParser::TokenTreeContext * /*ctx*/) override { }
  virtual void exitTokenTree(RustParser::TokenTreeContext * /*ctx*/) override { }

  virtual void enterTokenTreeToken(RustParser::TokenTreeTokenContext * /*ctx*/) override { }
  virtual void exitTokenTreeToken(RustParser::TokenTreeTokenContext * /*ctx*/) override { }

  virtual void enterMacroInvocationSemi(RustParser::MacroInvocationSemiContext * /*ctx*/) override { }
  virtual void exitMacroInvocationSemi(RustParser::MacroInvocationSemiContext * /*ctx*/) override { }

  virtual void enterMacroRulesDefinition(RustParser::MacroRulesDefinitionContext * /*ctx*/) override { }
  virtual void exitMacroRulesDefinition(RustParser::MacroRulesDefinitionContext * /*ctx*/) override { }

  virtual void enterMacroRulesDef(RustParser::MacroRulesDefContext * /*ctx*/) override { }
  virtual void exitMacroRulesDef(RustParser::MacroRulesDefContext * /*ctx*/) override { }

  virtual void enterMacroRules(RustParser::MacroRulesContext * /*ctx*/) override { }
  virtual void exitMacroRules(RustParser::MacroRulesContext * /*ctx*/) override { }

  virtual void enterMacroRule(RustParser::MacroRuleContext * /*ctx*/) override { }
  virtual void exitMacroRule(RustParser::MacroRuleContext * /*ctx*/) override { }

  virtual void enterMacroMatcher(RustParser::MacroMatcherContext * /*ctx*/) override { }
  virtual void exitMacroMatcher(RustParser::MacroMatcherContext * /*ctx*/) override { }

  virtual void enterMacroMatch(RustParser::MacroMatchContext * /*ctx*/) override { }
  virtual void exitMacroMatch(RustParser::MacroMatchContext * /*ctx*/) override { }

  virtual void enterMacroMatchToken(RustParser::MacroMatchTokenContext * /*ctx*/) override { }
  virtual void exitMacroMatchToken(RustParser::MacroMatchTokenContext * /*ctx*/) override { }

  virtual void enterMacroFragSpec(RustParser::MacroFragSpecContext * /*ctx*/) override { }
  virtual void exitMacroFragSpec(RustParser::MacroFragSpecContext * /*ctx*/) override { }

  virtual void enterMacroRepSep(RustParser::MacroRepSepContext * /*ctx*/) override { }
  virtual void exitMacroRepSep(RustParser::MacroRepSepContext * /*ctx*/) override { }

  virtual void enterMacroRepOp(RustParser::MacroRepOpContext * /*ctx*/) override { }
  virtual void exitMacroRepOp(RustParser::MacroRepOpContext * /*ctx*/) override { }

  virtual void enterMacroTranscriber(RustParser::MacroTranscriberContext * /*ctx*/) override { }
  virtual void exitMacroTranscriber(RustParser::MacroTranscriberContext * /*ctx*/) override { }

  virtual void enterItem(RustParser::ItemContext * /*ctx*/) override { }
  virtual void exitItem(RustParser::ItemContext * /*ctx*/) override { }

  virtual void enterVisItem(RustParser::VisItemContext * /*ctx*/) override { }
  virtual void exitVisItem(RustParser::VisItemContext * /*ctx*/) override { }

  virtual void enterMacroItem(RustParser::MacroItemContext * /*ctx*/) override { }
  virtual void exitMacroItem(RustParser::MacroItemContext * /*ctx*/) override { }

  virtual void enterModule(RustParser::ModuleContext * /*ctx*/) override { }
  virtual void exitModule(RustParser::ModuleContext * /*ctx*/) override { }

  virtual void enterExternCrate(RustParser::ExternCrateContext * /*ctx*/) override { }
  virtual void exitExternCrate(RustParser::ExternCrateContext * /*ctx*/) override { }

  virtual void enterCrateRef(RustParser::CrateRefContext * /*ctx*/) override { }
  virtual void exitCrateRef(RustParser::CrateRefContext * /*ctx*/) override { }

  virtual void enterAsClause(RustParser::AsClauseContext * /*ctx*/) override { }
  virtual void exitAsClause(RustParser::AsClauseContext * /*ctx*/) override { }

  virtual void enterUseDeclaration(RustParser::UseDeclarationContext * /*ctx*/) override { }
  virtual void exitUseDeclaration(RustParser::UseDeclarationContext * /*ctx*/) override { }

  virtual void enterUseTree(RustParser::UseTreeContext * /*ctx*/) override { }
  virtual void exitUseTree(RustParser::UseTreeContext * /*ctx*/) override { }

  virtual void enterFunction_(RustParser::Function_Context * /*ctx*/) override { }
  virtual void exitFunction_(RustParser::Function_Context * /*ctx*/) override { }

  virtual void enterFunctionQualifiers(RustParser::FunctionQualifiersContext * /*ctx*/) override { }
  virtual void exitFunctionQualifiers(RustParser::FunctionQualifiersContext * /*ctx*/) override { }

  virtual void enterAbi(RustParser::AbiContext * /*ctx*/) override { }
  virtual void exitAbi(RustParser::AbiContext * /*ctx*/) override { }

  virtual void enterFunctionParameters(RustParser::FunctionParametersContext * /*ctx*/) override { }
  virtual void exitFunctionParameters(RustParser::FunctionParametersContext * /*ctx*/) override { }

  virtual void enterSelfParam(RustParser::SelfParamContext * /*ctx*/) override { }
  virtual void exitSelfParam(RustParser::SelfParamContext * /*ctx*/) override { }

  virtual void enterShorthandSelf(RustParser::ShorthandSelfContext * /*ctx*/) override { }
  virtual void exitShorthandSelf(RustParser::ShorthandSelfContext * /*ctx*/) override { }

  virtual void enterTypedSelf(RustParser::TypedSelfContext * /*ctx*/) override { }
  virtual void exitTypedSelf(RustParser::TypedSelfContext * /*ctx*/) override { }

  virtual void enterFunctionParam(RustParser::FunctionParamContext * /*ctx*/) override { }
  virtual void exitFunctionParam(RustParser::FunctionParamContext * /*ctx*/) override { }

  virtual void enterFunctionParamPattern(RustParser::FunctionParamPatternContext * /*ctx*/) override { }
  virtual void exitFunctionParamPattern(RustParser::FunctionParamPatternContext * /*ctx*/) override { }

  virtual void enterFunctionReturnType(RustParser::FunctionReturnTypeContext * /*ctx*/) override { }
  virtual void exitFunctionReturnType(RustParser::FunctionReturnTypeContext * /*ctx*/) override { }

  virtual void enterTypeAlias(RustParser::TypeAliasContext * /*ctx*/) override { }
  virtual void exitTypeAlias(RustParser::TypeAliasContext * /*ctx*/) override { }

  virtual void enterStruct_(RustParser::Struct_Context * /*ctx*/) override { }
  virtual void exitStruct_(RustParser::Struct_Context * /*ctx*/) override { }

  virtual void enterStructStruct(RustParser::StructStructContext * /*ctx*/) override { }
  virtual void exitStructStruct(RustParser::StructStructContext * /*ctx*/) override { }

  virtual void enterTupleStruct(RustParser::TupleStructContext * /*ctx*/) override { }
  virtual void exitTupleStruct(RustParser::TupleStructContext * /*ctx*/) override { }

  virtual void enterStructFields(RustParser::StructFieldsContext * /*ctx*/) override { }
  virtual void exitStructFields(RustParser::StructFieldsContext * /*ctx*/) override { }

  virtual void enterStructField(RustParser::StructFieldContext * /*ctx*/) override { }
  virtual void exitStructField(RustParser::StructFieldContext * /*ctx*/) override { }

  virtual void enterTupleFields(RustParser::TupleFieldsContext * /*ctx*/) override { }
  virtual void exitTupleFields(RustParser::TupleFieldsContext * /*ctx*/) override { }

  virtual void enterTupleField(RustParser::TupleFieldContext * /*ctx*/) override { }
  virtual void exitTupleField(RustParser::TupleFieldContext * /*ctx*/) override { }

  virtual void enterEnumeration(RustParser::EnumerationContext * /*ctx*/) override { }
  virtual void exitEnumeration(RustParser::EnumerationContext * /*ctx*/) override { }

  virtual void enterEnumItems(RustParser::EnumItemsContext * /*ctx*/) override { }
  virtual void exitEnumItems(RustParser::EnumItemsContext * /*ctx*/) override { }

  virtual void enterEnumItem(RustParser::EnumItemContext * /*ctx*/) override { }
  virtual void exitEnumItem(RustParser::EnumItemContext * /*ctx*/) override { }

  virtual void enterEnumItemTuple(RustParser::EnumItemTupleContext * /*ctx*/) override { }
  virtual void exitEnumItemTuple(RustParser::EnumItemTupleContext * /*ctx*/) override { }

  virtual void enterEnumItemStruct(RustParser::EnumItemStructContext * /*ctx*/) override { }
  virtual void exitEnumItemStruct(RustParser::EnumItemStructContext * /*ctx*/) override { }

  virtual void enterEnumItemDiscriminant(RustParser::EnumItemDiscriminantContext * /*ctx*/) override { }
  virtual void exitEnumItemDiscriminant(RustParser::EnumItemDiscriminantContext * /*ctx*/) override { }

  virtual void enterUnion_(RustParser::Union_Context * /*ctx*/) override { }
  virtual void exitUnion_(RustParser::Union_Context * /*ctx*/) override { }

  virtual void enterConstantItem(RustParser::ConstantItemContext * /*ctx*/) override { }
  virtual void exitConstantItem(RustParser::ConstantItemContext * /*ctx*/) override { }

  virtual void enterStaticItem(RustParser::StaticItemContext * /*ctx*/) override { }
  virtual void exitStaticItem(RustParser::StaticItemContext * /*ctx*/) override { }

  virtual void enterTrait_(RustParser::Trait_Context * /*ctx*/) override { }
  virtual void exitTrait_(RustParser::Trait_Context * /*ctx*/) override { }

  virtual void enterImplementation(RustParser::ImplementationContext * /*ctx*/) override { }
  virtual void exitImplementation(RustParser::ImplementationContext * /*ctx*/) override { }

  virtual void enterInherentImpl(RustParser::InherentImplContext * /*ctx*/) override { }
  virtual void exitInherentImpl(RustParser::InherentImplContext * /*ctx*/) override { }

  virtual void enterTraitImpl(RustParser::TraitImplContext * /*ctx*/) override { }
  virtual void exitTraitImpl(RustParser::TraitImplContext * /*ctx*/) override { }

  virtual void enterExternBlock(RustParser::ExternBlockContext * /*ctx*/) override { }
  virtual void exitExternBlock(RustParser::ExternBlockContext * /*ctx*/) override { }

  virtual void enterExternalItem(RustParser::ExternalItemContext * /*ctx*/) override { }
  virtual void exitExternalItem(RustParser::ExternalItemContext * /*ctx*/) override { }

  virtual void enterGenericParams(RustParser::GenericParamsContext * /*ctx*/) override { }
  virtual void exitGenericParams(RustParser::GenericParamsContext * /*ctx*/) override { }

  virtual void enterGenericParam(RustParser::GenericParamContext * /*ctx*/) override { }
  virtual void exitGenericParam(RustParser::GenericParamContext * /*ctx*/) override { }

  virtual void enterLifetimeParam(RustParser::LifetimeParamContext * /*ctx*/) override { }
  virtual void exitLifetimeParam(RustParser::LifetimeParamContext * /*ctx*/) override { }

  virtual void enterTypeParam(RustParser::TypeParamContext * /*ctx*/) override { }
  virtual void exitTypeParam(RustParser::TypeParamContext * /*ctx*/) override { }

  virtual void enterConstParam(RustParser::ConstParamContext * /*ctx*/) override { }
  virtual void exitConstParam(RustParser::ConstParamContext * /*ctx*/) override { }

  virtual void enterWhereClause(RustParser::WhereClauseContext * /*ctx*/) override { }
  virtual void exitWhereClause(RustParser::WhereClauseContext * /*ctx*/) override { }

  virtual void enterWhereClauseItem(RustParser::WhereClauseItemContext * /*ctx*/) override { }
  virtual void exitWhereClauseItem(RustParser::WhereClauseItemContext * /*ctx*/) override { }

  virtual void enterLifetimeWhereClauseItem(RustParser::LifetimeWhereClauseItemContext * /*ctx*/) override { }
  virtual void exitLifetimeWhereClauseItem(RustParser::LifetimeWhereClauseItemContext * /*ctx*/) override { }

  virtual void enterTypeBoundWhereClauseItem(RustParser::TypeBoundWhereClauseItemContext * /*ctx*/) override { }
  virtual void exitTypeBoundWhereClauseItem(RustParser::TypeBoundWhereClauseItemContext * /*ctx*/) override { }

  virtual void enterForLifetimes(RustParser::ForLifetimesContext * /*ctx*/) override { }
  virtual void exitForLifetimes(RustParser::ForLifetimesContext * /*ctx*/) override { }

  virtual void enterAssociatedItem(RustParser::AssociatedItemContext * /*ctx*/) override { }
  virtual void exitAssociatedItem(RustParser::AssociatedItemContext * /*ctx*/) override { }

  virtual void enterInnerAttribute(RustParser::InnerAttributeContext * /*ctx*/) override { }
  virtual void exitInnerAttribute(RustParser::InnerAttributeContext * /*ctx*/) override { }

  virtual void enterOuterAttribute(RustParser::OuterAttributeContext * /*ctx*/) override { }
  virtual void exitOuterAttribute(RustParser::OuterAttributeContext * /*ctx*/) override { }

  virtual void enterAttr(RustParser::AttrContext * /*ctx*/) override { }
  virtual void exitAttr(RustParser::AttrContext * /*ctx*/) override { }

  virtual void enterAttrInput(RustParser::AttrInputContext * /*ctx*/) override { }
  virtual void exitAttrInput(RustParser::AttrInputContext * /*ctx*/) override { }

  virtual void enterStatement(RustParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(RustParser::StatementContext * /*ctx*/) override { }

  virtual void enterLetStatement(RustParser::LetStatementContext * /*ctx*/) override { }
  virtual void exitLetStatement(RustParser::LetStatementContext * /*ctx*/) override { }

  virtual void enterExpressionStatement(RustParser::ExpressionStatementContext * /*ctx*/) override { }
  virtual void exitExpressionStatement(RustParser::ExpressionStatementContext * /*ctx*/) override { }

  virtual void enterTypeCastExpression(RustParser::TypeCastExpressionContext * /*ctx*/) override { }
  virtual void exitTypeCastExpression(RustParser::TypeCastExpressionContext * /*ctx*/) override { }

  virtual void enterPathExpression_(RustParser::PathExpression_Context * /*ctx*/) override { }
  virtual void exitPathExpression_(RustParser::PathExpression_Context * /*ctx*/) override { }

  virtual void enterTupleExpression(RustParser::TupleExpressionContext * /*ctx*/) override { }
  virtual void exitTupleExpression(RustParser::TupleExpressionContext * /*ctx*/) override { }

  virtual void enterIndexExpression(RustParser::IndexExpressionContext * /*ctx*/) override { }
  virtual void exitIndexExpression(RustParser::IndexExpressionContext * /*ctx*/) override { }

  virtual void enterRangeExpression(RustParser::RangeExpressionContext * /*ctx*/) override { }
  virtual void exitRangeExpression(RustParser::RangeExpressionContext * /*ctx*/) override { }

  virtual void enterMacroInvocationAsExpression(RustParser::MacroInvocationAsExpressionContext * /*ctx*/) override { }
  virtual void exitMacroInvocationAsExpression(RustParser::MacroInvocationAsExpressionContext * /*ctx*/) override { }

  virtual void enterReturnExpression(RustParser::ReturnExpressionContext * /*ctx*/) override { }
  virtual void exitReturnExpression(RustParser::ReturnExpressionContext * /*ctx*/) override { }

  virtual void enterAwaitExpression(RustParser::AwaitExpressionContext * /*ctx*/) override { }
  virtual void exitAwaitExpression(RustParser::AwaitExpressionContext * /*ctx*/) override { }

  virtual void enterErrorPropagationExpression(RustParser::ErrorPropagationExpressionContext * /*ctx*/) override { }
  virtual void exitErrorPropagationExpression(RustParser::ErrorPropagationExpressionContext * /*ctx*/) override { }

  virtual void enterContinueExpression(RustParser::ContinueExpressionContext * /*ctx*/) override { }
  virtual void exitContinueExpression(RustParser::ContinueExpressionContext * /*ctx*/) override { }

  virtual void enterAssignmentExpression(RustParser::AssignmentExpressionContext * /*ctx*/) override { }
  virtual void exitAssignmentExpression(RustParser::AssignmentExpressionContext * /*ctx*/) override { }

  virtual void enterMethodCallExpression(RustParser::MethodCallExpressionContext * /*ctx*/) override { }
  virtual void exitMethodCallExpression(RustParser::MethodCallExpressionContext * /*ctx*/) override { }

  virtual void enterLiteralExpression_(RustParser::LiteralExpression_Context * /*ctx*/) override { }
  virtual void exitLiteralExpression_(RustParser::LiteralExpression_Context * /*ctx*/) override { }

  virtual void enterStructExpression_(RustParser::StructExpression_Context * /*ctx*/) override { }
  virtual void exitStructExpression_(RustParser::StructExpression_Context * /*ctx*/) override { }

  virtual void enterTupleIndexingExpression(RustParser::TupleIndexingExpressionContext * /*ctx*/) override { }
  virtual void exitTupleIndexingExpression(RustParser::TupleIndexingExpressionContext * /*ctx*/) override { }

  virtual void enterNegationExpression(RustParser::NegationExpressionContext * /*ctx*/) override { }
  virtual void exitNegationExpression(RustParser::NegationExpressionContext * /*ctx*/) override { }

  virtual void enterCallExpression(RustParser::CallExpressionContext * /*ctx*/) override { }
  virtual void exitCallExpression(RustParser::CallExpressionContext * /*ctx*/) override { }

  virtual void enterLazyBooleanExpression(RustParser::LazyBooleanExpressionContext * /*ctx*/) override { }
  virtual void exitLazyBooleanExpression(RustParser::LazyBooleanExpressionContext * /*ctx*/) override { }

  virtual void enterDereferenceExpression(RustParser::DereferenceExpressionContext * /*ctx*/) override { }
  virtual void exitDereferenceExpression(RustParser::DereferenceExpressionContext * /*ctx*/) override { }

  virtual void enterExpressionWithBlock_(RustParser::ExpressionWithBlock_Context * /*ctx*/) override { }
  virtual void exitExpressionWithBlock_(RustParser::ExpressionWithBlock_Context * /*ctx*/) override { }

  virtual void enterGroupedExpression(RustParser::GroupedExpressionContext * /*ctx*/) override { }
  virtual void exitGroupedExpression(RustParser::GroupedExpressionContext * /*ctx*/) override { }

  virtual void enterBreakExpression(RustParser::BreakExpressionContext * /*ctx*/) override { }
  virtual void exitBreakExpression(RustParser::BreakExpressionContext * /*ctx*/) override { }

  virtual void enterArithmeticOrLogicalExpression(RustParser::ArithmeticOrLogicalExpressionContext * /*ctx*/) override { }
  virtual void exitArithmeticOrLogicalExpression(RustParser::ArithmeticOrLogicalExpressionContext * /*ctx*/) override { }

  virtual void enterFieldExpression(RustParser::FieldExpressionContext * /*ctx*/) override { }
  virtual void exitFieldExpression(RustParser::FieldExpressionContext * /*ctx*/) override { }

  virtual void enterEnumerationVariantExpression_(RustParser::EnumerationVariantExpression_Context * /*ctx*/) override { }
  virtual void exitEnumerationVariantExpression_(RustParser::EnumerationVariantExpression_Context * /*ctx*/) override { }

  virtual void enterComparisonExpression(RustParser::ComparisonExpressionContext * /*ctx*/) override { }
  virtual void exitComparisonExpression(RustParser::ComparisonExpressionContext * /*ctx*/) override { }

  virtual void enterAttributedExpression(RustParser::AttributedExpressionContext * /*ctx*/) override { }
  virtual void exitAttributedExpression(RustParser::AttributedExpressionContext * /*ctx*/) override { }

  virtual void enterBorrowExpression(RustParser::BorrowExpressionContext * /*ctx*/) override { }
  virtual void exitBorrowExpression(RustParser::BorrowExpressionContext * /*ctx*/) override { }

  virtual void enterCompoundAssignmentExpression(RustParser::CompoundAssignmentExpressionContext * /*ctx*/) override { }
  virtual void exitCompoundAssignmentExpression(RustParser::CompoundAssignmentExpressionContext * /*ctx*/) override { }

  virtual void enterClosureExpression_(RustParser::ClosureExpression_Context * /*ctx*/) override { }
  virtual void exitClosureExpression_(RustParser::ClosureExpression_Context * /*ctx*/) override { }

  virtual void enterArrayExpression(RustParser::ArrayExpressionContext * /*ctx*/) override { }
  virtual void exitArrayExpression(RustParser::ArrayExpressionContext * /*ctx*/) override { }

  virtual void enterComparisonOperator(RustParser::ComparisonOperatorContext * /*ctx*/) override { }
  virtual void exitComparisonOperator(RustParser::ComparisonOperatorContext * /*ctx*/) override { }

  virtual void enterCompoundAssignOperator(RustParser::CompoundAssignOperatorContext * /*ctx*/) override { }
  virtual void exitCompoundAssignOperator(RustParser::CompoundAssignOperatorContext * /*ctx*/) override { }

  virtual void enterExpressionWithBlock(RustParser::ExpressionWithBlockContext * /*ctx*/) override { }
  virtual void exitExpressionWithBlock(RustParser::ExpressionWithBlockContext * /*ctx*/) override { }

  virtual void enterLiteralExpression(RustParser::LiteralExpressionContext * /*ctx*/) override { }
  virtual void exitLiteralExpression(RustParser::LiteralExpressionContext * /*ctx*/) override { }

  virtual void enterPathExpression(RustParser::PathExpressionContext * /*ctx*/) override { }
  virtual void exitPathExpression(RustParser::PathExpressionContext * /*ctx*/) override { }

  virtual void enterBlockExpression(RustParser::BlockExpressionContext * /*ctx*/) override { }
  virtual void exitBlockExpression(RustParser::BlockExpressionContext * /*ctx*/) override { }

  virtual void enterStatements(RustParser::StatementsContext * /*ctx*/) override { }
  virtual void exitStatements(RustParser::StatementsContext * /*ctx*/) override { }

  virtual void enterAsyncBlockExpression(RustParser::AsyncBlockExpressionContext * /*ctx*/) override { }
  virtual void exitAsyncBlockExpression(RustParser::AsyncBlockExpressionContext * /*ctx*/) override { }

  virtual void enterUnsafeBlockExpression(RustParser::UnsafeBlockExpressionContext * /*ctx*/) override { }
  virtual void exitUnsafeBlockExpression(RustParser::UnsafeBlockExpressionContext * /*ctx*/) override { }

  virtual void enterArrayElements(RustParser::ArrayElementsContext * /*ctx*/) override { }
  virtual void exitArrayElements(RustParser::ArrayElementsContext * /*ctx*/) override { }

  virtual void enterTupleElements(RustParser::TupleElementsContext * /*ctx*/) override { }
  virtual void exitTupleElements(RustParser::TupleElementsContext * /*ctx*/) override { }

  virtual void enterTupleIndex(RustParser::TupleIndexContext * /*ctx*/) override { }
  virtual void exitTupleIndex(RustParser::TupleIndexContext * /*ctx*/) override { }

  virtual void enterStructExpression(RustParser::StructExpressionContext * /*ctx*/) override { }
  virtual void exitStructExpression(RustParser::StructExpressionContext * /*ctx*/) override { }

  virtual void enterStructExprStruct(RustParser::StructExprStructContext * /*ctx*/) override { }
  virtual void exitStructExprStruct(RustParser::StructExprStructContext * /*ctx*/) override { }

  virtual void enterStructExprFields(RustParser::StructExprFieldsContext * /*ctx*/) override { }
  virtual void exitStructExprFields(RustParser::StructExprFieldsContext * /*ctx*/) override { }

  virtual void enterStructExprField(RustParser::StructExprFieldContext * /*ctx*/) override { }
  virtual void exitStructExprField(RustParser::StructExprFieldContext * /*ctx*/) override { }

  virtual void enterStructBase(RustParser::StructBaseContext * /*ctx*/) override { }
  virtual void exitStructBase(RustParser::StructBaseContext * /*ctx*/) override { }

  virtual void enterStructExprTuple(RustParser::StructExprTupleContext * /*ctx*/) override { }
  virtual void exitStructExprTuple(RustParser::StructExprTupleContext * /*ctx*/) override { }

  virtual void enterStructExprUnit(RustParser::StructExprUnitContext * /*ctx*/) override { }
  virtual void exitStructExprUnit(RustParser::StructExprUnitContext * /*ctx*/) override { }

  virtual void enterEnumerationVariantExpression(RustParser::EnumerationVariantExpressionContext * /*ctx*/) override { }
  virtual void exitEnumerationVariantExpression(RustParser::EnumerationVariantExpressionContext * /*ctx*/) override { }

  virtual void enterEnumExprStruct(RustParser::EnumExprStructContext * /*ctx*/) override { }
  virtual void exitEnumExprStruct(RustParser::EnumExprStructContext * /*ctx*/) override { }

  virtual void enterEnumExprFields(RustParser::EnumExprFieldsContext * /*ctx*/) override { }
  virtual void exitEnumExprFields(RustParser::EnumExprFieldsContext * /*ctx*/) override { }

  virtual void enterEnumExprField(RustParser::EnumExprFieldContext * /*ctx*/) override { }
  virtual void exitEnumExprField(RustParser::EnumExprFieldContext * /*ctx*/) override { }

  virtual void enterEnumExprTuple(RustParser::EnumExprTupleContext * /*ctx*/) override { }
  virtual void exitEnumExprTuple(RustParser::EnumExprTupleContext * /*ctx*/) override { }

  virtual void enterEnumExprFieldless(RustParser::EnumExprFieldlessContext * /*ctx*/) override { }
  virtual void exitEnumExprFieldless(RustParser::EnumExprFieldlessContext * /*ctx*/) override { }

  virtual void enterCallParams(RustParser::CallParamsContext * /*ctx*/) override { }
  virtual void exitCallParams(RustParser::CallParamsContext * /*ctx*/) override { }

  virtual void enterClosureExpression(RustParser::ClosureExpressionContext * /*ctx*/) override { }
  virtual void exitClosureExpression(RustParser::ClosureExpressionContext * /*ctx*/) override { }

  virtual void enterClosureParameters(RustParser::ClosureParametersContext * /*ctx*/) override { }
  virtual void exitClosureParameters(RustParser::ClosureParametersContext * /*ctx*/) override { }

  virtual void enterClosureParam(RustParser::ClosureParamContext * /*ctx*/) override { }
  virtual void exitClosureParam(RustParser::ClosureParamContext * /*ctx*/) override { }

  virtual void enterLoopExpression(RustParser::LoopExpressionContext * /*ctx*/) override { }
  virtual void exitLoopExpression(RustParser::LoopExpressionContext * /*ctx*/) override { }

  virtual void enterInfiniteLoopExpression(RustParser::InfiniteLoopExpressionContext * /*ctx*/) override { }
  virtual void exitInfiniteLoopExpression(RustParser::InfiniteLoopExpressionContext * /*ctx*/) override { }

  virtual void enterPredicateLoopExpression(RustParser::PredicateLoopExpressionContext * /*ctx*/) override { }
  virtual void exitPredicateLoopExpression(RustParser::PredicateLoopExpressionContext * /*ctx*/) override { }

  virtual void enterPredicatePatternLoopExpression(RustParser::PredicatePatternLoopExpressionContext * /*ctx*/) override { }
  virtual void exitPredicatePatternLoopExpression(RustParser::PredicatePatternLoopExpressionContext * /*ctx*/) override { }

  virtual void enterIteratorLoopExpression(RustParser::IteratorLoopExpressionContext * /*ctx*/) override { }
  virtual void exitIteratorLoopExpression(RustParser::IteratorLoopExpressionContext * /*ctx*/) override { }

  virtual void enterLoopLabel(RustParser::LoopLabelContext * /*ctx*/) override { }
  virtual void exitLoopLabel(RustParser::LoopLabelContext * /*ctx*/) override { }

  virtual void enterIfExpression(RustParser::IfExpressionContext * /*ctx*/) override { }
  virtual void exitIfExpression(RustParser::IfExpressionContext * /*ctx*/) override { }

  virtual void enterIfLetExpression(RustParser::IfLetExpressionContext * /*ctx*/) override { }
  virtual void exitIfLetExpression(RustParser::IfLetExpressionContext * /*ctx*/) override { }

  virtual void enterMatchExpression(RustParser::MatchExpressionContext * /*ctx*/) override { }
  virtual void exitMatchExpression(RustParser::MatchExpressionContext * /*ctx*/) override { }

  virtual void enterMatchArms(RustParser::MatchArmsContext * /*ctx*/) override { }
  virtual void exitMatchArms(RustParser::MatchArmsContext * /*ctx*/) override { }

  virtual void enterMatchArmExpression(RustParser::MatchArmExpressionContext * /*ctx*/) override { }
  virtual void exitMatchArmExpression(RustParser::MatchArmExpressionContext * /*ctx*/) override { }

  virtual void enterMatchArm(RustParser::MatchArmContext * /*ctx*/) override { }
  virtual void exitMatchArm(RustParser::MatchArmContext * /*ctx*/) override { }

  virtual void enterMatchArmGuard(RustParser::MatchArmGuardContext * /*ctx*/) override { }
  virtual void exitMatchArmGuard(RustParser::MatchArmGuardContext * /*ctx*/) override { }

  virtual void enterPattern(RustParser::PatternContext * /*ctx*/) override { }
  virtual void exitPattern(RustParser::PatternContext * /*ctx*/) override { }

  virtual void enterPatternNoTopAlt(RustParser::PatternNoTopAltContext * /*ctx*/) override { }
  virtual void exitPatternNoTopAlt(RustParser::PatternNoTopAltContext * /*ctx*/) override { }

  virtual void enterPatternWithoutRange(RustParser::PatternWithoutRangeContext * /*ctx*/) override { }
  virtual void exitPatternWithoutRange(RustParser::PatternWithoutRangeContext * /*ctx*/) override { }

  virtual void enterLiteralPattern(RustParser::LiteralPatternContext * /*ctx*/) override { }
  virtual void exitLiteralPattern(RustParser::LiteralPatternContext * /*ctx*/) override { }

  virtual void enterIdentifierPattern(RustParser::IdentifierPatternContext * /*ctx*/) override { }
  virtual void exitIdentifierPattern(RustParser::IdentifierPatternContext * /*ctx*/) override { }

  virtual void enterWildcardPattern(RustParser::WildcardPatternContext * /*ctx*/) override { }
  virtual void exitWildcardPattern(RustParser::WildcardPatternContext * /*ctx*/) override { }

  virtual void enterRestPattern(RustParser::RestPatternContext * /*ctx*/) override { }
  virtual void exitRestPattern(RustParser::RestPatternContext * /*ctx*/) override { }

  virtual void enterInclusiveRangePattern(RustParser::InclusiveRangePatternContext * /*ctx*/) override { }
  virtual void exitInclusiveRangePattern(RustParser::InclusiveRangePatternContext * /*ctx*/) override { }

  virtual void enterHalfOpenRangePattern(RustParser::HalfOpenRangePatternContext * /*ctx*/) override { }
  virtual void exitHalfOpenRangePattern(RustParser::HalfOpenRangePatternContext * /*ctx*/) override { }

  virtual void enterObsoleteRangePattern(RustParser::ObsoleteRangePatternContext * /*ctx*/) override { }
  virtual void exitObsoleteRangePattern(RustParser::ObsoleteRangePatternContext * /*ctx*/) override { }

  virtual void enterRangePatternBound(RustParser::RangePatternBoundContext * /*ctx*/) override { }
  virtual void exitRangePatternBound(RustParser::RangePatternBoundContext * /*ctx*/) override { }

  virtual void enterReferencePattern(RustParser::ReferencePatternContext * /*ctx*/) override { }
  virtual void exitReferencePattern(RustParser::ReferencePatternContext * /*ctx*/) override { }

  virtual void enterStructPattern(RustParser::StructPatternContext * /*ctx*/) override { }
  virtual void exitStructPattern(RustParser::StructPatternContext * /*ctx*/) override { }

  virtual void enterStructPatternElements(RustParser::StructPatternElementsContext * /*ctx*/) override { }
  virtual void exitStructPatternElements(RustParser::StructPatternElementsContext * /*ctx*/) override { }

  virtual void enterStructPatternFields(RustParser::StructPatternFieldsContext * /*ctx*/) override { }
  virtual void exitStructPatternFields(RustParser::StructPatternFieldsContext * /*ctx*/) override { }

  virtual void enterStructPatternField(RustParser::StructPatternFieldContext * /*ctx*/) override { }
  virtual void exitStructPatternField(RustParser::StructPatternFieldContext * /*ctx*/) override { }

  virtual void enterStructPatternEtCetera(RustParser::StructPatternEtCeteraContext * /*ctx*/) override { }
  virtual void exitStructPatternEtCetera(RustParser::StructPatternEtCeteraContext * /*ctx*/) override { }

  virtual void enterTupleStructPattern(RustParser::TupleStructPatternContext * /*ctx*/) override { }
  virtual void exitTupleStructPattern(RustParser::TupleStructPatternContext * /*ctx*/) override { }

  virtual void enterTupleStructItems(RustParser::TupleStructItemsContext * /*ctx*/) override { }
  virtual void exitTupleStructItems(RustParser::TupleStructItemsContext * /*ctx*/) override { }

  virtual void enterTuplePattern(RustParser::TuplePatternContext * /*ctx*/) override { }
  virtual void exitTuplePattern(RustParser::TuplePatternContext * /*ctx*/) override { }

  virtual void enterTuplePatternItems(RustParser::TuplePatternItemsContext * /*ctx*/) override { }
  virtual void exitTuplePatternItems(RustParser::TuplePatternItemsContext * /*ctx*/) override { }

  virtual void enterGroupedPattern(RustParser::GroupedPatternContext * /*ctx*/) override { }
  virtual void exitGroupedPattern(RustParser::GroupedPatternContext * /*ctx*/) override { }

  virtual void enterSlicePattern(RustParser::SlicePatternContext * /*ctx*/) override { }
  virtual void exitSlicePattern(RustParser::SlicePatternContext * /*ctx*/) override { }

  virtual void enterSlicePatternItems(RustParser::SlicePatternItemsContext * /*ctx*/) override { }
  virtual void exitSlicePatternItems(RustParser::SlicePatternItemsContext * /*ctx*/) override { }

  virtual void enterPathPattern(RustParser::PathPatternContext * /*ctx*/) override { }
  virtual void exitPathPattern(RustParser::PathPatternContext * /*ctx*/) override { }

  virtual void enterType_(RustParser::Type_Context * /*ctx*/) override { }
  virtual void exitType_(RustParser::Type_Context * /*ctx*/) override { }

  virtual void enterTypeNoBounds(RustParser::TypeNoBoundsContext * /*ctx*/) override { }
  virtual void exitTypeNoBounds(RustParser::TypeNoBoundsContext * /*ctx*/) override { }

  virtual void enterParenthesizedType(RustParser::ParenthesizedTypeContext * /*ctx*/) override { }
  virtual void exitParenthesizedType(RustParser::ParenthesizedTypeContext * /*ctx*/) override { }

  virtual void enterNeverType(RustParser::NeverTypeContext * /*ctx*/) override { }
  virtual void exitNeverType(RustParser::NeverTypeContext * /*ctx*/) override { }

  virtual void enterTupleType(RustParser::TupleTypeContext * /*ctx*/) override { }
  virtual void exitTupleType(RustParser::TupleTypeContext * /*ctx*/) override { }

  virtual void enterArrayType(RustParser::ArrayTypeContext * /*ctx*/) override { }
  virtual void exitArrayType(RustParser::ArrayTypeContext * /*ctx*/) override { }

  virtual void enterSliceType(RustParser::SliceTypeContext * /*ctx*/) override { }
  virtual void exitSliceType(RustParser::SliceTypeContext * /*ctx*/) override { }

  virtual void enterReferenceType(RustParser::ReferenceTypeContext * /*ctx*/) override { }
  virtual void exitReferenceType(RustParser::ReferenceTypeContext * /*ctx*/) override { }

  virtual void enterRawPointerType(RustParser::RawPointerTypeContext * /*ctx*/) override { }
  virtual void exitRawPointerType(RustParser::RawPointerTypeContext * /*ctx*/) override { }

  virtual void enterBareFunctionType(RustParser::BareFunctionTypeContext * /*ctx*/) override { }
  virtual void exitBareFunctionType(RustParser::BareFunctionTypeContext * /*ctx*/) override { }

  virtual void enterFunctionTypeQualifiers(RustParser::FunctionTypeQualifiersContext * /*ctx*/) override { }
  virtual void exitFunctionTypeQualifiers(RustParser::FunctionTypeQualifiersContext * /*ctx*/) override { }

  virtual void enterBareFunctionReturnType(RustParser::BareFunctionReturnTypeContext * /*ctx*/) override { }
  virtual void exitBareFunctionReturnType(RustParser::BareFunctionReturnTypeContext * /*ctx*/) override { }

  virtual void enterFunctionParametersMaybeNamedVariadic(RustParser::FunctionParametersMaybeNamedVariadicContext * /*ctx*/) override { }
  virtual void exitFunctionParametersMaybeNamedVariadic(RustParser::FunctionParametersMaybeNamedVariadicContext * /*ctx*/) override { }

  virtual void enterMaybeNamedFunctionParameters(RustParser::MaybeNamedFunctionParametersContext * /*ctx*/) override { }
  virtual void exitMaybeNamedFunctionParameters(RustParser::MaybeNamedFunctionParametersContext * /*ctx*/) override { }

  virtual void enterMaybeNamedParam(RustParser::MaybeNamedParamContext * /*ctx*/) override { }
  virtual void exitMaybeNamedParam(RustParser::MaybeNamedParamContext * /*ctx*/) override { }

  virtual void enterMaybeNamedFunctionParametersVariadic(RustParser::MaybeNamedFunctionParametersVariadicContext * /*ctx*/) override { }
  virtual void exitMaybeNamedFunctionParametersVariadic(RustParser::MaybeNamedFunctionParametersVariadicContext * /*ctx*/) override { }

  virtual void enterTraitObjectType(RustParser::TraitObjectTypeContext * /*ctx*/) override { }
  virtual void exitTraitObjectType(RustParser::TraitObjectTypeContext * /*ctx*/) override { }

  virtual void enterTraitObjectTypeOneBound(RustParser::TraitObjectTypeOneBoundContext * /*ctx*/) override { }
  virtual void exitTraitObjectTypeOneBound(RustParser::TraitObjectTypeOneBoundContext * /*ctx*/) override { }

  virtual void enterImplTraitType(RustParser::ImplTraitTypeContext * /*ctx*/) override { }
  virtual void exitImplTraitType(RustParser::ImplTraitTypeContext * /*ctx*/) override { }

  virtual void enterImplTraitTypeOneBound(RustParser::ImplTraitTypeOneBoundContext * /*ctx*/) override { }
  virtual void exitImplTraitTypeOneBound(RustParser::ImplTraitTypeOneBoundContext * /*ctx*/) override { }

  virtual void enterInferredType(RustParser::InferredTypeContext * /*ctx*/) override { }
  virtual void exitInferredType(RustParser::InferredTypeContext * /*ctx*/) override { }

  virtual void enterTypeParamBounds(RustParser::TypeParamBoundsContext * /*ctx*/) override { }
  virtual void exitTypeParamBounds(RustParser::TypeParamBoundsContext * /*ctx*/) override { }

  virtual void enterTypeParamBound(RustParser::TypeParamBoundContext * /*ctx*/) override { }
  virtual void exitTypeParamBound(RustParser::TypeParamBoundContext * /*ctx*/) override { }

  virtual void enterTraitBound(RustParser::TraitBoundContext * /*ctx*/) override { }
  virtual void exitTraitBound(RustParser::TraitBoundContext * /*ctx*/) override { }

  virtual void enterLifetimeBounds(RustParser::LifetimeBoundsContext * /*ctx*/) override { }
  virtual void exitLifetimeBounds(RustParser::LifetimeBoundsContext * /*ctx*/) override { }

  virtual void enterLifetime(RustParser::LifetimeContext * /*ctx*/) override { }
  virtual void exitLifetime(RustParser::LifetimeContext * /*ctx*/) override { }

  virtual void enterSimplePath(RustParser::SimplePathContext * /*ctx*/) override { }
  virtual void exitSimplePath(RustParser::SimplePathContext * /*ctx*/) override { }

  virtual void enterSimplePathSegment(RustParser::SimplePathSegmentContext * /*ctx*/) override { }
  virtual void exitSimplePathSegment(RustParser::SimplePathSegmentContext * /*ctx*/) override { }

  virtual void enterPathInExpression(RustParser::PathInExpressionContext * /*ctx*/) override { }
  virtual void exitPathInExpression(RustParser::PathInExpressionContext * /*ctx*/) override { }

  virtual void enterPathExprSegment(RustParser::PathExprSegmentContext * /*ctx*/) override { }
  virtual void exitPathExprSegment(RustParser::PathExprSegmentContext * /*ctx*/) override { }

  virtual void enterPathIdentSegment(RustParser::PathIdentSegmentContext * /*ctx*/) override { }
  virtual void exitPathIdentSegment(RustParser::PathIdentSegmentContext * /*ctx*/) override { }

  virtual void enterGenericArgs(RustParser::GenericArgsContext * /*ctx*/) override { }
  virtual void exitGenericArgs(RustParser::GenericArgsContext * /*ctx*/) override { }

  virtual void enterGenericArg(RustParser::GenericArgContext * /*ctx*/) override { }
  virtual void exitGenericArg(RustParser::GenericArgContext * /*ctx*/) override { }

  virtual void enterGenericArgsConst(RustParser::GenericArgsConstContext * /*ctx*/) override { }
  virtual void exitGenericArgsConst(RustParser::GenericArgsConstContext * /*ctx*/) override { }

  virtual void enterGenericArgsLifetimes(RustParser::GenericArgsLifetimesContext * /*ctx*/) override { }
  virtual void exitGenericArgsLifetimes(RustParser::GenericArgsLifetimesContext * /*ctx*/) override { }

  virtual void enterGenericArgsTypes(RustParser::GenericArgsTypesContext * /*ctx*/) override { }
  virtual void exitGenericArgsTypes(RustParser::GenericArgsTypesContext * /*ctx*/) override { }

  virtual void enterGenericArgsBindings(RustParser::GenericArgsBindingsContext * /*ctx*/) override { }
  virtual void exitGenericArgsBindings(RustParser::GenericArgsBindingsContext * /*ctx*/) override { }

  virtual void enterGenericArgsBinding(RustParser::GenericArgsBindingContext * /*ctx*/) override { }
  virtual void exitGenericArgsBinding(RustParser::GenericArgsBindingContext * /*ctx*/) override { }

  virtual void enterQualifiedPathInExpression(RustParser::QualifiedPathInExpressionContext * /*ctx*/) override { }
  virtual void exitQualifiedPathInExpression(RustParser::QualifiedPathInExpressionContext * /*ctx*/) override { }

  virtual void enterQualifiedPathType(RustParser::QualifiedPathTypeContext * /*ctx*/) override { }
  virtual void exitQualifiedPathType(RustParser::QualifiedPathTypeContext * /*ctx*/) override { }

  virtual void enterQualifiedPathInType(RustParser::QualifiedPathInTypeContext * /*ctx*/) override { }
  virtual void exitQualifiedPathInType(RustParser::QualifiedPathInTypeContext * /*ctx*/) override { }

  virtual void enterTypePath(RustParser::TypePathContext * /*ctx*/) override { }
  virtual void exitTypePath(RustParser::TypePathContext * /*ctx*/) override { }

  virtual void enterTypePathSegment(RustParser::TypePathSegmentContext * /*ctx*/) override { }
  virtual void exitTypePathSegment(RustParser::TypePathSegmentContext * /*ctx*/) override { }

  virtual void enterTypePathFn(RustParser::TypePathFnContext * /*ctx*/) override { }
  virtual void exitTypePathFn(RustParser::TypePathFnContext * /*ctx*/) override { }

  virtual void enterTypePathInputs(RustParser::TypePathInputsContext * /*ctx*/) override { }
  virtual void exitTypePathInputs(RustParser::TypePathInputsContext * /*ctx*/) override { }

  virtual void enterVisibility(RustParser::VisibilityContext * /*ctx*/) override { }
  virtual void exitVisibility(RustParser::VisibilityContext * /*ctx*/) override { }

  virtual void enterIdentifier(RustParser::IdentifierContext * /*ctx*/) override { }
  virtual void exitIdentifier(RustParser::IdentifierContext * /*ctx*/) override { }

  virtual void enterKeyword(RustParser::KeywordContext * /*ctx*/) override { }
  virtual void exitKeyword(RustParser::KeywordContext * /*ctx*/) override { }

  virtual void enterMacroIdentifierLikeToken(RustParser::MacroIdentifierLikeTokenContext * /*ctx*/) override { }
  virtual void exitMacroIdentifierLikeToken(RustParser::MacroIdentifierLikeTokenContext * /*ctx*/) override { }

  virtual void enterMacroLiteralToken(RustParser::MacroLiteralTokenContext * /*ctx*/) override { }
  virtual void exitMacroLiteralToken(RustParser::MacroLiteralTokenContext * /*ctx*/) override { }

  virtual void enterMacroPunctuationToken(RustParser::MacroPunctuationTokenContext * /*ctx*/) override { }
  virtual void exitMacroPunctuationToken(RustParser::MacroPunctuationTokenContext * /*ctx*/) override { }

  virtual void enterShl(RustParser::ShlContext * /*ctx*/) override { }
  virtual void exitShl(RustParser::ShlContext * /*ctx*/) override { }

  virtual void enterShr(RustParser::ShrContext * /*ctx*/) override { }
  virtual void exitShr(RustParser::ShrContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

