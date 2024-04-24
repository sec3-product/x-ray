
// Generated from /home/aaron/smallrace/utils/parser/Smalltalk.g4 by ANTLR 4.9


#include "SmalltalkListener.h"
#include "SmalltalkVisitor.h"

#include "SmalltalkParser.h"


using namespace antlrcpp;
using namespace antlr4;

SmalltalkParser::SmalltalkParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

SmalltalkParser::~SmalltalkParser() {
  delete _interpreter;
}

std::string SmalltalkParser::getGrammarFileName() const {
  return "Smalltalk.g4";
}

const std::vector<std::string>& SmalltalkParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& SmalltalkParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- ModuleContext ------------------------------------------------------------------

SmalltalkParser::ModuleContext::ModuleContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::FunctionContext* SmalltalkParser::ModuleContext::function() {
  return getRuleContext<SmalltalkParser::FunctionContext>(0);
}

SmalltalkParser::ScriptContext* SmalltalkParser::ModuleContext::script() {
  return getRuleContext<SmalltalkParser::ScriptContext>(0);
}


size_t SmalltalkParser::ModuleContext::getRuleIndex() const {
  return SmalltalkParser::RuleModule;
}

void SmalltalkParser::ModuleContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterModule(this);
}

void SmalltalkParser::ModuleContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitModule(this);
}


antlrcpp::Any SmalltalkParser::ModuleContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitModule(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ModuleContext* SmalltalkParser::module() {
  ModuleContext *_localctx = _tracker.createInstance<ModuleContext>(_ctx, getState());
  enterRule(_localctx, 0, SmalltalkParser::RuleModule);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(112);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(110);
      function();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(111);
      script();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionContext ------------------------------------------------------------------

SmalltalkParser::FunctionContext::FunctionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::FuncdeclContext* SmalltalkParser::FunctionContext::funcdecl() {
  return getRuleContext<SmalltalkParser::FuncdeclContext>(0);
}

tree::TerminalNode* SmalltalkParser::FunctionContext::EOF() {
  return getToken(SmalltalkParser::EOF, 0);
}

SmalltalkParser::ScriptContext* SmalltalkParser::FunctionContext::script() {
  return getRuleContext<SmalltalkParser::ScriptContext>(0);
}

SmalltalkParser::WsContext* SmalltalkParser::FunctionContext::ws() {
  return getRuleContext<SmalltalkParser::WsContext>(0);
}


size_t SmalltalkParser::FunctionContext::getRuleIndex() const {
  return SmalltalkParser::RuleFunction;
}

void SmalltalkParser::FunctionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunction(this);
}

void SmalltalkParser::FunctionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunction(this);
}


antlrcpp::Any SmalltalkParser::FunctionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitFunction(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::FunctionContext* SmalltalkParser::function() {
  FunctionContext *_localctx = _tracker.createInstance<FunctionContext>(_ctx, getState());
  enterRule(_localctx, 2, SmalltalkParser::RuleFunction);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(114);
    funcdecl();
    setState(116);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx)) {
    case 1: {
      setState(115);
      ws();
      break;
    }

    default:
      break;
    }
    setState(120);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SmalltalkParser::EOF: {
        setState(118);
        match(SmalltalkParser::EOF);
        break;
      }

      case SmalltalkParser::SEPARATOR:
      case SmalltalkParser::STRING:
      case SmalltalkParser::PRIMITIVE:
      case SmalltalkParser::COMMENT:
      case SmalltalkParser::BLOCK_START:
      case SmalltalkParser::OPEN_PAREN:
      case SmalltalkParser::PIPE2:
      case SmalltalkParser::PIPE:
      case SmalltalkParser::MINUS:
      case SmalltalkParser::RESERVED_WORD:
      case SmalltalkParser::IDENTIFIER:
      case SmalltalkParser::IDENTIFIER2:
      case SmalltalkParser::CARROT:
      case SmalltalkParser::UNDERSCORE:
      case SmalltalkParser::UNDERSCORE2:
      case SmalltalkParser::HASH:
      case SmalltalkParser::DOLLAR:
      case SmalltalkParser::HEXNUM:
      case SmalltalkParser::INTNUM:
      case SmalltalkParser::DYNDICT_START:
      case SmalltalkParser::DYNARR_START:
      case SmalltalkParser::CHARACTER_CONSTANT: {
        setState(119);
        script();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncdeclContext ------------------------------------------------------------------

SmalltalkParser::FuncdeclContext::FuncdeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::IDENTIFIER() {
  return getToken(SmalltalkParser::IDENTIFIER, 0);
}

SmalltalkParser::WsContext* SmalltalkParser::FuncdeclContext::ws() {
  return getRuleContext<SmalltalkParser::WsContext>(0);
}

SmalltalkParser::DeclPairsContext* SmalltalkParser::FuncdeclContext::declPairs() {
  return getRuleContext<SmalltalkParser::DeclPairsContext>(0);
}

SmalltalkParser::VariableContext* SmalltalkParser::FuncdeclContext::variable() {
  return getRuleContext<SmalltalkParser::VariableContext>(0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::MINUS() {
  return getToken(SmalltalkParser::MINUS, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::PIPE() {
  return getToken(SmalltalkParser::PIPE, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::PIPE2() {
  return getToken(SmalltalkParser::PIPE2, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::PIPE3() {
  return getToken(SmalltalkParser::PIPE3, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::AMP() {
  return getToken(SmalltalkParser::AMP, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::BINARY_SELECTOR() {
  return getToken(SmalltalkParser::BINARY_SELECTOR, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::NEQ() {
  return getToken(SmalltalkParser::NEQ, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::LLT() {
  return getToken(SmalltalkParser::LLT, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::GGT() {
  return getToken(SmalltalkParser::GGT, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::LTE() {
  return getToken(SmalltalkParser::LTE, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::GTE() {
  return getToken(SmalltalkParser::GTE, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::LT() {
  return getToken(SmalltalkParser::LT, 0);
}

tree::TerminalNode* SmalltalkParser::FuncdeclContext::GT() {
  return getToken(SmalltalkParser::GT, 0);
}


size_t SmalltalkParser::FuncdeclContext::getRuleIndex() const {
  return SmalltalkParser::RuleFuncdecl;
}

void SmalltalkParser::FuncdeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFuncdecl(this);
}

void SmalltalkParser::FuncdeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFuncdecl(this);
}


antlrcpp::Any SmalltalkParser::FuncdeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitFuncdecl(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::FuncdeclContext* SmalltalkParser::funcdecl() {
  FuncdeclContext *_localctx = _tracker.createInstance<FuncdeclContext>(_ctx, getState());
  enterRule(_localctx, 4, SmalltalkParser::RuleFuncdecl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(132);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(123);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(122);
        ws();
      }
      setState(125);
      match(SmalltalkParser::IDENTIFIER);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(126);
      declPairs();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(127);
      _la = _input->LA(1);
      if (!((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << SmalltalkParser::PIPE3)
        | (1ULL << SmalltalkParser::PIPE2)
        | (1ULL << SmalltalkParser::PIPE)
        | (1ULL << SmalltalkParser::LLT)
        | (1ULL << SmalltalkParser::GGT)
        | (1ULL << SmalltalkParser::LTE)
        | (1ULL << SmalltalkParser::GTE)
        | (1ULL << SmalltalkParser::NEQ)
        | (1ULL << SmalltalkParser::LT)
        | (1ULL << SmalltalkParser::GT)
        | (1ULL << SmalltalkParser::AMP)
        | (1ULL << SmalltalkParser::MINUS)
        | (1ULL << SmalltalkParser::BINARY_SELECTOR))) != 0))) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(129);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(128);
        ws();
      }
      setState(131);
      variable();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclPairsContext ------------------------------------------------------------------

SmalltalkParser::DeclPairsContext::DeclPairsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::DeclPairsContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::DeclPairsContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

std::vector<SmalltalkParser::DeclPairContext *> SmalltalkParser::DeclPairsContext::declPair() {
  return getRuleContexts<SmalltalkParser::DeclPairContext>();
}

SmalltalkParser::DeclPairContext* SmalltalkParser::DeclPairsContext::declPair(size_t i) {
  return getRuleContext<SmalltalkParser::DeclPairContext>(i);
}


size_t SmalltalkParser::DeclPairsContext::getRuleIndex() const {
  return SmalltalkParser::RuleDeclPairs;
}

void SmalltalkParser::DeclPairsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeclPairs(this);
}

void SmalltalkParser::DeclPairsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeclPairs(this);
}


antlrcpp::Any SmalltalkParser::DeclPairsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitDeclPairs(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::DeclPairsContext* SmalltalkParser::declPairs() {
  DeclPairsContext *_localctx = _tracker.createInstance<DeclPairsContext>(_ctx, getState());
  enterRule(_localctx, 6, SmalltalkParser::RuleDeclPairs);
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
    setState(135);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(134);
      ws();
    }
    setState(141); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(137);
      declPair();
      setState(139);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
      case 1: {
        setState(138);
        ws();
        break;
      }

      default:
        break;
      }
      setState(143); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == SmalltalkParser::KEYWORD);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclPairContext ------------------------------------------------------------------

SmalltalkParser::DeclPairContext::DeclPairContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::DeclPairContext::KEYWORD() {
  return getToken(SmalltalkParser::KEYWORD, 0);
}

SmalltalkParser::VariableContext* SmalltalkParser::DeclPairContext::variable() {
  return getRuleContext<SmalltalkParser::VariableContext>(0);
}

SmalltalkParser::WsContext* SmalltalkParser::DeclPairContext::ws() {
  return getRuleContext<SmalltalkParser::WsContext>(0);
}


size_t SmalltalkParser::DeclPairContext::getRuleIndex() const {
  return SmalltalkParser::RuleDeclPair;
}

void SmalltalkParser::DeclPairContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeclPair(this);
}

void SmalltalkParser::DeclPairContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeclPair(this);
}


antlrcpp::Any SmalltalkParser::DeclPairContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitDeclPair(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::DeclPairContext* SmalltalkParser::declPair() {
  DeclPairContext *_localctx = _tracker.createInstance<DeclPairContext>(_ctx, getState());
  enterRule(_localctx, 8, SmalltalkParser::RuleDeclPair);
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
    setState(145);
    match(SmalltalkParser::KEYWORD);
    setState(147);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(146);
      ws();
    }
    setState(149);
    variable();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ScriptContext ------------------------------------------------------------------

SmalltalkParser::ScriptContext::ScriptContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::SequenceContext* SmalltalkParser::ScriptContext::sequence() {
  return getRuleContext<SmalltalkParser::SequenceContext>(0);
}

tree::TerminalNode* SmalltalkParser::ScriptContext::EOF() {
  return getToken(SmalltalkParser::EOF, 0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::ScriptContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::ScriptContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

std::vector<SmalltalkParser::PrimitiveContext *> SmalltalkParser::ScriptContext::primitive() {
  return getRuleContexts<SmalltalkParser::PrimitiveContext>();
}

SmalltalkParser::PrimitiveContext* SmalltalkParser::ScriptContext::primitive(size_t i) {
  return getRuleContext<SmalltalkParser::PrimitiveContext>(i);
}


size_t SmalltalkParser::ScriptContext::getRuleIndex() const {
  return SmalltalkParser::RuleScript;
}

void SmalltalkParser::ScriptContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterScript(this);
}

void SmalltalkParser::ScriptContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitScript(this);
}


antlrcpp::Any SmalltalkParser::ScriptContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitScript(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ScriptContext* SmalltalkParser::script() {
  ScriptContext *_localctx = _tracker.createInstance<ScriptContext>(_ctx, getState());
  enterRule(_localctx, 10, SmalltalkParser::RuleScript);
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
    setState(152);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx)) {
    case 1: {
      setState(151);
      ws();
      break;
    }

    default:
      break;
    }
    setState(160);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(154);
        primitive();
        setState(156);
        _errHandler->sync(this);

        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
        case 1: {
          setState(155);
          ws();
          break;
        }

        default:
          break;
        } 
      }
      setState(162);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
    }
    setState(163);
    sequence();
    setState(165);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(164);
      ws();
    }
    setState(167);
    match(SmalltalkParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SequenceContext ------------------------------------------------------------------

SmalltalkParser::SequenceContext::SequenceContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::TempsContext* SmalltalkParser::SequenceContext::temps() {
  return getRuleContext<SmalltalkParser::TempsContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::SequenceContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::SequenceContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

SmalltalkParser::PrimitiveContext* SmalltalkParser::SequenceContext::primitive() {
  return getRuleContext<SmalltalkParser::PrimitiveContext>(0);
}

SmalltalkParser::StatementsContext* SmalltalkParser::SequenceContext::statements() {
  return getRuleContext<SmalltalkParser::StatementsContext>(0);
}


size_t SmalltalkParser::SequenceContext::getRuleIndex() const {
  return SmalltalkParser::RuleSequence;
}

void SmalltalkParser::SequenceContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSequence(this);
}

void SmalltalkParser::SequenceContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSequence(this);
}


antlrcpp::Any SmalltalkParser::SequenceContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitSequence(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::SequenceContext* SmalltalkParser::sequence() {
  SequenceContext *_localctx = _tracker.createInstance<SequenceContext>(_ctx, getState());
  enterRule(_localctx, 12, SmalltalkParser::RuleSequence);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(189);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SmalltalkParser::PIPE2:
      case SmalltalkParser::PIPE: {
        enterOuterAlt(_localctx, 1);
        setState(169);
        temps();
        setState(171);
        _errHandler->sync(this);

        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
        case 1: {
          setState(170);
          ws();
          break;
        }

        default:
          break;
        }
        setState(174);
        _errHandler->sync(this);

        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx)) {
        case 1: {
          setState(173);
          primitive();
          break;
        }

        default:
          break;
        }
        setState(177);
        _errHandler->sync(this);

        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
        case 1: {
          setState(176);
          ws();
          break;
        }

        default:
          break;
        }
        setState(180);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << SmalltalkParser::STRING)
          | (1ULL << SmalltalkParser::PRIMITIVE)
          | (1ULL << SmalltalkParser::BLOCK_START)
          | (1ULL << SmalltalkParser::OPEN_PAREN)
          | (1ULL << SmalltalkParser::MINUS)
          | (1ULL << SmalltalkParser::RESERVED_WORD)
          | (1ULL << SmalltalkParser::IDENTIFIER)
          | (1ULL << SmalltalkParser::IDENTIFIER2)
          | (1ULL << SmalltalkParser::CARROT)
          | (1ULL << SmalltalkParser::UNDERSCORE)
          | (1ULL << SmalltalkParser::UNDERSCORE2)
          | (1ULL << SmalltalkParser::HASH)
          | (1ULL << SmalltalkParser::DOLLAR)
          | (1ULL << SmalltalkParser::HEXNUM)
          | (1ULL << SmalltalkParser::INTNUM)
          | (1ULL << SmalltalkParser::DYNDICT_START)
          | (1ULL << SmalltalkParser::DYNARR_START)
          | (1ULL << SmalltalkParser::CHARACTER_CONSTANT))) != 0)) {
          setState(179);
          statements();
        }
        break;
      }

      case SmalltalkParser::SEPARATOR:
      case SmalltalkParser::STRING:
      case SmalltalkParser::PRIMITIVE:
      case SmalltalkParser::COMMENT:
      case SmalltalkParser::BLOCK_START:
      case SmalltalkParser::OPEN_PAREN:
      case SmalltalkParser::MINUS:
      case SmalltalkParser::RESERVED_WORD:
      case SmalltalkParser::IDENTIFIER:
      case SmalltalkParser::IDENTIFIER2:
      case SmalltalkParser::CARROT:
      case SmalltalkParser::UNDERSCORE:
      case SmalltalkParser::UNDERSCORE2:
      case SmalltalkParser::HASH:
      case SmalltalkParser::DOLLAR:
      case SmalltalkParser::HEXNUM:
      case SmalltalkParser::INTNUM:
      case SmalltalkParser::DYNDICT_START:
      case SmalltalkParser::DYNARR_START:
      case SmalltalkParser::CHARACTER_CONSTANT: {
        enterOuterAlt(_localctx, 2);
        setState(183);
        _errHandler->sync(this);

        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
        case 1: {
          setState(182);
          primitive();
          break;
        }

        default:
          break;
        }
        setState(186);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == SmalltalkParser::SEPARATOR

        || _la == SmalltalkParser::COMMENT) {
          setState(185);
          ws();
        }
        setState(188);
        statements();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- WsContext ------------------------------------------------------------------

SmalltalkParser::WsContext::WsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> SmalltalkParser::WsContext::SEPARATOR() {
  return getTokens(SmalltalkParser::SEPARATOR);
}

tree::TerminalNode* SmalltalkParser::WsContext::SEPARATOR(size_t i) {
  return getToken(SmalltalkParser::SEPARATOR, i);
}

std::vector<tree::TerminalNode *> SmalltalkParser::WsContext::COMMENT() {
  return getTokens(SmalltalkParser::COMMENT);
}

tree::TerminalNode* SmalltalkParser::WsContext::COMMENT(size_t i) {
  return getToken(SmalltalkParser::COMMENT, i);
}


size_t SmalltalkParser::WsContext::getRuleIndex() const {
  return SmalltalkParser::RuleWs;
}

void SmalltalkParser::WsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWs(this);
}

void SmalltalkParser::WsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWs(this);
}


antlrcpp::Any SmalltalkParser::WsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitWs(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::WsContext* SmalltalkParser::ws() {
  WsContext *_localctx = _tracker.createInstance<WsContext>(_ctx, getState());
  enterRule(_localctx, 14, SmalltalkParser::RuleWs);
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
    setState(192); 
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
              setState(191);
              _la = _input->LA(1);
              if (!(_la == SmalltalkParser::SEPARATOR

              || _la == SmalltalkParser::COMMENT)) {
              _errHandler->recoverInline(this);
              }
              else {
                _errHandler->reportMatch(this);
                consume();
              }
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(194); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TempsContext ------------------------------------------------------------------

SmalltalkParser::TempsContext::TempsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> SmalltalkParser::TempsContext::PIPE() {
  return getTokens(SmalltalkParser::PIPE);
}

tree::TerminalNode* SmalltalkParser::TempsContext::PIPE(size_t i) {
  return getToken(SmalltalkParser::PIPE, i);
}

std::vector<tree::TerminalNode *> SmalltalkParser::TempsContext::PIPE2() {
  return getTokens(SmalltalkParser::PIPE2);
}

tree::TerminalNode* SmalltalkParser::TempsContext::PIPE2(size_t i) {
  return getToken(SmalltalkParser::PIPE2, i);
}

std::vector<tree::TerminalNode *> SmalltalkParser::TempsContext::IDENTIFIER() {
  return getTokens(SmalltalkParser::IDENTIFIER);
}

tree::TerminalNode* SmalltalkParser::TempsContext::IDENTIFIER(size_t i) {
  return getToken(SmalltalkParser::IDENTIFIER, i);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::TempsContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::TempsContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}


size_t SmalltalkParser::TempsContext::getRuleIndex() const {
  return SmalltalkParser::RuleTemps;
}

void SmalltalkParser::TempsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTemps(this);
}

void SmalltalkParser::TempsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTemps(this);
}


antlrcpp::Any SmalltalkParser::TempsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitTemps(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::TempsContext* SmalltalkParser::temps() {
  TempsContext *_localctx = _tracker.createInstance<TempsContext>(_ctx, getState());
  enterRule(_localctx, 16, SmalltalkParser::RuleTemps);
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
    setState(211);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(196);
      _la = _input->LA(1);
      if (!(_la == SmalltalkParser::PIPE2

      || _la == SmalltalkParser::PIPE)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(203);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(198);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == SmalltalkParser::SEPARATOR

          || _la == SmalltalkParser::COMMENT) {
            setState(197);
            ws();
          }
          setState(200);
          match(SmalltalkParser::IDENTIFIER); 
        }
        setState(205);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx);
      }
      setState(207);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(206);
        ws();
      }
      setState(209);
      _la = _input->LA(1);
      if (!(_la == SmalltalkParser::PIPE2

      || _la == SmalltalkParser::PIPE)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(210);
      match(SmalltalkParser::PIPE2);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementsContext ------------------------------------------------------------------

SmalltalkParser::StatementsContext::StatementsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::AnswerContext* SmalltalkParser::StatementsContext::answer() {
  return getRuleContext<SmalltalkParser::AnswerContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::StatementsContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::StatementsContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

SmalltalkParser::ExpressionsContext* SmalltalkParser::StatementsContext::expressions() {
  return getRuleContext<SmalltalkParser::ExpressionsContext>(0);
}

tree::TerminalNode* SmalltalkParser::StatementsContext::PERIOD() {
  return getToken(SmalltalkParser::PERIOD, 0);
}


size_t SmalltalkParser::StatementsContext::getRuleIndex() const {
  return SmalltalkParser::RuleStatements;
}

void SmalltalkParser::StatementsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatements(this);
}

void SmalltalkParser::StatementsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatements(this);
}


antlrcpp::Any SmalltalkParser::StatementsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitStatements(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::StatementsContext* SmalltalkParser::statements() {
  StatementsContext *_localctx = _tracker.createInstance<StatementsContext>(_ctx, getState());
  enterRule(_localctx, 18, SmalltalkParser::RuleStatements);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(241);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(213);
      answer();
      setState(215);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 26, _ctx)) {
      case 1: {
        setState(214);
        ws();
        break;
      }

      default:
        break;
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(217);
      expressions();
      setState(219);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 27, _ctx)) {
      case 1: {
        setState(218);
        ws();
        break;
      }

      default:
        break;
      }
      setState(222);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::PERIOD) {
        setState(221);
        match(SmalltalkParser::PERIOD);
      }
      setState(225);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(224);
        ws();
      }
      setState(227);
      answer();
      setState(229);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx)) {
      case 1: {
        setState(228);
        ws();
        break;
      }

      default:
        break;
      }
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(231);
      expressions();
      setState(233);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx)) {
      case 1: {
        setState(232);
        ws();
        break;
      }

      default:
        break;
      }
      setState(236);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::PERIOD) {
        setState(235);
        match(SmalltalkParser::PERIOD);
      }
      setState(239);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx)) {
      case 1: {
        setState(238);
        ws();
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
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AnswerContext ------------------------------------------------------------------

SmalltalkParser::AnswerContext::AnswerContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::AnswerContext::CARROT() {
  return getToken(SmalltalkParser::CARROT, 0);
}

SmalltalkParser::ExpressionContext* SmalltalkParser::AnswerContext::expression() {
  return getRuleContext<SmalltalkParser::ExpressionContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::AnswerContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::AnswerContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

tree::TerminalNode* SmalltalkParser::AnswerContext::PERIOD() {
  return getToken(SmalltalkParser::PERIOD, 0);
}


size_t SmalltalkParser::AnswerContext::getRuleIndex() const {
  return SmalltalkParser::RuleAnswer;
}

void SmalltalkParser::AnswerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAnswer(this);
}

void SmalltalkParser::AnswerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAnswer(this);
}


antlrcpp::Any SmalltalkParser::AnswerContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitAnswer(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::AnswerContext* SmalltalkParser::answer() {
  AnswerContext *_localctx = _tracker.createInstance<AnswerContext>(_ctx, getState());
  enterRule(_localctx, 20, SmalltalkParser::RuleAnswer);
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
    setState(243);
    match(SmalltalkParser::CARROT);
    setState(245);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(244);
      ws();
    }
    setState(247);
    expression();
    setState(249);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 36, _ctx)) {
    case 1: {
      setState(248);
      ws();
      break;
    }

    default:
      break;
    }
    setState(252);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::PERIOD) {
      setState(251);
      match(SmalltalkParser::PERIOD);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

SmalltalkParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::AssignmentContext* SmalltalkParser::ExpressionContext::assignment() {
  return getRuleContext<SmalltalkParser::AssignmentContext>(0);
}

SmalltalkParser::CascadeContext* SmalltalkParser::ExpressionContext::cascade() {
  return getRuleContext<SmalltalkParser::CascadeContext>(0);
}

SmalltalkParser::KeywordSendContext* SmalltalkParser::ExpressionContext::keywordSend() {
  return getRuleContext<SmalltalkParser::KeywordSendContext>(0);
}

SmalltalkParser::BinarySendContext* SmalltalkParser::ExpressionContext::binarySend() {
  return getRuleContext<SmalltalkParser::BinarySendContext>(0);
}

SmalltalkParser::PrimitiveContext* SmalltalkParser::ExpressionContext::primitive() {
  return getRuleContext<SmalltalkParser::PrimitiveContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::ExpressionContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::ExpressionContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

tree::TerminalNode* SmalltalkParser::ExpressionContext::PERIOD() {
  return getToken(SmalltalkParser::PERIOD, 0);
}


size_t SmalltalkParser::ExpressionContext::getRuleIndex() const {
  return SmalltalkParser::RuleExpression;
}

void SmalltalkParser::ExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression(this);
}

void SmalltalkParser::ExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression(this);
}


antlrcpp::Any SmalltalkParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ExpressionContext* SmalltalkParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 22, SmalltalkParser::RuleExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(259);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 38, _ctx)) {
    case 1: {
      setState(254);
      assignment();
      break;
    }

    case 2: {
      setState(255);
      cascade();
      break;
    }

    case 3: {
      setState(256);
      keywordSend();
      break;
    }

    case 4: {
      setState(257);
      binarySend();
      break;
    }

    case 5: {
      setState(258);
      primitive();
      break;
    }

    default:
      break;
    }
    setState(262);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 39, _ctx)) {
    case 1: {
      setState(261);
      ws();
      break;
    }

    default:
      break;
    }
    setState(265);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 40, _ctx)) {
    case 1: {
      setState(264);
      match(SmalltalkParser::PERIOD);
      break;
    }

    default:
      break;
    }
    setState(268);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 41, _ctx)) {
    case 1: {
      setState(267);
      ws();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionsContext ------------------------------------------------------------------

SmalltalkParser::ExpressionsContext::ExpressionsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::ExpressionContext* SmalltalkParser::ExpressionsContext::expression() {
  return getRuleContext<SmalltalkParser::ExpressionContext>(0);
}

std::vector<SmalltalkParser::ExpressionListContext *> SmalltalkParser::ExpressionsContext::expressionList() {
  return getRuleContexts<SmalltalkParser::ExpressionListContext>();
}

SmalltalkParser::ExpressionListContext* SmalltalkParser::ExpressionsContext::expressionList(size_t i) {
  return getRuleContext<SmalltalkParser::ExpressionListContext>(i);
}


size_t SmalltalkParser::ExpressionsContext::getRuleIndex() const {
  return SmalltalkParser::RuleExpressions;
}

void SmalltalkParser::ExpressionsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressions(this);
}

void SmalltalkParser::ExpressionsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressions(this);
}


antlrcpp::Any SmalltalkParser::ExpressionsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitExpressions(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ExpressionsContext* SmalltalkParser::expressions() {
  ExpressionsContext *_localctx = _tracker.createInstance<ExpressionsContext>(_ctx, getState());
  enterRule(_localctx, 24, SmalltalkParser::RuleExpressions);

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
    setState(270);
    expression();
    setState(274);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 42, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(271);
        expressionList(); 
      }
      setState(276);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 42, _ctx);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionListContext ------------------------------------------------------------------

SmalltalkParser::ExpressionListContext::ExpressionListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::ExpressionListContext::PERIOD() {
  return getToken(SmalltalkParser::PERIOD, 0);
}

SmalltalkParser::ExpressionContext* SmalltalkParser::ExpressionListContext::expression() {
  return getRuleContext<SmalltalkParser::ExpressionContext>(0);
}

SmalltalkParser::WsContext* SmalltalkParser::ExpressionListContext::ws() {
  return getRuleContext<SmalltalkParser::WsContext>(0);
}


size_t SmalltalkParser::ExpressionListContext::getRuleIndex() const {
  return SmalltalkParser::RuleExpressionList;
}

void SmalltalkParser::ExpressionListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressionList(this);
}

void SmalltalkParser::ExpressionListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressionList(this);
}


antlrcpp::Any SmalltalkParser::ExpressionListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitExpressionList(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ExpressionListContext* SmalltalkParser::expressionList() {
  ExpressionListContext *_localctx = _tracker.createInstance<ExpressionListContext>(_ctx, getState());
  enterRule(_localctx, 26, SmalltalkParser::RuleExpressionList);
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
    setState(277);
    match(SmalltalkParser::PERIOD);
    setState(279);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(278);
      ws();
    }
    setState(281);
    expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CascadeContext ------------------------------------------------------------------

SmalltalkParser::CascadeContext::CascadeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::KeywordSendContext* SmalltalkParser::CascadeContext::keywordSend() {
  return getRuleContext<SmalltalkParser::KeywordSendContext>(0);
}

SmalltalkParser::BinarySendContext* SmalltalkParser::CascadeContext::binarySend() {
  return getRuleContext<SmalltalkParser::BinarySendContext>(0);
}

std::vector<tree::TerminalNode *> SmalltalkParser::CascadeContext::SEMI_COLON() {
  return getTokens(SmalltalkParser::SEMI_COLON);
}

tree::TerminalNode* SmalltalkParser::CascadeContext::SEMI_COLON(size_t i) {
  return getToken(SmalltalkParser::SEMI_COLON, i);
}

std::vector<SmalltalkParser::MessageContext *> SmalltalkParser::CascadeContext::message() {
  return getRuleContexts<SmalltalkParser::MessageContext>();
}

SmalltalkParser::MessageContext* SmalltalkParser::CascadeContext::message(size_t i) {
  return getRuleContext<SmalltalkParser::MessageContext>(i);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::CascadeContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::CascadeContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}


size_t SmalltalkParser::CascadeContext::getRuleIndex() const {
  return SmalltalkParser::RuleCascade;
}

void SmalltalkParser::CascadeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCascade(this);
}

void SmalltalkParser::CascadeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCascade(this);
}


antlrcpp::Any SmalltalkParser::CascadeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitCascade(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::CascadeContext* SmalltalkParser::cascade() {
  CascadeContext *_localctx = _tracker.createInstance<CascadeContext>(_ctx, getState());
  enterRule(_localctx, 28, SmalltalkParser::RuleCascade);
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
    setState(285);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 44, _ctx)) {
    case 1: {
      setState(283);
      keywordSend();
      break;
    }

    case 2: {
      setState(284);
      binarySend();
      break;
    }

    default:
      break;
    }
    setState(295); 
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
              setState(288);
              _errHandler->sync(this);

              _la = _input->LA(1);
              if (_la == SmalltalkParser::SEPARATOR

              || _la == SmalltalkParser::COMMENT) {
                setState(287);
                ws();
              }
              setState(290);
              match(SmalltalkParser::SEMI_COLON);
              setState(292);
              _errHandler->sync(this);

              switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 46, _ctx)) {
              case 1: {
                setState(291);
                ws();
                break;
              }

              default:
                break;
              }
              setState(294);
              message();
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(297); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 47, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MessageContext ------------------------------------------------------------------

SmalltalkParser::MessageContext::MessageContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::BinaryMessageContext* SmalltalkParser::MessageContext::binaryMessage() {
  return getRuleContext<SmalltalkParser::BinaryMessageContext>(0);
}

SmalltalkParser::UnaryMessageContext* SmalltalkParser::MessageContext::unaryMessage() {
  return getRuleContext<SmalltalkParser::UnaryMessageContext>(0);
}

SmalltalkParser::KeywordMessageContext* SmalltalkParser::MessageContext::keywordMessage() {
  return getRuleContext<SmalltalkParser::KeywordMessageContext>(0);
}


size_t SmalltalkParser::MessageContext::getRuleIndex() const {
  return SmalltalkParser::RuleMessage;
}

void SmalltalkParser::MessageContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMessage(this);
}

void SmalltalkParser::MessageContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMessage(this);
}


antlrcpp::Any SmalltalkParser::MessageContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitMessage(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::MessageContext* SmalltalkParser::message() {
  MessageContext *_localctx = _tracker.createInstance<MessageContext>(_ctx, getState());
  enterRule(_localctx, 30, SmalltalkParser::RuleMessage);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(302);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 48, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(299);
      binaryMessage();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(300);
      unaryMessage();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(301);
      keywordMessage();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignmentContext ------------------------------------------------------------------

SmalltalkParser::AssignmentContext::AssignmentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::VariableContext* SmalltalkParser::AssignmentContext::variable() {
  return getRuleContext<SmalltalkParser::VariableContext>(0);
}

tree::TerminalNode* SmalltalkParser::AssignmentContext::ASSIGNMENT() {
  return getToken(SmalltalkParser::ASSIGNMENT, 0);
}

SmalltalkParser::ExpressionContext* SmalltalkParser::AssignmentContext::expression() {
  return getRuleContext<SmalltalkParser::ExpressionContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::AssignmentContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::AssignmentContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}


size_t SmalltalkParser::AssignmentContext::getRuleIndex() const {
  return SmalltalkParser::RuleAssignment;
}

void SmalltalkParser::AssignmentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignment(this);
}

void SmalltalkParser::AssignmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignment(this);
}


antlrcpp::Any SmalltalkParser::AssignmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitAssignment(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::AssignmentContext* SmalltalkParser::assignment() {
  AssignmentContext *_localctx = _tracker.createInstance<AssignmentContext>(_ctx, getState());
  enterRule(_localctx, 32, SmalltalkParser::RuleAssignment);
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
    setState(304);
    variable();
    setState(306);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(305);
      ws();
    }
    setState(308);
    match(SmalltalkParser::ASSIGNMENT);
    setState(310);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(309);
      ws();
    }
    setState(312);
    expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VariableContext ------------------------------------------------------------------

SmalltalkParser::VariableContext::VariableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::VariableContext::IDENTIFIER() {
  return getToken(SmalltalkParser::IDENTIFIER, 0);
}

tree::TerminalNode* SmalltalkParser::VariableContext::IDENTIFIER2() {
  return getToken(SmalltalkParser::IDENTIFIER2, 0);
}

tree::TerminalNode* SmalltalkParser::VariableContext::UNDERSCORE() {
  return getToken(SmalltalkParser::UNDERSCORE, 0);
}

tree::TerminalNode* SmalltalkParser::VariableContext::UNDERSCORE2() {
  return getToken(SmalltalkParser::UNDERSCORE2, 0);
}


size_t SmalltalkParser::VariableContext::getRuleIndex() const {
  return SmalltalkParser::RuleVariable;
}

void SmalltalkParser::VariableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVariable(this);
}

void SmalltalkParser::VariableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVariable(this);
}


antlrcpp::Any SmalltalkParser::VariableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitVariable(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::VariableContext* SmalltalkParser::variable() {
  VariableContext *_localctx = _tracker.createInstance<VariableContext>(_ctx, getState());
  enterRule(_localctx, 34, SmalltalkParser::RuleVariable);
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
    setState(314);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SmalltalkParser::IDENTIFIER)
      | (1ULL << SmalltalkParser::IDENTIFIER2)
      | (1ULL << SmalltalkParser::UNDERSCORE)
      | (1ULL << SmalltalkParser::UNDERSCORE2))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BinarySendContext ------------------------------------------------------------------

SmalltalkParser::BinarySendContext::BinarySendContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::UnarySendContext* SmalltalkParser::BinarySendContext::unarySend() {
  return getRuleContext<SmalltalkParser::UnarySendContext>(0);
}

SmalltalkParser::BinaryTailContext* SmalltalkParser::BinarySendContext::binaryTail() {
  return getRuleContext<SmalltalkParser::BinaryTailContext>(0);
}


size_t SmalltalkParser::BinarySendContext::getRuleIndex() const {
  return SmalltalkParser::RuleBinarySend;
}

void SmalltalkParser::BinarySendContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinarySend(this);
}

void SmalltalkParser::BinarySendContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinarySend(this);
}


antlrcpp::Any SmalltalkParser::BinarySendContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitBinarySend(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::BinarySendContext* SmalltalkParser::binarySend() {
  BinarySendContext *_localctx = _tracker.createInstance<BinarySendContext>(_ctx, getState());
  enterRule(_localctx, 36, SmalltalkParser::RuleBinarySend);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(316);
    unarySend();
    setState(318);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 51, _ctx)) {
    case 1: {
      setState(317);
      binaryTail();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnarySendContext ------------------------------------------------------------------

SmalltalkParser::UnarySendContext::UnarySendContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::OperandContext* SmalltalkParser::UnarySendContext::operand() {
  return getRuleContext<SmalltalkParser::OperandContext>(0);
}

SmalltalkParser::WsContext* SmalltalkParser::UnarySendContext::ws() {
  return getRuleContext<SmalltalkParser::WsContext>(0);
}

SmalltalkParser::UnaryTailContext* SmalltalkParser::UnarySendContext::unaryTail() {
  return getRuleContext<SmalltalkParser::UnaryTailContext>(0);
}


size_t SmalltalkParser::UnarySendContext::getRuleIndex() const {
  return SmalltalkParser::RuleUnarySend;
}

void SmalltalkParser::UnarySendContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnarySend(this);
}

void SmalltalkParser::UnarySendContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnarySend(this);
}


antlrcpp::Any SmalltalkParser::UnarySendContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitUnarySend(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::UnarySendContext* SmalltalkParser::unarySend() {
  UnarySendContext *_localctx = _tracker.createInstance<UnarySendContext>(_ctx, getState());
  enterRule(_localctx, 38, SmalltalkParser::RuleUnarySend);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(320);
    operand();
    setState(322);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 52, _ctx)) {
    case 1: {
      setState(321);
      ws();
      break;
    }

    default:
      break;
    }
    setState(325);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 53, _ctx)) {
    case 1: {
      setState(324);
      unaryTail();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- KeywordSendContext ------------------------------------------------------------------

SmalltalkParser::KeywordSendContext::KeywordSendContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::BinarySendContext* SmalltalkParser::KeywordSendContext::binarySend() {
  return getRuleContext<SmalltalkParser::BinarySendContext>(0);
}

SmalltalkParser::KeywordMessageContext* SmalltalkParser::KeywordSendContext::keywordMessage() {
  return getRuleContext<SmalltalkParser::KeywordMessageContext>(0);
}


size_t SmalltalkParser::KeywordSendContext::getRuleIndex() const {
  return SmalltalkParser::RuleKeywordSend;
}

void SmalltalkParser::KeywordSendContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterKeywordSend(this);
}

void SmalltalkParser::KeywordSendContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitKeywordSend(this);
}


antlrcpp::Any SmalltalkParser::KeywordSendContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitKeywordSend(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::KeywordSendContext* SmalltalkParser::keywordSend() {
  KeywordSendContext *_localctx = _tracker.createInstance<KeywordSendContext>(_ctx, getState());
  enterRule(_localctx, 40, SmalltalkParser::RuleKeywordSend);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(327);
    binarySend();
    setState(328);
    keywordMessage();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- KeywordMessageContext ------------------------------------------------------------------

SmalltalkParser::KeywordMessageContext::KeywordMessageContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::KeywordMessageContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::KeywordMessageContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

std::vector<SmalltalkParser::KeywordPairContext *> SmalltalkParser::KeywordMessageContext::keywordPair() {
  return getRuleContexts<SmalltalkParser::KeywordPairContext>();
}

SmalltalkParser::KeywordPairContext* SmalltalkParser::KeywordMessageContext::keywordPair(size_t i) {
  return getRuleContext<SmalltalkParser::KeywordPairContext>(i);
}


size_t SmalltalkParser::KeywordMessageContext::getRuleIndex() const {
  return SmalltalkParser::RuleKeywordMessage;
}

void SmalltalkParser::KeywordMessageContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterKeywordMessage(this);
}

void SmalltalkParser::KeywordMessageContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitKeywordMessage(this);
}


antlrcpp::Any SmalltalkParser::KeywordMessageContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitKeywordMessage(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::KeywordMessageContext* SmalltalkParser::keywordMessage() {
  KeywordMessageContext *_localctx = _tracker.createInstance<KeywordMessageContext>(_ctx, getState());
  enterRule(_localctx, 42, SmalltalkParser::RuleKeywordMessage);
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
    setState(331);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(330);
      ws();
    }
    setState(337); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(333);
      keywordPair();
      setState(335);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 55, _ctx)) {
      case 1: {
        setState(334);
        ws();
        break;
      }

      default:
        break;
      }
      setState(339); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == SmalltalkParser::KEYWORD);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- KeywordPairContext ------------------------------------------------------------------

SmalltalkParser::KeywordPairContext::KeywordPairContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::KeywordPairContext::KEYWORD() {
  return getToken(SmalltalkParser::KEYWORD, 0);
}

SmalltalkParser::BinarySendContext* SmalltalkParser::KeywordPairContext::binarySend() {
  return getRuleContext<SmalltalkParser::BinarySendContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::KeywordPairContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::KeywordPairContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}


size_t SmalltalkParser::KeywordPairContext::getRuleIndex() const {
  return SmalltalkParser::RuleKeywordPair;
}

void SmalltalkParser::KeywordPairContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterKeywordPair(this);
}

void SmalltalkParser::KeywordPairContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitKeywordPair(this);
}


antlrcpp::Any SmalltalkParser::KeywordPairContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitKeywordPair(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::KeywordPairContext* SmalltalkParser::keywordPair() {
  KeywordPairContext *_localctx = _tracker.createInstance<KeywordPairContext>(_ctx, getState());
  enterRule(_localctx, 44, SmalltalkParser::RuleKeywordPair);
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
    setState(341);
    match(SmalltalkParser::KEYWORD);
    setState(343);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(342);
      ws();
    }
    setState(345);
    binarySend();
    setState(347);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 58, _ctx)) {
    case 1: {
      setState(346);
      ws();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OperandContext ------------------------------------------------------------------

SmalltalkParser::OperandContext::OperandContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::LiteralContext* SmalltalkParser::OperandContext::literal() {
  return getRuleContext<SmalltalkParser::LiteralContext>(0);
}

SmalltalkParser::ReferenceContext* SmalltalkParser::OperandContext::reference() {
  return getRuleContext<SmalltalkParser::ReferenceContext>(0);
}

SmalltalkParser::SubexpressionContext* SmalltalkParser::OperandContext::subexpression() {
  return getRuleContext<SmalltalkParser::SubexpressionContext>(0);
}


size_t SmalltalkParser::OperandContext::getRuleIndex() const {
  return SmalltalkParser::RuleOperand;
}

void SmalltalkParser::OperandContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOperand(this);
}

void SmalltalkParser::OperandContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOperand(this);
}


antlrcpp::Any SmalltalkParser::OperandContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitOperand(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::OperandContext* SmalltalkParser::operand() {
  OperandContext *_localctx = _tracker.createInstance<OperandContext>(_ctx, getState());
  enterRule(_localctx, 46, SmalltalkParser::RuleOperand);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(352);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SmalltalkParser::STRING:
      case SmalltalkParser::BLOCK_START:
      case SmalltalkParser::MINUS:
      case SmalltalkParser::RESERVED_WORD:
      case SmalltalkParser::HASH:
      case SmalltalkParser::DOLLAR:
      case SmalltalkParser::HEXNUM:
      case SmalltalkParser::INTNUM:
      case SmalltalkParser::DYNDICT_START:
      case SmalltalkParser::DYNARR_START:
      case SmalltalkParser::CHARACTER_CONSTANT: {
        enterOuterAlt(_localctx, 1);
        setState(349);
        literal();
        break;
      }

      case SmalltalkParser::IDENTIFIER:
      case SmalltalkParser::IDENTIFIER2:
      case SmalltalkParser::UNDERSCORE:
      case SmalltalkParser::UNDERSCORE2: {
        enterOuterAlt(_localctx, 2);
        setState(350);
        reference();
        break;
      }

      case SmalltalkParser::OPEN_PAREN: {
        enterOuterAlt(_localctx, 3);
        setState(351);
        subexpression();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SubexpressionContext ------------------------------------------------------------------

SmalltalkParser::SubexpressionContext::SubexpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::SubexpressionContext::OPEN_PAREN() {
  return getToken(SmalltalkParser::OPEN_PAREN, 0);
}

SmalltalkParser::ExpressionContext* SmalltalkParser::SubexpressionContext::expression() {
  return getRuleContext<SmalltalkParser::ExpressionContext>(0);
}

tree::TerminalNode* SmalltalkParser::SubexpressionContext::CLOSE_PAREN() {
  return getToken(SmalltalkParser::CLOSE_PAREN, 0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::SubexpressionContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::SubexpressionContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}


size_t SmalltalkParser::SubexpressionContext::getRuleIndex() const {
  return SmalltalkParser::RuleSubexpression;
}

void SmalltalkParser::SubexpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSubexpression(this);
}

void SmalltalkParser::SubexpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSubexpression(this);
}


antlrcpp::Any SmalltalkParser::SubexpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitSubexpression(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::SubexpressionContext* SmalltalkParser::subexpression() {
  SubexpressionContext *_localctx = _tracker.createInstance<SubexpressionContext>(_ctx, getState());
  enterRule(_localctx, 48, SmalltalkParser::RuleSubexpression);
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
    setState(354);
    match(SmalltalkParser::OPEN_PAREN);
    setState(356);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(355);
      ws();
    }
    setState(358);
    expression();
    setState(360);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(359);
      ws();
    }
    setState(362);
    match(SmalltalkParser::CLOSE_PAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LiteralContext ------------------------------------------------------------------

SmalltalkParser::LiteralContext::LiteralContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::RuntimeLiteralContext* SmalltalkParser::LiteralContext::runtimeLiteral() {
  return getRuleContext<SmalltalkParser::RuntimeLiteralContext>(0);
}

SmalltalkParser::ParsetimeLiteralContext* SmalltalkParser::LiteralContext::parsetimeLiteral() {
  return getRuleContext<SmalltalkParser::ParsetimeLiteralContext>(0);
}


size_t SmalltalkParser::LiteralContext::getRuleIndex() const {
  return SmalltalkParser::RuleLiteral;
}

void SmalltalkParser::LiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLiteral(this);
}

void SmalltalkParser::LiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLiteral(this);
}


antlrcpp::Any SmalltalkParser::LiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitLiteral(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::LiteralContext* SmalltalkParser::literal() {
  LiteralContext *_localctx = _tracker.createInstance<LiteralContext>(_ctx, getState());
  enterRule(_localctx, 50, SmalltalkParser::RuleLiteral);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(366);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SmalltalkParser::BLOCK_START:
      case SmalltalkParser::DYNDICT_START:
      case SmalltalkParser::DYNARR_START: {
        enterOuterAlt(_localctx, 1);
        setState(364);
        runtimeLiteral();
        break;
      }

      case SmalltalkParser::STRING:
      case SmalltalkParser::MINUS:
      case SmalltalkParser::RESERVED_WORD:
      case SmalltalkParser::HASH:
      case SmalltalkParser::DOLLAR:
      case SmalltalkParser::HEXNUM:
      case SmalltalkParser::INTNUM:
      case SmalltalkParser::CHARACTER_CONSTANT: {
        enterOuterAlt(_localctx, 2);
        setState(365);
        parsetimeLiteral();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RuntimeLiteralContext ------------------------------------------------------------------

SmalltalkParser::RuntimeLiteralContext::RuntimeLiteralContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::DynamicDictionaryContext* SmalltalkParser::RuntimeLiteralContext::dynamicDictionary() {
  return getRuleContext<SmalltalkParser::DynamicDictionaryContext>(0);
}

SmalltalkParser::DynamicArrayContext* SmalltalkParser::RuntimeLiteralContext::dynamicArray() {
  return getRuleContext<SmalltalkParser::DynamicArrayContext>(0);
}

SmalltalkParser::BlockContext* SmalltalkParser::RuntimeLiteralContext::block() {
  return getRuleContext<SmalltalkParser::BlockContext>(0);
}


size_t SmalltalkParser::RuntimeLiteralContext::getRuleIndex() const {
  return SmalltalkParser::RuleRuntimeLiteral;
}

void SmalltalkParser::RuntimeLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRuntimeLiteral(this);
}

void SmalltalkParser::RuntimeLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRuntimeLiteral(this);
}


antlrcpp::Any SmalltalkParser::RuntimeLiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitRuntimeLiteral(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::RuntimeLiteralContext* SmalltalkParser::runtimeLiteral() {
  RuntimeLiteralContext *_localctx = _tracker.createInstance<RuntimeLiteralContext>(_ctx, getState());
  enterRule(_localctx, 52, SmalltalkParser::RuleRuntimeLiteral);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(371);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SmalltalkParser::DYNDICT_START: {
        enterOuterAlt(_localctx, 1);
        setState(368);
        dynamicDictionary();
        break;
      }

      case SmalltalkParser::DYNARR_START: {
        enterOuterAlt(_localctx, 2);
        setState(369);
        dynamicArray();
        break;
      }

      case SmalltalkParser::BLOCK_START: {
        enterOuterAlt(_localctx, 3);
        setState(370);
        block();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockContext ------------------------------------------------------------------

SmalltalkParser::BlockContext::BlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::BlockContext::BLOCK_START() {
  return getToken(SmalltalkParser::BLOCK_START, 0);
}

tree::TerminalNode* SmalltalkParser::BlockContext::BLOCK_END() {
  return getToken(SmalltalkParser::BLOCK_END, 0);
}

tree::TerminalNode* SmalltalkParser::BlockContext::PIPE3() {
  return getToken(SmalltalkParser::PIPE3, 0);
}

SmalltalkParser::BlockParamListContext* SmalltalkParser::BlockContext::blockParamList() {
  return getRuleContext<SmalltalkParser::BlockParamListContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::BlockContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::BlockContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

SmalltalkParser::SequenceContext* SmalltalkParser::BlockContext::sequence() {
  return getRuleContext<SmalltalkParser::SequenceContext>(0);
}

tree::TerminalNode* SmalltalkParser::BlockContext::PIPE() {
  return getToken(SmalltalkParser::PIPE, 0);
}

SmalltalkParser::TempsContext* SmalltalkParser::BlockContext::temps() {
  return getRuleContext<SmalltalkParser::TempsContext>(0);
}

tree::TerminalNode* SmalltalkParser::BlockContext::PIPE2() {
  return getToken(SmalltalkParser::PIPE2, 0);
}


size_t SmalltalkParser::BlockContext::getRuleIndex() const {
  return SmalltalkParser::RuleBlock;
}

void SmalltalkParser::BlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBlock(this);
}

void SmalltalkParser::BlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBlock(this);
}


antlrcpp::Any SmalltalkParser::BlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitBlock(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::BlockContext* SmalltalkParser::block() {
  BlockContext *_localctx = _tracker.createInstance<BlockContext>(_ctx, getState());
  enterRule(_localctx, 54, SmalltalkParser::RuleBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(411);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 75, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(373);
      match(SmalltalkParser::BLOCK_START);
      setState(375);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 64, _ctx)) {
      case 1: {
        setState(374);
        blockParamList();
        break;
      }

      default:
        break;
      }
      setState(378);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 65, _ctx)) {
      case 1: {
        setState(377);
        ws();
        break;
      }

      default:
        break;
      }
      setState(390);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case SmalltalkParser::SEPARATOR:
        case SmalltalkParser::STRING:
        case SmalltalkParser::PRIMITIVE:
        case SmalltalkParser::COMMENT:
        case SmalltalkParser::BLOCK_START:
        case SmalltalkParser::BLOCK_END:
        case SmalltalkParser::OPEN_PAREN:
        case SmalltalkParser::PIPE2:
        case SmalltalkParser::PIPE:
        case SmalltalkParser::MINUS:
        case SmalltalkParser::RESERVED_WORD:
        case SmalltalkParser::IDENTIFIER:
        case SmalltalkParser::IDENTIFIER2:
        case SmalltalkParser::CARROT:
        case SmalltalkParser::UNDERSCORE:
        case SmalltalkParser::UNDERSCORE2:
        case SmalltalkParser::HASH:
        case SmalltalkParser::DOLLAR:
        case SmalltalkParser::HEXNUM:
        case SmalltalkParser::INTNUM:
        case SmalltalkParser::DYNDICT_START:
        case SmalltalkParser::DYNARR_START:
        case SmalltalkParser::CHARACTER_CONSTANT: {
          setState(381);
          _errHandler->sync(this);

          switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 66, _ctx)) {
          case 1: {
            setState(380);
            match(SmalltalkParser::PIPE);
            break;
          }

          default:
            break;
          }
          setState(384);
          _errHandler->sync(this);

          switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 67, _ctx)) {
          case 1: {
            setState(383);
            ws();
            break;
          }

          default:
            break;
          }
          setState(387);
          _errHandler->sync(this);

          switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 68, _ctx)) {
          case 1: {
            setState(386);
            temps();
            break;
          }

          default:
            break;
          }
          break;
        }

        case SmalltalkParser::PIPE3: {
          setState(389);
          match(SmalltalkParser::PIPE3);
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(393);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 70, _ctx)) {
      case 1: {
        setState(392);
        ws();
        break;
      }

      default:
        break;
      }
      setState(396);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << SmalltalkParser::SEPARATOR)
        | (1ULL << SmalltalkParser::STRING)
        | (1ULL << SmalltalkParser::PRIMITIVE)
        | (1ULL << SmalltalkParser::COMMENT)
        | (1ULL << SmalltalkParser::BLOCK_START)
        | (1ULL << SmalltalkParser::OPEN_PAREN)
        | (1ULL << SmalltalkParser::PIPE2)
        | (1ULL << SmalltalkParser::PIPE)
        | (1ULL << SmalltalkParser::MINUS)
        | (1ULL << SmalltalkParser::RESERVED_WORD)
        | (1ULL << SmalltalkParser::IDENTIFIER)
        | (1ULL << SmalltalkParser::IDENTIFIER2)
        | (1ULL << SmalltalkParser::CARROT)
        | (1ULL << SmalltalkParser::UNDERSCORE)
        | (1ULL << SmalltalkParser::UNDERSCORE2)
        | (1ULL << SmalltalkParser::HASH)
        | (1ULL << SmalltalkParser::DOLLAR)
        | (1ULL << SmalltalkParser::HEXNUM)
        | (1ULL << SmalltalkParser::INTNUM)
        | (1ULL << SmalltalkParser::DYNDICT_START)
        | (1ULL << SmalltalkParser::DYNARR_START)
        | (1ULL << SmalltalkParser::CHARACTER_CONSTANT))) != 0)) {
        setState(395);
        sequence();
      }
      setState(398);
      match(SmalltalkParser::BLOCK_END);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(399);
      match(SmalltalkParser::BLOCK_START);
      setState(401);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(400);
        ws();
      }
      setState(403);
      match(SmalltalkParser::PIPE2);
      setState(405);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 73, _ctx)) {
      case 1: {
        setState(404);
        ws();
        break;
      }

      default:
        break;
      }
      setState(408);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << SmalltalkParser::SEPARATOR)
        | (1ULL << SmalltalkParser::STRING)
        | (1ULL << SmalltalkParser::PRIMITIVE)
        | (1ULL << SmalltalkParser::COMMENT)
        | (1ULL << SmalltalkParser::BLOCK_START)
        | (1ULL << SmalltalkParser::OPEN_PAREN)
        | (1ULL << SmalltalkParser::PIPE2)
        | (1ULL << SmalltalkParser::PIPE)
        | (1ULL << SmalltalkParser::MINUS)
        | (1ULL << SmalltalkParser::RESERVED_WORD)
        | (1ULL << SmalltalkParser::IDENTIFIER)
        | (1ULL << SmalltalkParser::IDENTIFIER2)
        | (1ULL << SmalltalkParser::CARROT)
        | (1ULL << SmalltalkParser::UNDERSCORE)
        | (1ULL << SmalltalkParser::UNDERSCORE2)
        | (1ULL << SmalltalkParser::HASH)
        | (1ULL << SmalltalkParser::DOLLAR)
        | (1ULL << SmalltalkParser::HEXNUM)
        | (1ULL << SmalltalkParser::INTNUM)
        | (1ULL << SmalltalkParser::DYNDICT_START)
        | (1ULL << SmalltalkParser::DYNARR_START)
        | (1ULL << SmalltalkParser::CHARACTER_CONSTANT))) != 0)) {
        setState(407);
        sequence();
      }
      setState(410);
      match(SmalltalkParser::BLOCK_END);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockParamListContext ------------------------------------------------------------------

SmalltalkParser::BlockParamListContext::BlockParamListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> SmalltalkParser::BlockParamListContext::BLOCK_PARAM() {
  return getTokens(SmalltalkParser::BLOCK_PARAM);
}

tree::TerminalNode* SmalltalkParser::BlockParamListContext::BLOCK_PARAM(size_t i) {
  return getToken(SmalltalkParser::BLOCK_PARAM, i);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::BlockParamListContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::BlockParamListContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}


size_t SmalltalkParser::BlockParamListContext::getRuleIndex() const {
  return SmalltalkParser::RuleBlockParamList;
}

void SmalltalkParser::BlockParamListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBlockParamList(this);
}

void SmalltalkParser::BlockParamListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBlockParamList(this);
}


antlrcpp::Any SmalltalkParser::BlockParamListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitBlockParamList(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::BlockParamListContext* SmalltalkParser::blockParamList() {
  BlockParamListContext *_localctx = _tracker.createInstance<BlockParamListContext>(_ctx, getState());
  enterRule(_localctx, 56, SmalltalkParser::RuleBlockParamList);
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
    setState(417); 
    _errHandler->sync(this);
    alt = 1;
    do {
      switch (alt) {
        case 1: {
              setState(414);
              _errHandler->sync(this);

              _la = _input->LA(1);
              if (_la == SmalltalkParser::SEPARATOR

              || _la == SmalltalkParser::COMMENT) {
                setState(413);
                ws();
              }
              setState(416);
              match(SmalltalkParser::BLOCK_PARAM);
              break;
            }

      default:
        throw NoViableAltException(this);
      }
      setState(419); 
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 77, _ctx);
    } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DynamicDictionaryContext ------------------------------------------------------------------

SmalltalkParser::DynamicDictionaryContext::DynamicDictionaryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::DynamicDictionaryContext::DYNDICT_START() {
  return getToken(SmalltalkParser::DYNDICT_START, 0);
}

tree::TerminalNode* SmalltalkParser::DynamicDictionaryContext::DYNARR_END() {
  return getToken(SmalltalkParser::DYNARR_END, 0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::DynamicDictionaryContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::DynamicDictionaryContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

SmalltalkParser::ExpressionsContext* SmalltalkParser::DynamicDictionaryContext::expressions() {
  return getRuleContext<SmalltalkParser::ExpressionsContext>(0);
}


size_t SmalltalkParser::DynamicDictionaryContext::getRuleIndex() const {
  return SmalltalkParser::RuleDynamicDictionary;
}

void SmalltalkParser::DynamicDictionaryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDynamicDictionary(this);
}

void SmalltalkParser::DynamicDictionaryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDynamicDictionary(this);
}


antlrcpp::Any SmalltalkParser::DynamicDictionaryContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitDynamicDictionary(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::DynamicDictionaryContext* SmalltalkParser::dynamicDictionary() {
  DynamicDictionaryContext *_localctx = _tracker.createInstance<DynamicDictionaryContext>(_ctx, getState());
  enterRule(_localctx, 58, SmalltalkParser::RuleDynamicDictionary);
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
    setState(421);
    match(SmalltalkParser::DYNDICT_START);
    setState(423);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 78, _ctx)) {
    case 1: {
      setState(422);
      ws();
      break;
    }

    default:
      break;
    }
    setState(426);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SmalltalkParser::STRING)
      | (1ULL << SmalltalkParser::PRIMITIVE)
      | (1ULL << SmalltalkParser::BLOCK_START)
      | (1ULL << SmalltalkParser::OPEN_PAREN)
      | (1ULL << SmalltalkParser::MINUS)
      | (1ULL << SmalltalkParser::RESERVED_WORD)
      | (1ULL << SmalltalkParser::IDENTIFIER)
      | (1ULL << SmalltalkParser::IDENTIFIER2)
      | (1ULL << SmalltalkParser::UNDERSCORE)
      | (1ULL << SmalltalkParser::UNDERSCORE2)
      | (1ULL << SmalltalkParser::HASH)
      | (1ULL << SmalltalkParser::DOLLAR)
      | (1ULL << SmalltalkParser::HEXNUM)
      | (1ULL << SmalltalkParser::INTNUM)
      | (1ULL << SmalltalkParser::DYNDICT_START)
      | (1ULL << SmalltalkParser::DYNARR_START)
      | (1ULL << SmalltalkParser::CHARACTER_CONSTANT))) != 0)) {
      setState(425);
      expressions();
    }
    setState(429);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(428);
      ws();
    }
    setState(431);
    match(SmalltalkParser::DYNARR_END);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DynamicArrayContext ------------------------------------------------------------------

SmalltalkParser::DynamicArrayContext::DynamicArrayContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::DynamicArrayContext::DYNARR_START() {
  return getToken(SmalltalkParser::DYNARR_START, 0);
}

tree::TerminalNode* SmalltalkParser::DynamicArrayContext::DYNARR_END() {
  return getToken(SmalltalkParser::DYNARR_END, 0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::DynamicArrayContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::DynamicArrayContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

SmalltalkParser::ExpressionsContext* SmalltalkParser::DynamicArrayContext::expressions() {
  return getRuleContext<SmalltalkParser::ExpressionsContext>(0);
}


size_t SmalltalkParser::DynamicArrayContext::getRuleIndex() const {
  return SmalltalkParser::RuleDynamicArray;
}

void SmalltalkParser::DynamicArrayContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDynamicArray(this);
}

void SmalltalkParser::DynamicArrayContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDynamicArray(this);
}


antlrcpp::Any SmalltalkParser::DynamicArrayContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitDynamicArray(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::DynamicArrayContext* SmalltalkParser::dynamicArray() {
  DynamicArrayContext *_localctx = _tracker.createInstance<DynamicArrayContext>(_ctx, getState());
  enterRule(_localctx, 60, SmalltalkParser::RuleDynamicArray);
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
    setState(433);
    match(SmalltalkParser::DYNARR_START);
    setState(435);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 81, _ctx)) {
    case 1: {
      setState(434);
      ws();
      break;
    }

    default:
      break;
    }
    setState(438);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SmalltalkParser::STRING)
      | (1ULL << SmalltalkParser::PRIMITIVE)
      | (1ULL << SmalltalkParser::BLOCK_START)
      | (1ULL << SmalltalkParser::OPEN_PAREN)
      | (1ULL << SmalltalkParser::MINUS)
      | (1ULL << SmalltalkParser::RESERVED_WORD)
      | (1ULL << SmalltalkParser::IDENTIFIER)
      | (1ULL << SmalltalkParser::IDENTIFIER2)
      | (1ULL << SmalltalkParser::UNDERSCORE)
      | (1ULL << SmalltalkParser::UNDERSCORE2)
      | (1ULL << SmalltalkParser::HASH)
      | (1ULL << SmalltalkParser::DOLLAR)
      | (1ULL << SmalltalkParser::HEXNUM)
      | (1ULL << SmalltalkParser::INTNUM)
      | (1ULL << SmalltalkParser::DYNDICT_START)
      | (1ULL << SmalltalkParser::DYNARR_START)
      | (1ULL << SmalltalkParser::CHARACTER_CONSTANT))) != 0)) {
      setState(437);
      expressions();
    }
    setState(441);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(440);
      ws();
    }
    setState(443);
    match(SmalltalkParser::DYNARR_END);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ParsetimeLiteralContext ------------------------------------------------------------------

SmalltalkParser::ParsetimeLiteralContext::ParsetimeLiteralContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::PseudoVariableContext* SmalltalkParser::ParsetimeLiteralContext::pseudoVariable() {
  return getRuleContext<SmalltalkParser::PseudoVariableContext>(0);
}

SmalltalkParser::NumberContext* SmalltalkParser::ParsetimeLiteralContext::number() {
  return getRuleContext<SmalltalkParser::NumberContext>(0);
}

SmalltalkParser::CharConstantContext* SmalltalkParser::ParsetimeLiteralContext::charConstant() {
  return getRuleContext<SmalltalkParser::CharConstantContext>(0);
}

SmalltalkParser::LiteralArrayContext* SmalltalkParser::ParsetimeLiteralContext::literalArray() {
  return getRuleContext<SmalltalkParser::LiteralArrayContext>(0);
}

SmalltalkParser::StringContext* SmalltalkParser::ParsetimeLiteralContext::string() {
  return getRuleContext<SmalltalkParser::StringContext>(0);
}

SmalltalkParser::SymbolContext* SmalltalkParser::ParsetimeLiteralContext::symbol() {
  return getRuleContext<SmalltalkParser::SymbolContext>(0);
}

SmalltalkParser::ByteLiteralArrayContext* SmalltalkParser::ParsetimeLiteralContext::byteLiteralArray() {
  return getRuleContext<SmalltalkParser::ByteLiteralArrayContext>(0);
}


size_t SmalltalkParser::ParsetimeLiteralContext::getRuleIndex() const {
  return SmalltalkParser::RuleParsetimeLiteral;
}

void SmalltalkParser::ParsetimeLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParsetimeLiteral(this);
}

void SmalltalkParser::ParsetimeLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParsetimeLiteral(this);
}


antlrcpp::Any SmalltalkParser::ParsetimeLiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitParsetimeLiteral(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ParsetimeLiteralContext* SmalltalkParser::parsetimeLiteral() {
  ParsetimeLiteralContext *_localctx = _tracker.createInstance<ParsetimeLiteralContext>(_ctx, getState());
  enterRule(_localctx, 62, SmalltalkParser::RuleParsetimeLiteral);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(452);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 84, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(445);
      pseudoVariable();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(446);
      number();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(447);
      charConstant();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(448);
      literalArray();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(449);
      string();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(450);
      symbol();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(451);
      byteLiteralArray();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NumberContext ------------------------------------------------------------------

SmalltalkParser::NumberContext::NumberContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::NumberExpContext* SmalltalkParser::NumberContext::numberExp() {
  return getRuleContext<SmalltalkParser::NumberExpContext>(0);
}

SmalltalkParser::Hex_Context* SmalltalkParser::NumberContext::hex_() {
  return getRuleContext<SmalltalkParser::Hex_Context>(0);
}

SmalltalkParser::StFloatContext* SmalltalkParser::NumberContext::stFloat() {
  return getRuleContext<SmalltalkParser::StFloatContext>(0);
}

SmalltalkParser::StIntegerContext* SmalltalkParser::NumberContext::stInteger() {
  return getRuleContext<SmalltalkParser::StIntegerContext>(0);
}


size_t SmalltalkParser::NumberContext::getRuleIndex() const {
  return SmalltalkParser::RuleNumber;
}

void SmalltalkParser::NumberContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNumber(this);
}

void SmalltalkParser::NumberContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNumber(this);
}


antlrcpp::Any SmalltalkParser::NumberContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitNumber(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::NumberContext* SmalltalkParser::number() {
  NumberContext *_localctx = _tracker.createInstance<NumberContext>(_ctx, getState());
  enterRule(_localctx, 64, SmalltalkParser::RuleNumber);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(458);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 85, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(454);
      numberExp();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(455);
      hex_();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(456);
      stFloat();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(457);
      stInteger();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NumberExpContext ------------------------------------------------------------------

SmalltalkParser::NumberExpContext::NumberExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::NumberExpContext::EXP() {
  return getToken(SmalltalkParser::EXP, 0);
}

std::vector<SmalltalkParser::StIntegerContext *> SmalltalkParser::NumberExpContext::stInteger() {
  return getRuleContexts<SmalltalkParser::StIntegerContext>();
}

SmalltalkParser::StIntegerContext* SmalltalkParser::NumberExpContext::stInteger(size_t i) {
  return getRuleContext<SmalltalkParser::StIntegerContext>(i);
}

SmalltalkParser::StFloatContext* SmalltalkParser::NumberExpContext::stFloat() {
  return getRuleContext<SmalltalkParser::StFloatContext>(0);
}


size_t SmalltalkParser::NumberExpContext::getRuleIndex() const {
  return SmalltalkParser::RuleNumberExp;
}

void SmalltalkParser::NumberExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNumberExp(this);
}

void SmalltalkParser::NumberExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNumberExp(this);
}


antlrcpp::Any SmalltalkParser::NumberExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitNumberExp(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::NumberExpContext* SmalltalkParser::numberExp() {
  NumberExpContext *_localctx = _tracker.createInstance<NumberExpContext>(_ctx, getState());
  enterRule(_localctx, 66, SmalltalkParser::RuleNumberExp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(462);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 86, _ctx)) {
    case 1: {
      setState(460);
      stFloat();
      break;
    }

    case 2: {
      setState(461);
      stInteger();
      break;
    }

    default:
      break;
    }
    setState(464);
    match(SmalltalkParser::EXP);
    setState(465);
    stInteger();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CharConstantContext ------------------------------------------------------------------

SmalltalkParser::CharConstantContext::CharConstantContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::CharConstantContext::CHARACTER_CONSTANT() {
  return getToken(SmalltalkParser::CHARACTER_CONSTANT, 0);
}

tree::TerminalNode* SmalltalkParser::CharConstantContext::DOLLAR() {
  return getToken(SmalltalkParser::DOLLAR, 0);
}


size_t SmalltalkParser::CharConstantContext::getRuleIndex() const {
  return SmalltalkParser::RuleCharConstant;
}

void SmalltalkParser::CharConstantContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCharConstant(this);
}

void SmalltalkParser::CharConstantContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCharConstant(this);
}


antlrcpp::Any SmalltalkParser::CharConstantContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitCharConstant(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::CharConstantContext* SmalltalkParser::charConstant() {
  CharConstantContext *_localctx = _tracker.createInstance<CharConstantContext>(_ctx, getState());
  enterRule(_localctx, 68, SmalltalkParser::RuleCharConstant);
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
    setState(467);
    _la = _input->LA(1);
    if (!(_la == SmalltalkParser::DOLLAR

    || _la == SmalltalkParser::CHARACTER_CONSTANT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Hex_Context ------------------------------------------------------------------

SmalltalkParser::Hex_Context::Hex_Context(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::Hex_Context::HEXNUM() {
  return getToken(SmalltalkParser::HEXNUM, 0);
}

tree::TerminalNode* SmalltalkParser::Hex_Context::MINUS() {
  return getToken(SmalltalkParser::MINUS, 0);
}


size_t SmalltalkParser::Hex_Context::getRuleIndex() const {
  return SmalltalkParser::RuleHex_;
}

void SmalltalkParser::Hex_Context::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterHex_(this);
}

void SmalltalkParser::Hex_Context::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitHex_(this);
}


antlrcpp::Any SmalltalkParser::Hex_Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitHex_(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::Hex_Context* SmalltalkParser::hex_() {
  Hex_Context *_localctx = _tracker.createInstance<Hex_Context>(_ctx, getState());
  enterRule(_localctx, 70, SmalltalkParser::RuleHex_);
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
    setState(470);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::MINUS) {
      setState(469);
      match(SmalltalkParser::MINUS);
    }
    setState(472);
    match(SmalltalkParser::HEXNUM);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StIntegerContext ------------------------------------------------------------------

SmalltalkParser::StIntegerContext::StIntegerContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::StIntegerContext::INTNUM() {
  return getToken(SmalltalkParser::INTNUM, 0);
}

tree::TerminalNode* SmalltalkParser::StIntegerContext::MINUS() {
  return getToken(SmalltalkParser::MINUS, 0);
}


size_t SmalltalkParser::StIntegerContext::getRuleIndex() const {
  return SmalltalkParser::RuleStInteger;
}

void SmalltalkParser::StIntegerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStInteger(this);
}

void SmalltalkParser::StIntegerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStInteger(this);
}


antlrcpp::Any SmalltalkParser::StIntegerContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitStInteger(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::StIntegerContext* SmalltalkParser::stInteger() {
  StIntegerContext *_localctx = _tracker.createInstance<StIntegerContext>(_ctx, getState());
  enterRule(_localctx, 72, SmalltalkParser::RuleStInteger);
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
    setState(475);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::MINUS) {
      setState(474);
      match(SmalltalkParser::MINUS);
    }
    setState(477);
    match(SmalltalkParser::INTNUM);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StFloatContext ------------------------------------------------------------------

SmalltalkParser::StFloatContext::StFloatContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> SmalltalkParser::StFloatContext::INTNUM() {
  return getTokens(SmalltalkParser::INTNUM);
}

tree::TerminalNode* SmalltalkParser::StFloatContext::INTNUM(size_t i) {
  return getToken(SmalltalkParser::INTNUM, i);
}

tree::TerminalNode* SmalltalkParser::StFloatContext::PERIOD() {
  return getToken(SmalltalkParser::PERIOD, 0);
}

tree::TerminalNode* SmalltalkParser::StFloatContext::MINUS() {
  return getToken(SmalltalkParser::MINUS, 0);
}


size_t SmalltalkParser::StFloatContext::getRuleIndex() const {
  return SmalltalkParser::RuleStFloat;
}

void SmalltalkParser::StFloatContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStFloat(this);
}

void SmalltalkParser::StFloatContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStFloat(this);
}


antlrcpp::Any SmalltalkParser::StFloatContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitStFloat(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::StFloatContext* SmalltalkParser::stFloat() {
  StFloatContext *_localctx = _tracker.createInstance<StFloatContext>(_ctx, getState());
  enterRule(_localctx, 74, SmalltalkParser::RuleStFloat);
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
    setState(480);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::MINUS) {
      setState(479);
      match(SmalltalkParser::MINUS);
    }
    setState(482);
    match(SmalltalkParser::INTNUM);
    setState(483);
    match(SmalltalkParser::PERIOD);
    setState(484);
    match(SmalltalkParser::INTNUM);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PseudoVariableContext ------------------------------------------------------------------

SmalltalkParser::PseudoVariableContext::PseudoVariableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::PseudoVariableContext::RESERVED_WORD() {
  return getToken(SmalltalkParser::RESERVED_WORD, 0);
}


size_t SmalltalkParser::PseudoVariableContext::getRuleIndex() const {
  return SmalltalkParser::RulePseudoVariable;
}

void SmalltalkParser::PseudoVariableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPseudoVariable(this);
}

void SmalltalkParser::PseudoVariableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPseudoVariable(this);
}


antlrcpp::Any SmalltalkParser::PseudoVariableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitPseudoVariable(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::PseudoVariableContext* SmalltalkParser::pseudoVariable() {
  PseudoVariableContext *_localctx = _tracker.createInstance<PseudoVariableContext>(_ctx, getState());
  enterRule(_localctx, 76, SmalltalkParser::RulePseudoVariable);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(486);
    match(SmalltalkParser::RESERVED_WORD);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StringContext ------------------------------------------------------------------

SmalltalkParser::StringContext::StringContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::StringContext::STRING() {
  return getToken(SmalltalkParser::STRING, 0);
}


size_t SmalltalkParser::StringContext::getRuleIndex() const {
  return SmalltalkParser::RuleString;
}

void SmalltalkParser::StringContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterString(this);
}

void SmalltalkParser::StringContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitString(this);
}


antlrcpp::Any SmalltalkParser::StringContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitString(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::StringContext* SmalltalkParser::string() {
  StringContext *_localctx = _tracker.createInstance<StringContext>(_ctx, getState());
  enterRule(_localctx, 78, SmalltalkParser::RuleString);

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
    match(SmalltalkParser::STRING);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrimitiveContext ------------------------------------------------------------------

SmalltalkParser::PrimitiveContext::PrimitiveContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::PrimitiveContext::PRIMITIVE() {
  return getToken(SmalltalkParser::PRIMITIVE, 0);
}


size_t SmalltalkParser::PrimitiveContext::getRuleIndex() const {
  return SmalltalkParser::RulePrimitive;
}

void SmalltalkParser::PrimitiveContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrimitive(this);
}

void SmalltalkParser::PrimitiveContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrimitive(this);
}


antlrcpp::Any SmalltalkParser::PrimitiveContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitPrimitive(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::PrimitiveContext* SmalltalkParser::primitive() {
  PrimitiveContext *_localctx = _tracker.createInstance<PrimitiveContext>(_ctx, getState());
  enterRule(_localctx, 80, SmalltalkParser::RulePrimitive);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(490);
    match(SmalltalkParser::PRIMITIVE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SymbolContext ------------------------------------------------------------------

SmalltalkParser::SymbolContext::SymbolContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::SymbolContext::HASH() {
  return getToken(SmalltalkParser::HASH, 0);
}

SmalltalkParser::BareSymbolContext* SmalltalkParser::SymbolContext::bareSymbol() {
  return getRuleContext<SmalltalkParser::BareSymbolContext>(0);
}

SmalltalkParser::WsContext* SmalltalkParser::SymbolContext::ws() {
  return getRuleContext<SmalltalkParser::WsContext>(0);
}

tree::TerminalNode* SmalltalkParser::SymbolContext::SPECIAL_UNDERLINE_IDENTIFIER() {
  return getToken(SmalltalkParser::SPECIAL_UNDERLINE_IDENTIFIER, 0);
}


size_t SmalltalkParser::SymbolContext::getRuleIndex() const {
  return SmalltalkParser::RuleSymbol;
}

void SmalltalkParser::SymbolContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSymbol(this);
}

void SmalltalkParser::SymbolContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSymbol(this);
}


antlrcpp::Any SmalltalkParser::SymbolContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitSymbol(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::SymbolContext* SmalltalkParser::symbol() {
  SymbolContext *_localctx = _tracker.createInstance<SymbolContext>(_ctx, getState());
  enterRule(_localctx, 82, SmalltalkParser::RuleSymbol);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(501);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 92, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(492);
      match(SmalltalkParser::HASH);
      setState(494);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(493);
        ws();
      }
      setState(496);
      bareSymbol();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(497);
      match(SmalltalkParser::HASH);
      setState(499);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SPECIAL_UNDERLINE_IDENTIFIER) {
        setState(498);
        match(SmalltalkParser::SPECIAL_UNDERLINE_IDENTIFIER);
      }
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BareSymbolContext ------------------------------------------------------------------

SmalltalkParser::BareSymbolContext::BareSymbolContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::IDENTIFIER() {
  return getToken(SmalltalkParser::IDENTIFIER, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::UNDERSCORE() {
  return getToken(SmalltalkParser::UNDERSCORE, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::MINUS() {
  return getToken(SmalltalkParser::MINUS, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::PIPE() {
  return getToken(SmalltalkParser::PIPE, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::PIPE2() {
  return getToken(SmalltalkParser::PIPE2, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::PIPE3() {
  return getToken(SmalltalkParser::PIPE3, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::AMP() {
  return getToken(SmalltalkParser::AMP, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::BINARY_SELECTOR() {
  return getToken(SmalltalkParser::BINARY_SELECTOR, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::NEQ() {
  return getToken(SmalltalkParser::NEQ, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::LTE() {
  return getToken(SmalltalkParser::LTE, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::GTE() {
  return getToken(SmalltalkParser::GTE, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::LT() {
  return getToken(SmalltalkParser::LT, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::GT() {
  return getToken(SmalltalkParser::GT, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::GGT() {
  return getToken(SmalltalkParser::GGT, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::LLT() {
  return getToken(SmalltalkParser::LLT, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::OPEN_PAREN() {
  return getToken(SmalltalkParser::OPEN_PAREN, 0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::CLOSE_PAREN() {
  return getToken(SmalltalkParser::CLOSE_PAREN, 0);
}

std::vector<tree::TerminalNode *> SmalltalkParser::BareSymbolContext::KEYWORD() {
  return getTokens(SmalltalkParser::KEYWORD);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::KEYWORD(size_t i) {
  return getToken(SmalltalkParser::KEYWORD, i);
}

SmalltalkParser::StringContext* SmalltalkParser::BareSymbolContext::string() {
  return getRuleContext<SmalltalkParser::StringContext>(0);
}

tree::TerminalNode* SmalltalkParser::BareSymbolContext::RESERVED_WORD() {
  return getToken(SmalltalkParser::RESERVED_WORD, 0);
}


size_t SmalltalkParser::BareSymbolContext::getRuleIndex() const {
  return SmalltalkParser::RuleBareSymbol;
}

void SmalltalkParser::BareSymbolContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBareSymbol(this);
}

void SmalltalkParser::BareSymbolContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBareSymbol(this);
}


antlrcpp::Any SmalltalkParser::BareSymbolContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitBareSymbol(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::BareSymbolContext* SmalltalkParser::bareSymbol() {
  BareSymbolContext *_localctx = _tracker.createInstance<BareSymbolContext>(_ctx, getState());
  enterRule(_localctx, 84, SmalltalkParser::RuleBareSymbol);
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
    setState(511);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SmalltalkParser::CLOSE_PAREN:
      case SmalltalkParser::OPEN_PAREN:
      case SmalltalkParser::PIPE3:
      case SmalltalkParser::PIPE2:
      case SmalltalkParser::PIPE:
      case SmalltalkParser::LLT:
      case SmalltalkParser::GGT:
      case SmalltalkParser::LTE:
      case SmalltalkParser::GTE:
      case SmalltalkParser::NEQ:
      case SmalltalkParser::LT:
      case SmalltalkParser::GT:
      case SmalltalkParser::AMP:
      case SmalltalkParser::MINUS:
      case SmalltalkParser::BINARY_SELECTOR:
      case SmalltalkParser::IDENTIFIER:
      case SmalltalkParser::UNDERSCORE: {
        enterOuterAlt(_localctx, 1);
        setState(503);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << SmalltalkParser::CLOSE_PAREN)
          | (1ULL << SmalltalkParser::OPEN_PAREN)
          | (1ULL << SmalltalkParser::PIPE3)
          | (1ULL << SmalltalkParser::PIPE2)
          | (1ULL << SmalltalkParser::PIPE)
          | (1ULL << SmalltalkParser::LLT)
          | (1ULL << SmalltalkParser::GGT)
          | (1ULL << SmalltalkParser::LTE)
          | (1ULL << SmalltalkParser::GTE)
          | (1ULL << SmalltalkParser::NEQ)
          | (1ULL << SmalltalkParser::LT)
          | (1ULL << SmalltalkParser::GT)
          | (1ULL << SmalltalkParser::AMP)
          | (1ULL << SmalltalkParser::MINUS)
          | (1ULL << SmalltalkParser::BINARY_SELECTOR)
          | (1ULL << SmalltalkParser::IDENTIFIER)
          | (1ULL << SmalltalkParser::UNDERSCORE))) != 0))) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        break;
      }

      case SmalltalkParser::KEYWORD: {
        enterOuterAlt(_localctx, 2);
        setState(505); 
        _errHandler->sync(this);
        alt = 1;
        do {
          switch (alt) {
            case 1: {
                  setState(504);
                  match(SmalltalkParser::KEYWORD);
                  break;
                }

          default:
            throw NoViableAltException(this);
          }
          setState(507); 
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 93, _ctx);
        } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
        break;
      }

      case SmalltalkParser::STRING: {
        enterOuterAlt(_localctx, 3);
        setState(509);
        string();
        break;
      }

      case SmalltalkParser::RESERVED_WORD: {
        enterOuterAlt(_localctx, 4);
        setState(510);
        match(SmalltalkParser::RESERVED_WORD);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LiteralArrayContext ------------------------------------------------------------------

SmalltalkParser::LiteralArrayContext::LiteralArrayContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::LiteralArrayRestContext* SmalltalkParser::LiteralArrayContext::literalArrayRest() {
  return getRuleContext<SmalltalkParser::LiteralArrayRestContext>(0);
}

tree::TerminalNode* SmalltalkParser::LiteralArrayContext::HASH() {
  return getToken(SmalltalkParser::HASH, 0);
}

tree::TerminalNode* SmalltalkParser::LiteralArrayContext::OPEN_PAREN() {
  return getToken(SmalltalkParser::OPEN_PAREN, 0);
}

tree::TerminalNode* SmalltalkParser::LiteralArrayContext::BLOCK_START() {
  return getToken(SmalltalkParser::BLOCK_START, 0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::LiteralArrayContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::LiteralArrayContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

SmalltalkParser::DynamicDictionaryContext* SmalltalkParser::LiteralArrayContext::dynamicDictionary() {
  return getRuleContext<SmalltalkParser::DynamicDictionaryContext>(0);
}


size_t SmalltalkParser::LiteralArrayContext::getRuleIndex() const {
  return SmalltalkParser::RuleLiteralArray;
}

void SmalltalkParser::LiteralArrayContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLiteralArray(this);
}

void SmalltalkParser::LiteralArrayContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLiteralArray(this);
}


antlrcpp::Any SmalltalkParser::LiteralArrayContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitLiteralArray(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::LiteralArrayContext* SmalltalkParser::literalArray() {
  LiteralArrayContext *_localctx = _tracker.createInstance<LiteralArrayContext>(_ctx, getState());
  enterRule(_localctx, 86, SmalltalkParser::RuleLiteralArray);
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
    setState(523);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 97, _ctx)) {
    case 1: {
      setState(513);
      match(SmalltalkParser::HASH);
      setState(515);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(514);
        ws();
      }
      setState(517);
      match(SmalltalkParser::OPEN_PAREN);
      break;
    }

    case 2: {
      setState(518);
      match(SmalltalkParser::HASH);
      setState(520);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(519);
        ws();
      }
      setState(522);
      match(SmalltalkParser::BLOCK_START);
      break;
    }

    default:
      break;
    }
    setState(526);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 98, _ctx)) {
    case 1: {
      setState(525);
      ws();
      break;
    }

    default:
      break;
    }
    setState(529);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 99, _ctx)) {
    case 1: {
      setState(528);
      dynamicDictionary();
      break;
    }

    default:
      break;
    }
    setState(532);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 100, _ctx)) {
    case 1: {
      setState(531);
      ws();
      break;
    }

    default:
      break;
    }
    setState(534);
    literalArrayRest();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LiteralArrayRestContext ------------------------------------------------------------------

SmalltalkParser::LiteralArrayRestContext::LiteralArrayRestContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::LiteralArrayRestContext::CLOSE_PAREN() {
  return getToken(SmalltalkParser::CLOSE_PAREN, 0);
}

tree::TerminalNode* SmalltalkParser::LiteralArrayRestContext::BLOCK_END() {
  return getToken(SmalltalkParser::BLOCK_END, 0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::LiteralArrayRestContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::LiteralArrayRestContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

std::vector<SmalltalkParser::LiteralContext *> SmalltalkParser::LiteralArrayRestContext::literal() {
  return getRuleContexts<SmalltalkParser::LiteralContext>();
}

SmalltalkParser::LiteralContext* SmalltalkParser::LiteralArrayRestContext::literal(size_t i) {
  return getRuleContext<SmalltalkParser::LiteralContext>(i);
}

std::vector<SmalltalkParser::BareLiteralArrayContext *> SmalltalkParser::LiteralArrayRestContext::bareLiteralArray() {
  return getRuleContexts<SmalltalkParser::BareLiteralArrayContext>();
}

SmalltalkParser::BareLiteralArrayContext* SmalltalkParser::LiteralArrayRestContext::bareLiteralArray(size_t i) {
  return getRuleContext<SmalltalkParser::BareLiteralArrayContext>(i);
}

std::vector<SmalltalkParser::BareSymbolContext *> SmalltalkParser::LiteralArrayRestContext::bareSymbol() {
  return getRuleContexts<SmalltalkParser::BareSymbolContext>();
}

SmalltalkParser::BareSymbolContext* SmalltalkParser::LiteralArrayRestContext::bareSymbol(size_t i) {
  return getRuleContext<SmalltalkParser::BareSymbolContext>(i);
}

std::vector<SmalltalkParser::ByteLiteralArrayBodyContext *> SmalltalkParser::LiteralArrayRestContext::byteLiteralArrayBody() {
  return getRuleContexts<SmalltalkParser::ByteLiteralArrayBodyContext>();
}

SmalltalkParser::ByteLiteralArrayBodyContext* SmalltalkParser::LiteralArrayRestContext::byteLiteralArrayBody(size_t i) {
  return getRuleContext<SmalltalkParser::ByteLiteralArrayBodyContext>(i);
}


size_t SmalltalkParser::LiteralArrayRestContext::getRuleIndex() const {
  return SmalltalkParser::RuleLiteralArrayRest;
}

void SmalltalkParser::LiteralArrayRestContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLiteralArrayRest(this);
}

void SmalltalkParser::LiteralArrayRestContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLiteralArrayRest(this);
}


antlrcpp::Any SmalltalkParser::LiteralArrayRestContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitLiteralArrayRest(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::LiteralArrayRestContext* SmalltalkParser::literalArrayRest() {
  LiteralArrayRestContext *_localctx = _tracker.createInstance<LiteralArrayRestContext>(_ctx, getState());
  enterRule(_localctx, 88, SmalltalkParser::RuleLiteralArrayRest);
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
    setState(537);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(536);
      ws();
    }
    setState(550);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 104, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(543);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 102, _ctx)) {
        case 1: {
          setState(539);
          literal();
          break;
        }

        case 2: {
          setState(540);
          bareLiteralArray();
          break;
        }

        case 3: {
          setState(541);
          bareSymbol();
          break;
        }

        case 4: {
          setState(542);
          byteLiteralArrayBody();
          break;
        }

        default:
          break;
        }
        setState(546);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == SmalltalkParser::SEPARATOR

        || _la == SmalltalkParser::COMMENT) {
          setState(545);
          ws();
        } 
      }
      setState(552);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 104, _ctx);
    }
    setState(553);
    _la = _input->LA(1);
    if (!(_la == SmalltalkParser::BLOCK_END

    || _la == SmalltalkParser::CLOSE_PAREN)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ByteLiteralArrayContext ------------------------------------------------------------------

SmalltalkParser::ByteLiteralArrayContext::ByteLiteralArrayContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::ByteLiteralArrayContext::HASH() {
  return getToken(SmalltalkParser::HASH, 0);
}

SmalltalkParser::ByteLiteralArrayBodyContext* SmalltalkParser::ByteLiteralArrayContext::byteLiteralArrayBody() {
  return getRuleContext<SmalltalkParser::ByteLiteralArrayBodyContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::ByteLiteralArrayContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::ByteLiteralArrayContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}


size_t SmalltalkParser::ByteLiteralArrayContext::getRuleIndex() const {
  return SmalltalkParser::RuleByteLiteralArray;
}

void SmalltalkParser::ByteLiteralArrayContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterByteLiteralArray(this);
}

void SmalltalkParser::ByteLiteralArrayContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitByteLiteralArray(this);
}


antlrcpp::Any SmalltalkParser::ByteLiteralArrayContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitByteLiteralArray(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ByteLiteralArrayContext* SmalltalkParser::byteLiteralArray() {
  ByteLiteralArrayContext *_localctx = _tracker.createInstance<ByteLiteralArrayContext>(_ctx, getState());
  enterRule(_localctx, 90, SmalltalkParser::RuleByteLiteralArray);
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
    setState(555);
    match(SmalltalkParser::HASH);
    setState(557);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(556);
      ws();
    }
    setState(559);
    byteLiteralArrayBody();
    setState(561);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 106, _ctx)) {
    case 1: {
      setState(560);
      ws();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ByteLiteralArrayBodyContext ------------------------------------------------------------------

SmalltalkParser::ByteLiteralArrayBodyContext::ByteLiteralArrayBodyContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::ByteLiteralArrayBodyContext::BLOCK_START() {
  return getToken(SmalltalkParser::BLOCK_START, 0);
}

tree::TerminalNode* SmalltalkParser::ByteLiteralArrayBodyContext::BLOCK_END() {
  return getToken(SmalltalkParser::BLOCK_END, 0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::ByteLiteralArrayBodyContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::ByteLiteralArrayBodyContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

std::vector<tree::TerminalNode *> SmalltalkParser::ByteLiteralArrayBodyContext::INTNUM() {
  return getTokens(SmalltalkParser::INTNUM);
}

tree::TerminalNode* SmalltalkParser::ByteLiteralArrayBodyContext::INTNUM(size_t i) {
  return getToken(SmalltalkParser::INTNUM, i);
}


size_t SmalltalkParser::ByteLiteralArrayBodyContext::getRuleIndex() const {
  return SmalltalkParser::RuleByteLiteralArrayBody;
}

void SmalltalkParser::ByteLiteralArrayBodyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterByteLiteralArrayBody(this);
}

void SmalltalkParser::ByteLiteralArrayBodyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitByteLiteralArrayBody(this);
}


antlrcpp::Any SmalltalkParser::ByteLiteralArrayBodyContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitByteLiteralArrayBody(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ByteLiteralArrayBodyContext* SmalltalkParser::byteLiteralArrayBody() {
  ByteLiteralArrayBodyContext *_localctx = _tracker.createInstance<ByteLiteralArrayBodyContext>(_ctx, getState());
  enterRule(_localctx, 92, SmalltalkParser::RuleByteLiteralArrayBody);
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
    setState(563);
    match(SmalltalkParser::BLOCK_START);
    setState(565);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(564);
      ws();
    }
    setState(573);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SmalltalkParser::INTNUM) {
      setState(567);
      match(SmalltalkParser::INTNUM);
      setState(569);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SmalltalkParser::SEPARATOR

      || _la == SmalltalkParser::COMMENT) {
        setState(568);
        ws();
      }
      setState(575);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(576);
    match(SmalltalkParser::BLOCK_END);
    setState(578);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 110, _ctx)) {
    case 1: {
      setState(577);
      ws();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BareLiteralArrayContext ------------------------------------------------------------------

SmalltalkParser::BareLiteralArrayContext::BareLiteralArrayContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::BareLiteralArrayContext::OPEN_PAREN() {
  return getToken(SmalltalkParser::OPEN_PAREN, 0);
}

SmalltalkParser::LiteralArrayRestContext* SmalltalkParser::BareLiteralArrayContext::literalArrayRest() {
  return getRuleContext<SmalltalkParser::LiteralArrayRestContext>(0);
}


size_t SmalltalkParser::BareLiteralArrayContext::getRuleIndex() const {
  return SmalltalkParser::RuleBareLiteralArray;
}

void SmalltalkParser::BareLiteralArrayContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBareLiteralArray(this);
}

void SmalltalkParser::BareLiteralArrayContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBareLiteralArray(this);
}


antlrcpp::Any SmalltalkParser::BareLiteralArrayContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitBareLiteralArray(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::BareLiteralArrayContext* SmalltalkParser::bareLiteralArray() {
  BareLiteralArrayContext *_localctx = _tracker.createInstance<BareLiteralArrayContext>(_ctx, getState());
  enterRule(_localctx, 94, SmalltalkParser::RuleBareLiteralArray);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(580);
    match(SmalltalkParser::OPEN_PAREN);
    setState(581);
    literalArrayRest();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryTailContext ------------------------------------------------------------------

SmalltalkParser::UnaryTailContext::UnaryTailContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::UnaryMessageContext* SmalltalkParser::UnaryTailContext::unaryMessage() {
  return getRuleContext<SmalltalkParser::UnaryMessageContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::UnaryTailContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::UnaryTailContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}

SmalltalkParser::UnaryTailContext* SmalltalkParser::UnaryTailContext::unaryTail() {
  return getRuleContext<SmalltalkParser::UnaryTailContext>(0);
}


size_t SmalltalkParser::UnaryTailContext::getRuleIndex() const {
  return SmalltalkParser::RuleUnaryTail;
}

void SmalltalkParser::UnaryTailContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryTail(this);
}

void SmalltalkParser::UnaryTailContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryTail(this);
}


antlrcpp::Any SmalltalkParser::UnaryTailContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitUnaryTail(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::UnaryTailContext* SmalltalkParser::unaryTail() {
  UnaryTailContext *_localctx = _tracker.createInstance<UnaryTailContext>(_ctx, getState());
  enterRule(_localctx, 96, SmalltalkParser::RuleUnaryTail);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(583);
    unaryMessage();
    setState(585);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 111, _ctx)) {
    case 1: {
      setState(584);
      ws();
      break;
    }

    default:
      break;
    }
    setState(588);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 112, _ctx)) {
    case 1: {
      setState(587);
      unaryTail();
      break;
    }

    default:
      break;
    }
    setState(591);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 113, _ctx)) {
    case 1: {
      setState(590);
      ws();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryMessageContext ------------------------------------------------------------------

SmalltalkParser::UnaryMessageContext::UnaryMessageContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::UnarySelectorContext* SmalltalkParser::UnaryMessageContext::unarySelector() {
  return getRuleContext<SmalltalkParser::UnarySelectorContext>(0);
}

SmalltalkParser::WsContext* SmalltalkParser::UnaryMessageContext::ws() {
  return getRuleContext<SmalltalkParser::WsContext>(0);
}


size_t SmalltalkParser::UnaryMessageContext::getRuleIndex() const {
  return SmalltalkParser::RuleUnaryMessage;
}

void SmalltalkParser::UnaryMessageContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryMessage(this);
}

void SmalltalkParser::UnaryMessageContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryMessage(this);
}


antlrcpp::Any SmalltalkParser::UnaryMessageContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitUnaryMessage(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::UnaryMessageContext* SmalltalkParser::unaryMessage() {
  UnaryMessageContext *_localctx = _tracker.createInstance<UnaryMessageContext>(_ctx, getState());
  enterRule(_localctx, 98, SmalltalkParser::RuleUnaryMessage);
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
    setState(594);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(593);
      ws();
    }
    setState(596);
    unarySelector();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnarySelectorContext ------------------------------------------------------------------

SmalltalkParser::UnarySelectorContext::UnarySelectorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::UnarySelectorContext::IDENTIFIER() {
  return getToken(SmalltalkParser::IDENTIFIER, 0);
}


size_t SmalltalkParser::UnarySelectorContext::getRuleIndex() const {
  return SmalltalkParser::RuleUnarySelector;
}

void SmalltalkParser::UnarySelectorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnarySelector(this);
}

void SmalltalkParser::UnarySelectorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnarySelector(this);
}


antlrcpp::Any SmalltalkParser::UnarySelectorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitUnarySelector(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::UnarySelectorContext* SmalltalkParser::unarySelector() {
  UnarySelectorContext *_localctx = _tracker.createInstance<UnarySelectorContext>(_ctx, getState());
  enterRule(_localctx, 100, SmalltalkParser::RuleUnarySelector);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(598);
    match(SmalltalkParser::IDENTIFIER);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- KeywordsContext ------------------------------------------------------------------

SmalltalkParser::KeywordsContext::KeywordsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> SmalltalkParser::KeywordsContext::KEYWORD() {
  return getTokens(SmalltalkParser::KEYWORD);
}

tree::TerminalNode* SmalltalkParser::KeywordsContext::KEYWORD(size_t i) {
  return getToken(SmalltalkParser::KEYWORD, i);
}


size_t SmalltalkParser::KeywordsContext::getRuleIndex() const {
  return SmalltalkParser::RuleKeywords;
}

void SmalltalkParser::KeywordsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterKeywords(this);
}

void SmalltalkParser::KeywordsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitKeywords(this);
}


antlrcpp::Any SmalltalkParser::KeywordsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitKeywords(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::KeywordsContext* SmalltalkParser::keywords() {
  KeywordsContext *_localctx = _tracker.createInstance<KeywordsContext>(_ctx, getState());
  enterRule(_localctx, 102, SmalltalkParser::RuleKeywords);
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
    setState(601); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(600);
      match(SmalltalkParser::KEYWORD);
      setState(603); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == SmalltalkParser::KEYWORD);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReferenceContext ------------------------------------------------------------------

SmalltalkParser::ReferenceContext::ReferenceContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::VariableContext* SmalltalkParser::ReferenceContext::variable() {
  return getRuleContext<SmalltalkParser::VariableContext>(0);
}


size_t SmalltalkParser::ReferenceContext::getRuleIndex() const {
  return SmalltalkParser::RuleReference;
}

void SmalltalkParser::ReferenceContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReference(this);
}

void SmalltalkParser::ReferenceContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReference(this);
}


antlrcpp::Any SmalltalkParser::ReferenceContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitReference(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::ReferenceContext* SmalltalkParser::reference() {
  ReferenceContext *_localctx = _tracker.createInstance<ReferenceContext>(_ctx, getState());
  enterRule(_localctx, 104, SmalltalkParser::RuleReference);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(605);
    variable();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BinaryTailContext ------------------------------------------------------------------

SmalltalkParser::BinaryTailContext::BinaryTailContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SmalltalkParser::BinaryMessageContext* SmalltalkParser::BinaryTailContext::binaryMessage() {
  return getRuleContext<SmalltalkParser::BinaryMessageContext>(0);
}

SmalltalkParser::BinaryTailContext* SmalltalkParser::BinaryTailContext::binaryTail() {
  return getRuleContext<SmalltalkParser::BinaryTailContext>(0);
}


size_t SmalltalkParser::BinaryTailContext::getRuleIndex() const {
  return SmalltalkParser::RuleBinaryTail;
}

void SmalltalkParser::BinaryTailContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinaryTail(this);
}

void SmalltalkParser::BinaryTailContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinaryTail(this);
}


antlrcpp::Any SmalltalkParser::BinaryTailContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitBinaryTail(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::BinaryTailContext* SmalltalkParser::binaryTail() {
  BinaryTailContext *_localctx = _tracker.createInstance<BinaryTailContext>(_ctx, getState());
  enterRule(_localctx, 106, SmalltalkParser::RuleBinaryTail);

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
    binaryMessage();
    setState(609);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 116, _ctx)) {
    case 1: {
      setState(608);
      binaryTail();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BinaryMessageContext ------------------------------------------------------------------

SmalltalkParser::BinaryMessageContext::BinaryMessageContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::MINUS() {
  return getToken(SmalltalkParser::MINUS, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::PIPE() {
  return getToken(SmalltalkParser::PIPE, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::PIPE2() {
  return getToken(SmalltalkParser::PIPE2, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::PIPE3() {
  return getToken(SmalltalkParser::PIPE3, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::AMP() {
  return getToken(SmalltalkParser::AMP, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::BINARY_SELECTOR() {
  return getToken(SmalltalkParser::BINARY_SELECTOR, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::NEQ() {
  return getToken(SmalltalkParser::NEQ, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::LLT() {
  return getToken(SmalltalkParser::LLT, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::GGT() {
  return getToken(SmalltalkParser::GGT, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::LTE() {
  return getToken(SmalltalkParser::LTE, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::GTE() {
  return getToken(SmalltalkParser::GTE, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::LT() {
  return getToken(SmalltalkParser::LT, 0);
}

tree::TerminalNode* SmalltalkParser::BinaryMessageContext::GT() {
  return getToken(SmalltalkParser::GT, 0);
}

SmalltalkParser::UnarySendContext* SmalltalkParser::BinaryMessageContext::unarySend() {
  return getRuleContext<SmalltalkParser::UnarySendContext>(0);
}

SmalltalkParser::OperandContext* SmalltalkParser::BinaryMessageContext::operand() {
  return getRuleContext<SmalltalkParser::OperandContext>(0);
}

std::vector<SmalltalkParser::WsContext *> SmalltalkParser::BinaryMessageContext::ws() {
  return getRuleContexts<SmalltalkParser::WsContext>();
}

SmalltalkParser::WsContext* SmalltalkParser::BinaryMessageContext::ws(size_t i) {
  return getRuleContext<SmalltalkParser::WsContext>(i);
}


size_t SmalltalkParser::BinaryMessageContext::getRuleIndex() const {
  return SmalltalkParser::RuleBinaryMessage;
}

void SmalltalkParser::BinaryMessageContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinaryMessage(this);
}

void SmalltalkParser::BinaryMessageContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<SmalltalkListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinaryMessage(this);
}


antlrcpp::Any SmalltalkParser::BinaryMessageContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SmalltalkVisitor*>(visitor))
    return parserVisitor->visitBinaryMessage(this);
  else
    return visitor->visitChildren(this);
}

SmalltalkParser::BinaryMessageContext* SmalltalkParser::binaryMessage() {
  BinaryMessageContext *_localctx = _tracker.createInstance<BinaryMessageContext>(_ctx, getState());
  enterRule(_localctx, 108, SmalltalkParser::RuleBinaryMessage);
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
    setState(612);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(611);
      ws();
    }
    setState(614);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SmalltalkParser::PIPE3)
      | (1ULL << SmalltalkParser::PIPE2)
      | (1ULL << SmalltalkParser::PIPE)
      | (1ULL << SmalltalkParser::LLT)
      | (1ULL << SmalltalkParser::GGT)
      | (1ULL << SmalltalkParser::LTE)
      | (1ULL << SmalltalkParser::GTE)
      | (1ULL << SmalltalkParser::NEQ)
      | (1ULL << SmalltalkParser::LT)
      | (1ULL << SmalltalkParser::GT)
      | (1ULL << SmalltalkParser::AMP)
      | (1ULL << SmalltalkParser::MINUS)
      | (1ULL << SmalltalkParser::BINARY_SELECTOR))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(616);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SmalltalkParser::SEPARATOR

    || _la == SmalltalkParser::COMMENT) {
      setState(615);
      ws();
    }
    setState(620);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 119, _ctx)) {
    case 1: {
      setState(618);
      unarySend();
      break;
    }

    case 2: {
      setState(619);
      operand();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

// Static vars and initialization.
std::vector<dfa::DFA> SmalltalkParser::_decisionToDFA;
atn::PredictionContextCache SmalltalkParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN SmalltalkParser::_atn;
std::vector<uint16_t> SmalltalkParser::_serializedATN;

std::vector<std::string> SmalltalkParser::_ruleNames = {
  "module", "function", "funcdecl", "declPairs", "declPair", "script", "sequence", 
  "ws", "temps", "statements", "answer", "expression", "expressions", "expressionList", 
  "cascade", "message", "assignment", "variable", "binarySend", "unarySend", 
  "keywordSend", "keywordMessage", "keywordPair", "operand", "subexpression", 
  "literal", "runtimeLiteral", "block", "blockParamList", "dynamicDictionary", 
  "dynamicArray", "parsetimeLiteral", "number", "numberExp", "charConstant", 
  "hex_", "stInteger", "stFloat", "pseudoVariable", "string", "primitive", 
  "symbol", "bareSymbol", "literalArray", "literalArrayRest", "byteLiteralArray", 
  "byteLiteralArrayBody", "bareLiteralArray", "unaryTail", "unaryMessage", 
  "unarySelector", "keywords", "reference", "binaryTail", "binaryMessage"
};

std::vector<std::string> SmalltalkParser::_literalNames = {
  "", "", "", "", "", "", "'['", "']'", "')'", "'('", "'|||'", "'||'", "'|'", 
  "'.'", "';'", "", "", "", "", "", "", "", "'&amp;'", "'-'", "", "", "", 
  "", "", "'^'", "'_'", "'__'", "':='", "':'", "'#'", "'$'", "'e'", "", 
  "", "'16r'", "'#{'", "'}'", "'{'"
};

std::vector<std::string> SmalltalkParser::_symbolicNames = {
  "", "SEPARATOR", "STRING", "PRIMITIVE", "COMMENT", "ST_COMMENT", "BLOCK_START", 
  "BLOCK_END", "CLOSE_PAREN", "OPEN_PAREN", "PIPE3", "PIPE2", "PIPE", "PERIOD", 
  "SEMI_COLON", "LLT", "GGT", "LTE", "GTE", "NEQ", "LT", "GT", "AMP", "MINUS", 
  "BINARY_SELECTOR", "RESERVED_WORD", "IDENTIFIER", "IDENTIFIER2", "SPECIAL_UNDERLINE_IDENTIFIER", 
  "CARROT", "UNDERSCORE", "UNDERSCORE2", "ASSIGNMENT", "COLON", "HASH", 
  "DOLLAR", "EXP", "HEXNUM", "INTNUM", "HEX", "DYNDICT_START", "DYNARR_END", 
  "DYNARR_START", "HEXDIGIT", "KEYWORD", "BLOCK_PARAM", "CHARACTER_CONSTANT"
};

dfa::Vocabulary SmalltalkParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> SmalltalkParser::_tokenNames;

SmalltalkParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x30, 0x271, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 
    0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 
    0x1f, 0x9, 0x1f, 0x4, 0x20, 0x9, 0x20, 0x4, 0x21, 0x9, 0x21, 0x4, 0x22, 
    0x9, 0x22, 0x4, 0x23, 0x9, 0x23, 0x4, 0x24, 0x9, 0x24, 0x4, 0x25, 0x9, 
    0x25, 0x4, 0x26, 0x9, 0x26, 0x4, 0x27, 0x9, 0x27, 0x4, 0x28, 0x9, 0x28, 
    0x4, 0x29, 0x9, 0x29, 0x4, 0x2a, 0x9, 0x2a, 0x4, 0x2b, 0x9, 0x2b, 0x4, 
    0x2c, 0x9, 0x2c, 0x4, 0x2d, 0x9, 0x2d, 0x4, 0x2e, 0x9, 0x2e, 0x4, 0x2f, 
    0x9, 0x2f, 0x4, 0x30, 0x9, 0x30, 0x4, 0x31, 0x9, 0x31, 0x4, 0x32, 0x9, 
    0x32, 0x4, 0x33, 0x9, 0x33, 0x4, 0x34, 0x9, 0x34, 0x4, 0x35, 0x9, 0x35, 
    0x4, 0x36, 0x9, 0x36, 0x4, 0x37, 0x9, 0x37, 0x4, 0x38, 0x9, 0x38, 0x3, 
    0x2, 0x3, 0x2, 0x5, 0x2, 0x73, 0xa, 0x2, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 
    0x77, 0xa, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x7b, 0xa, 0x3, 0x3, 0x4, 
    0x5, 0x4, 0x7e, 0xa, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 
    0x4, 0x84, 0xa, 0x4, 0x3, 0x4, 0x5, 0x4, 0x87, 0xa, 0x4, 0x3, 0x5, 0x5, 
    0x5, 0x8a, 0xa, 0x5, 0x3, 0x5, 0x3, 0x5, 0x5, 0x5, 0x8e, 0xa, 0x5, 0x6, 
    0x5, 0x90, 0xa, 0x5, 0xd, 0x5, 0xe, 0x5, 0x91, 0x3, 0x6, 0x3, 0x6, 0x5, 
    0x6, 0x96, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x5, 0x7, 0x9b, 0xa, 
    0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0x9f, 0xa, 0x7, 0x7, 0x7, 0xa1, 0xa, 
    0x7, 0xc, 0x7, 0xe, 0x7, 0xa4, 0xb, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 
    0xa8, 0xa, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xae, 
    0xa, 0x8, 0x3, 0x8, 0x5, 0x8, 0xb1, 0xa, 0x8, 0x3, 0x8, 0x5, 0x8, 0xb4, 
    0xa, 0x8, 0x3, 0x8, 0x5, 0x8, 0xb7, 0xa, 0x8, 0x3, 0x8, 0x5, 0x8, 0xba, 
    0xa, 0x8, 0x3, 0x8, 0x5, 0x8, 0xbd, 0xa, 0x8, 0x3, 0x8, 0x5, 0x8, 0xc0, 
    0xa, 0x8, 0x3, 0x9, 0x6, 0x9, 0xc3, 0xa, 0x9, 0xd, 0x9, 0xe, 0x9, 0xc4, 
    0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xc9, 0xa, 0xa, 0x3, 0xa, 0x7, 0xa, 0xcc, 
    0xa, 0xa, 0xc, 0xa, 0xe, 0xa, 0xcf, 0xb, 0xa, 0x3, 0xa, 0x5, 0xa, 0xd2, 
    0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xd6, 0xa, 0xa, 0x3, 0xb, 0x3, 
    0xb, 0x5, 0xb, 0xda, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0xde, 0xa, 
    0xb, 0x3, 0xb, 0x5, 0xb, 0xe1, 0xa, 0xb, 0x3, 0xb, 0x5, 0xb, 0xe4, 0xa, 
    0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0xe8, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x5, 0xb, 0xec, 0xa, 0xb, 0x3, 0xb, 0x5, 0xb, 0xef, 0xa, 0xb, 0x3, 0xb, 
    0x5, 0xb, 0xf2, 0xa, 0xb, 0x5, 0xb, 0xf4, 0xa, 0xb, 0x3, 0xc, 0x3, 0xc, 
    0x5, 0xc, 0xf8, 0xa, 0xc, 0x3, 0xc, 0x3, 0xc, 0x5, 0xc, 0xfc, 0xa, 0xc, 
    0x3, 0xc, 0x5, 0xc, 0xff, 0xa, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 
    0xd, 0x3, 0xd, 0x5, 0xd, 0x106, 0xa, 0xd, 0x3, 0xd, 0x5, 0xd, 0x109, 
    0xa, 0xd, 0x3, 0xd, 0x5, 0xd, 0x10c, 0xa, 0xd, 0x3, 0xd, 0x5, 0xd, 0x10f, 
    0xa, 0xd, 0x3, 0xe, 0x3, 0xe, 0x7, 0xe, 0x113, 0xa, 0xe, 0xc, 0xe, 0xe, 
    0xe, 0x116, 0xb, 0xe, 0x3, 0xf, 0x3, 0xf, 0x5, 0xf, 0x11a, 0xa, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x5, 0x10, 0x120, 0xa, 0x10, 
    0x3, 0x10, 0x5, 0x10, 0x123, 0xa, 0x10, 0x3, 0x10, 0x3, 0x10, 0x5, 0x10, 
    0x127, 0xa, 0x10, 0x3, 0x10, 0x6, 0x10, 0x12a, 0xa, 0x10, 0xd, 0x10, 
    0xe, 0x10, 0x12b, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x5, 0x11, 0x131, 
    0xa, 0x11, 0x3, 0x12, 0x3, 0x12, 0x5, 0x12, 0x135, 0xa, 0x12, 0x3, 0x12, 
    0x3, 0x12, 0x5, 0x12, 0x139, 0xa, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 
    0x3, 0x13, 0x3, 0x14, 0x3, 0x14, 0x5, 0x14, 0x141, 0xa, 0x14, 0x3, 0x15, 
    0x3, 0x15, 0x5, 0x15, 0x145, 0xa, 0x15, 0x3, 0x15, 0x5, 0x15, 0x148, 
    0xa, 0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x5, 0x17, 0x14e, 
    0xa, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 0x152, 0xa, 0x17, 0x6, 0x17, 
    0x154, 0xa, 0x17, 0xd, 0x17, 0xe, 0x17, 0x155, 0x3, 0x18, 0x3, 0x18, 
    0x5, 0x18, 0x15a, 0xa, 0x18, 0x3, 0x18, 0x3, 0x18, 0x5, 0x18, 0x15e, 
    0xa, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x5, 0x19, 0x163, 0xa, 0x19, 
    0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 0x167, 0xa, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 
    0x5, 0x1a, 0x16b, 0xa, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 
    0x5, 0x1b, 0x171, 0xa, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x5, 0x1c, 
    0x176, 0xa, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x17a, 0xa, 0x1d, 
    0x3, 0x1d, 0x5, 0x1d, 0x17d, 0xa, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x180, 
    0xa, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x183, 0xa, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 
    0x186, 0xa, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x189, 0xa, 0x1d, 0x3, 0x1d, 
    0x5, 0x1d, 0x18c, 0xa, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x18f, 0xa, 0x1d, 
    0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x194, 0xa, 0x1d, 0x3, 0x1d, 
    0x3, 0x1d, 0x5, 0x1d, 0x198, 0xa, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x19b, 
    0xa, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x19e, 0xa, 0x1d, 0x3, 0x1e, 0x5, 0x1e, 
    0x1a1, 0xa, 0x1e, 0x3, 0x1e, 0x6, 0x1e, 0x1a4, 0xa, 0x1e, 0xd, 0x1e, 
    0xe, 0x1e, 0x1a5, 0x3, 0x1f, 0x3, 0x1f, 0x5, 0x1f, 0x1aa, 0xa, 0x1f, 
    0x3, 0x1f, 0x5, 0x1f, 0x1ad, 0xa, 0x1f, 0x3, 0x1f, 0x5, 0x1f, 0x1b0, 
    0xa, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x5, 0x20, 0x1b6, 
    0xa, 0x20, 0x3, 0x20, 0x5, 0x20, 0x1b9, 0xa, 0x20, 0x3, 0x20, 0x5, 0x20, 
    0x1bc, 0xa, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x5, 0x21, 0x1c7, 0xa, 0x21, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x5, 0x22, 0x1cd, 0xa, 0x22, 
    0x3, 0x23, 0x3, 0x23, 0x5, 0x23, 0x1d1, 0xa, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x24, 0x3, 0x24, 0x3, 0x25, 0x5, 0x25, 0x1d9, 0xa, 0x25, 
    0x3, 0x25, 0x3, 0x25, 0x3, 0x26, 0x5, 0x26, 0x1de, 0xa, 0x26, 0x3, 0x26, 
    0x3, 0x26, 0x3, 0x27, 0x5, 0x27, 0x1e3, 0xa, 0x27, 0x3, 0x27, 0x3, 0x27, 
    0x3, 0x27, 0x3, 0x27, 0x3, 0x28, 0x3, 0x28, 0x3, 0x29, 0x3, 0x29, 0x3, 
    0x2a, 0x3, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x5, 0x2b, 0x1f1, 0xa, 0x2b, 
    0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x5, 0x2b, 0x1f6, 0xa, 0x2b, 0x5, 0x2b, 
    0x1f8, 0xa, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x6, 0x2c, 0x1fc, 0xa, 0x2c, 
    0xd, 0x2c, 0xe, 0x2c, 0x1fd, 0x3, 0x2c, 0x3, 0x2c, 0x5, 0x2c, 0x202, 
    0xa, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x5, 0x2d, 0x206, 0xa, 0x2d, 0x3, 0x2d, 
    0x3, 0x2d, 0x3, 0x2d, 0x5, 0x2d, 0x20b, 0xa, 0x2d, 0x3, 0x2d, 0x5, 0x2d, 
    0x20e, 0xa, 0x2d, 0x3, 0x2d, 0x5, 0x2d, 0x211, 0xa, 0x2d, 0x3, 0x2d, 
    0x5, 0x2d, 0x214, 0xa, 0x2d, 0x3, 0x2d, 0x5, 0x2d, 0x217, 0xa, 0x2d, 
    0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2e, 0x5, 0x2e, 0x21c, 0xa, 0x2e, 0x3, 0x2e, 
    0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x5, 0x2e, 0x222, 0xa, 0x2e, 0x3, 0x2e, 
    0x5, 0x2e, 0x225, 0xa, 0x2e, 0x7, 0x2e, 0x227, 0xa, 0x2e, 0xc, 0x2e, 
    0xe, 0x2e, 0x22a, 0xb, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 
    0x5, 0x2f, 0x230, 0xa, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x5, 0x2f, 0x234, 
    0xa, 0x2f, 0x3, 0x30, 0x3, 0x30, 0x5, 0x30, 0x238, 0xa, 0x30, 0x3, 0x30, 
    0x3, 0x30, 0x5, 0x30, 0x23c, 0xa, 0x30, 0x7, 0x30, 0x23e, 0xa, 0x30, 
    0xc, 0x30, 0xe, 0x30, 0x241, 0xb, 0x30, 0x3, 0x30, 0x3, 0x30, 0x5, 0x30, 
    0x245, 0xa, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x32, 0x3, 0x32, 
    0x5, 0x32, 0x24c, 0xa, 0x32, 0x3, 0x32, 0x5, 0x32, 0x24f, 0xa, 0x32, 
    0x3, 0x32, 0x5, 0x32, 0x252, 0xa, 0x32, 0x3, 0x33, 0x5, 0x33, 0x255, 
    0xa, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x34, 0x3, 0x34, 0x3, 0x35, 0x6, 
    0x35, 0x25c, 0xa, 0x35, 0xd, 0x35, 0xe, 0x35, 0x25d, 0x3, 0x36, 0x3, 
    0x36, 0x3, 0x37, 0x3, 0x37, 0x5, 0x37, 0x264, 0xa, 0x37, 0x3, 0x38, 
    0x5, 0x38, 0x267, 0xa, 0x38, 0x3, 0x38, 0x3, 0x38, 0x5, 0x38, 0x26b, 
    0xa, 0x38, 0x3, 0x38, 0x3, 0x38, 0x5, 0x38, 0x26f, 0xa, 0x38, 0x3, 0x38, 
    0x2, 0x2, 0x39, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 
    0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 
    0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 
    0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 
    0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x2, 0x9, 0x4, 
    0x2, 0xc, 0xe, 0x11, 0x1a, 0x4, 0x2, 0x3, 0x3, 0x6, 0x6, 0x3, 0x2, 0xd, 
    0xe, 0x4, 0x2, 0x1c, 0x1d, 0x20, 0x21, 0x4, 0x2, 0x25, 0x25, 0x30, 0x30, 
    0x6, 0x2, 0xa, 0xe, 0x11, 0x1a, 0x1c, 0x1c, 0x20, 0x20, 0x3, 0x2, 0x9, 
    0xa, 0x2, 0x2c4, 0x2, 0x72, 0x3, 0x2, 0x2, 0x2, 0x4, 0x74, 0x3, 0x2, 
    0x2, 0x2, 0x6, 0x86, 0x3, 0x2, 0x2, 0x2, 0x8, 0x89, 0x3, 0x2, 0x2, 0x2, 
    0xa, 0x93, 0x3, 0x2, 0x2, 0x2, 0xc, 0x9a, 0x3, 0x2, 0x2, 0x2, 0xe, 0xbf, 
    0x3, 0x2, 0x2, 0x2, 0x10, 0xc2, 0x3, 0x2, 0x2, 0x2, 0x12, 0xd5, 0x3, 
    0x2, 0x2, 0x2, 0x14, 0xf3, 0x3, 0x2, 0x2, 0x2, 0x16, 0xf5, 0x3, 0x2, 
    0x2, 0x2, 0x18, 0x105, 0x3, 0x2, 0x2, 0x2, 0x1a, 0x110, 0x3, 0x2, 0x2, 
    0x2, 0x1c, 0x117, 0x3, 0x2, 0x2, 0x2, 0x1e, 0x11f, 0x3, 0x2, 0x2, 0x2, 
    0x20, 0x130, 0x3, 0x2, 0x2, 0x2, 0x22, 0x132, 0x3, 0x2, 0x2, 0x2, 0x24, 
    0x13c, 0x3, 0x2, 0x2, 0x2, 0x26, 0x13e, 0x3, 0x2, 0x2, 0x2, 0x28, 0x142, 
    0x3, 0x2, 0x2, 0x2, 0x2a, 0x149, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x14d, 0x3, 
    0x2, 0x2, 0x2, 0x2e, 0x157, 0x3, 0x2, 0x2, 0x2, 0x30, 0x162, 0x3, 0x2, 
    0x2, 0x2, 0x32, 0x164, 0x3, 0x2, 0x2, 0x2, 0x34, 0x170, 0x3, 0x2, 0x2, 
    0x2, 0x36, 0x175, 0x3, 0x2, 0x2, 0x2, 0x38, 0x19d, 0x3, 0x2, 0x2, 0x2, 
    0x3a, 0x1a3, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x1a7, 0x3, 0x2, 0x2, 0x2, 0x3e, 
    0x1b3, 0x3, 0x2, 0x2, 0x2, 0x40, 0x1c6, 0x3, 0x2, 0x2, 0x2, 0x42, 0x1cc, 
    0x3, 0x2, 0x2, 0x2, 0x44, 0x1d0, 0x3, 0x2, 0x2, 0x2, 0x46, 0x1d5, 0x3, 
    0x2, 0x2, 0x2, 0x48, 0x1d8, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x1dd, 0x3, 0x2, 
    0x2, 0x2, 0x4c, 0x1e2, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x1e8, 0x3, 0x2, 0x2, 
    0x2, 0x50, 0x1ea, 0x3, 0x2, 0x2, 0x2, 0x52, 0x1ec, 0x3, 0x2, 0x2, 0x2, 
    0x54, 0x1f7, 0x3, 0x2, 0x2, 0x2, 0x56, 0x201, 0x3, 0x2, 0x2, 0x2, 0x58, 
    0x20d, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x21b, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x22d, 
    0x3, 0x2, 0x2, 0x2, 0x5e, 0x235, 0x3, 0x2, 0x2, 0x2, 0x60, 0x246, 0x3, 
    0x2, 0x2, 0x2, 0x62, 0x249, 0x3, 0x2, 0x2, 0x2, 0x64, 0x254, 0x3, 0x2, 
    0x2, 0x2, 0x66, 0x258, 0x3, 0x2, 0x2, 0x2, 0x68, 0x25b, 0x3, 0x2, 0x2, 
    0x2, 0x6a, 0x25f, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x261, 0x3, 0x2, 0x2, 0x2, 
    0x6e, 0x266, 0x3, 0x2, 0x2, 0x2, 0x70, 0x73, 0x5, 0x4, 0x3, 0x2, 0x71, 
    0x73, 0x5, 0xc, 0x7, 0x2, 0x72, 0x70, 0x3, 0x2, 0x2, 0x2, 0x72, 0x71, 
    0x3, 0x2, 0x2, 0x2, 0x73, 0x3, 0x3, 0x2, 0x2, 0x2, 0x74, 0x76, 0x5, 
    0x6, 0x4, 0x2, 0x75, 0x77, 0x5, 0x10, 0x9, 0x2, 0x76, 0x75, 0x3, 0x2, 
    0x2, 0x2, 0x76, 0x77, 0x3, 0x2, 0x2, 0x2, 0x77, 0x7a, 0x3, 0x2, 0x2, 
    0x2, 0x78, 0x7b, 0x7, 0x2, 0x2, 0x3, 0x79, 0x7b, 0x5, 0xc, 0x7, 0x2, 
    0x7a, 0x78, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x79, 0x3, 0x2, 0x2, 0x2, 0x7b, 
    0x5, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x7e, 0x5, 0x10, 0x9, 0x2, 0x7d, 0x7c, 
    0x3, 0x2, 0x2, 0x2, 0x7d, 0x7e, 0x3, 0x2, 0x2, 0x2, 0x7e, 0x7f, 0x3, 
    0x2, 0x2, 0x2, 0x7f, 0x87, 0x7, 0x1c, 0x2, 0x2, 0x80, 0x87, 0x5, 0x8, 
    0x5, 0x2, 0x81, 0x83, 0x9, 0x2, 0x2, 0x2, 0x82, 0x84, 0x5, 0x10, 0x9, 
    0x2, 0x83, 0x82, 0x3, 0x2, 0x2, 0x2, 0x83, 0x84, 0x3, 0x2, 0x2, 0x2, 
    0x84, 0x85, 0x3, 0x2, 0x2, 0x2, 0x85, 0x87, 0x5, 0x24, 0x13, 0x2, 0x86, 
    0x7d, 0x3, 0x2, 0x2, 0x2, 0x86, 0x80, 0x3, 0x2, 0x2, 0x2, 0x86, 0x81, 
    0x3, 0x2, 0x2, 0x2, 0x87, 0x7, 0x3, 0x2, 0x2, 0x2, 0x88, 0x8a, 0x5, 
    0x10, 0x9, 0x2, 0x89, 0x88, 0x3, 0x2, 0x2, 0x2, 0x89, 0x8a, 0x3, 0x2, 
    0x2, 0x2, 0x8a, 0x8f, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x8d, 0x5, 0xa, 0x6, 
    0x2, 0x8c, 0x8e, 0x5, 0x10, 0x9, 0x2, 0x8d, 0x8c, 0x3, 0x2, 0x2, 0x2, 
    0x8d, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x90, 0x3, 0x2, 0x2, 0x2, 0x8f, 
    0x8b, 0x3, 0x2, 0x2, 0x2, 0x90, 0x91, 0x3, 0x2, 0x2, 0x2, 0x91, 0x8f, 
    0x3, 0x2, 0x2, 0x2, 0x91, 0x92, 0x3, 0x2, 0x2, 0x2, 0x92, 0x9, 0x3, 
    0x2, 0x2, 0x2, 0x93, 0x95, 0x7, 0x2e, 0x2, 0x2, 0x94, 0x96, 0x5, 0x10, 
    0x9, 0x2, 0x95, 0x94, 0x3, 0x2, 0x2, 0x2, 0x95, 0x96, 0x3, 0x2, 0x2, 
    0x2, 0x96, 0x97, 0x3, 0x2, 0x2, 0x2, 0x97, 0x98, 0x5, 0x24, 0x13, 0x2, 
    0x98, 0xb, 0x3, 0x2, 0x2, 0x2, 0x99, 0x9b, 0x5, 0x10, 0x9, 0x2, 0x9a, 
    0x99, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x9b, 0x3, 0x2, 0x2, 0x2, 0x9b, 0xa2, 
    0x3, 0x2, 0x2, 0x2, 0x9c, 0x9e, 0x5, 0x52, 0x2a, 0x2, 0x9d, 0x9f, 0x5, 
    0x10, 0x9, 0x2, 0x9e, 0x9d, 0x3, 0x2, 0x2, 0x2, 0x9e, 0x9f, 0x3, 0x2, 
    0x2, 0x2, 0x9f, 0xa1, 0x3, 0x2, 0x2, 0x2, 0xa0, 0x9c, 0x3, 0x2, 0x2, 
    0x2, 0xa1, 0xa4, 0x3, 0x2, 0x2, 0x2, 0xa2, 0xa0, 0x3, 0x2, 0x2, 0x2, 
    0xa2, 0xa3, 0x3, 0x2, 0x2, 0x2, 0xa3, 0xa5, 0x3, 0x2, 0x2, 0x2, 0xa4, 
    0xa2, 0x3, 0x2, 0x2, 0x2, 0xa5, 0xa7, 0x5, 0xe, 0x8, 0x2, 0xa6, 0xa8, 
    0x5, 0x10, 0x9, 0x2, 0xa7, 0xa6, 0x3, 0x2, 0x2, 0x2, 0xa7, 0xa8, 0x3, 
    0x2, 0x2, 0x2, 0xa8, 0xa9, 0x3, 0x2, 0x2, 0x2, 0xa9, 0xaa, 0x7, 0x2, 
    0x2, 0x3, 0xaa, 0xd, 0x3, 0x2, 0x2, 0x2, 0xab, 0xad, 0x5, 0x12, 0xa, 
    0x2, 0xac, 0xae, 0x5, 0x10, 0x9, 0x2, 0xad, 0xac, 0x3, 0x2, 0x2, 0x2, 
    0xad, 0xae, 0x3, 0x2, 0x2, 0x2, 0xae, 0xb0, 0x3, 0x2, 0x2, 0x2, 0xaf, 
    0xb1, 0x5, 0x52, 0x2a, 0x2, 0xb0, 0xaf, 0x3, 0x2, 0x2, 0x2, 0xb0, 0xb1, 
    0x3, 0x2, 0x2, 0x2, 0xb1, 0xb3, 0x3, 0x2, 0x2, 0x2, 0xb2, 0xb4, 0x5, 
    0x10, 0x9, 0x2, 0xb3, 0xb2, 0x3, 0x2, 0x2, 0x2, 0xb3, 0xb4, 0x3, 0x2, 
    0x2, 0x2, 0xb4, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xb7, 0x5, 0x14, 0xb, 
    0x2, 0xb6, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb6, 0xb7, 0x3, 0x2, 0x2, 0x2, 
    0xb7, 0xc0, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xba, 0x5, 0x52, 0x2a, 0x2, 0xb9, 
    0xb8, 0x3, 0x2, 0x2, 0x2, 0xb9, 0xba, 0x3, 0x2, 0x2, 0x2, 0xba, 0xbc, 
    0x3, 0x2, 0x2, 0x2, 0xbb, 0xbd, 0x5, 0x10, 0x9, 0x2, 0xbc, 0xbb, 0x3, 
    0x2, 0x2, 0x2, 0xbc, 0xbd, 0x3, 0x2, 0x2, 0x2, 0xbd, 0xbe, 0x3, 0x2, 
    0x2, 0x2, 0xbe, 0xc0, 0x5, 0x14, 0xb, 0x2, 0xbf, 0xab, 0x3, 0x2, 0x2, 
    0x2, 0xbf, 0xb9, 0x3, 0x2, 0x2, 0x2, 0xc0, 0xf, 0x3, 0x2, 0x2, 0x2, 
    0xc1, 0xc3, 0x9, 0x3, 0x2, 0x2, 0xc2, 0xc1, 0x3, 0x2, 0x2, 0x2, 0xc3, 
    0xc4, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc2, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc5, 
    0x3, 0x2, 0x2, 0x2, 0xc5, 0x11, 0x3, 0x2, 0x2, 0x2, 0xc6, 0xcd, 0x9, 
    0x4, 0x2, 0x2, 0xc7, 0xc9, 0x5, 0x10, 0x9, 0x2, 0xc8, 0xc7, 0x3, 0x2, 
    0x2, 0x2, 0xc8, 0xc9, 0x3, 0x2, 0x2, 0x2, 0xc9, 0xca, 0x3, 0x2, 0x2, 
    0x2, 0xca, 0xcc, 0x7, 0x1c, 0x2, 0x2, 0xcb, 0xc8, 0x3, 0x2, 0x2, 0x2, 
    0xcc, 0xcf, 0x3, 0x2, 0x2, 0x2, 0xcd, 0xcb, 0x3, 0x2, 0x2, 0x2, 0xcd, 
    0xce, 0x3, 0x2, 0x2, 0x2, 0xce, 0xd1, 0x3, 0x2, 0x2, 0x2, 0xcf, 0xcd, 
    0x3, 0x2, 0x2, 0x2, 0xd0, 0xd2, 0x5, 0x10, 0x9, 0x2, 0xd1, 0xd0, 0x3, 
    0x2, 0x2, 0x2, 0xd1, 0xd2, 0x3, 0x2, 0x2, 0x2, 0xd2, 0xd3, 0x3, 0x2, 
    0x2, 0x2, 0xd3, 0xd6, 0x9, 0x4, 0x2, 0x2, 0xd4, 0xd6, 0x7, 0xd, 0x2, 
    0x2, 0xd5, 0xc6, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd4, 0x3, 0x2, 0x2, 0x2, 
    0xd6, 0x13, 0x3, 0x2, 0x2, 0x2, 0xd7, 0xd9, 0x5, 0x16, 0xc, 0x2, 0xd8, 
    0xda, 0x5, 0x10, 0x9, 0x2, 0xd9, 0xd8, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xda, 
    0x3, 0x2, 0x2, 0x2, 0xda, 0xf4, 0x3, 0x2, 0x2, 0x2, 0xdb, 0xdd, 0x5, 
    0x1a, 0xe, 0x2, 0xdc, 0xde, 0x5, 0x10, 0x9, 0x2, 0xdd, 0xdc, 0x3, 0x2, 
    0x2, 0x2, 0xdd, 0xde, 0x3, 0x2, 0x2, 0x2, 0xde, 0xe0, 0x3, 0x2, 0x2, 
    0x2, 0xdf, 0xe1, 0x7, 0xf, 0x2, 0x2, 0xe0, 0xdf, 0x3, 0x2, 0x2, 0x2, 
    0xe0, 0xe1, 0x3, 0x2, 0x2, 0x2, 0xe1, 0xe3, 0x3, 0x2, 0x2, 0x2, 0xe2, 
    0xe4, 0x5, 0x10, 0x9, 0x2, 0xe3, 0xe2, 0x3, 0x2, 0x2, 0x2, 0xe3, 0xe4, 
    0x3, 0x2, 0x2, 0x2, 0xe4, 0xe5, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xe7, 0x5, 
    0x16, 0xc, 0x2, 0xe6, 0xe8, 0x5, 0x10, 0x9, 0x2, 0xe7, 0xe6, 0x3, 0x2, 
    0x2, 0x2, 0xe7, 0xe8, 0x3, 0x2, 0x2, 0x2, 0xe8, 0xf4, 0x3, 0x2, 0x2, 
    0x2, 0xe9, 0xeb, 0x5, 0x1a, 0xe, 0x2, 0xea, 0xec, 0x5, 0x10, 0x9, 0x2, 
    0xeb, 0xea, 0x3, 0x2, 0x2, 0x2, 0xeb, 0xec, 0x3, 0x2, 0x2, 0x2, 0xec, 
    0xee, 0x3, 0x2, 0x2, 0x2, 0xed, 0xef, 0x7, 0xf, 0x2, 0x2, 0xee, 0xed, 
    0x3, 0x2, 0x2, 0x2, 0xee, 0xef, 0x3, 0x2, 0x2, 0x2, 0xef, 0xf1, 0x3, 
    0x2, 0x2, 0x2, 0xf0, 0xf2, 0x5, 0x10, 0x9, 0x2, 0xf1, 0xf0, 0x3, 0x2, 
    0x2, 0x2, 0xf1, 0xf2, 0x3, 0x2, 0x2, 0x2, 0xf2, 0xf4, 0x3, 0x2, 0x2, 
    0x2, 0xf3, 0xd7, 0x3, 0x2, 0x2, 0x2, 0xf3, 0xdb, 0x3, 0x2, 0x2, 0x2, 
    0xf3, 0xe9, 0x3, 0x2, 0x2, 0x2, 0xf4, 0x15, 0x3, 0x2, 0x2, 0x2, 0xf5, 
    0xf7, 0x7, 0x1f, 0x2, 0x2, 0xf6, 0xf8, 0x5, 0x10, 0x9, 0x2, 0xf7, 0xf6, 
    0x3, 0x2, 0x2, 0x2, 0xf7, 0xf8, 0x3, 0x2, 0x2, 0x2, 0xf8, 0xf9, 0x3, 
    0x2, 0x2, 0x2, 0xf9, 0xfb, 0x5, 0x18, 0xd, 0x2, 0xfa, 0xfc, 0x5, 0x10, 
    0x9, 0x2, 0xfb, 0xfa, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xfc, 0x3, 0x2, 0x2, 
    0x2, 0xfc, 0xfe, 0x3, 0x2, 0x2, 0x2, 0xfd, 0xff, 0x7, 0xf, 0x2, 0x2, 
    0xfe, 0xfd, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xff, 0x3, 0x2, 0x2, 0x2, 0xff, 
    0x17, 0x3, 0x2, 0x2, 0x2, 0x100, 0x106, 0x5, 0x22, 0x12, 0x2, 0x101, 
    0x106, 0x5, 0x1e, 0x10, 0x2, 0x102, 0x106, 0x5, 0x2a, 0x16, 0x2, 0x103, 
    0x106, 0x5, 0x26, 0x14, 0x2, 0x104, 0x106, 0x5, 0x52, 0x2a, 0x2, 0x105, 
    0x100, 0x3, 0x2, 0x2, 0x2, 0x105, 0x101, 0x3, 0x2, 0x2, 0x2, 0x105, 
    0x102, 0x3, 0x2, 0x2, 0x2, 0x105, 0x103, 0x3, 0x2, 0x2, 0x2, 0x105, 
    0x104, 0x3, 0x2, 0x2, 0x2, 0x106, 0x108, 0x3, 0x2, 0x2, 0x2, 0x107, 
    0x109, 0x5, 0x10, 0x9, 0x2, 0x108, 0x107, 0x3, 0x2, 0x2, 0x2, 0x108, 
    0x109, 0x3, 0x2, 0x2, 0x2, 0x109, 0x10b, 0x3, 0x2, 0x2, 0x2, 0x10a, 
    0x10c, 0x7, 0xf, 0x2, 0x2, 0x10b, 0x10a, 0x3, 0x2, 0x2, 0x2, 0x10b, 
    0x10c, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10e, 0x3, 0x2, 0x2, 0x2, 0x10d, 
    0x10f, 0x5, 0x10, 0x9, 0x2, 0x10e, 0x10d, 0x3, 0x2, 0x2, 0x2, 0x10e, 
    0x10f, 0x3, 0x2, 0x2, 0x2, 0x10f, 0x19, 0x3, 0x2, 0x2, 0x2, 0x110, 0x114, 
    0x5, 0x18, 0xd, 0x2, 0x111, 0x113, 0x5, 0x1c, 0xf, 0x2, 0x112, 0x111, 
    0x3, 0x2, 0x2, 0x2, 0x113, 0x116, 0x3, 0x2, 0x2, 0x2, 0x114, 0x112, 
    0x3, 0x2, 0x2, 0x2, 0x114, 0x115, 0x3, 0x2, 0x2, 0x2, 0x115, 0x1b, 0x3, 
    0x2, 0x2, 0x2, 0x116, 0x114, 0x3, 0x2, 0x2, 0x2, 0x117, 0x119, 0x7, 
    0xf, 0x2, 0x2, 0x118, 0x11a, 0x5, 0x10, 0x9, 0x2, 0x119, 0x118, 0x3, 
    0x2, 0x2, 0x2, 0x119, 0x11a, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x11b, 0x3, 
    0x2, 0x2, 0x2, 0x11b, 0x11c, 0x5, 0x18, 0xd, 0x2, 0x11c, 0x1d, 0x3, 
    0x2, 0x2, 0x2, 0x11d, 0x120, 0x5, 0x2a, 0x16, 0x2, 0x11e, 0x120, 0x5, 
    0x26, 0x14, 0x2, 0x11f, 0x11d, 0x3, 0x2, 0x2, 0x2, 0x11f, 0x11e, 0x3, 
    0x2, 0x2, 0x2, 0x120, 0x129, 0x3, 0x2, 0x2, 0x2, 0x121, 0x123, 0x5, 
    0x10, 0x9, 0x2, 0x122, 0x121, 0x3, 0x2, 0x2, 0x2, 0x122, 0x123, 0x3, 
    0x2, 0x2, 0x2, 0x123, 0x124, 0x3, 0x2, 0x2, 0x2, 0x124, 0x126, 0x7, 
    0x10, 0x2, 0x2, 0x125, 0x127, 0x5, 0x10, 0x9, 0x2, 0x126, 0x125, 0x3, 
    0x2, 0x2, 0x2, 0x126, 0x127, 0x3, 0x2, 0x2, 0x2, 0x127, 0x128, 0x3, 
    0x2, 0x2, 0x2, 0x128, 0x12a, 0x5, 0x20, 0x11, 0x2, 0x129, 0x122, 0x3, 
    0x2, 0x2, 0x2, 0x12a, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x12b, 0x129, 0x3, 
    0x2, 0x2, 0x2, 0x12b, 0x12c, 0x3, 0x2, 0x2, 0x2, 0x12c, 0x1f, 0x3, 0x2, 
    0x2, 0x2, 0x12d, 0x131, 0x5, 0x6e, 0x38, 0x2, 0x12e, 0x131, 0x5, 0x64, 
    0x33, 0x2, 0x12f, 0x131, 0x5, 0x2c, 0x17, 0x2, 0x130, 0x12d, 0x3, 0x2, 
    0x2, 0x2, 0x130, 0x12e, 0x3, 0x2, 0x2, 0x2, 0x130, 0x12f, 0x3, 0x2, 
    0x2, 0x2, 0x131, 0x21, 0x3, 0x2, 0x2, 0x2, 0x132, 0x134, 0x5, 0x24, 
    0x13, 0x2, 0x133, 0x135, 0x5, 0x10, 0x9, 0x2, 0x134, 0x133, 0x3, 0x2, 
    0x2, 0x2, 0x134, 0x135, 0x3, 0x2, 0x2, 0x2, 0x135, 0x136, 0x3, 0x2, 
    0x2, 0x2, 0x136, 0x138, 0x7, 0x22, 0x2, 0x2, 0x137, 0x139, 0x5, 0x10, 
    0x9, 0x2, 0x138, 0x137, 0x3, 0x2, 0x2, 0x2, 0x138, 0x139, 0x3, 0x2, 
    0x2, 0x2, 0x139, 0x13a, 0x3, 0x2, 0x2, 0x2, 0x13a, 0x13b, 0x5, 0x18, 
    0xd, 0x2, 0x13b, 0x23, 0x3, 0x2, 0x2, 0x2, 0x13c, 0x13d, 0x9, 0x5, 0x2, 
    0x2, 0x13d, 0x25, 0x3, 0x2, 0x2, 0x2, 0x13e, 0x140, 0x5, 0x28, 0x15, 
    0x2, 0x13f, 0x141, 0x5, 0x6c, 0x37, 0x2, 0x140, 0x13f, 0x3, 0x2, 0x2, 
    0x2, 0x140, 0x141, 0x3, 0x2, 0x2, 0x2, 0x141, 0x27, 0x3, 0x2, 0x2, 0x2, 
    0x142, 0x144, 0x5, 0x30, 0x19, 0x2, 0x143, 0x145, 0x5, 0x10, 0x9, 0x2, 
    0x144, 0x143, 0x3, 0x2, 0x2, 0x2, 0x144, 0x145, 0x3, 0x2, 0x2, 0x2, 
    0x145, 0x147, 0x3, 0x2, 0x2, 0x2, 0x146, 0x148, 0x5, 0x62, 0x32, 0x2, 
    0x147, 0x146, 0x3, 0x2, 0x2, 0x2, 0x147, 0x148, 0x3, 0x2, 0x2, 0x2, 
    0x148, 0x29, 0x3, 0x2, 0x2, 0x2, 0x149, 0x14a, 0x5, 0x26, 0x14, 0x2, 
    0x14a, 0x14b, 0x5, 0x2c, 0x17, 0x2, 0x14b, 0x2b, 0x3, 0x2, 0x2, 0x2, 
    0x14c, 0x14e, 0x5, 0x10, 0x9, 0x2, 0x14d, 0x14c, 0x3, 0x2, 0x2, 0x2, 
    0x14d, 0x14e, 0x3, 0x2, 0x2, 0x2, 0x14e, 0x153, 0x3, 0x2, 0x2, 0x2, 
    0x14f, 0x151, 0x5, 0x2e, 0x18, 0x2, 0x150, 0x152, 0x5, 0x10, 0x9, 0x2, 
    0x151, 0x150, 0x3, 0x2, 0x2, 0x2, 0x151, 0x152, 0x3, 0x2, 0x2, 0x2, 
    0x152, 0x154, 0x3, 0x2, 0x2, 0x2, 0x153, 0x14f, 0x3, 0x2, 0x2, 0x2, 
    0x154, 0x155, 0x3, 0x2, 0x2, 0x2, 0x155, 0x153, 0x3, 0x2, 0x2, 0x2, 
    0x155, 0x156, 0x3, 0x2, 0x2, 0x2, 0x156, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x157, 
    0x159, 0x7, 0x2e, 0x2, 0x2, 0x158, 0x15a, 0x5, 0x10, 0x9, 0x2, 0x159, 
    0x158, 0x3, 0x2, 0x2, 0x2, 0x159, 0x15a, 0x3, 0x2, 0x2, 0x2, 0x15a, 
    0x15b, 0x3, 0x2, 0x2, 0x2, 0x15b, 0x15d, 0x5, 0x26, 0x14, 0x2, 0x15c, 
    0x15e, 0x5, 0x10, 0x9, 0x2, 0x15d, 0x15c, 0x3, 0x2, 0x2, 0x2, 0x15d, 
    0x15e, 0x3, 0x2, 0x2, 0x2, 0x15e, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x15f, 0x163, 
    0x5, 0x34, 0x1b, 0x2, 0x160, 0x163, 0x5, 0x6a, 0x36, 0x2, 0x161, 0x163, 
    0x5, 0x32, 0x1a, 0x2, 0x162, 0x15f, 0x3, 0x2, 0x2, 0x2, 0x162, 0x160, 
    0x3, 0x2, 0x2, 0x2, 0x162, 0x161, 0x3, 0x2, 0x2, 0x2, 0x163, 0x31, 0x3, 
    0x2, 0x2, 0x2, 0x164, 0x166, 0x7, 0xb, 0x2, 0x2, 0x165, 0x167, 0x5, 
    0x10, 0x9, 0x2, 0x166, 0x165, 0x3, 0x2, 0x2, 0x2, 0x166, 0x167, 0x3, 
    0x2, 0x2, 0x2, 0x167, 0x168, 0x3, 0x2, 0x2, 0x2, 0x168, 0x16a, 0x5, 
    0x18, 0xd, 0x2, 0x169, 0x16b, 0x5, 0x10, 0x9, 0x2, 0x16a, 0x169, 0x3, 
    0x2, 0x2, 0x2, 0x16a, 0x16b, 0x3, 0x2, 0x2, 0x2, 0x16b, 0x16c, 0x3, 
    0x2, 0x2, 0x2, 0x16c, 0x16d, 0x7, 0xa, 0x2, 0x2, 0x16d, 0x33, 0x3, 0x2, 
    0x2, 0x2, 0x16e, 0x171, 0x5, 0x36, 0x1c, 0x2, 0x16f, 0x171, 0x5, 0x40, 
    0x21, 0x2, 0x170, 0x16e, 0x3, 0x2, 0x2, 0x2, 0x170, 0x16f, 0x3, 0x2, 
    0x2, 0x2, 0x171, 0x35, 0x3, 0x2, 0x2, 0x2, 0x172, 0x176, 0x5, 0x3c, 
    0x1f, 0x2, 0x173, 0x176, 0x5, 0x3e, 0x20, 0x2, 0x174, 0x176, 0x5, 0x38, 
    0x1d, 0x2, 0x175, 0x172, 0x3, 0x2, 0x2, 0x2, 0x175, 0x173, 0x3, 0x2, 
    0x2, 0x2, 0x175, 0x174, 0x3, 0x2, 0x2, 0x2, 0x176, 0x37, 0x3, 0x2, 0x2, 
    0x2, 0x177, 0x179, 0x7, 0x8, 0x2, 0x2, 0x178, 0x17a, 0x5, 0x3a, 0x1e, 
    0x2, 0x179, 0x178, 0x3, 0x2, 0x2, 0x2, 0x179, 0x17a, 0x3, 0x2, 0x2, 
    0x2, 0x17a, 0x17c, 0x3, 0x2, 0x2, 0x2, 0x17b, 0x17d, 0x5, 0x10, 0x9, 
    0x2, 0x17c, 0x17b, 0x3, 0x2, 0x2, 0x2, 0x17c, 0x17d, 0x3, 0x2, 0x2, 
    0x2, 0x17d, 0x188, 0x3, 0x2, 0x2, 0x2, 0x17e, 0x180, 0x7, 0xe, 0x2, 
    0x2, 0x17f, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x17f, 0x180, 0x3, 0x2, 0x2, 
    0x2, 0x180, 0x182, 0x3, 0x2, 0x2, 0x2, 0x181, 0x183, 0x5, 0x10, 0x9, 
    0x2, 0x182, 0x181, 0x3, 0x2, 0x2, 0x2, 0x182, 0x183, 0x3, 0x2, 0x2, 
    0x2, 0x183, 0x185, 0x3, 0x2, 0x2, 0x2, 0x184, 0x186, 0x5, 0x12, 0xa, 
    0x2, 0x185, 0x184, 0x3, 0x2, 0x2, 0x2, 0x185, 0x186, 0x3, 0x2, 0x2, 
    0x2, 0x186, 0x189, 0x3, 0x2, 0x2, 0x2, 0x187, 0x189, 0x7, 0xc, 0x2, 
    0x2, 0x188, 0x17f, 0x3, 0x2, 0x2, 0x2, 0x188, 0x187, 0x3, 0x2, 0x2, 
    0x2, 0x189, 0x18b, 0x3, 0x2, 0x2, 0x2, 0x18a, 0x18c, 0x5, 0x10, 0x9, 
    0x2, 0x18b, 0x18a, 0x3, 0x2, 0x2, 0x2, 0x18b, 0x18c, 0x3, 0x2, 0x2, 
    0x2, 0x18c, 0x18e, 0x3, 0x2, 0x2, 0x2, 0x18d, 0x18f, 0x5, 0xe, 0x8, 
    0x2, 0x18e, 0x18d, 0x3, 0x2, 0x2, 0x2, 0x18e, 0x18f, 0x3, 0x2, 0x2, 
    0x2, 0x18f, 0x190, 0x3, 0x2, 0x2, 0x2, 0x190, 0x19e, 0x7, 0x9, 0x2, 
    0x2, 0x191, 0x193, 0x7, 0x8, 0x2, 0x2, 0x192, 0x194, 0x5, 0x10, 0x9, 
    0x2, 0x193, 0x192, 0x3, 0x2, 0x2, 0x2, 0x193, 0x194, 0x3, 0x2, 0x2, 
    0x2, 0x194, 0x195, 0x3, 0x2, 0x2, 0x2, 0x195, 0x197, 0x7, 0xd, 0x2, 
    0x2, 0x196, 0x198, 0x5, 0x10, 0x9, 0x2, 0x197, 0x196, 0x3, 0x2, 0x2, 
    0x2, 0x197, 0x198, 0x3, 0x2, 0x2, 0x2, 0x198, 0x19a, 0x3, 0x2, 0x2, 
    0x2, 0x199, 0x19b, 0x5, 0xe, 0x8, 0x2, 0x19a, 0x199, 0x3, 0x2, 0x2, 
    0x2, 0x19a, 0x19b, 0x3, 0x2, 0x2, 0x2, 0x19b, 0x19c, 0x3, 0x2, 0x2, 
    0x2, 0x19c, 0x19e, 0x7, 0x9, 0x2, 0x2, 0x19d, 0x177, 0x3, 0x2, 0x2, 
    0x2, 0x19d, 0x191, 0x3, 0x2, 0x2, 0x2, 0x19e, 0x39, 0x3, 0x2, 0x2, 0x2, 
    0x19f, 0x1a1, 0x5, 0x10, 0x9, 0x2, 0x1a0, 0x19f, 0x3, 0x2, 0x2, 0x2, 
    0x1a0, 0x1a1, 0x3, 0x2, 0x2, 0x2, 0x1a1, 0x1a2, 0x3, 0x2, 0x2, 0x2, 
    0x1a2, 0x1a4, 0x7, 0x2f, 0x2, 0x2, 0x1a3, 0x1a0, 0x3, 0x2, 0x2, 0x2, 
    0x1a4, 0x1a5, 0x3, 0x2, 0x2, 0x2, 0x1a5, 0x1a3, 0x3, 0x2, 0x2, 0x2, 
    0x1a5, 0x1a6, 0x3, 0x2, 0x2, 0x2, 0x1a6, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x1a7, 
    0x1a9, 0x7, 0x2a, 0x2, 0x2, 0x1a8, 0x1aa, 0x5, 0x10, 0x9, 0x2, 0x1a9, 
    0x1a8, 0x3, 0x2, 0x2, 0x2, 0x1a9, 0x1aa, 0x3, 0x2, 0x2, 0x2, 0x1aa, 
    0x1ac, 0x3, 0x2, 0x2, 0x2, 0x1ab, 0x1ad, 0x5, 0x1a, 0xe, 0x2, 0x1ac, 
    0x1ab, 0x3, 0x2, 0x2, 0x2, 0x1ac, 0x1ad, 0x3, 0x2, 0x2, 0x2, 0x1ad, 
    0x1af, 0x3, 0x2, 0x2, 0x2, 0x1ae, 0x1b0, 0x5, 0x10, 0x9, 0x2, 0x1af, 
    0x1ae, 0x3, 0x2, 0x2, 0x2, 0x1af, 0x1b0, 0x3, 0x2, 0x2, 0x2, 0x1b0, 
    0x1b1, 0x3, 0x2, 0x2, 0x2, 0x1b1, 0x1b2, 0x7, 0x2b, 0x2, 0x2, 0x1b2, 
    0x3d, 0x3, 0x2, 0x2, 0x2, 0x1b3, 0x1b5, 0x7, 0x2c, 0x2, 0x2, 0x1b4, 
    0x1b6, 0x5, 0x10, 0x9, 0x2, 0x1b5, 0x1b4, 0x3, 0x2, 0x2, 0x2, 0x1b5, 
    0x1b6, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b8, 0x3, 0x2, 0x2, 0x2, 0x1b7, 
    0x1b9, 0x5, 0x1a, 0xe, 0x2, 0x1b8, 0x1b7, 0x3, 0x2, 0x2, 0x2, 0x1b8, 
    0x1b9, 0x3, 0x2, 0x2, 0x2, 0x1b9, 0x1bb, 0x3, 0x2, 0x2, 0x2, 0x1ba, 
    0x1bc, 0x5, 0x10, 0x9, 0x2, 0x1bb, 0x1ba, 0x3, 0x2, 0x2, 0x2, 0x1bb, 
    0x1bc, 0x3, 0x2, 0x2, 0x2, 0x1bc, 0x1bd, 0x3, 0x2, 0x2, 0x2, 0x1bd, 
    0x1be, 0x7, 0x2b, 0x2, 0x2, 0x1be, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x1bf, 
    0x1c7, 0x5, 0x4e, 0x28, 0x2, 0x1c0, 0x1c7, 0x5, 0x42, 0x22, 0x2, 0x1c1, 
    0x1c7, 0x5, 0x46, 0x24, 0x2, 0x1c2, 0x1c7, 0x5, 0x58, 0x2d, 0x2, 0x1c3, 
    0x1c7, 0x5, 0x50, 0x29, 0x2, 0x1c4, 0x1c7, 0x5, 0x54, 0x2b, 0x2, 0x1c5, 
    0x1c7, 0x5, 0x5c, 0x2f, 0x2, 0x1c6, 0x1bf, 0x3, 0x2, 0x2, 0x2, 0x1c6, 
    0x1c0, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1c1, 0x3, 0x2, 0x2, 0x2, 0x1c6, 
    0x1c2, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1c3, 0x3, 0x2, 0x2, 0x2, 0x1c6, 
    0x1c4, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1c5, 0x3, 0x2, 0x2, 0x2, 0x1c7, 
    0x41, 0x3, 0x2, 0x2, 0x2, 0x1c8, 0x1cd, 0x5, 0x44, 0x23, 0x2, 0x1c9, 
    0x1cd, 0x5, 0x48, 0x25, 0x2, 0x1ca, 0x1cd, 0x5, 0x4c, 0x27, 0x2, 0x1cb, 
    0x1cd, 0x5, 0x4a, 0x26, 0x2, 0x1cc, 0x1c8, 0x3, 0x2, 0x2, 0x2, 0x1cc, 
    0x1c9, 0x3, 0x2, 0x2, 0x2, 0x1cc, 0x1ca, 0x3, 0x2, 0x2, 0x2, 0x1cc, 
    0x1cb, 0x3, 0x2, 0x2, 0x2, 0x1cd, 0x43, 0x3, 0x2, 0x2, 0x2, 0x1ce, 0x1d1, 
    0x5, 0x4c, 0x27, 0x2, 0x1cf, 0x1d1, 0x5, 0x4a, 0x26, 0x2, 0x1d0, 0x1ce, 
    0x3, 0x2, 0x2, 0x2, 0x1d0, 0x1cf, 0x3, 0x2, 0x2, 0x2, 0x1d1, 0x1d2, 
    0x3, 0x2, 0x2, 0x2, 0x1d2, 0x1d3, 0x7, 0x26, 0x2, 0x2, 0x1d3, 0x1d4, 
    0x5, 0x4a, 0x26, 0x2, 0x1d4, 0x45, 0x3, 0x2, 0x2, 0x2, 0x1d5, 0x1d6, 
    0x9, 0x6, 0x2, 0x2, 0x1d6, 0x47, 0x3, 0x2, 0x2, 0x2, 0x1d7, 0x1d9, 0x7, 
    0x19, 0x2, 0x2, 0x1d8, 0x1d7, 0x3, 0x2, 0x2, 0x2, 0x1d8, 0x1d9, 0x3, 
    0x2, 0x2, 0x2, 0x1d9, 0x1da, 0x3, 0x2, 0x2, 0x2, 0x1da, 0x1db, 0x7, 
    0x27, 0x2, 0x2, 0x1db, 0x49, 0x3, 0x2, 0x2, 0x2, 0x1dc, 0x1de, 0x7, 
    0x19, 0x2, 0x2, 0x1dd, 0x1dc, 0x3, 0x2, 0x2, 0x2, 0x1dd, 0x1de, 0x3, 
    0x2, 0x2, 0x2, 0x1de, 0x1df, 0x3, 0x2, 0x2, 0x2, 0x1df, 0x1e0, 0x7, 
    0x28, 0x2, 0x2, 0x1e0, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x1e1, 0x1e3, 0x7, 
    0x19, 0x2, 0x2, 0x1e2, 0x1e1, 0x3, 0x2, 0x2, 0x2, 0x1e2, 0x1e3, 0x3, 
    0x2, 0x2, 0x2, 0x1e3, 0x1e4, 0x3, 0x2, 0x2, 0x2, 0x1e4, 0x1e5, 0x7, 
    0x28, 0x2, 0x2, 0x1e5, 0x1e6, 0x7, 0xf, 0x2, 0x2, 0x1e6, 0x1e7, 0x7, 
    0x28, 0x2, 0x2, 0x1e7, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x1e8, 0x1e9, 0x7, 
    0x1b, 0x2, 0x2, 0x1e9, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x1ea, 0x1eb, 0x7, 
    0x4, 0x2, 0x2, 0x1eb, 0x51, 0x3, 0x2, 0x2, 0x2, 0x1ec, 0x1ed, 0x7, 0x5, 
    0x2, 0x2, 0x1ed, 0x53, 0x3, 0x2, 0x2, 0x2, 0x1ee, 0x1f0, 0x7, 0x24, 
    0x2, 0x2, 0x1ef, 0x1f1, 0x5, 0x10, 0x9, 0x2, 0x1f0, 0x1ef, 0x3, 0x2, 
    0x2, 0x2, 0x1f0, 0x1f1, 0x3, 0x2, 0x2, 0x2, 0x1f1, 0x1f2, 0x3, 0x2, 
    0x2, 0x2, 0x1f2, 0x1f8, 0x5, 0x56, 0x2c, 0x2, 0x1f3, 0x1f5, 0x7, 0x24, 
    0x2, 0x2, 0x1f4, 0x1f6, 0x7, 0x1e, 0x2, 0x2, 0x1f5, 0x1f4, 0x3, 0x2, 
    0x2, 0x2, 0x1f5, 0x1f6, 0x3, 0x2, 0x2, 0x2, 0x1f6, 0x1f8, 0x3, 0x2, 
    0x2, 0x2, 0x1f7, 0x1ee, 0x3, 0x2, 0x2, 0x2, 0x1f7, 0x1f3, 0x3, 0x2, 
    0x2, 0x2, 0x1f8, 0x55, 0x3, 0x2, 0x2, 0x2, 0x1f9, 0x202, 0x9, 0x7, 0x2, 
    0x2, 0x1fa, 0x1fc, 0x7, 0x2e, 0x2, 0x2, 0x1fb, 0x1fa, 0x3, 0x2, 0x2, 
    0x2, 0x1fc, 0x1fd, 0x3, 0x2, 0x2, 0x2, 0x1fd, 0x1fb, 0x3, 0x2, 0x2, 
    0x2, 0x1fd, 0x1fe, 0x3, 0x2, 0x2, 0x2, 0x1fe, 0x202, 0x3, 0x2, 0x2, 
    0x2, 0x1ff, 0x202, 0x5, 0x50, 0x29, 0x2, 0x200, 0x202, 0x7, 0x1b, 0x2, 
    0x2, 0x201, 0x1f9, 0x3, 0x2, 0x2, 0x2, 0x201, 0x1fb, 0x3, 0x2, 0x2, 
    0x2, 0x201, 0x1ff, 0x3, 0x2, 0x2, 0x2, 0x201, 0x200, 0x3, 0x2, 0x2, 
    0x2, 0x202, 0x57, 0x3, 0x2, 0x2, 0x2, 0x203, 0x205, 0x7, 0x24, 0x2, 
    0x2, 0x204, 0x206, 0x5, 0x10, 0x9, 0x2, 0x205, 0x204, 0x3, 0x2, 0x2, 
    0x2, 0x205, 0x206, 0x3, 0x2, 0x2, 0x2, 0x206, 0x207, 0x3, 0x2, 0x2, 
    0x2, 0x207, 0x20e, 0x7, 0xb, 0x2, 0x2, 0x208, 0x20a, 0x7, 0x24, 0x2, 
    0x2, 0x209, 0x20b, 0x5, 0x10, 0x9, 0x2, 0x20a, 0x209, 0x3, 0x2, 0x2, 
    0x2, 0x20a, 0x20b, 0x3, 0x2, 0x2, 0x2, 0x20b, 0x20c, 0x3, 0x2, 0x2, 
    0x2, 0x20c, 0x20e, 0x7, 0x8, 0x2, 0x2, 0x20d, 0x203, 0x3, 0x2, 0x2, 
    0x2, 0x20d, 0x208, 0x3, 0x2, 0x2, 0x2, 0x20e, 0x210, 0x3, 0x2, 0x2, 
    0x2, 0x20f, 0x211, 0x5, 0x10, 0x9, 0x2, 0x210, 0x20f, 0x3, 0x2, 0x2, 
    0x2, 0x210, 0x211, 0x3, 0x2, 0x2, 0x2, 0x211, 0x213, 0x3, 0x2, 0x2, 
    0x2, 0x212, 0x214, 0x5, 0x3c, 0x1f, 0x2, 0x213, 0x212, 0x3, 0x2, 0x2, 
    0x2, 0x213, 0x214, 0x3, 0x2, 0x2, 0x2, 0x214, 0x216, 0x3, 0x2, 0x2, 
    0x2, 0x215, 0x217, 0x5, 0x10, 0x9, 0x2, 0x216, 0x215, 0x3, 0x2, 0x2, 
    0x2, 0x216, 0x217, 0x3, 0x2, 0x2, 0x2, 0x217, 0x218, 0x3, 0x2, 0x2, 
    0x2, 0x218, 0x219, 0x5, 0x5a, 0x2e, 0x2, 0x219, 0x59, 0x3, 0x2, 0x2, 
    0x2, 0x21a, 0x21c, 0x5, 0x10, 0x9, 0x2, 0x21b, 0x21a, 0x3, 0x2, 0x2, 
    0x2, 0x21b, 0x21c, 0x3, 0x2, 0x2, 0x2, 0x21c, 0x228, 0x3, 0x2, 0x2, 
    0x2, 0x21d, 0x222, 0x5, 0x34, 0x1b, 0x2, 0x21e, 0x222, 0x5, 0x60, 0x31, 
    0x2, 0x21f, 0x222, 0x5, 0x56, 0x2c, 0x2, 0x220, 0x222, 0x5, 0x5e, 0x30, 
    0x2, 0x221, 0x21d, 0x3, 0x2, 0x2, 0x2, 0x221, 0x21e, 0x3, 0x2, 0x2, 
    0x2, 0x221, 0x21f, 0x3, 0x2, 0x2, 0x2, 0x221, 0x220, 0x3, 0x2, 0x2, 
    0x2, 0x222, 0x224, 0x3, 0x2, 0x2, 0x2, 0x223, 0x225, 0x5, 0x10, 0x9, 
    0x2, 0x224, 0x223, 0x3, 0x2, 0x2, 0x2, 0x224, 0x225, 0x3, 0x2, 0x2, 
    0x2, 0x225, 0x227, 0x3, 0x2, 0x2, 0x2, 0x226, 0x221, 0x3, 0x2, 0x2, 
    0x2, 0x227, 0x22a, 0x3, 0x2, 0x2, 0x2, 0x228, 0x226, 0x3, 0x2, 0x2, 
    0x2, 0x228, 0x229, 0x3, 0x2, 0x2, 0x2, 0x229, 0x22b, 0x3, 0x2, 0x2, 
    0x2, 0x22a, 0x228, 0x3, 0x2, 0x2, 0x2, 0x22b, 0x22c, 0x9, 0x8, 0x2, 
    0x2, 0x22c, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x22d, 0x22f, 0x7, 0x24, 0x2, 
    0x2, 0x22e, 0x230, 0x5, 0x10, 0x9, 0x2, 0x22f, 0x22e, 0x3, 0x2, 0x2, 
    0x2, 0x22f, 0x230, 0x3, 0x2, 0x2, 0x2, 0x230, 0x231, 0x3, 0x2, 0x2, 
    0x2, 0x231, 0x233, 0x5, 0x5e, 0x30, 0x2, 0x232, 0x234, 0x5, 0x10, 0x9, 
    0x2, 0x233, 0x232, 0x3, 0x2, 0x2, 0x2, 0x233, 0x234, 0x3, 0x2, 0x2, 
    0x2, 0x234, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x235, 0x237, 0x7, 0x8, 0x2, 0x2, 
    0x236, 0x238, 0x5, 0x10, 0x9, 0x2, 0x237, 0x236, 0x3, 0x2, 0x2, 0x2, 
    0x237, 0x238, 0x3, 0x2, 0x2, 0x2, 0x238, 0x23f, 0x3, 0x2, 0x2, 0x2, 
    0x239, 0x23b, 0x7, 0x28, 0x2, 0x2, 0x23a, 0x23c, 0x5, 0x10, 0x9, 0x2, 
    0x23b, 0x23a, 0x3, 0x2, 0x2, 0x2, 0x23b, 0x23c, 0x3, 0x2, 0x2, 0x2, 
    0x23c, 0x23e, 0x3, 0x2, 0x2, 0x2, 0x23d, 0x239, 0x3, 0x2, 0x2, 0x2, 
    0x23e, 0x241, 0x3, 0x2, 0x2, 0x2, 0x23f, 0x23d, 0x3, 0x2, 0x2, 0x2, 
    0x23f, 0x240, 0x3, 0x2, 0x2, 0x2, 0x240, 0x242, 0x3, 0x2, 0x2, 0x2, 
    0x241, 0x23f, 0x3, 0x2, 0x2, 0x2, 0x242, 0x244, 0x7, 0x9, 0x2, 0x2, 
    0x243, 0x245, 0x5, 0x10, 0x9, 0x2, 0x244, 0x243, 0x3, 0x2, 0x2, 0x2, 
    0x244, 0x245, 0x3, 0x2, 0x2, 0x2, 0x245, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x246, 
    0x247, 0x7, 0xb, 0x2, 0x2, 0x247, 0x248, 0x5, 0x5a, 0x2e, 0x2, 0x248, 
    0x61, 0x3, 0x2, 0x2, 0x2, 0x249, 0x24b, 0x5, 0x64, 0x33, 0x2, 0x24a, 
    0x24c, 0x5, 0x10, 0x9, 0x2, 0x24b, 0x24a, 0x3, 0x2, 0x2, 0x2, 0x24b, 
    0x24c, 0x3, 0x2, 0x2, 0x2, 0x24c, 0x24e, 0x3, 0x2, 0x2, 0x2, 0x24d, 
    0x24f, 0x5, 0x62, 0x32, 0x2, 0x24e, 0x24d, 0x3, 0x2, 0x2, 0x2, 0x24e, 
    0x24f, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x251, 0x3, 0x2, 0x2, 0x2, 0x250, 
    0x252, 0x5, 0x10, 0x9, 0x2, 0x251, 0x250, 0x3, 0x2, 0x2, 0x2, 0x251, 
    0x252, 0x3, 0x2, 0x2, 0x2, 0x252, 0x63, 0x3, 0x2, 0x2, 0x2, 0x253, 0x255, 
    0x5, 0x10, 0x9, 0x2, 0x254, 0x253, 0x3, 0x2, 0x2, 0x2, 0x254, 0x255, 
    0x3, 0x2, 0x2, 0x2, 0x255, 0x256, 0x3, 0x2, 0x2, 0x2, 0x256, 0x257, 
    0x5, 0x66, 0x34, 0x2, 0x257, 0x65, 0x3, 0x2, 0x2, 0x2, 0x258, 0x259, 
    0x7, 0x1c, 0x2, 0x2, 0x259, 0x67, 0x3, 0x2, 0x2, 0x2, 0x25a, 0x25c, 
    0x7, 0x2e, 0x2, 0x2, 0x25b, 0x25a, 0x3, 0x2, 0x2, 0x2, 0x25c, 0x25d, 
    0x3, 0x2, 0x2, 0x2, 0x25d, 0x25b, 0x3, 0x2, 0x2, 0x2, 0x25d, 0x25e, 
    0x3, 0x2, 0x2, 0x2, 0x25e, 0x69, 0x3, 0x2, 0x2, 0x2, 0x25f, 0x260, 0x5, 
    0x24, 0x13, 0x2, 0x260, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x261, 0x263, 0x5, 
    0x6e, 0x38, 0x2, 0x262, 0x264, 0x5, 0x6c, 0x37, 0x2, 0x263, 0x262, 0x3, 
    0x2, 0x2, 0x2, 0x263, 0x264, 0x3, 0x2, 0x2, 0x2, 0x264, 0x6d, 0x3, 0x2, 
    0x2, 0x2, 0x265, 0x267, 0x5, 0x10, 0x9, 0x2, 0x266, 0x265, 0x3, 0x2, 
    0x2, 0x2, 0x266, 0x267, 0x3, 0x2, 0x2, 0x2, 0x267, 0x268, 0x3, 0x2, 
    0x2, 0x2, 0x268, 0x26a, 0x9, 0x2, 0x2, 0x2, 0x269, 0x26b, 0x5, 0x10, 
    0x9, 0x2, 0x26a, 0x269, 0x3, 0x2, 0x2, 0x2, 0x26a, 0x26b, 0x3, 0x2, 
    0x2, 0x2, 0x26b, 0x26e, 0x3, 0x2, 0x2, 0x2, 0x26c, 0x26f, 0x5, 0x28, 
    0x15, 0x2, 0x26d, 0x26f, 0x5, 0x30, 0x19, 0x2, 0x26e, 0x26c, 0x3, 0x2, 
    0x2, 0x2, 0x26e, 0x26d, 0x3, 0x2, 0x2, 0x2, 0x26f, 0x6f, 0x3, 0x2, 0x2, 
    0x2, 0x7a, 0x72, 0x76, 0x7a, 0x7d, 0x83, 0x86, 0x89, 0x8d, 0x91, 0x95, 
    0x9a, 0x9e, 0xa2, 0xa7, 0xad, 0xb0, 0xb3, 0xb6, 0xb9, 0xbc, 0xbf, 0xc4, 
    0xc8, 0xcd, 0xd1, 0xd5, 0xd9, 0xdd, 0xe0, 0xe3, 0xe7, 0xeb, 0xee, 0xf1, 
    0xf3, 0xf7, 0xfb, 0xfe, 0x105, 0x108, 0x10b, 0x10e, 0x114, 0x119, 0x11f, 
    0x122, 0x126, 0x12b, 0x130, 0x134, 0x138, 0x140, 0x144, 0x147, 0x14d, 
    0x151, 0x155, 0x159, 0x15d, 0x162, 0x166, 0x16a, 0x170, 0x175, 0x179, 
    0x17c, 0x17f, 0x182, 0x185, 0x188, 0x18b, 0x18e, 0x193, 0x197, 0x19a, 
    0x19d, 0x1a0, 0x1a5, 0x1a9, 0x1ac, 0x1af, 0x1b5, 0x1b8, 0x1bb, 0x1c6, 
    0x1cc, 0x1d0, 0x1d8, 0x1dd, 0x1e2, 0x1f0, 0x1f5, 0x1f7, 0x1fd, 0x201, 
    0x205, 0x20a, 0x20d, 0x210, 0x213, 0x216, 0x21b, 0x221, 0x224, 0x228, 
    0x22f, 0x233, 0x237, 0x23b, 0x23f, 0x244, 0x24b, 0x24e, 0x251, 0x254, 
    0x25d, 0x263, 0x266, 0x26a, 0x26e, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

SmalltalkParser::Initializer SmalltalkParser::_init;
