#ifndef MLIR_ST_PARSER_VISITOR_H_
#define MLIR_ST_PARSER_VISITOR_H_

#include "SmalltalkBaseVisitor.h"
#include "ast/AST.h"

using namespace std;
using namespace stx;

class STParserVisitor : public SmalltalkBaseVisitor {
 private:
  const size_t MAX_TEXT_LENGTH = 20;
  std::string fileName;
  std::string fnName;
  size_t line_base;

  Location getLoc(antlr4::ParserRuleContext *ctx) {
    size_t line = 0;
    size_t pos = 0;
    if (ctx) {
      line = ctx->getStart()->getLine();
      pos = ctx->getStart()->getCharPositionInLine();
    }
    Location loc({fileName, line + line_base, pos});
    return loc;
  }

  bool existInCurScope(FunctionAST *scope, string name) {
    if (scope->temp_vars.size() == 0) return false;
    for (auto it = scope->temp_vars.begin(); it < scope->temp_vars.end();
         ++it) {
      if (it->getName() == name) return true;
    }
    return false;
  }

 public:
  STParserVisitor(std::string fileName, std::string funcName, size_t line)
      : fileName(fileName), fnName(funcName), line_base(line) {
    // llvm::outs() << "----visitor filename: " << fileName << "-------"<< "\n";
  }
  SEM sem;
  virtual antlrcpp::Any visitModule(
      SmalltalkParser::ModuleContext *ctx) override {
    FunctionAST *func;
    if (ctx->function()) {
      func = visitFunction(ctx->function());
    } else {
      cout << "workspace file: " << fileName << endl;
      auto loc = getLoc(ctx);
      if (ctx->script()) {
        func = visitSequenceX(ctx->script()->sequence());
      } else {
        func = new FunctionAST(loc);
      }
      std::vector<VarDeclExprAST> args;
      auto protoAST = new PrototypeAST(loc, fnName, args);
      func->proto = protoAST;
    }
    func->function_name = fnName;
    return func;
  }

  virtual antlrcpp::Any visitFunction(
      SmalltalkParser::FunctionContext *ctx) override {
    string func_name = ctx->funcdecl()->getText();
    string delimiter = "\n";
    string token = func_name.substr(0, func_name.find(delimiter));
    // cout << "Visit Function - " << token << endl;

    sem.curScopeName.push_back(func_name);
    // function arguments
    auto id = ctx->funcdecl()->IDENTIFIER();
    auto loc = getLoc(ctx);
    VarType type;
    type.name = "*i8";
    std::vector<VarDeclExprAST> args;  // TODO: add arg0 as this
    // always add "this" to non-main function
    VarDeclExprAST arg0(loc, "this", type);
    args.push_back(arg0);

    if (!id) {
      if (ctx->funcdecl()->declPairs()) {
        auto keywords = ctx->funcdecl()->declPairs()->declPair();
        // int count = 1;
        for (auto kw : keywords) {
          VarDeclExprAST arg(loc, kw->variable()->getText(), type);
          args.push_back(arg);
        }
      } else if (ctx->funcdecl()->variable()) {
        VarDeclExprAST arg(loc, ctx->funcdecl()->variable()->getText(), type);
        args.push_back(arg);
      }
    }
    FunctionAST *func;
    if (ctx->script()) {
      func = visitSequenceX(ctx->script()->sequence());
    } else {
      func = new FunctionAST(loc);
    }

    auto protoAST = new PrototypeAST(loc, fnName, args, func->return_val);
    func->proto = protoAST;
    return func;
  }

  virtual antlrcpp::Any visitFuncdecl(
      SmalltalkParser::FuncdeclContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclPairs(
      SmalltalkParser::DeclPairsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclPair(
      SmalltalkParser::DeclPairContext *ctx) override {
    return visitChildren(ctx);
  }

  FunctionAST *visitScriptX(SmalltalkParser::ScriptContext *ctx) {
    FunctionAST *func = visitSequenceX(ctx->sequence());
    return func;
  }

  FunctionAST *visitSequenceX(SmalltalkParser::SequenceContext *ctx) {
    FunctionAST *func = new FunctionAST(getLoc(ctx));
    //   <C: char *  primLamDecryptPrtStringWChecksum(char * arg1)>
    // ^self externalAccessFailed
    if (!ctx) return func;  // in case of parsing error for <C:
    func->source = ctx->getText();
    sem.Scopes.push_back(func);
    // temp var
    if (ctx->temps()) {
      // cout << "DEBUG visitSequenceX temps: " << func->getName() << endl;
      func->temp_vars = visitTempsX(ctx->temps());
    }

    if (auto newctx = ctx->statements()) {
      func->source = newctx->getText();
      // cout << "processing statements in " << sem.curScopeName << endl;
      // cout << "statement source " << func->source << endl;
      if (newctx->expressions()) {
        vector<ExprAST *> res = visitExpressionsX(newctx->expressions());
        func->body = res;
      }

      if (newctx->answer()) {
        // TODO: return value can be complicated
        auto result = visitAnswerX(newctx->answer());
        if (result != nullptr) func->return_val = result;
      }
    }
    sem.Scopes.pop_back();
    return func;
    //            return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitWs(SmalltalkParser::WsContext *ctx) override {
    return visitChildren(ctx);
  }

  vector<VarDeclExprAST> visitTempsX(SmalltalkParser::TempsContext *ctx) {
    vector<VarDeclExprAST> tmps;
    // if (auto curctx = ctx->temps2())
    {
      for (auto temp : ctx->IDENTIFIER()) {
        // cout << "DEBUG visitTempsX TEMP VAR: " << temp->getText() << endl;
        Location loc({fileName, temp->getSymbol()->getLine() + line_base,
                      temp->getSymbol()->getCharPositionInLine()});
        VarType type;
        type.name = "*i8";
        VarDeclExprAST var(loc, temp->getSymbol()->getText(), type);
        tmps.push_back(var);
      }
    }

    return tmps;
    //            return visitChildren(ctx);
  }

  // virtual antlrcpp::Any visitTemps1(
  //     SmalltalkParser::TempsContext *ctx) override {
  //   return visitChildren(ctx);
  // }

  // virtual antlrcpp::Any visitTemps2(
  //     SmalltalkParser::TempsContext *ctx) override {
  //   return visitChildren(ctx);
  // }

  ExprAST *visitAnswerX(SmalltalkParser::AnswerContext *ctx) {
    //            ExprAST * res = new ExprAST();
    ExprAST *res = visitExpressionX(ctx->expression());
    return res;
    //            return visitChildren(ctx);
  }
  // virtual antlrcpp::Any
  ExprAST *visitAssignmentX(
      SmalltalkParser::AssignmentContext *ctx) {  // override
    // return assign expr
    // AssignExprAST(Location loc, llvm::StringRef name, ExprAST* valueExpr)
    auto varname = ctx->variable()->getText();
    // check write var
    {
      sem.Scopes.back()->writeVars.insert(varname);
      sem.Scopes.back()->usedVars.insert(varname);
    }
    auto *var = new VariableExprAST(getLoc(ctx), varname);
    ExprAST *valueExpr = visitExpressionX(ctx->expression());
    auto assign_expr = new AssignExprAST(getLoc(ctx), var, valueExpr);
    ExprAST *expr = assign_expr;
    // cout << "Return Assign AST" << endl;
    return expr;
  }
  // virtual antlrcpp::Any
  ExprAST *visitExpressionX(
      SmalltalkParser::ExpressionContext *ctx) {  // override
    // return ExprAST
    ExprAST *expr = nullptr;
    if (ctx == nullptr) return expr;
    if (auto ass = ctx->assignment()) {
      ExprAST *assign = visitAssignmentX(ctx->assignment());
      expr = assign;
    }

    if (auto cas = ctx->cascade()) {
      ExprAST *cascade = visitCascadeX(ctx->cascade());
      expr = cascade;
    }

    if (auto kws = ctx->keywordSend()) {
      ExprAST *kwsend = visitKeywordSendX(kws);
      // cout << "Getting KeywordFunctionCallAST" << endl;
      expr = kwsend;
    }
    if (auto bs = ctx->binarySend()) {
      ExprAST *binsend = visitBinarySendX(ctx->binarySend());
      // cout << "Getting BinaryFunctionCallAST" << endl;
      expr = binsend;
    }
    if (auto pri = ctx->primitive()) {
      std::string text = pri->getText();
      if (text.size() > MAX_TEXT_LENGTH) text = text.substr(0, MAX_TEXT_LENGTH);
      auto *primitive = new PrimitiveExprAST(getLoc(ctx), text);
      expr = primitive;
    }
    // if (auto embed = ctx->EMBEDC()) {
    //   auto *embedc = new EmbedCExprAST(getLoc(ctx), embed->getText());
    //   expr = embedc;
    // }
    if (expr) {
      if (!ctx->isEmpty()) expr->source = ctx->getText();
    } else {
      // parser errors
      expr = new ParserErrorAST(getLoc(ctx), ctx->getText());
    }
    return expr;
  }

  vector<ExprAST *> visitExpressionsX(
      SmalltalkParser::ExpressionsContext *ctx) {
    // std::cout << "visitExpressions: " << ctx->getText() << "\n";

    // return a vector<ExprAST *>
    vector<ExprAST *> ret;
    if (ctx->expression()) {
      ExprAST *expr = visitExpressionX(ctx->expression());
      ret.push_back(expr);
    }
    for (auto expl : ctx->expressionList()) {
      if (expl->expression()) {
        // std::cout << "expressionList: " << expl->expression()->getText()
        //           << "\n";
        ret.push_back(visitExpressionX(expl->expression()));
      }
    }
    // std::cout << "return visitExpressions: size=" << ret.size() << "\n";

    return ret;
  }

  virtual antlrcpp::Any visitExpressionList(
      SmalltalkParser::ExpressionListContext *ctx) override {
    return visitExpressionX(ctx->expression());
  }

  virtual antlrcpp::Any visitMessage(
      SmalltalkParser::MessageContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVariable(
      SmalltalkParser::VariableContext *ctx) override {
    // cout << "Variable Defined: " << ctx->getText() << endl;
    return ctx->getText();
  }

  BinaryFunctionCallAST *visitBinarySendX(
      SmalltalkParser::BinarySendContext *ctx) {
    auto nctx = ctx->unarySend();
    auto umsg = visitUnarySendX(nctx);
    BinaryFunctionCallAST *fcc = new BinaryFunctionCallAST(getLoc(ctx), umsg);

    if (nctx->unaryTail()) {
      vector<string> ut = visitUnaryTail(nctx->unaryTail());
      fcc->umessagesTails = ut;
    }

    if (ctx->binaryTail()) {
      fcc->bmessages = visitBinaryTailX(ctx->binaryTail());
    }
    return fcc;
    //            return visitChildren(ctx);
  }

  UnaryMessageAST *visitUnarySendX(SmalltalkParser::UnarySendContext *ctx) {  //

    UnaryMessageAST *mes = new UnaryMessageAST(getLoc(ctx));
    mes->startFlag = true;
    mes->source = ctx->operand()->getText();
    mes->expr = visitOperandX(ctx->operand());

    // cout << "visitUnarySendX\n";
    // cout << mes->source << endl;
    if (ctx->unaryTail()) {
      vector<string> ut = visitUnaryTail(ctx->unaryTail());
      mes->unaryTails = ut;
    }

    return mes;
  }

  CascadeFunctionCallAST *visitCascadeX(SmalltalkParser::CascadeContext *ctx) {
    // cout << "CascadeFunctionCallAST\n";

    CascadeFunctionCallAST *fcall = new CascadeFunctionCallAST(getLoc(ctx));
    if (ctx->keywordSend()) {
      fcall->callType = "key";
      fcall->subCall = visitKeywordSendX(ctx->keywordSend());
    } else if (ctx->binarySend()) {
      fcall->callType = "bin";
      fcall->subCall = visitBinarySendX(ctx->binarySend());
    }

    // handle keyword message
    // cout << "TODO: handle keyword messages\n";
    for (auto msg : ctx->message()) {
      if (msg->keywordMessage()) {
        // only handle one keyword message for now
        vector<KeywordPairAST *> kwMsgs =
            visitKeywordMessageX(msg->keywordMessage());

        string func_name = "";
        for (auto kwp : msg->keywordMessage()->keywordPair()) {
          func_name = func_name + kwp->KEYWORD()->getText();
        }
        auto kwmsg = new KeywordMessageAST(getLoc(msg->keywordMessage()),
                                           kwMsgs, func_name);
        fcall->addKeyWordMessage(kwmsg);
      } else if (msg->unaryMessage()) {
        std::string umsg = msg->unaryMessage()->unarySelector()->getText();
        fcall->addUnaryMessage(umsg);
      }
    }
    return fcall;

    //            return visitChildren(ctx);
  }

  KeywordFunctionCallAST *visitKeywordSendX(
      SmalltalkParser::KeywordSendContext *ctx) {
    auto binSend = visitBinarySendX(ctx->binarySend());
    vector<KeywordPairAST *> kwMsgs =
        visitKeywordMessageX(ctx->keywordMessage());

    string func_name = "";
    for (auto kwp : ctx->keywordMessage()->keywordPair()) {
      func_name = func_name + kwp->KEYWORD()->getText();
    }
    KeywordFunctionCallAST *ret =
        new KeywordFunctionCallAST(getLoc(ctx), binSend, kwMsgs, func_name);

    return ret;
    //            return visitChildren(ctx);
  }

  vector<KeywordPairAST *> visitKeywordMessageX(
      SmalltalkParser::KeywordMessageContext *ctx) {
    // cout << "keywordMessage: " << ctx->getText() << "\n";

    vector<KeywordPairAST *> kwp;

    for (auto kp : ctx->keywordPair()) {
      KeywordPairAST *cur = new KeywordPairAST(getLoc(ctx));
      cur->method_name = kp->KEYWORD()->getText();
      cur->params = visitBinarySendX(
          kp->binarySend());  //  cur->params =  kp->binarySend()->getText();
      kwp.push_back(cur);
    }
    return kwp;
  }

  virtual antlrcpp::Any visitKeywordPair(
      SmalltalkParser::KeywordPairContext *ctx) override {
    return visitChildren(ctx);
  }

  ExprAST *visitOperandX(SmalltalkParser::OperandContext *ctx) {
    if (auto sub = ctx->subexpression()) {
      return visitSubexpressionX(sub);

    } else if (auto ref = ctx->reference()) {
      string varname = visitVariable(ref->variable());
      // check read var
      { sem.Scopes.back()->usedVars.insert(varname); }

      ExprAST *ret;
      auto *var = new VariableExprAST(getLoc(ctx), varname);
      ret = var;
      return ret;

    } else if (auto lit = ctx->literal()) {
      return visitLiteralX(lit);
    }
  }

  ExprAST *visitSubexpressionX(SmalltalkParser::SubexpressionContext *ctx) {
    return visitExpressionX(ctx->expression());
  }

  ExprAST *visitLiteralX(SmalltalkParser::LiteralContext *ctx) {
    if (ctx->runtimeLiteral()) {
      return visitRuntimeLiteralX(ctx->runtimeLiteral());
    } else  //      (ctx->parsetimeLiteral())
    {
      return visitParsetimeLiteralX(ctx->parsetimeLiteral());
    }
  }

  ExprAST *visitRuntimeLiteralX(SmalltalkParser::RuntimeLiteralContext *ctx) {
    if (ctx->dynamicDictionary()) {
      auto exprVec = visitExpressionsX(ctx->dynamicDictionary()->expressions());
      auto dictExpr = new DynamicDictionaryExprAST(getLoc(ctx), exprVec);
      return dictExpr;
    } else if (ctx->dynamicArray()) {
      auto exprVec = visitExpressionsX(ctx->dynamicArray()->expressions());
      auto arrExpr = new DynamicDictionaryExprAST(getLoc(ctx), exprVec);
      return arrExpr;
    } else if (ctx->block()) {
      return visitBlockX(ctx->block());
    }
  }

  BlockExprAST *visitBlockX(SmalltalkParser::BlockContext *ctx) {
    auto loc = getLoc(ctx);
    BlockExprAST *blk = new BlockExprAST(loc, ctx->getText());
    // always add "this" to block code
    // VarType type;
    // type.name = "*i8";
    // VarDeclExprAST arg0(loc, "this", type);
    // blk->addParameter(arg0);
    if (ctx->blockParamList()) {
      auto blkParams = visitBlockParamListX(ctx->blockParamList());
      for (auto param : blkParams) blk->addParameter(param);
    }

    if (ctx->temps()) {
      auto blkTemps = visitTempsX(ctx->temps());
      for (auto local : blkTemps) {
        // cout << "DEBUG BlockExprAST addLocal: " << local.getName() << endl;
        blk->addLocal(local);
      }
    }

    // if(sem.curScope == NULL)
    //    cout << "bug at visitBlock" << endl;
    blk->parentScope = sem.Scopes.back();
    // cout << "Creating block, parent scope: " << sem.curScopeName.back()
    //      << endl;  //<< sem.curScope->function_name << endl;

    if (ctx->sequence()) {
      sem.curScopeName.push_back("AnonymousFunc" + to_string(sem.blockNum));
      FunctionAST *func = visitSequenceX(ctx->sequence());
      auto funcName = "AnonymousFunc" + to_string(++sem.blockNum);
      func->function_name = funcName;
      blk->func = func;
      // cout << "Adding block anonymous function " << funcName << endl;
      sem.curScopeName.pop_back();

      // find outer var: not in locals or params
      for (auto varname : func->usedVars) {
        if (blk->isLocal(varname) || blk->isParameter(varname) ||
            blk->isBuiltInClass(varname))
          continue;
        // add to parent scope
        blk->parentScope->addUsedVar(varname);
        if (!existInCurScope(func, varname)) {
          // cout << "DEBUG BlockExprAST outerVars: " << varname << endl;
          blk->outerVars.insert(varname);
        }
      }
    }

    return blk;
    //            return visitChildren(ctx);
  }

  vector<VarDeclExprAST> visitBlockParamListX(
      SmalltalkParser::BlockParamListContext *ctx) {
    vector<VarDeclExprAST> blkparam;
    for (auto param : ctx->BLOCK_PARAM()) {
      string para = param->getText();
      // cout << "Block Parameter: " << para << endl;
      para = para.substr(1);  // skip :
      VarType type;
      type.name = "*i8";
      VarDeclExprAST arg(getLoc(ctx), para, type);
      blkparam.push_back(arg);
    }
    return blkparam;
    //            return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDynamicDictionary(
      SmalltalkParser::DynamicDictionaryContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDynamicArray(
      SmalltalkParser::DynamicArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  ExprAST *visitParsetimeLiteralX(
      SmalltalkParser::ParsetimeLiteralContext *ctx) {
    std::string text = ctx->getText();
    if (text.size() > MAX_TEXT_LENGTH) text = text.substr(0, MAX_TEXT_LENGTH);

    if (ctx->number()) {
      // auto expr = new NumberExprAST(getLoc(ctx), std::stol(ctx->getText()));
      // change NumberExprAST to StringExprAST, deal with stol and others
      auto expr = new StringExprAST(getLoc(ctx), text);
      return expr;
    } else if (ctx->symbol()) {
      auto expr = new SymbolExprAST(getLoc(ctx), text);
      return expr;
    } else if (ctx->pseudoVariable()) {
      auto expr = new ReservedKeywordExprAST(getLoc(ctx), ctx->getText());
      return expr;
    } else if (ctx->string()) {
      auto expr = new StringExprAST(getLoc(ctx), text);
      return expr;
    } else {
      auto expr = new LiteralExprAST(getLoc(ctx), text);
      return expr;
    }
    //            return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNumber(
      SmalltalkParser::NumberContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNumberExp(
      SmalltalkParser::NumberExpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCharConstant(
      SmalltalkParser::CharConstantContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitHex_(SmalltalkParser::Hex_Context *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStInteger(
      SmalltalkParser::StIntegerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStFloat(
      SmalltalkParser::StFloatContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPseudoVariable(
      SmalltalkParser::PseudoVariableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitString(
      SmalltalkParser::StringContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSymbol(
      SmalltalkParser::SymbolContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPrimitive(
      SmalltalkParser::PrimitiveContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBareSymbol(
      SmalltalkParser::BareSymbolContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLiteralArray(
      SmalltalkParser::LiteralArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLiteralArrayRest(
      SmalltalkParser::LiteralArrayRestContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBareLiteralArray(
      SmalltalkParser::BareLiteralArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnaryTail(
      SmalltalkParser::UnaryTailContext *ctx) override {
    vector<string> ret;
    ret.push_back(visitUnaryMessage(ctx->unaryMessage()));
    if (ctx->unaryTail()) {
      auto nctx = ctx->unaryTail();
      while (nctx) {
        ret.push_back(visitUnaryMessage(nctx->unaryMessage()));
        nctx = nctx->unaryTail();
      }
    }
    //                mes->next = visitUnaryTail(ctx->unaryTail());
    return ret;
    //            return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnaryMessage(
      SmalltalkParser::UnaryMessageContext *ctx) override {
    auto text = ctx->unarySelector()->IDENTIFIER()->getText();
    // if (text.size() > MAX_TEXT_LENGTH) text = text.substr(0,
    // MAX_TEXT_LENGTH);

    return text;
    //            return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnarySelector(
      SmalltalkParser::UnarySelectorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitKeywords(
      SmalltalkParser::KeywordsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitReference(
      SmalltalkParser::ReferenceContext *ctx) override {
    return visitChildren(ctx);
  }

  BinaryMessageAST *visitBinaryTailX(SmalltalkParser::BinaryTailContext *ctx) {
    //            string tmpopd = ctx->binaryMessage()->unarySend()->getText();
    //            string tmpopr;
    //            mes->opd = ctx->binaryMessage()->unarySend();
    BinaryMessageAST *mes = visitBinaryMessage(ctx->binaryMessage());
    if (ctx->binaryTail()) {
      mes->next = visitBinaryTailX(ctx->binaryTail());
    }
    return mes;
  }

  virtual antlrcpp::Any visitBinaryMessage(
      SmalltalkParser::BinaryMessageContext *ctx) override {
    BinaryMessageAST *mes = new BinaryMessageAST(getLoc(ctx));

    if (ctx->MINUS()) {
      mes->opr = ctx->MINUS()->getText();
    } else if (ctx->PIPE()) {
      mes->opr = ctx->PIPE()->getText();
    } else if (ctx->PIPE2()) {
      mes->opr = ctx->PIPE2()->getText();
    } else if (ctx->PIPE3()) {
      mes->opr = ctx->PIPE3()->getText();
    } else if (ctx->AMP()) {
      mes->opr = ctx->AMP()->getText();
    } else if (ctx->LLT()) {
      mes->opr = ctx->LLT()->getText();
    } else if (ctx->GGT()) {
      mes->opr = ctx->GGT()->getText();
    } else if (ctx->LTE()) {
      mes->opr = ctx->LTE()->getText();
    } else if (ctx->GTE()) {
      mes->opr = ctx->GTE()->getText();
    } else if (ctx->LT()) {
      mes->opr = ctx->LT()->getText();
    } else if (ctx->GT()) {
      mes->opr = ctx->GT()->getText();
    } else if (ctx->NEQ()) {
      mes->opr = ctx->NEQ()->getText();
    } else {
      mes->opr = ctx->BINARY_SELECTOR()->getText();
      //                tmpopr =
      //                ctx->binaryMessage()->BINARY_SELECTOR()->getText();
    }
    if (mes->opr.size() > MAX_TEXT_LENGTH)
      mes->opr = mes->opr.substr(0, MAX_TEXT_LENGTH);
    if (ctx->operand()) {
      visitOperandX(ctx->operand());
    }
    if (ctx->unarySend()) {
      mes->operand = visitUnarySendX(ctx->unarySend());
    }
    return mes;
    //            return visitChildren(ctx);
  }
};

#endif  // MLIR_ST_PARSER_VISITOR_H_