#pragma once

#include <functional>
#include <map>
#include <utility>
#include <vector>

#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include "Graph/Event.h"
#include "SVE.h"
#include "SourceInfo.h"

namespace xray {

class StaticThread;

using FunctionFieldsMap =
    std::map<const llvm::Function *,
             std::vector<std::pair<llvm::StringRef, llvm::StringRef>>>;
using FuncArgTypesMap =
    std::map<const llvm::Function *,
             std::vector<std::pair<llvm::StringRef, llvm::StringRef>>>;

// RuleContext encapsulates the context information when evaluating a rule.
class RuleContext {
public:
  using CollectUnsafeOperationFunc =
      std::function<void(Event *, SVE::Type, int)>;
  using CreateReadEventFunc = std::function<Event *(const llvm::Instruction *)>;
  using IsInLoopFunc = std::function<bool()>;

  RuleContext(const llvm::Function *func, const llvm::Instruction *inst,
              FuncArgTypesMap &funcArgTypesMap, StaticThread *thread,
              CreateReadEventFunc createReadEvent, IsInLoopFunc isInLoop,
              CollectUnsafeOperationFunc collectUnsafeOperation)
      : Func(func), Inst(inst), FuncArgTypesMap(funcArgTypesMap),
        Thread(thread), CreateReadEvent(createReadEvent), IsInLoop(isInLoop),
        CollectUnsafeOperation(collectUnsafeOperation) {}

  const llvm::Function *getFunc() const { return Func; }
  const llvm::Instruction *getInst() const { return Inst; }
  FuncArgTypesMap &getFuncArgTypesMap() const { return FuncArgTypesMap; }
  StaticThread *getThread() const { return Thread; }
  Event *createReadEvent() const { return CreateReadEvent(Inst); }

  virtual bool isSafeType(const llvm::Value *value) const;
  virtual bool isSafeVariable(const llvm::Value *value) const;
  virtual bool hasValueLessMoreThan(const llvm::Value *value,
                                    bool isLess) const;
  virtual bool isInLoop() const { return IsInLoop(); }

  virtual void collectUnsafeOperation(SVE::Type type, int size) const {
    auto e = createReadEvent();
    CollectUnsafeOperation(e, type, size);
  }

private:
  const llvm::Function *Func;
  const llvm::Instruction *Inst;
  FuncArgTypesMap &FuncArgTypesMap;
  StaticThread *Thread;
  CreateReadEventFunc CreateReadEvent;
  IsInLoopFunc IsInLoop;
  CollectUnsafeOperationFunc CollectUnsafeOperation;
};

class Rule {
public:
  using Matcher = std::function<bool(const CallSite &)>;
  using Handler = std::function<void(const RuleContext &, const CallSite &)>;

  Rule(Matcher matcher, Handler handler)
      : MatcherFunc(matcher), HandlerFunc(handler) {}

  bool match(const CallSite &CS) const { return MatcherFunc(CS); }
  void handle(const RuleContext &RC, const CallSite &CS) const {
    HandlerFunc(RC, CS);
  }

private:
  Matcher MatcherFunc;
  Handler HandlerFunc;
};

// Utility functions.
bool isUpper(const std::string &s);
bool isAllCapitalOrNumber(const std::string &s);
bool isAllCapital(const std::string &s);
bool isNumber(const std::string &s);
bool isConstant(const std::string &s);

} // namespace xray
