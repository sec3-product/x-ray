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

using CollectUnsafeOperationFunc = std::function<void(Event *, SVE::Type, int)>;
using CollectUntrustfulAccountFunc = std::function<void(
    llvm::StringRef, const Event *, SVE::Type, int, std::string)>;

// RuleContext encapsulates the context information when evaluating a rule.
class RuleContext {
public:
  using CreateReadEventFunc = std::function<Event *(const llvm::Instruction *)>;
  using IsInLoopFunc = std::function<bool()>;
  using GetLastInstFunc = std::function<const llvm::Instruction *()>;

  RuleContext(const llvm::Function *func, const llvm::Instruction *inst,
              FuncArgTypesMap &funcArgTypesMap, StaticThread *thread,
              CreateReadEventFunc createReadEvent, IsInLoopFunc isInLoop,
              GetLastInstFunc getLastInst,
              CollectUnsafeOperationFunc collectUnsafeOperation,
              CollectUntrustfulAccountFunc collectUntrustfulAccount)
      : Func(func), Inst(inst), FuncArgTypesMap(funcArgTypesMap),
        Thread(thread), CreateReadEvent(createReadEvent), IsInLoop(isInLoop),
        GetLastInst(getLastInst),
        CollectUnsafeOperation(collectUnsafeOperation),
        CollectUntrustfulAccount(collectUntrustfulAccount) {}

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
  virtual const llvm::Instruction *getLastInst() const { return GetLastInst(); }

  virtual void collectUnsafeOperation(SVE::Type type, int size) const {
    auto e = createReadEvent();
    CollectUnsafeOperation(e, type, size);
  }
  virtual void collectUntrustfulAccount(llvm::StringRef name, SVE::Type type,
                                        int size,
                                        std::string additionalInfo) const {
    auto e = createReadEvent();
    CollectUntrustfulAccount(name, e, type, size, additionalInfo);
  }

private:
  const llvm::Function *Func;
  const llvm::Instruction *Inst;
  FuncArgTypesMap &FuncArgTypesMap;
  StaticThread *Thread;
  CreateReadEventFunc CreateReadEvent;
  IsInLoopFunc IsInLoop;
  GetLastInstFunc GetLastInst;
  CollectUnsafeOperationFunc CollectUnsafeOperation;
  CollectUntrustfulAccountFunc CollectUntrustfulAccount;
};

class Rule {
public:
  using Handler = std::function<bool(const RuleContext &, const CallSite &)>;

  Rule(Handler handler) : HandlerFunc(handler) {}

  // Checks whether the given CallSite matches the rule entrypoint and
  // processes the rule if it does. It returns true if the CallSite was fully
  // handled and no further rules should be processed, false otherwise.
  bool handle(const RuleContext &RC, const CallSite &CS) const {
    return HandlerFunc(RC, CS);
  }

private:
  Handler HandlerFunc;
};

// Utility functions.
bool isUpper(const std::string &s);
bool isAllCapitalOrNumber(const std::string &s);
bool isAllCapital(const std::string &s);
bool isNumber(const std::string &s);
bool isConstant(const std::string &s);

} // namespace xray
