#pragma once

#include <stdint.h>

#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include <llvm/Analysis/ScopedNoAliasAA.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/IR/Module.h>
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/Transforms/Utils/Local.h>

#include "DebugFlags.h"
#include "Graph/Event.h"
#include "PTAModels/GraphBLASModel.h"
#include "PointerAnalysis/PointerAnalysisPass.h"
#include "Rules/Ruleset.h"
#include "SVE.h"
#include "StaticThread.h"

namespace xray {

class Event;
class ReachGraph;

class SolanaAnalysisPass : public llvm::ModulePass {
public:
  explicit SolanaAnalysisPass() : llvm::ModulePass(ID) {}
  static char ID;

  bool runOnModule(llvm::Module &module) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<PointerAnalysisPass<PTA>>(); // need pointer analysis
    AU.addRequired<llvm::TypeBasedAAWrapperPass>();
    AU.setPreservesAll(); // does not transform the LLVM module.
  }

  void initialize(SVE::Database sves, int limit);

  void traverseFunction(
      const xray::ctx *ctx, const llvm::Function *f, StaticThread *thread,
      std::vector<const llvm::Function *> &callStack,
      std::map<uint8_t, const llvm::Constant *> *constArgs = nullptr);

  PTA *pta;

private:
  void traverseFunctionWrapper(
      const xray::ctx *ctx, StaticThread *thread,
      std::vector<const llvm::Function *> &callStack,
      const llvm::Instruction *inst, const llvm::Function *f,
      std::map<uint8_t, const llvm::Constant *> *constArgs = nullptr);

  void handleRustModelAPI(const xray::ctx *ctx, TID tid, llvm::Function *func,
                          const llvm::Instruction *inst, StaticThread *thread,
                          CallSite CS, bool isMacroArrayRefUsedInFunction);
  void handleNonRustModelAPI(const xray::ctx *ctx, TID tid,
                             llvm::Function *func, const Instruction *inst,
                             StaticThread *thread, CallSite CS);

  void detectUntrustfulAccounts();
  void detectCosplayAccounts(const xray::ctx *ctx, TID tid);

  std::set<const llvm::Function *> threadStartFunctions;
  StaticThread *forkNewThread(ForkEvent *forkEvent);
  TID addNewThread(ForkEvent *fork);
  bool addThreadStartFunction(const llvm::Function *func);
  bool hasThreadStartInitFunction(std::string symbol) const;
  bool isInitFunction(llvm::StringRef funcName) const;

  void initStructFunctions();
  void updateAccountStates(StaticThread *curThread);

  // Global state accounts.
  std::set<llvm::StringRef> globalStateAccounts;

  // FunctionFieldsMap is defined in Rules/CosplayDetector.h.
  FunctionFieldsMap anchorStructFunctionFieldsMap;
  FunctionFieldsMap normalStructFunctionFieldsMap;

  bool accountTypeContainsMoreThanOneMint(llvm::StringRef structName) const;
  bool isAnchorDataAccount(llvm::StringRef accountName) const;

  std::map<llvm::StringRef, std::vector<llvm::StringRef>> accountsPDASeedsMap;

  std::set<llvm::StringRef> potentialOwnerAccounts;
  std::map<llvm::StringRef, const Event *> accountsPDAMap;
  std::map<llvm::StringRef, const Event *> accountsSeedProgramAddressMap;

  bool isAccountPDA(llvm::StringRef accountName) const;
  bool isAccountUsedInSeedsProgramAddress(llvm::StringRef accountName) const;

  void updateKeyEqualMap(StaticThread *thread, const Event *e, bool isEqual,
                         llvm::StringRef valueName1,
                         llvm::StringRef valueName2);

  void addCheckKeyEqual(const xray::ctx *ctx, TID tid,
                        const llvm::Instruction *inst, StaticThread *thread,
                        CallSite &CS);
  void handleConditionalCheck0(const xray::ctx *ctx, TID tid,
                               const llvm::Function *func,
                               const llvm::Instruction *inst,
                               StaticThread *thread, const llvm::Value *value);

  const llvm::Function *findCallStackNonAnonFunc(const Event *e);
  llvm::StringRef findCallStackAccountAliasName(const llvm::Function *func,
                                                const Event *e,
                                                llvm::StringRef valueName,
                                                bool stripKey = true);

  llvm::StringRef findNewStructAccountName(TID tid,
                                           const llvm::Instruction *inst,
                                           const llvm::StringRef name);

  const llvm::Module *thisModule;
  // Type-based Alias Analysis
  llvm::TypeBasedAAResult *tbaa;
  // used by tbaa
  llvm::SimpleAAQueryInfo aaqi;
  bool hasFoundThread = false;
  // collection of all static threads
  std::vector<StaticThread *> threadSet;
  // list of forked thread that haven't been visited yet
  std::queue<StaticThread *> threadList;
  FuncArgTypesMap funcArgTypesMap;

  uint8_t for_loop_counter = 0;
  bool isInLoop() { return for_loop_counter > 0; }

  // A per-thread callEventTrace. Each callEvent has an endID. endID represents
  // the last EventID belongs to the current function. The call stack of an
  // event is computed in the following way:
  // 1. get the callEventTrace from the current thread
  // 2. traverse the callEvents from the start of the trace
  // 3. the endID of a callEvent >= the target event ID
  // 4. push this callEvent into the stack trace
  // 5. stop when we hit a callEvent whose ID is larger than the target event ID
  std::map<TID, std::vector<CallEvent *>> callEventTraces;
  std::vector<const Function *> callStack;

  std::map<const llvm::Value *, ForkEvent *> threadIDValueMap;
  std::map<const llvm::Function *, ForkEvent *> threadIDFunctionMap;

  // reachability graph (static happens-before graph)
  ReachGraph *graph;

  Ruleset nonRustModelRuleset;
  Ruleset rustModelRuleset;
};

extern void computeCargoTomlConfig(llvm::Module *module);

extern std::string CONFIG_OUTPUT_PATH;
extern std::string TARGET_MODULE_PATH;
extern unsigned int NUM_OF_ATTACK_VECTORS;
extern unsigned int NUM_OF_IR_LINES;
extern int FUNC_COUNT_BUDGET;

extern bool ConfigCheckUncheckedAccount;
extern bool hasOverFlowChecks;
extern bool anchorVersionTooOld;
extern bool splVersionTooOld;
extern bool solanaVersionTooOld;

} // namespace xray
