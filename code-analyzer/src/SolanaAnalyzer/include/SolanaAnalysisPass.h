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

namespace aser {

class Event;
class ReachGraph;

class SolanaAnalysisPass : public llvm::ModulePass {
public:
  explicit SolanaAnalysisPass() : llvm::ModulePass(ID) {}
  static char ID;

  void initialize(SVE::Database sves, int limit);

  // A per thread callEventTrace
  // each callEvent has an endID
  // endID represents the last EventID belongs to the current function
  // the call stack of an event is computed in the following way:
  // 1. get the callEventTrace from the current thread
  // 2. traverse the callEvents from the start of the trace
  // 3. the endID of a callEvent >= the target event ID
  // 4. push this callEvent into the stack trace
  // 5. stop when we hit a callEvent whose ID is larger than the target event ID
  std::map<TID, std::vector<CallEvent *>> callEventTraces;
  std::vector<const Function *> callStack;

  // Abstract Object ~> Thread Id ~> Event (read/write insts)
  std::vector<const ObjTy *> objs;
  llvm::DenseMap<const ObjTy *, unsigned int> objIdxCache;
  std::map<unsigned int, std::map<TID, std::vector<MemAccessEvent *>>>
      memWrites;
  std::map<unsigned int, std::map<TID, std::vector<MemAccessEvent *>>> memReads;

  std::vector<unsigned int> sharedObjIdxs;
  std::map<unsigned int, std::map<TID, std::vector<MemAccessEvent *>>>
      memWritesMask;
  std::map<unsigned int, std::map<TID, std::vector<MemAccessEvent *>>>
      memReadsMask;

  std::map<const llvm::Value *, ForkEvent *> threadIDValueMap;
  std::map<const llvm::Value *, std::set<ForkEvent *>> threadIDValue2ndMap;
  std::map<StringRef, ForkEvent *> threadIDValueNameMap;
  std::map<TID, TID> twinTids; // if two identical threads are created in a loop
  std::map<const llvm::Function *, ForkEvent *> threadIDFunctionMap;

  PTA *pta;
  // reachability graph (static happens-before graph)
  ReachGraph *graph;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<PointerAnalysisPass<PTA>>(); // need pointer analysis
    AU.addRequired<llvm::TypeBasedAAWrapperPass>();
    AU.setPreservesAll(); // does not transform the LLVM module.
  }

  bool runOnModule(llvm::Module &module) override;

  StaticThread *forkNewThread(ForkEvent *forkEvent);

  void detectRaceCondition(const aser::ctx *ctx, TID tid);
  void detectDeadCode(const aser::ctx *ctx, TID tid);
  void detectAccountsCosplay(const aser::ctx *ctx, TID tid);

  void printStatistics();

  void traverseFunction(
      const aser::ctx *ctx, const llvm::Function *f, StaticThread *thread,
      std::vector<const llvm::Function *> &callStack,
      std::map<uint8_t, const llvm::Constant *> *constArgs = nullptr);

private:
  const llvm::Module *thisModule;
  // Type-based Alias Analysis
  llvm::TypeBasedAAResult *tbaa;
  // used by tbaa
  llvm::SimpleAAQueryInfo aaqi;
  bool hasFoundThread = false;
  // collection of all static threads
  std::vector<StaticThread *> threadSet;
  uint8_t for_loop_counter = 0;
  bool isInLoop() { return for_loop_counter > 0; }

  std::set<const llvm::Function *> exploredIndirectTarget;
  std::set<const llvm::Function *> threadStartFunctions;

  bool addThreadStartFunction(const llvm::Function *func) {
    if (threadStartFunctions.find(func) != threadStartFunctions.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "addThreadStartFunction false: " << func->getName()
                     << "\n";
      return false;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "addThreadStartFunction: " << func->getName() << "\n";
      threadStartFunctions.insert(func);
      return true;
    }
  }
  bool hasThreadStartInitFunction(std::string symbol) {
    for (auto func : threadStartFunctions) {
      if (func->getName().contains(symbol)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "hasThreadStartInitFunction: " << func->getName()
                       << " " << symbol << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "hasThreadStartInitFunction false: " << symbol << "\n";
    return false;
  }
  std::set<llvm::StringRef> globalStateAccounts;
  bool isGlobalStateAccount(llvm::StringRef accountName) {
    if (globalStateAccounts.find(accountName) != globalStateAccounts.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isGlobalStateAccount: " << accountName << "\n";

      return true;
    } else {
      if (DEBUG_RUST_API)
        llvm::outs() << "isGlobalStateAccount false: " << accountName << "\n";
    }
    return false;
  }

  FuncArgTypesMap funcArgTypesMap;

  // list of forked thread that haven't been visited yet
  std::queue<StaticThread *> threadList;
  std::map<const llvm::Function *,
           std::vector<std::pair<llvm::StringRef, llvm::StringRef>>>
      normalStructFunctionFieldsMap;
  std::map<const llvm::Function *,
           std::vector<std::pair<llvm::StringRef, llvm::StringRef>>>
      anchorStructFunctionFieldsMap;
  void initStructFunctions();

  bool accountTypeContainsMoreThanOneMint(llvm::StringRef struct_name) {
    auto num_mints = 0;
    std::string func_struct_name =
        "sol.model.struct.anchor." + struct_name.str();
    auto func_struct = thisModule->getFunction(func_struct_name);
    if (func_struct) {
      auto fieldTypes = anchorStructFunctionFieldsMap[func_struct];
      for (auto pair : fieldTypes) {
        if (pair.first.contains("mint") && pair.second.equals("Pubkey"))
          num_mints++;
      }
    }
    if (num_mints > 1)
      return true;

    return false;
  }

  bool isAnchorStructFunction(const llvm::Function *func) {
    return anchorStructFunctionFieldsMap.find(func) !=
           anchorStructFunctionFieldsMap.end();
  }

  bool isAnchorTokenProgram(llvm::StringRef accountName) {
    for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
      for (auto pair : fieldTypes) {
        if (pair.first.equals(accountName) &&
            pair.second.equals("Program<'info, Token>"))
          return true;
      }
    }
    return false;
  }
  bool isAnchorTokenAccount(llvm::StringRef accountName) {
    for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
      for (auto pair : fieldTypes) {
        if (pair.first.equals(accountName) &&
            pair.second.contains("TokenAccount>"))
          return true;
      }
    }
    return false;
  }

  bool isAnchorValidatedAccount(llvm::StringRef accountName) {
    for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
      for (auto pair : fieldTypes) {
        if (pair.first.equals(accountName) &&
            (pair.second.contains("Program<") ||
             pair.second.contains("Sysvar<")))
          return true;
      }
    }
    return false;
  }
  bool isAnchorDataAccount(llvm::StringRef accountName) {
    for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
      for (auto pair : fieldTypes) {
        if (pair.first.equals(accountName) &&
            (pair.second.startswith("Box<Account<'info") ||
             pair.second.startswith("Account<'info")))
          return true;
      }
    }
    return false;
  }

  std::map<llvm::StringRef, const Event *> accessControlMap;
  bool isAccessControlInstruction(llvm::StringRef sig) {
    for (auto [name, e] : accessControlMap) {
      if (name.contains(sig)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccessControlInstruction: " << sig << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccessControlInstruction false: " << sig << "\n";

    return false;
  }

  std::map<const llvm::Function *, std::set<llvm::StringRef>>
      potentialInitOrderRelatedAccountsMap;
  std::map<const llvm::Function *, std::set<llvm::StringRef>>
      potentialCancelOrderRelatedAccountsMap;
  std::map<const llvm::Function *, std::set<llvm::StringRef>>
      potentialExchangeOrderRelatedAccountsMap;

  std::map<llvm::StringRef, std::vector<llvm::StringRef>> accountsPDASeedsMap;
  bool isCompatibleSeeds(llvm::StringRef seed, llvm::StringRef seed2) {
    // llvm::outs() << "isCompatibleSeeds seed: " << seed << " seed2: " << seed2
    // << "\n";

    if (seed == seed2)
      return false; // skip identical seeds
    if (seed.startswith("b\"") && seed2.startswith("b\"")) {
      seed = seed.substr(0, seed.find_last_of("\""));
      seed2 = seed2.substr(0, seed2.find_last_of("\""));
      if (!seed.contains(seed2) && !seed2.contains(seed))
        return false;
    } else if (seed.contains("as_ref()") && seed2.contains("as_ref()")) {
      seed = seed.substr(0, seed.find_last_of("."));
      seed2 = seed2.substr(0, seed2.find_last_of("."));
      // llvm::outs() << "isCompatibleSeeds2 seed: " << seed << " seed2: " <<
      // seed2 << "\n";

      if (!seed.contains(seed2) && !seed2.contains(seed))
        return false;
    } else if (seed.contains("as_bytes()") && seed2.contains("as_bytes()")) {
      // TODO: check string constants
      // pub const CREATOR_SEED: &str = "creator";
      seed = seed.substr(0, seed.find_last_of("."));
      seed2 = seed2.substr(0, seed2.find_last_of("."));
      if (!seed.contains(seed2) && !seed2.contains(seed))
        return false;
    }

    return true;
  }
  std::set<llvm::StringRef> userProvidedInputStringPDAAccounts;

  std::set<llvm::StringRef> potentialOwnerAccounts;
  std::map<llvm::StringRef, const Event *> accountsPDAMap;
  bool isAccountPDA(llvm::StringRef accountName) {
    for (auto [accountPda, inst] : accountsPDAMap) {
      if (accountPda.contains(accountName)) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountPDA: " << accountName << "\n";
        return true;
      }
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountPDA false: " << accountName << "\n";

    return false;
  }
  std::map<llvm::StringRef, const Event *> accountsSeedProgramAddressMap;
  bool isAccountUsedInSeedsProgramAddress(llvm::StringRef accountName) {
    if (accountsSeedProgramAddressMap.find(accountName) !=
        accountsSeedProgramAddressMap.end()) {
      if (DEBUG_RUST_API)
        llvm::outs() << "isAccountUsedInSeedsProgramAddress: " << accountName
                     << "\n";
      return true;
    }
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountUsedInSeedsProgramAddress false: "
                   << accountName << "\n";

    return false;
  }
  std::map<std::pair<llvm::StringRef, llvm::StringRef>, const Event *>
      assertKeyNotEqualMap;
  bool isAccountKeyNotEqual(llvm::StringRef accountName1,
                            llvm::StringRef accountName2) {
    for (auto [pair, inst] : assertKeyNotEqualMap) {
      if ((pair.first.contains(accountName1.str() + ".key") &&
           pair.second.contains(accountName2.str() + ".key")) ||
          (pair.first.contains(accountName2.str() + ".key") &&
           pair.second.contains(accountName1.str() + ".key"))) {
        if (DEBUG_RUST_API)
          llvm::outs() << "isAccountKeyNotEqual: " << accountName1 << " "
                       << accountName2 << "\n";
        return true;
      }
    }

    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountKeyNotEqual false: " << accountName1 << " "
                   << accountName2 << "\n";

    return false;
  }

  void updateKeyEqualMap(StaticThread *thread, const Event *e, bool isEqual,
                         bool isNotEqual, llvm::StringRef valueName1,
                         llvm::StringRef valueName2);
  void detectUntrustfulAccounts(TID tid);
  TID addNewThread(ForkEvent *fork);

  void addCheckKeyEqual(const aser::ctx *ctx, TID tid,
                        const llvm::Instruction *inst, StaticThread *thread,
                        CallSite &CS);
  void handleConditionalCheck0(const aser::ctx *ctx, TID tid,
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

  void traverseFunctionWrapper(
      const aser::ctx *ctx, StaticThread *thread,
      std::vector<const llvm::Function *> &callStack,
      const llvm::Instruction *inst, const llvm::Function *f,
      std::map<uint8_t, const llvm::Constant *> *constArgs = nullptr);

  std::map<const llvm::Function *, std::set<const llvm::Function *>>
      exclusiveFunctionsMap;
  bool mayBeExclusive(const Event *const e1, const Event *const e2);

  void handleRustModelAPI(const aser::ctx *ctx, TID tid, llvm::Function *func,
                          const llvm::Instruction *inst, StaticThread *thread,
                          CallSite CS, bool isMacroArrayRefUsedInFunction);
  void handleNonRustModelAPI(const aser::ctx *ctx, TID tid,
                             llvm::Function *func, const Instruction *inst,
                             StaticThread *thread, CallSite CS);

  Ruleset nonRustModelRuleset;
};

extern void computeCargoTomlConfig(llvm::Module *module);

extern std::string CONFIG_OUTPUT_PATH;
extern std::string TARGET_MODULE_PATH;
extern unsigned int NUM_OF_ATTACK_VECTORS;
extern unsigned int NUM_OF_IR_LINES;

} // namespace aser
