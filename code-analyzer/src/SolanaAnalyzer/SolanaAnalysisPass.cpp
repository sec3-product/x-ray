#include "SolanaAnalysisPass.h"

#include <chrono>
#include <queue>

#include <Logger/Logger.h>
#include <PointerAnalysis/Program/CallSite.h>
#include <llvm/ADT/IndexedMap.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>

#include "Collectors/CosplayAccounts.h"
#include "Collectors/Output.h"
#include "Collectors/UnsafeOperation.h"
#include "Collectors/UntrustfulAccount.h"
#include "DebugFlags.h"
#include "Graph/Event.h"
#include "Graph/ReachGraph.h"
#include "Graph/Trie.h"
#include "PTAModels/GraphBLASModel.h"
#include "Rules/Ruleset.h"
#include "SVE.h"
#include "StaticThread.h"

using namespace llvm;
using namespace xray;

namespace xray {

std::string CONFIG_OUTPUT_PATH;
std::string TARGET_MODULE_PATH;
unsigned int NUM_OF_IR_LINES;
unsigned int NUM_OF_ATTACK_VECTORS;
int FUNC_COUNT_BUDGET;

bool ConfigCheckUncheckedAccount;
bool hasOverFlowChecks;
bool anchorVersionTooOld;
bool splVersionTooOld;
bool solanaVersionTooOld;

} // namespace xray

EventID Event::ID_counter = 0;
llvm::StringRef stripSelfAccountName(llvm::StringRef account_name) {
  if (account_name.find("self.") != std::string::npos)
    account_name = account_name.substr(5);
  return account_name;
}

llvm::StringRef stripCtxAccountsName(llvm::StringRef account_name) {
  if (account_name.contains("ctx.accounts.")) {
    // for anchor if not matched..
    auto found = account_name.find("ctx.accounts.");
    account_name = account_name.substr(found + 13);
  } else if (account_name.contains("ctx.")) {
    auto found = account_name.find("ctx.");
    account_name = account_name.substr(found + 4);
  }
  return account_name;
}

llvm::StringRef stripToAccountInfo(llvm::StringRef account_name) {
  auto found = account_name.find(".to_account_info()");
  if (found != std::string::npos) {
    account_name = account_name.substr(0, found);
  }
  return account_name;
}

llvm::StringRef getStructName(llvm::StringRef valueName) {
  // Context<'_, '_, '_, 'info, Auth<'info>>
  auto found_left = valueName.find("<");
  auto found_right = valueName.find(">");
  auto struct_name =
      valueName.substr(found_left + 1, found_right - found_left - 1);
  // parse further
  if (struct_name.contains("<")) {
    struct_name = struct_name.substr(0, struct_name.find("<"));
    llvm::SmallVector<StringRef, 8> struct_name_vec;
    struct_name.split(struct_name_vec, ',', -1, false);
    struct_name = struct_name_vec.back();
  }
  return struct_name;
}

llvm::StringRef stripDotKey(llvm::StringRef valueName) {
  auto foundDotKey = valueName.find(".key");
  if (foundDotKey != std::string::npos)
    valueName = valueName.substr(0, foundDotKey);
  return valueName;
}

llvm::StringRef stripAtErrorAccountName(llvm::StringRef valueName) {
  auto foundAt = valueName.find("@");
  if (foundAt != std::string::npos)
    valueName = valueName.substr(0, foundAt);
  return valueName;
}

llvm::StringRef stripAccountName(llvm::StringRef valueName) {
  if (valueName.empty())
    return valueName;
  while (valueName.front() == '*' || valueName.front() == '&' ||
         valueName.front() == '(')
    valueName = valueName.substr(1);
  return valueName;
}

llvm::StringRef findMacroAccount(llvm::StringRef account) {
  // find account
  account = stripAccountName(account);
  {
    auto found = account.find_last_of(",");
    if (found != std::string::npos)
      account = account.substr(found + 1);
  }
  {
    auto found = account.find_last_of(".");
    if (found != std::string::npos)
      account = account.substr(found + 1);
  }
  return account;
}

llvm::StringRef stripAll(llvm::StringRef account_name) {
  account_name = stripAccountName(account_name);
  account_name = stripCtxAccountsName(account_name);
  account_name = stripToAccountInfo(account_name);
  return account_name;
}

llvm::StringRef SolanaAnalysisPass::findNewStructAccountName(
    TID tid, const llvm::Instruction *inst, const llvm::StringRef name) {
  llvm::StringRef account_name;
  // from: self.order_vault_account.to_account_info().clone(),
  auto lastInst = inst->getPrevNonDebugInstruction();
  CallSite CS2(lastInst);
  while (!CS2.getTargetFunction()->getName().startswith(
             "sol.model.opaqueAssign") ||
         !LangModel::findGlobalString(CS2.getArgOperand(0)).equals(name)) {
    lastInst = lastInst->getPrevNonDebugInstruction();
    if (lastInst) {
      // llvm::outs() << "lastInst: " << *lastInst << "\n";
      CallSite CS_(lastInst);
      CS2 = CS_;
    } else
      break;
  }
  auto value2 = CS2.getArgOperand(1); // sol.clone
  if (auto arg = llvm::dyn_cast<llvm::Argument>(value2))
    account_name = funcArgTypesMap[inst->getFunction()][arg->getArgNo()].first;
  else if (auto callInst = llvm::dyn_cast<CallBase>(value2)) {
    CallSite CS3(callInst);
    if (CS3.getNumArgOperands() > 0) {
      auto value3 = CS3.getArgOperand(0); // sol.to_account_info
      if (auto arg = llvm::dyn_cast<llvm::Argument>(value3))
        account_name =
            funcArgTypesMap[inst->getFunction()][arg->getArgNo()].first;
      else {
        if (auto callInst4 = dyn_cast<CallBase>(value3)) {
          CallSite CS4(callInst4);
          if (CS4.getNumArgOperands() > 0) {
            auto value4 = CS4.getArgOperand(0);
            account_name = LangModel::findGlobalString(value4);
          }
        } else
          account_name = LangModel::findGlobalString(value3);
      }
    }
  }
  account_name = stripSelfAccountName(account_name);
  if (DEBUG_RUST_API)
    llvm::outs() << "tid: " << tid << " " << name << ": " << account_name
                 << " call: " << *inst << "\n";
  return stripCtxAccountsName(account_name);
}

std::pair<llvm::StringRef, const llvm::Instruction *>
getProgramIdAccountName(const llvm::Instruction *inst) {
  if (auto previousInst =
          dyn_cast<CallBase>(inst->getPrevNonDebugInstruction())) {
    auto prevCallName = previousInst->getCalledFunction()->getName();
    if (prevCallName.startswith("sol.model.struct.new.Instruction.3")) {
      auto value = previousInst->getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      if (valueName == "program_id") {
        // find all the prev sol.model.opaqueAssign with program_id
        auto prevInst = inst->getPrevNonDebugInstruction();
        while (prevInst) {
          if (auto callValue2 = dyn_cast<CallBase>(prevInst)) {
            CallSite CS2(callValue2);
            if (CS2.getTargetFunction()->getName().equals(
                    "sol.model.opaqueAssign")) {
              auto value1 = CS2.getArgOperand(0);
              auto valueName1 = LangModel::findGlobalString(value1);
              if (valueName1 == "program_id") {
                auto value2 = CS2.getArgOperand(1);
                auto valueName2 = LangModel::findGlobalString(value2);
                // ctx.accounts.candy_machine_program.key
                auto account = stripCtxAccountsName(valueName2);
                auto found = account.find_last_of(".");
                if (found != std::string::npos)
                  account = account.substr(0, found);
                return std::make_pair(account, prevInst);
              }
            }
          }
          prevInst = prevInst->getPrevNonDebugInstruction();
        }
      }
    }
  }
  return std::make_pair("", inst);
}

std::map<llvm::StringRef, const llvm::Function *> FUNC_NAME_MAP;
static const llvm::Function *
getFunctionFromPartialName(llvm::StringRef partialName) {
  for (auto [name, func] : FUNC_NAME_MAP) {
    if (name.contains(partialName) && !name.contains(".anon."))
      return func;
  }
  return nullptr;
}

static const llvm::Function *
getFunctionMatchStartEndName(llvm::StringRef startName,
                             llvm::StringRef endName) {
  for (auto [name, func] : FUNC_NAME_MAP) {
    if (name.startswith(startName) && name.endswith(endName) &&
        !name.contains(".anon."))
      return func;
  }
  return nullptr;
}

// Method to compare two versions.
// Returns 1 if v2 is smaller, -1
// if v1 is smaller, 0 if equal
static int versionCompare(std::string v1,
                          std::string v2) {     // v1: the real version: ^0.20.1
  std::replace(v1.begin(), v1.end(), '*', '0'); // replace all '*' to '0'
  v1.erase(std::remove_if(
               v1.begin(), v1.end(),
               [](char c) { return !(c >= '0' && c <= '9') && c != '.'; }),
           v1.end());
  // llvm::outs() << "v1: " << v1 << "\n";
  // llvm::outs() << "v2: " << v2 << "\n";
  // v2: the target version 3.1.1
  // vnum stores each numeric
  // part of version
  int vnum1 = 0, vnum2 = 0;
  // loop until both string are
  // processed
  for (int i = 0, j = 0; (i < v1.length() || j < v2.length());) {
    // storing numeric part of
    // version 1 in vnum1
    while (i < v1.length() && v1[i] != '.') {
      vnum1 = vnum1 * 10 + (v1[i] - '0');
      i++;
    }
    // storing numeric part of
    // version 2 in vnum2
    while (j < v2.length() && v2[j] != '.') {
      vnum2 = vnum2 * 10 + (v2[j] - '0');
      j++;
    }
    if (vnum1 > vnum2)
      return 1;
    if (vnum2 > vnum1)
      return -1;
    // if equal, reset variables and
    // go for next numeric part
    vnum1 = vnum2 = 0;
    i++;
    j++;
  }
  return 0;
}

static std::map<StringRef, StringRef> CARGO_TOML_CONFIG_MAP;

void xray::computeCargoTomlConfig(llvm::Module *module) {
  auto f = module->getFunction("sol.model.cargo.toml");
  if (!f) {
    return;
  }
  for (auto &BB : *f) {
    for (auto &I : BB) {
      if (isa<CallBase>(&I)) {
        xray::CallSite CS(&I);
        if (CS.getNumArgOperands() < 2)
          continue;
        auto v1 = CS.getArgOperand(0);
        auto v2 = CS.getArgOperand(1);

        auto valueName1 = LangModel::findGlobalString(v1);
        auto valueName2 = LangModel::findGlobalString(v2);
        auto overflow_checks = "profile.release.overflow-checks";
        if (valueName1 == overflow_checks) {
          if (valueName2.contains("1"))
            hasOverFlowChecks = true;
          llvm::outs() << "overflow_checks: " << hasOverFlowChecks << "\n";
        }

        auto anchor_lang_version = "dependencies.anchor-lang.version";
        // prior to 0.24.x, insecure init_if_needed
        if (valueName1 == anchor_lang_version) {
          if (versionCompare(valueName2.str(), "0.24.2") == -1)
            anchorVersionTooOld = true;
          llvm::outs() << "anchor_lang_version: " << valueName2
                       << " anchorVersionTooOld: " << anchorVersionTooOld
                       << "\n";
        }
        auto anchor_spl_version = "dependencies.anchor-spl.version";
        auto spl_token_version = "dependencies.spl-token.version";
        if (valueName1 == spl_token_version) {
          if (versionCompare(valueName2.str(), "3.1.1") == -1)
            splVersionTooOld = true;
          llvm::outs() << "spl_version: " << valueName2
                       << " splVersionTooOld: " << splVersionTooOld << "\n";
        }
        // prior to v3.1.1, insecure spl invoke

        auto solana_program_version = "dependencies.solana_program.version";
        if (valueName1 == solana_program_version) {
          if (versionCompare(valueName2.str(), "1.10.29") == -1)
            solanaVersionTooOld = true;
          llvm::outs() << "solana_version: " << valueName2
                       << " solanaVersionTooOld: " << solanaVersionTooOld
                       << "\n";
        }

        CARGO_TOML_CONFIG_MAP[valueName1] = valueName2;
      }
    }
  }
}

void SolanaAnalysisPass::initialize(SVE::Database sves, int limit) {
  SVE::init(sves);

  // Initialize the rulesets.
  nonRustModelRuleset = Ruleset::createRustModelRuleset();

  auto unlimited = (limit == -1);
  UntrustfulAccount::init(limit, unlimited);
  UnsafeOperation::init(limit, unlimited);
  CosplayAccounts::init(limit, unlimited);
}

static bool isUpper(const std::string &s) {
  return std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return std::isupper(c) || c == '_';
  });
}

const llvm::Function *
SolanaAnalysisPass::findCallStackNonAnonFunc(const Event *e) {
  auto func = e->getInst()->getFunction();
  // skip .anon.
  if (func->getName().contains(".anon.")) {
    auto callTrace = getCallEventStack(e, callEventTraces);
    int i = callTrace.size() - 1; // start index
    for (; i >= 0; i--) {
      func = callTrace.at(i)->getInst()->getFunction();
      if (!func->getName().contains(".anon."))
        break;
    }
  }
  return func;
}

llvm::StringRef SolanaAnalysisPass::findCallStackAccountAliasName(
    const llvm::Function *func, const Event *e, llvm::StringRef valueName,
    bool stripKey) {
  if (DEBUG_RUST_API)
    llvm::outs() << "finding alias account for: " << valueName << "\n";
  if (valueName.empty())
    return valueName;
  // strip & or *
  while (valueName.front() == '*' || valueName.front() == '&')
    valueName = valueName.substr(1);
  auto callTrace = getCallEventStack(e, callEventTraces);
  int i = callTrace.size() - 1; // start index
  // skip .anon.
  if (func->getName().contains(".anon.")) {
    for (; i >= 0; i--) {
      func = callTrace.at(i)->getInst()->getFunction();
      if (!func->getName().contains(".anon."))
        break;
    }
  }

  auto ArgNo = 0;
  auto size = funcArgTypesMap[func].size();
  for (; ArgNo < size; ArgNo++) {
    auto pair = funcArgTypesMap[func][ArgNo];
    if (pair.first.equals(valueName))
      break;
    if (valueName.startswith("ctx.") && pair.first.equals("ctx"))
      break;
    // skip ctx.accounts.
    if (valueName.contains(".")) {
      auto valueName2 = valueName.substr(0, valueName.find("."));
      if (pair.first.equals(valueName2))
        break;
    }
  }
  if (ArgNo < size) {
    if (DEBUG_RUST_API)
      llvm::outs() << "OK - passed as parameter account: " << valueName << "\n";
    auto callTrace = getCallEventStack(e, callEventTraces);
    for (; i >= 0; i--) {
      CallEvent *ce = callTrace.at(i);
      auto *e_caller = ce->getInst();
      if (auto call_site = dyn_cast<CallBase>(e_caller)) {
        CallSite CS4(call_site);
        if (DEBUG_RUST_API) {
          llvm::outs() << "call_site: " << *call_site << "\n";
          llvm::outs() << "ArgNo: " << ArgNo << "\n";
        }

        // skip anon block
        if (CS4.getNumArgOperands() < ArgNo + 1)
          continue;
        if (CS4.getCalledFunction()->getName().contains(".anon."))
          continue;

        auto value_j = CS4.getArgOperand(ArgNo);
        if (auto arg_j = dyn_cast<Argument>(value_j)) {
          ArgNo = arg_j->getArgNo();
        } else {
          // if callbase
          if (auto callValue_j = dyn_cast<CallBase>(value_j)) {
            if (callValue_j->getCalledFunction()->getName().startswith(
                    "sol.clone.") ||
                callValue_j->getCalledFunction()->getName().startswith(
                    "sol.to_account_info.") ||
                callValue_j->getCalledFunction()->getName().startswith(
                    "sol.to_owned.")) {
              value_j = callValue_j->getArgOperand(0);

              if (auto arg_j = dyn_cast<Argument>(value_j)) {
                ArgNo = arg_j->getArgNo();
                continue;
              }

            } else if (callValue_j->arg_empty()) {
              // sol.spl_token::id.0"
              auto valueName_j = callValue_j->getCalledFunction()->getName();
              if (valueName_j.startswith("sol."))
                valueName_j = valueName_j.substr(4);
              if (valueName_j.endswith(".0"))
                valueName_j =
                    valueName_j.substr(0, valueName_j.str().length() - 2);
              if (DEBUG_RUST_API)
                llvm::outs() << "Passed account1: " << valueName_j << "\n";
              return valueName_j;
            }
          }

          auto valueName_j = LangModel::findGlobalString(value_j);
          // llvm::outs() << "value_j: " << *value_j << "\n";
          // llvm::outs() << "valueName_j: " << valueName_j << "\n";
          auto found = valueName_j.find(".key");

          if (found != std::string::npos)
            if (stripKey)
              valueName_j = valueName_j.substr(0, found);

          valueName_j = stripAll(valueName_j);

          if (DEBUG_RUST_API)
            llvm::outs() << "Passed account2: " << valueName_j << "\n";
          return valueName_j;
          // break;
        }
      }
    }
  }
  valueName = stripCtxAccountsName(valueName);
  return valueName;
}

void SolanaAnalysisPass::addCheckKeyEqual(const xray::ctx *ctx, TID tid,
                                          const llvm::Instruction *inst,
                                          StaticThread *thread, CallSite &CS) {
  if (DEBUG_RUST_API)
    llvm::outs() << "assert_eq: " << *inst << "\n";
  // only deal with two or more arguments
  if (CS.getNumArgOperands() < 2)
    return;

  auto value1 = CS.getArgOperand(0);
  auto valueName1 = LangModel::findGlobalString(value1);
  // llvm::outs() << "data1: " << valueName1 << "\n";
  auto value2 = CS.getArgOperand(1);
  auto valueName2 = LangModel::findGlobalString(value2);
  // llvm::outs() << "data2: " << valueName2 << "\n";
  // TODO: argument
  if (auto arg = dyn_cast<Argument>(value1)) {
    // llvm::outs() << "TODO argument data1: " << *value1 << "\n";
    valueName1 = "program_id";
  }
  if (auto arg = dyn_cast<Argument>(value2)) {
    // llvm::outs() << "TODO argument data2: " << *value2 << "\n";
    valueName2 = "program_id";
  }
  auto e = graph->createReadEvent(ctx, inst, tid);
  bool valueKey1 = false;
  bool valueKey2 = false;
  if (valueName1.contains(".key"))
    valueKey1 = true;
  if (valueName2.contains(".key"))
    valueKey2 = true;

  // parcl: require_keys_eq!(self.fee_vault.owner, self.authority.key());
  // saber: assert_keys_eq!(self.mm.pool, self.pool, "mm.pool");
  valueName1 = stripSelfAccountName(valueName1);
  valueName1 = stripAll(valueName1);
  // self.mm,\22miner.authority\22
  auto foundComma = valueName2.find(",");
  if (foundComma != std::string::npos)
    valueName2 = valueName2.substr(0, foundComma);
  valueName2 = stripSelfAccountName(valueName2);
  valueName2 = stripAll(valueName2);

  bool addedKeyEqual = false;
  bool addedOwnerEqual = false;

  if (valueKey1) {
    auto foundDot1 = valueName1.find_last_of(".");
    if (foundDot1 != std::string::npos)
      valueName1 = valueName1.substr(0, foundDot1);
    auto pair = std::make_pair(valueName1, valueName2);
    thread->assertKeyEqualMap[pair] = e;
    addedKeyEqual = true;
  }
  if (valueKey2) {
    auto foundDot2 = valueName2.find_last_of(".");
    if (foundDot2 != std::string::npos)
      valueName2 = valueName2.substr(0, foundDot2);
    auto pair = std::make_pair(valueName2, valueName1);
    thread->assertKeyEqualMap[pair] = e;
    addedKeyEqual = true;
  }

  if (valueName1.endswith(".owner")) {
    auto pair = std::make_pair(valueName1, valueName2);
    thread->assertOwnerEqualMap[pair] = e;
    addedOwnerEqual = true;
  }
  if (valueName2.endswith(".owner")) {
    auto pair = std::make_pair(valueName2, valueName1);
    thread->assertOwnerEqualMap[pair] = e;
    addedOwnerEqual = true;
  }
  if (!addedKeyEqual && !addedOwnerEqual) {
    // llvm::outs() << "addCheckKeyEqual: " << valueName1 << " - " << valueName2
    // << "\n";

    auto pair = std::make_pair(valueName1, valueName2);
    thread->assertOtherEqualMap[pair] = e;

    if (valueName1.contains(".") && !valueName2.contains(".")) {
      auto valueName1_ = valueName1.substr(0, valueName1.find("."));
      auto pair = std::make_pair(valueName1_, valueName2);
      thread->assertKeyEqualMap[pair] = e;
      // if (thread->isInAccountsMap(valueName2))
      {
        auto pair2 = std::make_pair(valueName2, valueName1_);
        thread->assertKeyEqualMap[pair2] = e;
      }
      // if (valueName2.contains("mint") || valueName1.contains("_key")) {
      //     auto pair2 = std::make_pair(valueName2, valueName1_);
      //     thread->assertKeyEqualMap[pair2] = e;
      // }

    } else if (!valueName1.contains(".") && valueName2.contains(".")) {
      auto valueName2_ = valueName2.substr(0, valueName2.find("."));
      auto pair = std::make_pair(valueName2_, valueName1);
      thread->assertKeyEqualMap[pair] = e;
      // if (thread->isInAccountsMap(valueName1))
      {
        auto pair1 = std::make_pair(valueName1, valueName2_);
        thread->assertKeyEqualMap[pair1] = e;
      }
      // if (valueName1.contains("mint") || valueName2.contains("_key")) {
      //     auto pair2 = std::make_pair(valueName1, valueName2_);
      //     thread->assertKeyEqualMap[pair2] = e;
      // }
    }
  }
}

void SolanaAnalysisPass::handleConditionalCheck0(const xray::ctx *ctx, TID tid,
                                                 const llvm::Function *func,
                                                 const llvm::Instruction *inst,
                                                 StaticThread *thread,
                                                 const llvm::Value *value) {
  bool isAssertKeyAdded = false;
  if (auto callValue = dyn_cast<CallBase>(value)) {
    CallSite CS2(callValue);
    if (CS2.getTargetFunction()->getName().equals("sol.!=") ||
        CS2.getTargetFunction()->getName().equals("sol.==")) {
      auto value1 = CS2.getArgOperand(0);
      auto valueName1 = LangModel::findGlobalString(value1);
      auto value2 = CS2.getArgOperand(1);
      auto valueName2 = LangModel::findGlobalString(value2);

      // check for arguments
      if (auto arg1 = dyn_cast<Argument>(value1))
        valueName1 = funcArgTypesMap[func][arg1->getArgNo()].first;
      if (auto arg2 = dyn_cast<Argument>(value2))
        valueName2 = funcArgTypesMap[func][arg2->getArgNo()].first;

      auto e = graph->createReadEvent(ctx, inst, tid);
      // if ctx.accounts.authority.key != &token.owner

      // while, it could be passed as f(account.key)...
      auto account1_raw =
          findCallStackAccountAliasName(func, e, valueName1, false);
      auto account2_raw =
          findCallStackAccountAliasName(func, e, valueName2, false);

      if (DEBUG_RUST_API)
        llvm::outs() << "sol.if data1: " << valueName1
                     << " data2: " << valueName2
                     << " account1_raw: " << account1_raw
                     << " account2_raw: " << account2_raw << "\n";

      bool valueKey1 = false;
      bool valueKey2 = false;
      if (valueName1.contains(".key") || account1_raw.contains(".key"))
        valueKey1 = true;
      if (valueName2.contains(".key") || account2_raw.contains(".key"))
        valueKey2 = true;
      if (valueName1.empty()) {
        if (auto callValue1 = dyn_cast<CallBase>(value1)) {
          if (callValue1->getCalledFunction()->getName().startswith(
                  "sol.key.")) {
            valueName1 =
                LangModel::findGlobalString(callValue1->getArgOperand(0));
            valueKey1 = true;
          }
        } else if (callValue1->getCalledFunction()->getName().startswith(
                       "sol.match.")) {
          valueName1 =
              LangModel::findGlobalString(callValue1->getArgOperand(0));
        }
      }
      if (valueName2.empty()) {
        if (auto callValue2 = dyn_cast<CallBase>(value2)) {
          if (callValue2->getCalledFunction()->getName().startswith(
                  "sol.key.")) {
            valueName2 =
                LangModel::findGlobalString(callValue2->getArgOperand(0));
            valueKey2 = true;
          } else if (callValue2->getCalledFunction()->getName().startswith(
                         "sol.match.")) {
            valueName2 =
                LangModel::findGlobalString(callValue2->getArgOperand(0));
          }
        }
      }
      valueName1 = stripAll(valueName1);
      valueName2 = stripAll(valueName2);
      if (valueKey1 || valueKey2) {
        valueName1 = stripDotKey(valueName1);
        valueName2 = stripDotKey(valueName2);
        //&spl_token::ID == ctx.accounts.token_program.key
        auto pair1 = std::make_pair(valueName1, valueName2);
        auto pair2 = std::make_pair(valueName2, valueName1);
        if (valueKey1)
          thread->assertKeyEqualMap[pair1] = e;
        if (valueKey2)
          thread->assertKeyEqualMap[pair2] = e;

        // can be token_owner_record_info.key != proposal_owner
        auto account1 = valueName1;
        auto account2 = valueName2;
        account1 = stripAll(account1);
        account2 = stripAll(account2);

        auto foundDot1 = account1.find_last_of(".");
        if (foundDot1 != std::string::npos)
          account1 = account1.substr(0, foundDot1);
        if (!thread->isInAccountsMap(account1)) {
          account1 = findCallStackAccountAliasName(func, e, account1);
        }
        auto foundDot2 = account2.find_last_of(".");
        if (foundDot2 != std::string::npos)
          account2 = account2.substr(0, foundDot2);
        if (!thread->isInAccountsMap(account2)) {
          account2 = findCallStackAccountAliasName(func, e, account2);
        }

        if (valueKey1 && valueKey2) {
          auto pairx = std::make_pair(account1, account2);
          thread->checkDuplicateAccountKeyEqualMap[pairx] = e;
          if (DEBUG_RUST_API)
            llvm::outs() << "checkDuplicateAccountKeyEqualMap account1: "
                         << account1 << " account2: " << account2 << "\n";
        }

        if (account1 != valueName1 || account2 != valueName2) {
          auto pair_real1 = std::make_pair(account1, account1);
          auto pair_real2 = std::make_pair(account2, account2);
          if (valueKey1) {
            thread->assertKeyEqualMap[pair_real1] = e;
            if (account2.contains("pda"))
              accountsPDAMap[account1] = e;
          }
          if (valueKey2) {
            thread->assertKeyEqualMap[pair_real2] = e;
            if (account1.contains("pda") || account1.contains("canon_"))
              accountsPDAMap[account2] = e;
          }
        }
      }
      if (valueName1.contains(".owner") || account1_raw.contains(".owner")) {
        // skip user-level .owner
        //&token.owner

        valueName1 = stripAccountName(valueName1);
        auto found1 = valueName1.find_last_of("."); // for Anchor
        if (found1 != std::string::npos) {
          valueName1 = valueName1.substr(0, found1);
        }
        if (!thread->isInAccountsMap(valueName1)) {
          valueName1 = findCallStackAccountAliasName(func, e, valueName1);
          if (!thread->isInAccountsMap(valueName1)) {
            valueName1 = account1_raw;
          }
        }

        if (!thread->isAccountsUnpackedData(valueName1)) {
          auto pair = std::make_pair(valueName1, valueName2);
          thread->assertOwnerEqualMap[pair] = e;
        } else {
        }
      } else if (valueName2.contains(".owner") ||
                 account2_raw.contains(".owner")) {
        // skip user-level .owner
        valueName2 = stripAccountName(valueName2);
        auto found2 = valueName2.find_last_of("."); // for Anchor
        if (found2 != std::string::npos) {
          valueName2 = valueName2.substr(0, found2);
        }
        if (!thread->isInAccountsMap(valueName2)) {
          valueName2 = findCallStackAccountAliasName(func, e, valueName2);
          if (!thread->isInAccountsMap(valueName2)) {
            valueName2 = account2_raw;
          }
        }
        if (!thread->isAccountsUnpackedData(valueName2)) {
          auto pair = std::make_pair(valueName1, valueName2);
          thread->assertOwnerEqualMap[pair] = e;
        }
      } else if (valueName1.contains("bump") || valueName2.contains("bump") ||
                 valueName1.contains("expected_address") ||
                 valueName2.contains("expected_address")) {
        auto pair = std::make_pair(valueName1, valueName2);
        thread->assertBumpEqualMap[pair] = e;
      } else if (valueName1.contains("addr") || valueName2.contains("addr")) {
        auto pair = std::make_pair(valueName1, valueName2);
        thread->assertKeyEqualMap[pair] = e;
        bool accountNew = false;
        if (!thread->isInAccountsMap(valueName1)) {
          valueName1 = findCallStackAccountAliasName(func, e, valueName1);
          // llvm::outs() << "valueName1: " << valueName1 << "\n";
          if (!valueName1.empty())
            accountNew = true;
        }
        if (!thread->isInAccountsMap(valueName2)) {
          valueName2 = findCallStackAccountAliasName(func, e, valueName2);
          // llvm::outs() << "valueName2: " << valueName2 << "\n";
          if (!valueName2.empty())
            accountNew = true;
        }
        if (accountNew) {
          auto pair_new = std::make_pair(valueName1, valueName2);
          thread->assertKeyEqualMap[pair] = e;
        }

      } else {
        auto pair = std::make_pair(valueName1, valueName2);
        thread->assertOtherEqualMap[pair] = e;
      }

      // check PDA here
      // sol.Pubkey::create_program_address
      // if prev inst is sol.model.opaqueAssign
      auto prevInst = callValue->getPrevNonDebugInstruction();
      if (prevInst) {
        if (auto callValue3 = dyn_cast<CallBase>(prevInst)) {
          CallSite CS3(callValue3);
          if (CS3.getTargetFunction()->getName().equals(
                  "sol.model.opaqueAssign")) {
            auto prevPrevInst = prevInst->getPrevNonDebugInstruction();
            if (prevPrevInst == CS3.getArgOperand(1)) {
              if (auto callValue4 = dyn_cast<CallBase>(prevPrevInst)) {
                if (callValue4->getCalledFunction()->getName().contains(
                        ".map_err.")) {
                  CallSite CS4(callValue4);
                  callValue4 = dyn_cast<CallBase>(CS4.getArgOperand(0));
                }

                CallSite CS4(callValue4);
                if (CS4.getTargetFunction()->getName().contains(
                        "::create_program_address.") ||
                    CS4.getTargetFunction()->getName().contains(
                        "::find_program_address.")) {
                  // auto valueName_x =
                  //     LangModel::findGlobalString(CS3.getArgOperand(0));
                  // if (valueName1.contains(valueName_x))
                  //     valueName_x = valueName2;
                  // if (valueName2.contains(valueName_x))
                  //     valueName_x = valueName1;
                  std::array<llvm::StringRef, 2> valueNames = {valueName1,
                                                               valueName2};
                  for (auto valueName_x : valueNames) {
                    valueName_x = stripAll(valueName_x);
                    if (valueName_x.contains(".key"))
                      valueName_x =
                          valueName_x.substr(0, valueName_x.find(".key"));

                    if (DEBUG_RUST_API)
                      llvm::outs()
                          << "PDA valueName_x1: " << valueName_x << "\n";
                    accountsPDAMap[valueName_x] = e;

                    if (!thread->isInAccountsMap(valueName_x)) {
                      valueName_x =
                          findCallStackAccountAliasName(func, e, valueName_x);
                      if (DEBUG_RUST_API)
                        llvm::outs()
                            << "PDA valueName_x2: " << valueName_x << "\n";
                      accountsPDAMap[valueName_x] = e;
                    }
                  }
                }
              }
            }
          }
        }
      }

      isAssertKeyAdded = true;
    } else {
      // set value to real condition
      if (CS2.getNumArgOperands() > 0)
        value = CS2.getArgOperand(0);
    }
  }
  if (!isAssertKeyAdded) {
    auto valueName = LangModel::findGlobalString(value);
    if (valueName.contains("initialized") || valueName.contains(".discrimi"))
      thread->hasInitializedCheck = true;
    //@ctx.accounts.authority.is_signer
    // llvm::outs()
    //     << "ctx.accounts.authority.is_signer valueName : " << valueName <<
    //     "\n";
    auto foundDot = valueName.find_last_of(".");
    if (foundDot != std::string::npos) {
      auto account = valueName.substr(0, foundDot);
      account = stripAll(account);
      auto e = graph->createReadEvent(ctx, inst, tid);
      if (!thread->isInAccountsMap(account)) {
        account = findCallStackAccountAliasName(func, e, account);
      }
      if (account.empty() && func->getName().contains(".anon.")) {
        // hack for if block
        auto prevInst = inst->getPrevNonDebugInstruction();
        while (prevInst) {
          if (auto callValue2 = dyn_cast<CallBase>(prevInst)) {
            CallSite CS2(callValue2);
            if (CS2.getTargetFunction()->getName().equals(
                    "sol.model.opaqueAssign")) {
              auto value1 = CS2.getArgOperand(0);
              auto valueName1 = LangModel::findGlobalString(value1);
              auto value2 = CS2.getArgOperand(1);
              auto valueName2 = LangModel::findGlobalString(value2);

              if (valueName.contains(valueName1))
                account = valueName2;
              if (valueName.contains(valueName2))
                account = valueName1;
              break;
            }
          }
          prevInst = prevInst->getPrevNonDebugInstruction();
        }
        // now check again
        if (!thread->isInAccountsMap(account)) {
          auto funcx = callStack.back();
          for (int k = callStack.size() - 1; k >= 0; k--) {
            funcx = callStack[k];
            if (!funcx->getName().contains(".anon."))
              break;
          }
          account = findCallStackAccountAliasName(funcx, e, account);
        }
      }
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.if valueName: " << valueName
                     << " account: " << account << "\n";
      if (!account.empty()) {
        if (valueName.find(".is_signer") != std::string::npos) {
          thread->asssertSignerMap[account] = e;
        } else if (valueName.find("crimi") != std::string::npos) {
          thread->asssertDiscriminatorMap[account] = e;
        } else {
          thread->asssertOtherMap[account] = e;
        }
      }
    }
  }
}

void SolanaAnalysisPass::updateKeyEqualMap(StaticThread *thread, const Event *e,
                                           bool isEqual, bool isNotEqual,
                                           llvm::StringRef valueName1,
                                           llvm::StringRef valueName2) {
  valueName1 = stripAccountName(valueName1);
  valueName2 = stripAccountName(valueName2);
  if (DEBUG_RUST_API)
    llvm::outs() << "updateKeyEqualMap constraint: " << valueName1 << " - "
                 << valueName2 << "\n";
  bool valueKey1 = false;
  bool valueKey2 = false;
  if (valueName1.contains(".key"))
    valueKey1 = true;
  if (valueName2.contains(".key"))
    valueKey2 = true;
  if (valueKey1 || valueKey2) {
    if (isEqual) {
      if (valueKey1) {
        valueName1 = valueName1.substr(0, valueName1.find(".key"));
        valueName1 = stripAll(valueName1);
        auto pair = std::make_pair(valueName1, valueName2);
        thread->assertKeyEqualMap[pair] = e;
      }
      if (valueKey2) {
        valueName2 = valueName2.substr(0, valueName2.find(".key"));
        valueName2 = stripAll(valueName2);
        auto pair = std::make_pair(valueName2, valueName1);
        thread->assertKeyEqualMap[pair] = e;
      }
    } else if (isNotEqual) {
      // global info
      auto pair = std::make_pair(valueName1, valueName2);
      assertKeyNotEqualMap[pair] = e;
    }

    // for owner_usdh_account.owner == payer.key()
    if (valueName1.endswith(".owner") || valueName2.endswith(".owner")) {
      auto pair_owner = std::make_pair(valueName1, valueName2);
      thread->assertOwnerEqualMap[pair_owner] = e;
    }
  } else if (valueName1.endswith(".owner") || valueName2.endswith(".owner")) {
    auto pair = std::make_pair(valueName1, valueName2);
    thread->assertOwnerEqualMap[pair] = e;
  } else {
    auto pair = std::make_pair(valueName1, valueName2);
    thread->assertOtherEqualMap[pair] = e;
  }
}

static std::map<TID, unsigned int> threadNFuncMap;
static xray::trie::TrieNode *cur_trie;

bool SolanaAnalysisPass::mayBeExclusive(const Event *const e1,
                                        const Event *const e2) {
  std::vector<CallEvent *> callStack1, callStack2;
  getCallEventStackUntilMain(e1, callEventTraces, callStack1);
  getCallEventStackUntilMain(e2, callEventTraces, callStack2);
  if (DEBUG_RUST_API) {
    printCallEventStackTrace(callStack1);
    printCallEventStackTrace(callStack2);
  }
  // find the first divergent insts
  // (invariant: the first divergent insts must be in the same method)
  const Instruction *inst1, *inst2;
  size_t size1 = callStack1.size();
  size_t size2 = callStack2.size();
  size_t i, size = std::min(size1, size2);
  // i should start from 0
  // the 0-th entry for API race detection is not "main" function
  for (i = 0; i < size; i++) {
    if (callStack1[i]->getInst() != callStack2[i]->getInst()) {
      inst1 = callStack1[i]->getInst();
      inst2 = callStack2[i]->getInst();
      break;
    }
  }
  // the two callEventStacks share the same prefix
  if (i == size) {
    inst1 = (i == size1) ? e1->getInst() : callStack1[i]->getInst();
    inst2 = (i == size2) ? e2->getInst() : callStack2[i]->getInst();
  }
  // self race, why bother?
  if (inst1 == inst2) {
    return false;
  }
  if (DEBUG_RUST_API) {
    llvm::outs() << "inst1: " << *inst1 << "\n";
    llvm::outs() << "function1: " << inst1->getFunction()->getName() << "\n";
    llvm::outs() << "inst2: " << *inst2 << "\n";
    llvm::outs() << "function2: " << inst2->getFunction()->getName() << "\n";
  }
  if (inst1->getFunction() == inst2->getFunction()) {
    auto func = inst1->getFunction();
    if (auto target1 = llvm::dyn_cast<llvm::CallBase>(inst1))
      if (auto target2 = llvm::dyn_cast<llvm::CallBase>(inst2)) {
        // find the next sol.match

        auto nextInst = inst1->getNextNonDebugInstruction();
        while (nextInst) {
          if (auto callValue = dyn_cast<CallBase>(nextInst)) {
            CallSite CS(callValue);
            if (CS.getTargetFunction()->getName().startswith("sol.match.")) {
              if (CS.getNumArgOperands() > 1) {
                if (auto funStart0 =
                        llvm::dyn_cast<llvm::CallBase>(CS.getArgOperand(0))) {
                  if (funStart0->getCalledFunction()->getName().startswith(
                          "sol.ifTrue.anon.") ||
                      funStart0->getCalledFunction()->getName().startswith(
                          "sol.ifFalse.anon.")) {
                    funStart0 = llvm::dyn_cast<llvm::CallBase>(
                        funStart0->getArgOperand(0));

                    for (auto i = 1; i < CS.getNumArgOperands(); i++) {
                      if (auto funStart_i = llvm::dyn_cast<llvm::CallBase>(
                              CS.getArgOperand(i))) {
                        if (funStart_i->getCalledFunction()
                                ->getName()
                                .startswith("sol.ifTrue.anon.") ||
                            funStart_i->getCalledFunction()
                                ->getName()
                                .startswith("sol.ifFalse.anon.")) {
                          funStart_i = llvm::dyn_cast<llvm::CallBase>(
                              funStart_i->getArgOperand(0));

                          if (i < 2) {
                            if (DEBUG_RUST_API)
                              llvm::outs()
                                  << "sol.match in: " << func->getName()
                                  << "\n";
                            if (DEBUG_RUST_API)
                              llvm::outs()
                                  << "exclusive func: "
                                  << funStart0->getCalledFunction()->getName()
                                  << "\n";
                            exclusiveFunctionsMap[func].insert(
                                funStart0->getCalledFunction());
                          }
                          if (DEBUG_RUST_API)
                            llvm::outs()
                                << "exclusive func: "
                                << funStart_i->getCalledFunction()->getName()
                                << "\n";
                          exclusiveFunctionsMap[func].insert(
                              funStart_i->getCalledFunction());
                        }
                      }
                    }
                  }
                }
              }

              auto exclusiveFunctions =
                  exclusiveFunctionsMap[inst1->getFunction()];
              if (exclusiveFunctions.find(target1->getCalledFunction()) !=
                      exclusiveFunctions.end() &&
                  exclusiveFunctions.find(target2->getCalledFunction()) !=
                      exclusiveFunctions.end()) {
                if (DEBUG_RUST_API)
                  llvm::outs()
                      << "OKOKOK exclusive functions: "
                      << target1->getCalledFunction()->getName() << " X "
                      << target2->getCalledFunction()->getName() << "\n";
                return true;
              }
              break;
            }
          }
          nextInst = nextInst->getNextNonDebugInstruction();
        }
      }
  }
  return false;
}

void SolanaAnalysisPass::handleRustModelAPI(
    const xray::ctx *ctx, TID tid, llvm::Function *func,
    const llvm::Instruction *inst, StaticThread *thread, CallSite CS,
    bool isMacroArrayRefUsedInFunction) {
  auto calledFuncName = CS.getCalledFunction()->getName();
  if (calledFuncName.startswith("sol.model.opaqueAssign")) {
    auto value = CS.getArgOperand(0);
    auto valueName = LangModel::findGlobalString(value);
    // llvm::outs() << "data: " << valueName << "\n";
    if (isa<CallBase>(CS.getArgOperand(1))) {
      auto callInst = llvm::cast<CallBase>(CS.getArgOperand(1));
      if (callInst->getCalledFunction()->getName().startswith("sol.clone.1")) {
        // skip clone
        CallSite CS2(callInst);
        auto value2 = CS2.getArgOperand(0);
        if (isa<CallBase>(CS2.getArgOperand(0))) {
          auto callInst2 = llvm::cast<CallBase>(CS2.getArgOperand(0));
          if (callInst2->getCalledFunction()->getName().contains(
                  "_account_info.")) {
            CallSite CS3(callInst2);
            value2 = CS3.getArgOperand(0);
          }
        }

        auto valueName2 = LangModel::findGlobalString(value2);
        valueName2 = stripSelfAccountName(valueName2);
        auto account = stripAll(valueName2);
        if (!account.empty() && account != valueName) {
          thread->accountAliasesMap[account].insert(valueName);
          if (DEBUG_RUST_API)
            llvm::outs() << "account: " << account
                         << " aliasAccount: " << valueName << "\n";
        }

      } else if (callInst->getCalledFunction()->getName().startswith(
                     "sol.next_account_info.")) {
        // llvm::outs() << "next_account_info: " << valueName <<
        // "\n";
        auto e = graph->createReadEvent(ctx, inst, tid);
        if (!valueName.empty())
          thread->accountsMap[valueName] = e;

      } else if (callInst->getCalledFunction()->getName().contains(
                     "_account_info.")) {
        // llvm::outs() << "calling: " <<
        // callInst->getCalledFunction()->getName() <<
        // "\n";
        auto aliasAccount =
            thread->findReturnAliasAccount(callInst->getCalledFunction());
        if (!aliasAccount.empty() && aliasAccount != valueName) {
          if (DEBUG_RUST_API)
            llvm::outs() << "account: " << valueName
                         << " aliasAccount: " << aliasAccount << "\n";
          thread->accountAliasesMap[valueName].insert(aliasAccount);
        }
      } else if (callInst->getCalledFunction()->getName().startswith(
                     "sol.Account::try_from.1")) {
        CallSite CS2(callInst);
        auto value2 = CS2.getArgOperand(0);
        auto account = LangModel::findGlobalString(value2);
        if (DEBUG_RUST_API)
          llvm::outs() << "sol.Account::try_from.1 account: " << account
                       << " aliasAccount: " << valueName << "\n";
        if (!valueName.empty() && account != valueName) {
          thread->accountAliasesMap[account].insert(valueName);
        }
      } else if (callInst->getCalledFunction()->getName().contains(
                     "::unpack.")) {
        // unpacked data - could be untrustful
        auto e = graph->createReadEvent(ctx, inst, tid);
        thread->accountsUnpackedDataMap[valueName] = e;
      } else if (callInst->getCalledFunction()->getName().contains(
                     "sol.key.")) {
        CallSite CS2(callInst);
        auto value2 = CS2.getArgOperand(0);
        auto valueName2 = LangModel::findGlobalString(value2);
        auto account = stripAll(valueName2);
        if (!account.empty() && account != valueName) {
          thread->accountAliasesMap[account].insert(valueName);
          if (DEBUG_RUST_API)
            llvm::outs() << "account: " << account
                         << " aliasAccount: " << valueName << "\n";
        }
      } else {
        // llvm::outs() << valueName << " == " <<
        // callInst->getCalledFunction()->getName() << "\n";
      }
    } else {
      auto valueName1 = LangModel::findGlobalString(CS.getArgOperand(1));
      if (valueName.contains("seeds")) {
        auto e = graph->createReadEvent(ctx, inst, tid);
        if (valueName1.contains(",")) {
          llvm::SmallVector<StringRef, 8> valueName1_vec;
          valueName1.split(valueName1_vec, ',', -1, false);
          for (auto value_ : valueName1_vec) {
            auto found = value_.find(".key");
            if (found != std::string::npos) {
              auto account = value_.substr(0, found);
              account = stripAll(account);
              auto pair = std::make_pair(account, account);
              thread->assertKeyEqualMap[pair] = e;
            }
          }
        } else if (valueName1.contains("pool.") &&
                   !valueName1.contains("fee_") &&
                   !valueName1.contains("dest")) {
          // pda_sharing_secure
          // llvm::errs()<< "==============VULNERABLE: insecure PDA
          // sharing!============\n";
          UntrustfulAccount::collect(valueName1, e, callEventTraces,
                                     SVE::Type::INSECURE_PDA_SHARING, 5);
        }
      } else if (valueName1.contains(".accounts.")) {
        valueName1 = stripCtxAccountsName(valueName1);
        auto foundDot = valueName1.find(".");
        if (foundDot != std::string::npos)
          valueName1 = valueName1.substr(0, foundDot);
        else {
          // account alias
          // let store_account = &mut ctx.accounts.store
          auto account = valueName1;
          if (valueName != account) {
            thread->accountAliasesMap[valueName].insert(account);

            if (DEBUG_RUST_API)
              llvm::outs() << "account: " << valueName
                           << " aliasAccount: " << account << "\n";
          }
        }
        auto e = graph->createReadEvent(ctx, inst, tid);
        if (!thread->isInAccountOrStructAccountMap(valueName1))
          thread->accountsMap[valueName1] = e;

        // TODO this account data is written
        thread->accountsDataWrittenMap[valueName1] = e;
      } else if (valueName1.equals("0")) {
        // check closing_accounts insecure
        if (valueName.endswith("lamports.borrow_mut()")) {
          auto e = graph->createReadEvent(ctx, inst, tid);
          thread->accountsCloseMap[valueName] = e;

          // traverse the next few instructions, find sol_memset
          auto nextInst = inst->getNextNonDebugInstruction();
          while (nextInst) {
            if (auto callValue2 = dyn_cast<CallBase>(nextInst)) {
              CallSite CS2(callValue2);
              if (CS2.getTargetFunction()->getName().startswith(
                      "sol.sol_memset.3")) {
                valueName = stripAccountName(valueName);
                auto foundDot = valueName.find(".");
                if (foundDot != std::string::npos)
                  valueName = valueName.substr(0, foundDot);
                thread->memsetDataMap[valueName] = e;
                // llvm::outs() << "sol_memset valueName: " <<
                // valueName <<
                // "\n";
                break;
              }
            }
            nextInst = nextInst->getNextNonDebugInstruction();
          }
        }
      }
    }

    // for mango: it uses array_ref!
    if (isMacroArrayRefUsedInFunction) {
      if (auto prevInst = dyn_cast<CallBase>(CS.getArgOperand(0))) {
        auto prevCallName = prevInst->getCalledFunction()->getName();
        if (prevCallName.startswith("sol.model.slice.")) {
          CallSite CS2(prevInst);
          for (int i = 0; i < CS2.getNumArgOperands(); i++) {
            auto value = CS2.getArgOperand(i);
            if (auto callValue = dyn_cast<CallBase>(value)) {
              CallSite CS3(callValue);
              if (CS3.getTargetFunction()->getName().startswith(
                      "sol.model.slice.item.")) {
                auto account =
                    LangModel::findGlobalString(CS3.getArgOperand(0));
                auto e = graph->createReadEvent(ctx, callValue, tid);
                if (!account.empty())
                  if (!thread->isInAccountOrStructAccountMap(account))
                    thread->accountsMap[account] = e;
              }
            }
          }
        }

        // llvm::outs() << "check.mango: valueName: " << valueName
        // << "\n";
        bool isMangoAccounts = false;
        auto valueName1 = LangModel::findGlobalString(CS.getArgOperand(1));
        if (valueName1.contains("accounts") || valueName1.contains("_ais") ||
            valueName1.contains("array_ref!")) {
          isMangoAccounts = true;
        } else if (auto prevInst =
                       dyn_cast<CallBase>(inst->getPrevNonDebugInstruction())) {
          auto prevCallName = prevInst->getCalledFunction()->getName();
          if (prevCallName.contains("array_ref!"))
            isMangoAccounts = true;
        }
        // llvm::outs() << "check.mango: isMangoAccounts: " <<
        // isMangoAccounts << "\n";

        if (isMangoAccounts) {
          auto valueNameX = valueName.substr(1, valueName.size() - 2);
          if (DEBUG_RUST_API)
            llvm::outs() << "sol.mango: valueNameX: " << valueNameX << "\n";
          llvm::SmallVector<StringRef, 8> value_vec;
          valueNameX.split(value_vec, ',', -1, false);
          auto e = graph->createReadEvent(ctx, inst, tid);
          for (auto value_ : value_vec) {
            if (DEBUG_RUST_API)
              llvm::outs() << "sol.mango: value_: " << value_ << "\n";
            // if (false)
            if (!value_.empty())
              thread->accountsMap[value_] = e;
          }
        }
      }
    }

  } else if (calledFuncName.startswith("sol.model.macro.array_ref!.")) {
    isMacroArrayRefUsedInFunction = true;
  } else if (calledFuncName.startswith("sol.model.struct.new.")) {
    // sol.model.struct.new.Transfer.
    auto calleeName = calledFuncName;
    auto e = graph->createReadEvent(ctx, inst, tid);
    auto foundTransferType = (calleeName.find("Transfer") != std::string::npos);
    if (foundTransferType) {
      // find all sol.model.opaqueAssign @authority
      auto from_name = findNewStructAccountName(tid, inst, "from");
      auto to_name = findNewStructAccountName(tid, inst, "to");
      auto authority_name = findNewStructAccountName(tid, inst, "authority");
      if (!from_name.empty()) {
        thread->accountsInvokedMap[from_name] = e;
      }

      if (!to_name.empty()) {
        if (DEBUG_RUST_API)
          llvm::outs() << "Transfer accountsInvokedMap: from_name: "
                       << from_name << " to_name: " << to_name << "\n";

        thread->accountsInvokedMap[to_name] = e;
        auto pair = std::make_pair(from_name, to_name);
        thread->tokenTransferFromToMap[pair] = e;
      }
      if (!authority_name.empty()) {
        thread->accountsInvokedMap[authority_name] = e;
        auto pair = std::make_pair(authority_name, from_name);
        thread->accountsInvokeAuthorityMap[pair] = e;
      }
    }
    auto foundMintType = (calleeName.find("MintTo.") != std::string::npos);
    auto foundBurnType = (calleeName.find("Burn.") != std::string::npos);
    if (foundMintType || foundBurnType) {
      // find all sol.model.opaqueAssign @authority
      auto mint_name = findNewStructAccountName(tid, inst, "mint");
      auto from_name = findNewStructAccountName(tid, inst, "from");
      auto to_name = findNewStructAccountName(tid, inst, "to");
      auto authority_name = findNewStructAccountName(tid, inst, "authority");
      // skip mint as it may not be trusted
      // if (!mint_name.empty())
      // thread->accountsInvokedMap[mint_name] = e;
      if (!to_name.empty())
        thread->accountsInvokedMap[to_name] = e;
      // if (foundMintType) {
      //     llvm::outs() << "tid: " << thread->getTID()
      //                  << " MintTo accountsInvokedMap:  to_name:
      //                  " << to_name << "\n";
      // }
      if (!from_name.empty()) {
        thread->accountsInvokedMap[from_name] = e;

        if (!thread->isInAccountsMap(from_name)) {
          auto account_from_name =
              findCallStackAccountAliasName(func, e, from_name);
          if (DEBUG_RUST_API)
            llvm::outs() << "DEBUG accountsInvokedMap: from_name: " << from_name
                         << " account_from_name: " << account_from_name << "\n";
          thread->accountsInvokedMap[account_from_name] = e;
        }
      }

      if (!authority_name.empty()) {
        thread->accountsInvokedMap[authority_name] = e;
        auto pair1 = std::make_pair(authority_name, from_name);
        thread->accountsInvokeAuthorityMap[pair1] = e;
        if (foundBurnType) {
          if (DEBUG_RUST_API)
            llvm::outs() << "Burn accountsInvokeAuthorityMap: "
                            "authority_name: "
                         << authority_name << " from_name: " << from_name
                         << " to_name: " << to_name << "\n";

          thread->accountsBurnAuthorityMap[pair1] = e;
          auto pair2 = std::make_pair(authority_name, to_name);
          thread->accountsBurnAuthorityMap[pair2] = e;
        }
      }
    }

    // CPI Serum
    auto foundSerumCPIType = (calleeName.find(".dex:") != std::string::npos);
    if (foundSerumCPIType) {
      for (int i = 0; i < CS.getNumArgOperands(); i++) {
        auto value_i = CS.getArgOperand(i);
        auto valuename_i = LangModel::findGlobalString(value_i);
        thread->accountsInvokedMap[valuename_i] = e;
        if (DEBUG_RUST_API)
          llvm::outs() << "foundSerumCPIType valuename_i: " << valuename_i
                       << "\n";
      }
    }

  } else if (calledFuncName.startswith("sol.model.funcArg")) {
    // let's check each arg and their type
    // if (thread->getTID() > 0)
    {
      auto value1 = CS.getArgOperand(0);
      auto name = LangModel::findGlobalString(value1);
      auto value2 = CS.getArgOperand(1);
      auto type = LangModel::findGlobalString(value2);
      if (type.front() == '&')
        type = type.substr(1);
      if (DEBUG_RUST_API)
        llvm::outs() << "funcArg name: " << name << " type: " << type << "\n";
      auto pair = std::make_pair(name, type);
      funcArgTypesMap[func].push_back(pair);
    }
  } else if (calledFuncName.startswith("sol.model.struct.field")) {
    // let's check each field accounts and their constraints
    auto value1 = CS.getArgOperand(0);
    auto field = LangModel::findGlobalString(value1);
    auto value2 = CS.getArgOperand(1);
    auto type = LangModel::findGlobalString(value2);

    // get the lastest forked thread
    auto lastThread = threadList.back();
    lastThread->anchorStructFunctionFields.push_back(
        std::make_pair(field, type));
    // skip UncheckedAccount
    if (type.contains("UncheckedAccount") && !ConfigCheckUncheckedAccount) {
      return;
    }

    auto e = graph->createForkEvent(ctx, inst, lastThread->getTID());
    if (type.startswith("Signer")) {
      lastThread->asssertSignerMap[field] = e;
    }
    lastThread->addStructAccountType(func, field, type, e);
    if (DEBUG_RUST_API)
      llvm::outs() << "field account: " << field << " type: " << type
                   << " thread: " << thread->getTID()
                   << " lastThread: " << lastThread->getTID() << "\n";
    // for non-account type: BrrrCommon
    auto struct_name = type;
    auto found = type.find("<");
    if (found != std::string::npos)
      struct_name = struct_name.substr(0, found);
    std::string func_struct_name =
        "sol.model.struct.anchor." + struct_name.str();
    // checking issues in @sol.model.struct.anchor.LogMessage
    auto func_struct = thisModule->getFunction(func_struct_name);
    if (func_struct) {
      if (DEBUG_RUST_API)
        llvm::outs() << "found additional struct accounts: " << struct_name
                     << "\n";
      lastThread->structAccountsMap[field] = e;
      traverseFunctionWrapper(ctx, thread, callStack, inst, func_struct);
    } else {
      lastThread->accountsMap[field] = e;
    }
    // TODO: check is_signer in function body
  } else if (calledFuncName.startswith("sol.model.struct.constraint")) {
    // let's check each constraint
    //@"authority.key==&token.owner"
    auto cons_all = LangModel::findGlobalString(CS.getArgOperand(0));
    auto lastThread = threadList.back();
    auto nextInst = inst->getNextNonDebugInstruction();
    auto e = graph->createReadEvent(ctx, nextInst, lastThread->getTID());
    auto callInst = llvm::cast<CallBase>(nextInst);
    auto e2 = graph->createForkEvent(ctx, callInst, lastThread->getTID());
    auto value = callInst->getOperand(0);
    // find the next instruction, which has e.g., user.authority
    auto valueName = LangModel::findGlobalString(value);
    bool isMutable =
        (cons_all.find("mut,") != std::string::npos) || (cons_all == "mut");
    if (isMutable) {
      if (DEBUG_RUST_API)
        llvm::outs() << "accountsMutMap: " << valueName << "\n";
      lastThread->accountsMutMap[valueName] = true;
    }
    //"init,seeds=[b\22cdp_vault\22.as_ref(),cdp_vault_type.cdp_state.as_ref(),cdp_vault_type.key().as_ref(),cdp_vault_owner.key().as_ref()],bump=bump,payer=payer,space=8+std::mem::size_of::<CdpVault>()"
    auto foundInit = cons_all.find("init,") != std::string::npos;
    auto foundInitIfNeeded =
        cons_all.find("init_if_needed,") != std::string::npos;
    if (foundInit || foundInitIfNeeded) {
      accountsPDAMap[valueName] = e2;
      lastThread->accountsPDAInitInInstructionMap[valueName] = e2;
      if (DEBUG_RUST_API)
        llvm::outs() << "init: " << valueName << "\n";
    }

    auto foundSeeds = cons_all.find("seeds=[");
    if (foundSeeds != std::string::npos) {
      // add the next instruction account to PDA account
      auto account_pda = LangModel::findGlobalString(callInst->getOperand(0));
      accountsPDAMap[account_pda] = e2;
      lastThread->accountsPDAInInstructionMap[account_pda] = e2;
      if (isMutable) {
        lastThread->accountsPDAMutMap[account_pda] = e2;
      }
      auto cons_seeds = cons_all.substr(foundSeeds + 7);
      auto found = cons_seeds.find_last_of("]");
      if (found != std::string::npos) {
        cons_seeds = cons_seeds.substr(0, found);
        llvm::SmallVector<StringRef, 8> cons_vec;
        cons_seeds.split(cons_vec, ',', -1, false);
        auto e = graph->createForkEvent(ctx, inst, lastThread->getTID());
        bool isConstantOrSigner = true;
        bool isFreshPDA =
            accountsPDASeedsMap.find(account_pda) == accountsPDASeedsMap.end();
        if (lastThread->isUserProvidedString(cons_seeds))
          userProvidedInputStringPDAAccounts.insert(account_pda);
        for (auto cons : cons_vec) {
          if (DEBUG_RUST_API)
            llvm::outs() << "seeds: " << cons << "\n";
          // cdp_vault_owner.key()
          // currency_mint.to_account_info().key.as_ref()
          if (isFreshPDA)
            accountsPDASeedsMap[account_pda].push_back(cons);

          found = cons.find(".key");
          if (found != std::string::npos) {
            auto account = cons.substr(0, found);
            account = stripAll(account);
            lastThread->accountsSeedsMap[account_pda].insert(account);
          } else if (cons.contains(".")) {
            found = cons.find(".");
            auto account = cons.substr(0, found);
            if (account_pda.equals(account)) {
              lastThread->accountsSelfSeedsMap[account_pda] = e2;
            }
          }
          if (isConstantOrSigner && !cons.startswith("b\"") &&
              !isUpper(cons.str()))
            isConstantOrSigner = false;
        }
        // onceOnly instruction
        if (foundInit && isConstantOrSigner) {
          lastThread->isOnceOnlyOwnerOnlyInstruction = true;
          if (DEBUG_RUST_API)
            llvm::outs() << "T" << lastThread->getTID()
                         << " isOnceOnlyOwnerOnlyInstruction: " << true << "\n"
                         << cons_all << "\n";
        }
      }
    }

    // split by ','
    llvm::SmallVector<StringRef, 8> cons_vec;
    cons_all.split(cons_vec, ',', -1, false);
    for (auto cons : cons_vec) {
      if (DEBUG_RUST_API)
        llvm::outs() << "cons: " << cons << "\n";
      auto foundAddress = cons.find("address=");
      if (foundAddress != std::string::npos) {
        auto cons_address = cons.substr(foundAddress + 8);
        // add the next instruction account to validated account
        {
          auto pair = std::make_pair(valueName, cons_address);
          lastThread->assertKeyEqualMap[pair] = e2;
          // if (DEBUG_RUST_API)
          llvm::outs() << "address: " << valueName << " == " << cons_address
                       << "\n";
        }
      }
      // TODO: handle multiple constraints
      auto foundCons = cons.find("constraint=");
      if (foundCons != std::string::npos) {
        auto cons_full = cons.substr(foundCons + 11);
        // pool.token_x_reserve ==
        // token_src_reserve.key()||pool.token_y_reserve==token_src_reserve.key()
        llvm::SmallVector<StringRef, 8> cons_vec_full;
        cons_full.split(cons_vec_full, "||", -1, false);
        for (auto cons_ : cons_vec_full) {
          auto foundErrorAt = cons_.find("@");
          if (foundErrorAt != std::string::npos)
            cons_ = cons_.substr(0, foundErrorAt);
          auto equalStr = "==";
          auto notEqualStr = "!=";
          auto lessOrEqualStr = "<=";
          auto largerOrEqualStr = ">=";
          auto lessThanStr = "<";
          auto largerThanStr = ">";
          auto valueName1 = cons_;
          auto valueName2 = cons_;
          // must exist
          auto foundEqual = cons_.find(equalStr);
          auto foundNotEqual = cons_.find(notEqualStr);
          auto foundLessOrEqual = cons_.find(lessOrEqualStr);
          auto foundLessThan = cons_.find(lessThanStr);
          auto foundLargerOrEqual = cons_.find(largerOrEqualStr);
          auto foundLargerThan = cons_.find(largerThanStr);
          bool isEqual = foundEqual != std::string::npos;
          bool isNotEqual = foundNotEqual != std::string::npos;
          bool isLessOrEqual = foundLessOrEqual != std::string::npos;
          bool isLessThan = foundLessThan != std::string::npos;
          bool isLargerOrEqual = foundLargerOrEqual != std::string::npos;
          bool isLargerThan = foundLargerThan != std::string::npos;
          auto found = std::string::npos;
          if (isEqual)
            found = foundEqual;
          else if (isNotEqual)
            found = foundNotEqual;
          else if (isLessOrEqual)
            found = foundLessOrEqual;
          else if (isLargerOrEqual)
            found = foundLargerOrEqual;
          else if (isLessThan)
            found = foundLessThan;
          else if (isLargerThan)
            found = foundLargerThan;
          else {
            llvm::outs() << "unknown constraint: " << cons_ << "\n";
          }

          if (found != std::string::npos) {
            size_t op_len = 2;
            if (isLessThan || isLargerThan)
              op_len = 1;
            valueName1 = cons_.substr(0, found);
            valueName2 = cons_.substr(found + op_len);
            if (isEqual || isNotEqual)
              updateKeyEqualMap(lastThread, e, isEqual, isNotEqual, valueName1,
                                valueName2);
          }

          // add to constraint map
          {
            lastThread->assertAccountConstraintsMap[valueName].insert(cons_);
            // token_dest.mint == pool.token_x_mint
            if (valueName1.endswith(".mint") || valueName2.endswith(".mint")) {
              auto value2 = callInst->getOperand(1);
              auto type = LangModel::findGlobalString(value2);
              if (type.contains("TokenAccount")) {
                auto pair = std::make_pair(valueName, valueName2);
                lastThread->assertMintEqualMap[pair] = e;
              }
            }
          }
        }
      }

      auto foundHasOne = cons.find("has_one=");
      if (foundHasOne != std::string::npos) {
        // #[account(mut,
        //           has_one = market,
        //           has_one = fee_note_vault,
        //           has_one = pyth_oracle_price)]
        // pub reserve: Loader<'info, Reserve>,

        auto account = cons.substr(foundHasOne + 8);
        // has_one = staked_mint
        // @StakingCampaignError::InvalidStakedMint
        account = stripAtErrorAccountName(account);
        // get the lastest forked thread
        lastThread->assertHasOneFieldMap[valueName].insert(account);
        auto e = graph->createForkEvent(ctx, nextInst, lastThread->getTID());
        // JEFF: if mut, then account is trustful
        // if (isMutable)
        // {
        //     auto pair = std::make_pair(valueName, account);
        //     lastThread->assertHasOneFieldMap[pair] = e;
        // }

        // auto value_str = valueName.str();
        // // these are stack variables and will be deallocated when
        // the function returns.. auto account_key = account.str() +
        // ".key"; auto cons_account_key = value_str + "." +
        // account_key;
        // // UNDERSTAND WHY?
        // auto pair = std::make_pair(cons_account_key,
        // account_key); lastThread->assertKeyEqualMap[pair] = e;
        // auto pair2 = std::make_pair(account, account_key);
        // lastThread->assertKeyEqualMap[pair2] = e;
        // llvm::outs()
        //     << "T" << lastThread->getTID() << "
        //     assertKeyEqualMap: " << cons_account_key
        //     << " == " << account_key << "\n";

        // add  has_one = vault_owner
        if (account.find("owner") != std::string::npos) {
          auto pair_owner = std::make_pair(valueName, account);
          lastThread->assertOwnerEqualMap[pair_owner] = e;
        }
      }
      auto account = LangModel::findGlobalString(value);
      auto foundClose = (cons.find("close=") != std::string::npos);
      auto foundZero = (cons.find("zero") != std::string::npos);
      // anchor account cannot be reinitialized
      if (foundClose || foundZero) {
        // find the next instruction, which has e.g., user

        if (DEBUG_RUST_API) {
          if (foundZero)
            llvm::outs() << "zero: " << account << "\n";
          if (foundClose)
            llvm::outs() << "close: " << account << "\n";
        }
        // get the lastest forked thread
        auto lastThread = threadList.back();
        if (foundZero)
          lastThread->asssertDiscriminatorMap[account] = e;
        if (foundClose)
          lastThread->accountsAnchorCloseMap[account] = e;
      }
      auto foundSigner = cons.find("signer");
      auto foundSigner_ = cons.find("signer_");
      auto foundSignerDot = cons.find("signer.");
      auto foundSignerPayer = cons.find("payer=signer");
      // the account is signer
      if (foundSigner != std::string::npos &&
          foundSigner_ == std::string::npos &&
          foundSignerDot == std::string::npos &&
          foundSignerPayer == std::string::npos) {
        // find the next instruction, which has e.g.,
        //>12|    #[account(signer)]
        // 13|    pub owner: AccountInfo<'info>
        llvm::outs() << "signer cons: " << cons << " account: " << account
                     << "\n";
        if (DEBUG_RUST_API)
          llvm::outs() << "signer: " << account << "\n";
        lastThread->asssertSignerMap[account] = e;
      }
      auto foundAssociatedToken = cons.find("associated_token::authority");
      // the account is associated_token
      if (foundAssociatedToken != std::string::npos) {
        // associated_token::authority = user,
        // pub user_redeemable: Box<Account<'info, TokenAccount>>,
        if (DEBUG_RUST_API)
          llvm::outs() << "associated_token: " << account << "\n";
        auto pair = std::make_pair(account, account);
        lastThread->assertKeyEqualMap[pair] = e;
      }
      auto foundTokenAuthority = cons.find("token::authority");
      if (foundTokenAuthority != std::string::npos) {
        if (DEBUG_RUST_API)
          llvm::outs() << "token::authority: " << account << "\n";
        auto pair = std::make_pair(account, account);
        lastThread->assertKeyEqualMap[pair] = e;
      }
      auto foundTokenMint = cons.find("token::mint");
      if (foundTokenMint != std::string::npos) {
        if (DEBUG_RUST_API)
          llvm::outs() << "token::mint: " << account << "\n";
        auto account_mint = cons.substr(foundTokenMint + 12);
        auto pair = std::make_pair(account, account_mint);
        lastThread->assertMintEqualMap[pair] = e;
      }

      // bump
      auto foundBump = cons.find("bump=");
      if (foundBump != std::string::npos) {
        if (!foundInit) {
          auto bumpSeed = cons.substr(foundBump + 5);
          if (!bumpSeed.contains(".")) {
            // llvm::outs() << "LIKELY BUMP SEED VULNERABILITY: " <<
            // cons <<
            // "\n";
            auto pair = std::make_pair(bumpSeed, account);
            lastThread->accountsBumpMap[pair] = e;
          }
        }
      }
    }

  } else if (calledFuncName.startswith("sol.model.loop")) {
    if (DEBUG_RUST_API)
      llvm::outs() << "sol.model.loop: " << *inst << "\n";
    if (CS.getNumArgOperands() > 0) {
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto funcName = valueName;
      // auto funcName = valueName.substr(0,
      // valueName.find_last_of(".")); llvm::outs() << "func_loop: "
      // << funcName << "\n";

      auto func_loop = thisModule->getFunction(funcName);
      if (func_loop) {
        // llvm::outs() << "found func_loop: " << funcName << "\n";
        for_loop_counter++;
        traverseFunctionWrapper(ctx, thread, callStack, inst, func_loop);
        for_loop_counter--;
      }
    }
  } else if (calledFuncName.startswith("sol.model.break")) {
    if (DEBUG_RUST_API)
      llvm::outs() << "sol.model.break: " << *inst << "\n";
    if (isInLoop()) {
      // check potential break => continue
      // caller instruction
      auto inst0 = callEventTraces[tid].back()->getInst();
      if (auto inst1 = inst0->getPrevNonDebugInstruction()) {
        if (auto callValue2 = dyn_cast<CallBase>(inst1)) {
          CallSite CS2(callValue2);
          if (CS2.getTargetFunction()->getName().equals("sol.if")) {
            if (auto inst2 = inst1->getPrevNonDebugInstruction()) {
              if (auto inst3 = inst2->getPrevNonDebugInstruction()) {
                if (auto callValue3 = dyn_cast<CallBase>(inst3)) {
                  CallSite CS3(callValue3);
                  if (CS3.getTargetFunction()->getName().startswith(
                          "sol.Pubkey::default.")) {
                    // sol.Pubkey::default.0
                    auto e = graph->createReadEvent(ctx, inst, tid);
                    UnsafeOperation::collect(e, callEventTraces,
                                             SVE::Type::INCORRECT_BREAK_LOGIC,
                                             10);
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (calledFuncName.startswith("sol.model.macro.")) {
    if (DEBUG_RUST_API)
      llvm::outs() << "sol.model.macro: " << *inst << "\n";
    // sol.model.macro.placeholder.
    if (calledFuncName.startswith("sol.model.macro.placeholder.")) {
      auto func_namespace_name = func->getName().str();
      {
        auto found = func_namespace_name.find("::");
        if (found != std::string::npos)
          func_namespace_name = func_namespace_name.substr(found + 2);
      }
      {
        auto found = func_namespace_name.find(".");
        if (found != std::string::npos)
          func_namespace_name = func_namespace_name.substr(0, found);
      }
      auto func_handler_name = func_namespace_name + "::handler.";
      auto func_handler = getFunctionFromPartialName(func_handler_name);
      if (DEBUG_RUST_API)
        llvm::outs() << "placeholder func_handler_name: " << func_handler_name
                     << "\n";
      if (func_handler) {
        // macro invariants
        auto macro_handler = getFunctionFromPartialName("lib::accounts.2");
        if (macro_handler) {
          traverseFunctionWrapper(ctx, thread, callStack, inst, macro_handler);
        }
        auto macro_validate =
            getFunctionMatchStartEndName(func_namespace_name, "::validate.1");
        if (macro_validate) {
          traverseFunctionWrapper(ctx, thread, callStack, inst, macro_validate);
        }
        traverseFunctionWrapper(ctx, thread, callStack, inst, func_handler);
      } else {
        if (DEBUG_RUST_API)
          llvm::outs() << "func_handler: NULL\n";
      }
    } else {
      auto value = CS.getArgOperand(0);
      auto params = LangModel::findGlobalString(value);
      auto e = graph->createForkEvent(ctx, inst, tid);
      if (DEBUG_RUST_API)
        llvm::outs() << "macro: " << calledFuncName << " params: " << params
                     << "\n";
      if (calledFuncName.startswith("sol.model.macro.authority_constraint.")) {
        llvm::SmallVector<StringRef, 8> accounts_vec;
        params.split(accounts_vec, ',', -1, false);
        if (accounts_vec.size() > 1) {
          auto account = accounts_vec[1];
          if (DEBUG_RUST_API)
            llvm::outs() << "authority_constraint authority: " << account
                         << "\n";
          auto pair = std::make_pair(account, account);
          thread->assertKeyEqualMap[pair] = e;
        }

      } else if (calledFuncName.startswith(
                     "sol.model.macro.parent_child_constraint.")) {
        llvm::SmallVector<StringRef, 8> accounts_vec;
        params.split(accounts_vec, ',', -1, false);
        if (accounts_vec.size() > 0) {
          auto account = stripAccountName(accounts_vec[0]);
          if (DEBUG_RUST_API)
            llvm::outs() << "parent_child_constraint parent: " << account
                         << "\n";
          auto pair = std::make_pair(account, account);
          thread->assertKeyEqualMap[pair] = e;
        }

      } else if (calledFuncName.startswith("//sol.model.macro.quote.")) {
      } else {
        {
          auto params_tmp = params;
          auto found = params_tmp.find(".is_signer");
          while (found != std::string::npos) {
            auto account = params_tmp.substr(0, found);
            account = findMacroAccount(account);
            thread->asssertSignerMap[account] = e;
            if (!thread->isInAccountsMap(account)) {
              account = findCallStackAccountAliasName(func, e, account);
              if (DEBUG_RUST_API)
                llvm::outs() << "macro is_signer: " << account << "\n";
              if (!account.empty())
                thread->asssertSignerMap[account] = e;
            }
            params_tmp = params_tmp.substr(found + 1);
            found = params_tmp.find(".is_signer");
          }
        }
        {
          auto params_tmp = params;
          auto found = params_tmp.find(".owner");
          while (found != std::string::npos) {
            auto account = params_tmp.substr(0, found);
            account = findMacroAccount(account);
            auto account2 = account;
            if (!thread->isInAccountsMap(account)) {
              account2 = findCallStackAccountAliasName(func, e, account);
            }
            if (DEBUG_RUST_API)
              llvm::outs() << "macro owner: " << account << "\n";
            if (!account.contains("=")) {
              auto pair = std::make_pair(account, account2);
              thread->assertOwnerEqualMap[pair] = e;
            }
            params_tmp = params_tmp.substr(found + 1);
            found = params_tmp.find(".owner");
          }
        }
        {
          auto params_tmp = params;
          auto found = params_tmp.find(".key");
          while (found != std::string::npos) {
            auto account = params_tmp.substr(0, found);
            account = findMacroAccount(account);
            auto account2 = account;
            if (!thread->isInAccountsMap(account)) {
              account2 = findCallStackAccountAliasName(func, e, account);
            }
            if (DEBUG_RUST_API)
              llvm::outs() << "macro key: " << account << "\n";
            auto pair = std::make_pair(account, account2);
            thread->assertKeyEqualMap[pair] = e;
            auto pair2 = std::make_pair(account2, account);
            thread->assertKeyEqualMap[pair2] = e;

            params_tmp = params_tmp.substr(found + 1);
            found = params_tmp.find(".key");
          }
        }
      }
    }

  } else if (calledFuncName.equals("sol.model.access_control")) {
    auto value = CS.getArgOperand(0);
    auto calleeName = LangModel::findGlobalString(value);
    auto found = calleeName.find("(");
    if (found != std::string::npos)
      calleeName = calleeName.substr(0, found);
    if (calleeName.startswith("ctx.accounts.")) {
      auto foundDot = calleeName.find_last_of(".");
      if (foundDot != std::string::npos)
        calleeName = calleeName.substr(foundDot + 1);
    }
    std::string func_struct_name = "::" + calleeName.str() + ".";
    if (calleeName == "validate") {
      auto arg0Type = funcArgTypesMap[func][0].second;
      auto struct_name = getStructName(arg0Type);
      func_struct_name = "::" + struct_name.str() + func_struct_name;
    }
    auto func_struct = getFunctionFromPartialName(func_struct_name);
    if (DEBUG_RUST_API) {
      llvm::outs() << "access_control func_struct_name: " << func_struct_name
                   << "\n";
    }
    if (func_struct) {
      traverseFunctionWrapper(ctx, thread, callStack, inst, func_struct);
    }
  } else {
    traverseFunctionWrapper(ctx, thread, callStack, inst,
                            CS.getCalledFunction());
  }
}

void SolanaAnalysisPass::handleNonRustModelAPI(const xray::ctx *ctx, TID tid,
                                               Function *func,
                                               const Instruction *inst,
                                               StaticThread *thread,
                                               CallSite CS) {
  if (DEBUG_RUST_API) {
    llvm::outs() << "analyzing sol call: " << CS.getCalledFunction()->getName()
                 << "\n";
  }

  auto createReadEventFunc = std::bind(&xray::ReachGraph::createReadEvent,
                                       graph, ctx, std::placeholders::_1, tid);
  auto isInLoop = std::bind(&SolanaAnalysisPass::isInLoop, this);
  auto collectUnsafeOperationFunc =
      std::bind(&UnsafeOperation::collect, std::placeholders::_1,
                callEventTraces, std::placeholders::_2, std::placeholders::_3);
  RuleContext RC(func, inst, funcArgTypesMap, thread, createReadEventFunc,
                 isInLoop, collectUnsafeOperationFunc);

  if (!nonRustModelRuleset.evaluate(RC, CS)) {
    // No matching rules. Traverse into the callee.
    // traverseFunctionWrapper(ctx, thread, callStack, inst,
    //                         CS.getCalledFunction());

    auto targetFuncName = CS.getCalledFunction()->getName();
    // TODO: Drop the following if-s and just use the above lines
    // instead.
    if (func->getName().startswith("sol.model.anchor.program.")) {
      // IMPORTANT: creating a new thread
      auto e = graph->createForkEvent(ctx, inst, tid);
      thread->addForkSite(e);
      auto newtid = addNewThread(e);
      // for the new thread, set funcArgTypesMap
      if (newtid > 0) {
        auto newThread = StaticThread::getThreadByTID(newtid);
        bool added = addThreadStartFunction(newThread->startFunc);
        for (int i = 0; i < CS.getNumArgOperands(); i++) {
          auto value_i = CS.getArgOperand(i);
          auto valuename_i = LangModel::findGlobalString(value_i);
          auto found = valuename_i.find(":");
          if (found != std::string::npos) {
            auto name = valuename_i.substr(0, found);
            auto type = valuename_i.substr(found + 1);
            if (DEBUG_RUST_API)
              llvm::outs() << "New funcArg name: " << name << " type: " << type
                           << "\n";
            auto pair = std::make_pair(name, type);
            funcArgTypesMap[newThread->startFunc].push_back(pair);
            // user-provided std::strings
            if (type.contains("String")) {
              newThread->userProvidedInputStrings.insert(name);
            }
          }
        }
      }

      //@"ctx:Context<LogMessage>"
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto struct_name = getStructName(valueName);
      if (DEBUG_RUST_API)
        llvm::outs() << "anchor struct_name: " << struct_name << "\n";
      std::string func_struct_name =
          "sol.model.struct.anchor." + struct_name.str();
      // checking issues in @sol.model.struct.anchor.LogMessage
      auto func_struct = thisModule->getFunction(func_struct_name);
      if (func_struct) {
        auto lastThread = threadList.back();
        lastThread->anchor_struct_function = func_struct;
        traverseFunctionWrapper(ctx, thread, callStack, inst, func_struct);
      }
    } else if (targetFuncName.startswith("sol.sol_memset.3")) {
      auto value = CS.getArgOperand(0);
      // TODO source_account_data
      if (auto callValue = dyn_cast<CallBase>(value)) {
        CallSite CS2(callValue);
        auto value1 = CS2.getArgOperand(0);
        auto valueName = LangModel::findGlobalString(value1);
        if (valueName.find(".data") != std::string::npos) {
          auto e = graph->createReadEvent(ctx, inst, tid);
          auto account = valueName;
          auto found = valueName.find(".");
          if (found != std::string::npos)
            account = valueName.substr(0, found);
          if (!thread->isInAccountsMap(account))
            account = findCallStackAccountAliasName(func, e, account);
          thread->memsetDataMap[account] = e;
          auto found1 = account.find(".accounts.");
          if (found1 != std::string::npos) {
            auto account1 = account.substr(found1 + 10);
            thread->memsetDataMap[account1] = e;
          }
        }
      }
    } else if (targetFuncName.equals("sol.position.2")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.position: " << *inst << "\n";

      auto value = CS.getArgOperand(1);
      if (auto callValue = dyn_cast<CallBase>(value)) {
        handleConditionalCheck0(ctx, tid, func, inst, thread, value);
      }
    } else if (targetFuncName.startswith("sol.Ok.1")) {
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto e = graph->createReadEvent(ctx, inst, tid);
      auto funcx = findCallStackNonAnonFunc(e);
      if (!valueName.empty())
        thread->updateMostRecentFuncReturn(funcx, valueName);
    } else if (targetFuncName.contains("::keccak::hashv.")) {
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      if (valueName.contains(",")) {
        auto e = graph->createReadEvent(ctx, inst, tid);
        llvm::SmallVector<StringRef, 8> valueName_vec;
        valueName.split(valueName_vec, ',', -1, false);
        for (auto value_ : valueName_vec) {
          auto found = value_.find(".key");
          if (found != std::string::npos) {
            auto account = value_.substr(0, found);
            account = stripAll(account);
            auto pair = std::make_pair(account, account);
            thread->assertKeyEqualMap[pair] = e;
          }
        }
      }
    } else if (targetFuncName.equals("sol.if")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.if: " << *inst << "\n";

      auto value = CS.getArgOperand(0);
      bool handled = false;
      // for && and ||
      if (auto callValue = dyn_cast<CallBase>(value)) {
        CallSite CS2(callValue);
        if (CS2.getTargetFunction()->getName().equals("sol.&&") ||
            CS2.getTargetFunction()->getName().equals("sol.||")) {
          handleConditionalCheck0(ctx, tid, func, inst, thread,
                                  CS2.getArgOperand(0));
          handleConditionalCheck0(ctx, tid, func, inst, thread,
                                  CS2.getArgOperand(1));
          handled = true;
        }
      }
      if (!handled)
        handleConditionalCheck0(ctx, tid, func, inst, thread, value);
    } else if (targetFuncName.startswith("sol.&&")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.&&: " << *inst << "\n";
      auto e = graph->createReadEvent(ctx, inst, tid);
      for (int i = 0; i <= 1; i++) {
        auto value = CS.getArgOperand(i);
        auto valueName = LangModel::findGlobalString(value);
        if (valueName.find(".is_signer") != std::string::npos) {
          auto account = valueName;
          auto found = valueName.find(".");
          if (found != std::string::npos)
            account = valueName.substr(0, found);

          if (!thread->isInAccountsMap(account)) {
            account = findCallStackAccountAliasName(func, e, account);
          }
          thread->asssertSignerMap[account] = e;
        }
      }

    } else if (targetFuncName.startswith("sol.!")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.!: " << *inst << "\n";
      auto e = graph->createReadEvent(ctx, inst, tid);
      for (int i = 0; i <= 0; i++) {
        auto value = CS.getArgOperand(i);
        auto valueName = LangModel::findGlobalString(value);
        if (valueName.find(".is_signer") != std::string::npos) {
          auto account = valueName;
          auto found = valueName.find(".");
          if (found != std::string::npos)
            account = valueName.substr(0, found);

          if (!thread->isInAccountsMap(account)) {
            account = findCallStackAccountAliasName(func, e, account);
          }
          thread->asssertSignerMap[account] = e;
        }
      }
    } else if (targetFuncName.startswith("sol.get_account_data.2")) {
      auto e = graph->createReadEvent(ctx, inst, tid);

      auto value1 = CS.getArgOperand(0);
      auto account1 = LangModel::findGlobalString(value1);
      if (auto arg1 = dyn_cast<Argument>(value1))
        account1 = funcArgTypesMap[func][arg1->getArgNo()].first;

      auto value2 = CS.getArgOperand(1);
      auto account2 = LangModel::findGlobalString(value2);
      if (auto arg2 = dyn_cast<Argument>(value2))
        account2 = funcArgTypesMap[func][arg2->getArgNo()].first;

      if (!thread->isInAccountsMap(account1)) {
        account1 = findCallStackAccountAliasName(func, e, account1);
      }
      if (!thread->isInAccountsMap(account2)) {
        account2 = findCallStackAccountAliasName(func, e, account2);
      }
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.get_account_data: account1: " << account1
                     << " account2: " << account2 << "\n";

      auto pair = std::make_pair(account1, account1);
      thread->assertOwnerEqualMap[pair] = e;

    } else if (targetFuncName.startswith("sol.next_account_info.")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "next_account_info: " << *inst << "\n";
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);

    } else if (targetFuncName.startswith("sol.Pubkey::find_program_address.")) {
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto e = graph->createReadEvent(ctx, inst, tid);

      // seeds: "[key.to_le_bytes().as_ref(),&[bump]]
      // llvm::outs() << "account: " << valueName << "\n";
      auto nextInst = inst->getNextNonDebugInstruction();
      if (auto callValue = dyn_cast<CallBase>(nextInst)) {
        CallSite CS2(callValue);
        if (CS2.getTargetFunction()->getName().startswith(
                "sol.model.opaqueAssign")) {
          auto bumpName = LangModel::findGlobalString(CS2.getArgOperand(0));
          //(claim_account_key,claim_account_bump)
          auto found = bumpName.find(",");
          if (found != std::string::npos) {
            auto pda_address = bumpName.substr(0, found).drop_front();
            if (DEBUG_RUST_API)
              llvm::outs() << "pda_address: " << pda_address << "\n";
            if (pda_address != "_")
              thread->asssertProgramAddressMap[pda_address] = e;
            bumpName = bumpName.substr(found + 1);
            bumpName = bumpName.drop_back();
          }
          auto nextInst2 = nextInst->getNextNonDebugInstruction();
          if (auto callValue2 = dyn_cast<CallBase>(nextInst2)) {
            CallSite CS3(callValue2);
            if (CS3.getTargetFunction()->getName().startswith(
                    "sol.require.!2")) {
              auto cons = LangModel::findGlobalString(CS3.getArgOperand(0));
              auto found = cons.find(bumpName);
              if (found != std::string::npos) {
                auto pair = std::make_pair(bumpName, cons);
                thread->assertBumpEqualMap[pair] = e;
              }
            }
          }
        }
      }

    } else if (targetFuncName.startswith(
                   "sol.Pubkey::create_program_address.")) {
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto e = graph->createReadEvent(ctx, inst, tid);
      if (valueName.contains("bump")) {
        bool bumpAdded = false;
        auto nextInst = inst->getNextNonDebugInstruction();
        if (auto callValue = dyn_cast<CallBase>(nextInst)) {
          CallSite CS2(callValue);
          if (CS2.getTargetFunction()->getName().startswith(
                  "sol.model.opaqueAssign")) {
            auto bumpName = LangModel::findGlobalString(CS2.getArgOperand(0));
            auto pair = std::make_pair(bumpName, valueName);
            thread->accountsBumpMap[pair] = e;
            bumpAdded = true;
          }
        }
        if (!bumpAdded) {
          auto pair = std::make_pair("bump", valueName);
          thread->accountsBumpMap[pair] = e;
        }
      }
      if (valueName.startswith("[")) {
        llvm::SmallVector<StringRef, 8> accounts_vec;
        valueName.split(accounts_vec, ',', -1, false);
        for (auto valueName_x : accounts_vec) {
          if (valueName_x.contains(".key")) {
            valueName_x = valueName_x.substr(0, valueName_x.find(".key"));
            valueName_x = stripAll(valueName_x);
            accountsSeedProgramAddressMap[valueName_x] = e;
          }
        }
      }
    } else if (targetFuncName.startswith(
                   "sol.spl_token::instruction::transfer") ||
               targetFuncName.startswith("sol.spl_token::instruction::burn") ||
               targetFuncName.startswith(
                   "sol.spl_token::instruction::mint_to")) {
      if (DEBUG_RUST_API) {
        llvm::outs() << "spl_token::instruction:: " << *inst << "\n";
      }
      auto e = graph->createReadEvent(ctx, inst, tid);
      if (CS.getNumArgOperands() > 3) {
        std::vector<llvm::StringRef> accountNames;
        for (int i = 1; i < 4; i++) {
          auto value_i = CS.getArgOperand(i);
          auto accountName_i = LangModel::findGlobalString(value_i);
          auto foundDot = accountName_i.find_last_of(".");
          // strip .key
          if (foundDot != std::string::npos)
            accountName_i = accountName_i.substr(0, foundDot);
          accountName_i = stripAll(accountName_i);
          thread->accountsInvokedMap[accountName_i] = e;
          // spl_token::instruction::burn
          if (!thread->isInAccountsMap(accountName_i)) {
            accountName_i =
                findCallStackAccountAliasName(func, e, accountName_i);
            thread->accountsInvokedMap[accountName_i] = e;
          }

          // for anchor
          auto foundAnchor = accountName_i.find(".accounts.");
          if (foundAnchor != std::string::npos) {
            // accountName_i = accountName_i.substr(0,
            // accountName_i.find_last_of(".")); strip ctx.accounts.
            accountName_i = stripCtxAccountsName(accountName_i);
            thread->accountsInvokedMap[accountName_i] = e;
          }
          accountNames.push_back(accountName_i);
          if (DEBUG_RUST_API)
            llvm::outs() << "accountsInvokedMap accountName_i: "
                         << accountName_i << "\n";
        }
        // now check if source_token_account.key ==
        // destination_token_account.key
        if (targetFuncName.startswith("sol.spl_token::instruction::transfer")) {
          auto from = accountNames[0];
          auto to = accountNames[1];
          // TODO: check from and to have equality constraints
          // llvm::outs() << "TODO from: " << from << " to: " << to << "\n";
          auto pair = std::make_pair(from, to);
          thread->tokenTransferFromToMap[pair] = e;
        }
      }
    } else if (targetFuncName.startswith(
                   "sol.spl_token::instruction::mint_to")) {
      // Not handled.
    } else if (targetFuncName.startswith(
                   "sol.spl_token::instruction::approve")) {
      // Not handled.
    } else if (targetFuncName.startswith("sol.spl_token::instruction::transfer_"
                                         "checked")) {
      // Not handled.
    } else if (targetFuncName.startswith(
                   "sol.spl_token::instruction::mint_to_checked")) {
      // Not handled.
    } else if (targetFuncName.startswith(
                   "sol.spl_token::instruction::approve_checked")) {
      // Not handled.
    } else if (targetFuncName.startswith("sol.Rent::from_account_info.") ||
               targetFuncName.startswith("sol.Clock::from_account_info.")) {
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      if (auto arg = dyn_cast<Argument>(value))
        valueName = funcArgTypesMap[func][arg->getArgNo()].first;

      if (!valueName.empty()) {
        auto e = graph->createReadEvent(ctx, inst, tid);
        if (!thread->isInAccountsMap(valueName)) {
          valueName = findCallStackAccountAliasName(func, e, valueName);
        }
        auto pair = std::make_pair(valueName, valueName);
        thread->assertKeyEqualMap[pair] = e;
      }
    } else if (targetFuncName.contains("program::invoke")) {
      auto value = CS.getArgOperand(0);
      if (auto callValue = dyn_cast<CallBase>(value)) {
        CallSite CS2(callValue);
        if (CS2.getTargetFunction()->getName().startswith(
                "sol.create_associated_token_account.")) {
          auto value1 = CS.getArgOperand(1);
          auto valueName = LangModel::findGlobalString(value1);
          // split
          llvm::SmallVector<StringRef, 8> accounts_vec;
          valueName.split(accounts_vec, ',', -1, false);
          if (accounts_vec.size() > 3) {
            auto associate_account = accounts_vec[2];
            associate_account = stripToAccountInfo(associate_account);
            auto foundClone = associate_account.find(".clone()");
            if (foundClone != std::string::npos)
              associate_account = associate_account.substr(0, foundClone);
            if (DEBUG_RUST_API)
              llvm::outs() << "sol.create_associated_token_account: "
                           << associate_account << "\n";
            auto e = graph->createReadEvent(ctx, inst, tid);
            // pool_associated_staked_token_account.to_account_info().clone()
            thread->accountsInvokedMap[associate_account] = e;
          }
        }
      }

    } else if (targetFuncName.equals("sol.stake::instruction::split.4")) {
      auto e = graph->createReadEvent(ctx, inst, tid);
      for (int i = 1; i <= 1; i++) {
        auto value = CS.getArgOperand(i);
        auto valueName = LangModel::findGlobalString(value);
        // if this is parameter
        if (valueName.empty()) {
          if (auto arg = dyn_cast<Argument>(value)) {
            valueName = funcArgTypesMap[func][arg->getArgNo()].first;
          }
        }
        thread->accountsInvokedMap[valueName] = e;

        if (DEBUG_RUST_API)
          llvm::outs() << "sol.stake::instruction::split: " << valueName
                       << "\n";
        // if not in accounts, find aliases..
        if (!thread->isInAccountsMap(valueName)) {
          auto valueName_j = findCallStackAccountAliasName(func, e, valueName);
          // from_user_lamports_info
          thread->accountsInvokedMap[valueName_j] = e;
        }
      }

    } else if (targetFuncName.equals(
                   "sol.stake::instruction::delegate_stake.3")) {
      auto e = graph->createReadEvent(ctx, inst, tid);
      for (int i = 0; i <= 2; i++) {
        auto value = CS.getArgOperand(i);
        auto valueName = LangModel::findGlobalString(value);
        // if this is parameter
        if (valueName.empty()) {
          if (auto arg = dyn_cast<Argument>(value)) {
            valueName = funcArgTypesMap[func][arg->getArgNo()].first;
          }
        }
        thread->accountsInvokedMap[valueName] = e;

        if (DEBUG_RUST_API)
          llvm::outs() << "sol.stake::instruction:: " << valueName << "\n";
        // if not in accounts, find aliases..
        if (!thread->isInAccountsMap(valueName)) {
          auto valueName_j = findCallStackAccountAliasName(func, e, valueName);
          // from_user_lamports_info
          thread->accountsInvokedMap[valueName_j] = e;
        }
      }

    } else if (targetFuncName.equals("sol.stake::instruction::authorize.5")) {
      auto e = graph->createReadEvent(ctx, inst, tid);
      for (int i = 0; i <= 2; i++) {
        auto value = CS.getArgOperand(i);
        auto valueName = LangModel::findGlobalString(value);
        // if this is parameter
        if (valueName.empty()) {
          if (auto arg = dyn_cast<Argument>(value)) {
            valueName = funcArgTypesMap[func][arg->getArgNo()].first;
          }
        }

        thread->accountsInvokedMap[valueName] = e;

        if (DEBUG_RUST_API)
          llvm::outs() << "sol.stake::instruction::authorize: " << valueName
                       << "\n";
        // if not in accounts, find aliases..
        if (!thread->isInAccountsMap(valueName)) {
          auto valueName_j = findCallStackAccountAliasName(func, e, valueName);
          // from_user_lamports_info
          thread->accountsInvokedMap[valueName_j] = e;
        }
      }

    } else if (targetFuncName.equals("sol.stake::instruction::withdraw.5")) {
      auto e = graph->createReadEvent(ctx, inst, tid);
      // withdrawer_pubkey: &Pubkey,
      // destination_lamports_info
      for (int i = 1; i <= 2; i++) {
        auto value = CS.getArgOperand(i);
        auto valueName = LangModel::findGlobalString(value);
        // if this is parameter
        if (valueName.empty()) {
          if (auto arg = dyn_cast<Argument>(value)) {
            valueName = funcArgTypesMap[func][arg->getArgNo()].first;
          }
        }
        thread->accountsInvokedMap[valueName] = e;

        if (DEBUG_RUST_API)
          llvm::outs() << "sol.stake::instruction::withdraw: " << valueName
                       << "\n";
        // if not in accounts, find aliases..
        if (!thread->isInAccountsMap(valueName)) {
          auto valueName_j = findCallStackAccountAliasName(func, e, valueName);
          // from_user_lamports_info
          thread->accountsInvokedMap[valueName_j] = e;
        }
      }

    } else if (targetFuncName.contains("system_instruction::transfer")) {
      // TODO track parameter passing
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto found = valueName.find_last_of(".");
      if (found != std::string::npos)
        valueName = valueName.substr(0, found);
      // if this is parameter
      if (valueName.empty()) {
        if (auto arg = dyn_cast<Argument>(value)) {
          valueName = funcArgTypesMap[func][arg->getArgNo()].first;
        }
      }
      auto e = graph->createReadEvent(ctx, inst, tid);
      thread->accountsInvokedMap[valueName] = e;

      if (DEBUG_RUST_API)
        llvm::outs() << "system_instruction::transfer source: " << valueName
                     << "\n";
      // if not in accounts, find aliases..
      if (!thread->isInAccountsMap(valueName)) {
        auto valueName_j = findCallStackAccountAliasName(func, e, valueName);
        // from_user_lamports_info
        thread->accountsInvokedMap[valueName_j] = e;
      }

    } else if (targetFuncName.startswith("//sol.transfer_ctx")) {
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      if (DEBUG_RUST_API)
        llvm::outs() << "TODO sol.transfer_ctx: " << valueName << "\n";
      //
      auto e = graph->createReadEvent(ctx, inst, tid);
      thread->accountsInvokedMap["authority"] = e;

    } else if (targetFuncName.startswith("sol.CpiContext::new.")) {
      auto value = CS.getArgOperand(1);
      auto valueName = LangModel::findGlobalString(value);
      auto found = valueName.find("authority:");
      // authority:self.authority.to_account_info()
      if (found != std::string::npos)
        valueName = valueName.substr(found);
      valueName = stripToAccountInfo(valueName);
      // self.market_authority.clone(),
      found = valueName.find(".clone()");
      if (found != std::string::npos)
        valueName = valueName.substr(0, found);
      valueName = stripSelfAccountName(valueName);
      auto account = valueName;

      if (DEBUG_RUST_API)
        llvm::outs() << "sol.CpiContext::new.: " << valueName << "\n";
      //
      auto e = graph->createReadEvent(ctx, inst, tid);
      thread->accountsInvokedMap[account] = e;
      if (!thread->isInAccountsMap(account)) {
        auto valueName_j = findCallStackAccountAliasName(func, e, account);
        // from_user_lamports_info
        thread->accountsInvokedMap[valueName_j] = e;
      }

    } else if (targetFuncName.startswith("sol.borrow_mut.")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "borrow_mut: " << *inst << "\n";
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      // llvm::outs() << "data: " << valueName << "\n";
      auto e = graph->createReadEvent(ctx, inst, tid);
      // ctx.accounts.candy_machine
      auto account = stripCtxAccountsName(valueName);
      auto found = account.find_last_of(".");
      if (found != std::string::npos)
        account = account.substr(0, found);
      if (!thread->isInAccountsMap(account))
        account = findCallStackAccountAliasName(func, e, account);

      if (valueName.find(".data") != std::string::npos) {
        thread->borrowDataMutMap[account] = e;
        // llvm::outs()
        //     << "borrow_mut account: " << account << " valueName:
        //     "
        //     << valueName << "\n";

        auto found1 = account.find(".accounts.");
        if (found1 != std::string::npos) {
          auto account1 = stripCtxAccountsName(account);
          thread->borrowDataMutMap[account1] = e;
        }
      } else if (valueName.find(".lamports") != std::string::npos) {
        thread->borrowLamportsMutMap[account] = e;

      } else {
        thread->borrowOtherMutMap[account] = e;
      }

      // if (auto user = dyn_cast<llvm::User>(value)) {
      //     auto globalVar = user->getOperand(0);
      //     if (auto gv =
      //     dyn_cast_or_null<GlobalVariable>(globalVar)) {
      //         // llvm::outs() << "global variable: " << *gv <<
      //         "\n"; if (auto globalData =
      //                 dyn_cast_or_null<ConstantDataArray>(gv->getInitializer()))
      //                 {
      //             auto valueName = globalData->getAsString();
      //             llvm::outs() << "data: " << valueName << "\n";
      //         }
      //     }
      // }
      // important
      // could be passed from call parameters
      auto account_ = findCallStackAccountAliasName(func, e, account);
      if (!account_.empty())
        account = account_;
      if (!account.empty())
        if (!thread->isInAccountOrStructAccountMap(account))
          thread->accountsMap[account] = e;

    } else if (targetFuncName.startswith("sol.serialize.")) {
      // create an object of a class
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.serialize: " << *inst << "\n";
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto e = graph->createReadEvent(ctx, inst, tid);
      auto account = valueName;
      thread->borrowDataMutMap[account] = e;
    } else if (targetFuncName.startswith("sol.try_borrow_mut_data.")) {
      // create an object of a class
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.try_borrow_mut_data: " << *inst << "\n";
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto e = graph->createReadEvent(ctx, inst, tid);
      auto account = valueName;
      thread->borrowDataMutMap[account] = e;

      auto account_ = findCallStackAccountAliasName(func, e, account);
      if (!account_.empty())
        account = account_;
      // important
      account = stripCtxAccountsName(valueName);
      if (!account.empty())
        if (!thread->isInAccountOrStructAccountMap(account))
          thread->accountsMap[account] = e;

    } else if (targetFuncName.startswith("sol.borrow.") ||
               targetFuncName.startswith("sol.try_borrow_data.")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.borrow: " << *inst << "\n";
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      auto e = graph->createReadEvent(ctx, inst, tid);
      // ctx.accounts.candy_machine
      auto account = stripCtxAccountsName(valueName);
      auto found = account.find_last_of(".");
      if (found != std::string::npos)
        account = account.substr(0, found);

      if (DEBUG_RUST_API)
        llvm::outs() << "valueName: " << valueName << " account: " << account
                     << "\n";

      if (valueName.find(".data") != std::string::npos) {
        bool isBorrow = true;
        if (valueName.find("borrow_mut") != std::string::npos)
          isBorrow = false;
        if (isBorrow)
          thread->borrowDataMap[account] = e;
        else
          thread->borrowDataMutMap[account] = e;
        // for Anchor: extract accounts from Anchor names
        auto found1 = account.find(".accounts.");
        if (found1 != std::string::npos) {
          auto account1 = stripCtxAccountsName(account);
          if (isBorrow)
            thread->borrowDataMap[account1] = e;
          else
            thread->borrowDataMutMap[account1] = e;
          if (DEBUG_RUST_API)
            llvm::outs() << "account1: " << account1 << "\n";
        }

      } else if (valueName.find(".lamports") != std::string::npos) {
        thread->borrowLamportsMap[account] = e;

      } else {
        thread->borrowOtherMap[account] = e;
      }

      auto account_ = findCallStackAccountAliasName(func, e, account);
      if (!account_.empty())
        account = account_;

      // ctx.accounts.pyth_oracle_price.try_borrow_data()?;
      // accs.instruction_acc.try_borrow_mut_data()?
      // important
      account = stripCtxAccountsName(account);
      if (!account.empty()) {
        if (!thread->isInAccountOrStructAccountMap(account)) {
          // find where account is defined
          // let fee_collector_account_b =
          // ctx.remaining_accounts.get(1).unwrap();
          bool remainingAccountChecked = false;
          if (auto prevInst = inst->getPrevNonDebugInstruction()) {
            if (auto callValue2 = dyn_cast<CallBase>(prevInst)) {
              CallSite CS2(callValue2);
              if (CS2.getTargetFunction()->getName().startswith(
                      "sol.model.opaqueAssign")) {
                auto value1 = CS2.getArgOperand(0);
                auto valueName1 = LangModel::findGlobalString(value1);
                if (account == valueName1) {
                  auto value2 = CS2.getArgOperand(1);
                  if (auto callValue3 = dyn_cast<CallBase>(value2)) {
                    CallSite CS3(callValue3);
                    if (CS3.getTargetFunction()->getName().startswith(
                            "sol.unwrap.1")) {
                      auto value3 = CS3.getArgOperand(0);
                      if (auto callValue4 = dyn_cast<CallBase>(value3)) {
                        CallSite CS4(callValue4);
                        if (CS4.getTargetFunction()->getName().startswith(
                                "sol.get.2")) {
                          auto value41 = CS4.getArgOperand(0);
                          auto valueName41 =
                              LangModel::findGlobalString(value41);
                          auto value42 = CS4.getArgOperand(1);
                          auto valueName42 =
                              LangModel::findGlobalString(value42);
                          auto accountSig =
                              valueName41.str() + ".get(" + valueName42.str();
                          if (thread->isAccountOwnerValidated(accountSig) ||
                              isAnchorDataAccount(accountSig)) {
                            remainingAccountChecked = true;
                            // llvm::outs()
                            //     << "remainingAccountChecked: "
                            //     << remainingAccountChecked <<
                            //     "\n";
                          }
                        }
                      }
                    } else if (CS3.getTargetFunction()->getName().startswith(
                                   "sol.to_account_info.1")) {
                      // let instruction_sysvar_account =
                      // &ctx.accounts.instruction_sysvar_account;
                      // let instruction_sysvar_account_info =
                      // instruction_sysvar_account.to_account_info();
                      // let instruction_sysvar =
                      // instruction_sysvar_account_info.data.borrow();
                      auto value3 = CS3.getArgOperand(0);
                      auto valueName3 = LangModel::findGlobalString(value3);
                      valueName3 = stripAll(valueName3);
                      if (thread->isAccountKeyValidated(valueName3) ||
                          isAnchorDataAccount(valueName3)) {
                        remainingAccountChecked = true;
                      }
                      thread->accountAliasesMap[account].insert(valueName3);
                      if (DEBUG_RUST_API)
                        llvm::outs() << "account: " << account
                                     << " aliasAccount: " << valueName3 << "\n";
                      // llvm::outs() << "valueName3: " <<
                      // valueName3
                      // << "\n";
                    }
                  }
                }
              }
            }
          }
          if (!remainingAccountChecked)
            thread->accountsMap[account] = e;
        }
      }

    } else if (targetFuncName.startswith("sol.require.!2")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.require.!: " << *inst << "\n";
      // merkle_proof::verify(proof,distributor.root,node.0)
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      if (valueName.contains("::verify(")) {
        auto found = valueName.find("(");
        auto params = valueName.substr(found);
        llvm::SmallVector<StringRef, 8> value_vec;
        params.split(value_vec, ',', -1, false);

        auto funcName = valueName.substr(0, found).str() + "." +
                        std::to_string(value_vec.size());
        // llvm::outs() << "func_require: " << funcName << "\n";
        auto func_require = thisModule->getFunction(funcName);
        if (func_require) {
          // llvm::outs() << "found func_require: " << funcName <<
          // "\n";
          traverseFunctionWrapper(ctx, thread, callStack, inst, func_require);
        }
      }
      // TODO parse more

    } else if (targetFuncName.startswith("sol.assert.")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "assert: " << *inst << "\n";
      auto value = CS.getArgOperand(0);
      auto valueName = LangModel::findGlobalString(value);
      // llvm::outs() << "data: " << valueName << "\n";
      // self.user_underlying_token_account.is_native()
      auto account = stripSelfAccountName(valueName);
      auto found = account.find(".");
      if (found != std::string::npos)
        account = account.substr(0, found);

      auto e = graph->createReadEvent(ctx, inst, tid);
      if (!thread->isInAccountsMap(account)) {
        account = findCallStackAccountAliasName(func, e, account);
      }
      if (valueName.find(".is_signer") != std::string::npos) {
        thread->asssertSignerMap[account] = e;
      } else if (valueName.find(".data_is_empty") != std::string::npos) {
        thread->hasInitializedCheck = true;
      } else {
        thread->asssertOtherMap[account] = e;
      }
    } else if (targetFuncName.startswith("sol.assert_eq.") ||
               targetFuncName.startswith("sol.ne.2") ||
               targetFuncName.startswith("sol.eq.2") ||
               targetFuncName.contains("_keys_eq.") ||
               targetFuncName.contains("_keys_neq.") ||
               targetFuncName.contains("_keys_equal.") ||
               targetFuncName.contains("_keys_not_equal.")) {
      // targetFuncName.startswith("sol.assert_eq.")
      // ||
      //        targetFuncName.startswith("sol.assert_keys_eq.2")
      //        ||
      //        targetFuncName.startswith("sol.require_keys_eq.2")
      //        ||
      //        targetFuncName.startswith("sol.require_keys_neq.2")
      //        ||
      //        targetFuncName.startswith("sol.check_keys_equal.2")
      //        ||
      //        targetFuncName.startswith("sol.check_keys_not_equal.2")
      addCheckKeyEqual(ctx, tid, inst, thread, CS);

    } else if (targetFuncName.startswith("sol.assert_owner.")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "assert_owner: " << *inst << "\n";
      if (CS.getNumArgOperands() >= 2) {
        auto value1 = CS.getArgOperand(0);
        auto valueName1 = LangModel::findGlobalString(value1);
        valueName1 = stripAll(valueName1);
        auto value2 = CS.getArgOperand(1);
        auto valueName2 = LangModel::findGlobalString(value2);
        auto pair = std::make_pair(valueName1, valueName2);
        auto e = graph->createReadEvent(ctx, inst, tid);
        thread->assertOwnerEqualMap[pair] = e;
      }
    } else if (targetFuncName.startswith("sol.assert_signer.")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "assert_signer: " << *inst << "\n";
      if (CS.getNumArgOperands() >= 1) {
        auto value1 = CS.getArgOperand(0);
        auto valueName1 = LangModel::findGlobalString(value1);
        auto account = stripAll(valueName1);
        auto e = graph->createReadEvent(ctx, inst, tid);
        thread->asssertSignerMap[account] = e;
      }
    } else if (targetFuncName.startswith("sol.get.2")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.get.2: " << *inst << "\n";
      auto value1 = CS.getArgOperand(0);
      auto valueName1 = LangModel::findGlobalString(value1);
      if (valueName1.contains(".remaining_accounts")) {
        if (auto nextInst = inst->getNextNonDebugInstruction()) {
          if (auto callValue2 = dyn_cast<CallBase>(nextInst)) {
            CallSite CS2(callValue2);
            if (CS2.getTargetFunction()->getName().startswith("sol.unwrap.1") &&
                CS2.getArgOperand(0) == inst) {
              if (auto nextInst2 = nextInst->getNextNonDebugInstruction()) {
                if (auto callValue3 = dyn_cast<CallBase>(nextInst2)) {
                  CallSite CS3(callValue3);
                  if (CS3.getTargetFunction()->getName().startswith(
                          "sol.model.opaqueAssign") &&
                      CS3.getArgOperand(1) == nextInst) {
                    // check
                    auto value2 = CS.getArgOperand(1);
                    auto valueName2 = LangModel::findGlobalString(value2);
                    auto accountSig =
                        ".remaining_accounts.get(" + valueName2.str();
                    if (thread->isAccountOwnerValidated(accountSig)) {
                      auto value3 = CS3.getArgOperand(0);
                      auto valueName3 = LangModel::findGlobalString(value3);
                      auto pair = std::make_pair(valueName3, accountSig);
                      auto e = graph->createReadEvent(ctx, nextInst2, tid);
                      thread->assertOwnerEqualMap[pair] = e;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else if (targetFuncName.startswith("sol.validate.!2")) {
      //            let want_destination_information_addr =
      //            self.get_destination_information_address(ctx);
      // validate!(
      //     ctx.accounts.destination_platform_information.key(),
      //     &want_destination_information_addr
      // );
      auto value1 = CS.getArgOperand(0);
      auto valueName1 = LangModel::findGlobalString(value1);

      // @"ctx.accounts.receiving_underlying_token_account.key().ne(&ctx.accounts.vault_withdraw_queue.key())

      auto value2 = CS.getArgOperand(1);
      auto valueName2 = LangModel::findGlobalString(value2);
      // llvm::outs() << "sol.validate.2 valueName1: " << valueName1
      //              << " valueName2: " << valueName2 << "\n";
      auto e = graph->createReadEvent(ctx, inst, tid);
      if (valueName2.endswith("true") || valueName2.endswith("false")) {
        if (valueName1.contains(".ne(")) {
          auto x1 = valueName1.substr(0, valueName1.find(".ne("));
          auto x2 = valueName1.substr(valueName1.find(".ne(") + 4);
          // llvm::outs() << "updateKeyEqualMap x1: " << x1
          //       << " x2: " << x2 << "\n";
          updateKeyEqualMap(thread, e, false, true, x1, x2);
        }
      } else {
        addCheckKeyEqual(ctx, tid, inst, thread, CS);
        if (valueName2.endswith("_addr") || valueName2.endswith("_address")) {
          if (auto prevInst = inst->getPrevNonDebugInstruction()) {
            if (auto callValue3 = dyn_cast<CallBase>(prevInst)) {
              CallSite CS3(callValue3);
              if (CS3.getTargetFunction()->getName().equals(
                      "sol.model.opaqueAssign")) {
                auto account = stripCtxAccountsName(valueName1);
                auto found = account.find_last_of(".");
                if (found != std::string::npos)
                  account = account.substr(0, found);
                accountsPDAMap[account] = e;
                if (DEBUG_RUST_API)
                  llvm::outs() << "accountsPDAMap account: " << account << "\n";
              }
            }
          }
        }
      }
    } else if (targetFuncName.startswith("sol.>=") ||
               targetFuncName.startswith("sol.<=")) {
      if (DEBUG_RUST_API) {
        llvm::outs() << "sol.>=: " << *inst << "\n";
      }
      auto value1 = CS.getArgOperand(0);
      auto value2 = CS.getArgOperand(1);
      auto valueName1 = LangModel::findGlobalString(value1);
      auto valueName2 = LangModel::findGlobalString(value2);

      if (valueName1.contains("slot") && valueName2.contains("slot") &&
          !valueName2.contains("price")) {
        auto e = graph->createReadEvent(ctx, inst, tid);
        UntrustfulAccount::collect(valueName1, e, callEventTraces,
                                   SVE::Type::MALICIOUS_SIMULATION, 11);
      }

    } else if (targetFuncName.equals("sol.==")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.==: " << *inst << "\n";
      auto value1 = CS.getArgOperand(0);
      auto value2 = CS.getArgOperand(1);
      if (auto key1_inst = dyn_cast<CallBase>(value1)) {
        if (auto key2_inst = dyn_cast<CallBase>(value2)) {
          CallSite CS1(key1_inst);
          CallSite CS2(key2_inst);
          if (CS1.getTargetFunction()->getName().startswith("sol.key.1") &&
              CS2.getTargetFunction()->getName().startswith("sol.key.1")) {
            auto valueName1 = LangModel::findGlobalString(CS1.getArgOperand(0));
            auto valueName2 = LangModel::findGlobalString(CS2.getArgOperand(0));
            auto account1 = stripAll(valueName1);
            auto account2 = stripAll(valueName2);
            auto pairx = std::make_pair(account1, account2);
            auto e = graph->createReadEvent(ctx, inst, tid);
            thread->checkDuplicateAccountKeyEqualMap[pairx] = e;
          }
        }
      }
    } else if (targetFuncName.equals("sol.contains.2")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.contains.2: " << *inst << "\n";
      if (func->getName().contains("::verify")) {
        auto e = graph->createReadEvent(ctx, inst, tid);
        auto value1 = CS.getArgOperand(0);
        auto value2 = CS.getArgOperand(1);
        auto valueName1 = LangModel::findGlobalString(value1);
        auto valueName2 = LangModel::findGlobalString(value2);
        if (valueName1.empty() && isa<Argument>(value1)) {
          if (auto arg1 = dyn_cast<Argument>(value1)) {
            valueName1 = funcArgTypesMap[func][arg1->getArgNo()].first;
          }
        }
        if (valueName2.empty() && isa<Argument>(value2)) {
          if (auto arg2 = dyn_cast<Argument>(value2)) {
            valueName2 = funcArgTypesMap[func][arg2->getArgNo()].first;
          }
        }
        valueName1 = findCallStackAccountAliasName(func, e, valueName1);
        valueName2 = findCallStackAccountAliasName(func, e, valueName2);
        if (DEBUG_RUST_API)
          llvm::outs() << "sol.contains.2 verified: " << valueName2 << " "
                       << valueName1 << "\n";
        auto pairx = std::make_pair(valueName2, valueName1);
        thread->accountContainsVerifiedMap[pairx] = e;
      }
    } else if (targetFuncName.equals("sol.+=")) {
      // No-op; this is handled by the rulset.
    } else if (targetFuncName.equals("sol.-=")) {
      // No-op; this is handled by the rulset.
    } else if (targetFuncName.equals("sol.+")) {
      // No-op; this is handled by the rulset.
    } else if (targetFuncName.equals("sol.-")) {
      // No-op; this is handled by the rulset.
    } else if (targetFuncName.equals("sol.*")) {
      // No-op; this is handled by the rulset.
    } else if (targetFuncName.equals("sol./")) {
      // No-op; this is handled by the rulset.
    } else if (targetFuncName.startswith("sol.checked_div.")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.checked_div: " << *inst << "\n";

      if (isInLoop() && CS.getNumArgOperands() > 1) {
        auto value = CS.getArgOperand(1);
        if (auto denominator_inst = dyn_cast<CallBase>(value)) {
          CallSite CS2(denominator_inst);
          if (CS2.getTargetFunction()->getName().startswith("sol.checked_")) {
            auto e = graph->createReadEvent(ctx, inst, tid);
            UnsafeOperation::collect(e, callEventTraces,
                                     SVE::Type::INCORRECT_DIVISION_LOGIC, 8);
          }
        }
      }
    } else if (targetFuncName.contains("::check_account_owner.2")) {
      auto value1 = CS.getArgOperand(0); // program_id
      auto valueName1 = LangModel::findGlobalString(value1);

      if (auto arg1 = dyn_cast<Argument>(value1)) {
        valueName1 = funcArgTypesMap[func][arg1->getArgNo()].first;
        if (DEBUG_RUST_API)
          llvm::outs() << "check_account_owner valueName1: " << valueName1
                       << "\n";
      }
      auto value2 = CS.getArgOperand(1); // account
      auto valueName2 = LangModel::findGlobalString(value2);
      if (auto arg2 = dyn_cast<Argument>(value2)) {
        valueName2 = funcArgTypesMap[func][arg2->getArgNo()].first;
        if (DEBUG_RUST_API)
          llvm::outs() << "check_account_owner valueName2: " << valueName2
                       << "\n";
      }

      auto pair = std::make_pair(valueName1, valueName2);
      auto e = graph->createReadEvent(ctx, inst, tid);
      thread->assertOwnerEqualMap[pair] = e;

    } else if (targetFuncName.contains("::validate_owner.")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "validate_owner: " << *inst << "\n";
      if (CS.getNumArgOperands() > 2) {
        auto value1 = CS.getArgOperand(0); // program_id
        auto value2 = CS.getArgOperand(1); // authority
        auto value3 = CS.getArgOperand(2); // authority_info
        // auto value4 = CS.getArgOperand(3);  //
        auto valueName2 = LangModel::findGlobalString(value2);
        if (auto arg2 = dyn_cast<Argument>(value2))
          valueName2 = funcArgTypesMap[func][arg2->getArgNo()].first;
        auto valueName3 = LangModel::findGlobalString(value3);
        if (auto arg3 = dyn_cast<Argument>(value3))
          valueName3 = funcArgTypesMap[func][arg3->getArgNo()].first;

        if (!valueName2.empty() || !valueName3.empty()) {
          auto pair = std::make_pair(valueName2, valueName3);
          auto e = graph->createReadEvent(ctx, inst, tid);
          thread->assertOwnerEqualMap[pair] = e;
          thread->assertKeyEqualMap[pair] = e;
          // in addition, owner_account_info.is_signer)
          thread->asssertSignerMap[valueName3] = e;

          if (DEBUG_RUST_API)
            llvm::outs() << "valueName2: " << valueName2
                         << " valueName3: " << valueName3 << "\n";
        }
      }
    } else if (targetFuncName.startswith("sol.require_eq.!")) {
      auto value1 = CS.getArgOperand(0); // mint_info.key
      auto value2 = CS.getArgOperand(1); // dest_account.base.mint
      auto valueName1 = LangModel::findGlobalString(value1);
      if (auto arg1 = dyn_cast<Argument>(value1))
        valueName1 = funcArgTypesMap[func][arg1->getArgNo()].first;
      auto valueName2 = LangModel::findGlobalString(value2);
      if (auto arg2 = dyn_cast<Argument>(value2))
        valueName2 = funcArgTypesMap[func][arg2->getArgNo()].first;

      if (!valueName1.empty() && !valueName2.empty()) {
        if (valueName1.contains(".owner") && valueName2.contains("::ID")) {
          auto e = graph->createReadEvent(ctx, inst, tid);
          valueName1 = stripAll(valueName1);
          auto foundComma = valueName2.find(",");
          if (foundComma != std::string::npos)
            valueName2 = valueName2.substr(0, foundComma);
          auto pair = std::make_pair(valueName1, valueName2);
          thread->assertOwnerEqualMap[pair] = e;
        }
      }
    } else if (targetFuncName.contains("cmp_pubkeys.2")) {
      auto value1 = CS.getArgOperand(0); // mint_info.key
      auto value2 = CS.getArgOperand(1); // dest_account.base.mint
      auto valueName1 = LangModel::findGlobalString(value1);
      if (auto arg1 = dyn_cast<Argument>(value1))
        valueName1 = funcArgTypesMap[func][arg1->getArgNo()].first;
      auto valueName2 = LangModel::findGlobalString(value2);
      if (auto arg2 = dyn_cast<Argument>(value2))
        valueName2 = funcArgTypesMap[func][arg2->getArgNo()].first;

      if (!valueName1.empty() && !valueName2.empty()) {
        auto e = graph->createReadEvent(ctx, inst, tid);

        if (DEBUG_RUST_API)
          llvm::outs() << "sol.cmp_pubkeys.2 valueName1: " << valueName1
                       << " valueName2: " << valueName2 << "\n";

        auto pair = std::make_pair(valueName1, valueName2);
        thread->assertKeyEqualMap[pair] = e;
        auto pair2 = std::make_pair(valueName2, valueName1);
        thread->assertKeyEqualMap[pair2] = e;
        // TODO: owner check
        if (valueName1.contains(".owner") || valueName2.contains(".owner"))
          thread->assertOwnerEqualMap[pair] = e;
      }
    } else if (targetFuncName.startswith("sol.invoke_signed.") ||
               targetFuncName.startswith("sol.invoke.")) {
      auto value = CS.getArgOperand(0);
      // find all accounts invoked
      if (!isa<CallBase>(value))
        value = inst->getPrevNonDebugInstruction();
      auto e = graph->createReadEvent(ctx, inst, tid);

      if (auto call_inst = dyn_cast<CallBase>(value)) {
        auto value2 = value;
        CallSite CS2(call_inst);
        if (CS2.getTargetFunction()->getName().startswith(
                "sol.model.opaqueAssign")) {
          value2 = CS2.getArgOperand(1);
        }
        if (auto call_inst2 = dyn_cast<CallBase>(value2)) {
          CallSite CS3(call_inst2);
          if (CS3.getTargetFunction()->getName().contains("instruction::")) {
            auto valuex = CS.getArgOperand(1);
            auto valueNameX = LangModel::findGlobalString(valuex);
            // split
            valueNameX = valueNameX.substr(1, valueNameX.size() - 2);
            if (DEBUG_RUST_API)
              llvm::outs() << "sol.invoke: valueNameX: " << valueNameX << "\n";
            llvm::SmallVector<StringRef, 8> value_vec;
            valueNameX.split(value_vec, ',', -1, false);
            for (auto value_ : value_vec) {
              auto foundClone = value_.find(".clone");
              if (foundClone != std::string::npos)
                value_ = value_.substr(0, foundClone);
              if (DEBUG_RUST_API)
                llvm::outs() << "sol.invoke: value_: " << value_ << "\n";
              thread->accountsInvokedMap[value_] = e;
              if (!thread->isInAccountsMap(value_)) {
                value_ = findCallStackAccountAliasName(func, e, value_);
                thread->accountsInvokedMap[value_] = e;
              }
            }
          }
        }
      } else {
        if (DEBUG_RUST_API) {
          llvm::outs() << "sol.invoke: failed to find invoked accounts: "
                       << *value << "\n";
        }
      }
    } else if (targetFuncName.contains("//initialized")) {
      if (DEBUG_RUST_API)
        llvm::outs() << "sol.initialized: " << *inst << "\n";
      thread->hasInitializedCheck = true;

    } else {
      if (targetFuncName.contains("initialized")) {
        if (DEBUG_RUST_API) {
          llvm::outs() << "sol.initialized: " << *inst << "\n";
        }
        thread->hasInitializedCheck = true;
      }

      if (targetFuncName.startswith("sol.check_") ||
          targetFuncName.startswith("sol.validate_")) {
        if (DEBUG_RUST_API) {
          llvm::outs() << "sol.check_: " << *inst << "\n";
        }

        // heuristics for interprocedural call
        // for any parameter that is an account name, set it
        // validated
        if (CS.getNumArgOperands() > 0) {
          auto e = graph->createReadEvent(ctx, inst, tid);
          for (auto i = 0; i < CS.getNumArgOperands(); i++) {
            auto value = CS.getArgOperand(i);
            auto valueName = LangModel::findGlobalString(value);
            if (!valueName.empty()) {
              auto pair = std::make_pair(valueName, valueName);
              thread->assertKeyEqualMap[pair] = e;
            }
            if (DEBUG_RUST_API)
              llvm::outs() << targetFuncName << " check_account: " << valueName
                           << "\n";
          }
        }
        // let's still check its content - for signer
      }
      traverseFunctionWrapper(ctx, thread, callStack, inst,
                              CS.getCalledFunction());
    }
  }
}

constexpr unsigned int FUNC_COUNT_PROGRESS_THESHOLD = 10000;

void SolanaAnalysisPass::traverseFunction(
    const xray::ctx *ctx, const Function *func0, StaticThread *thread,
    std::vector<const Function *> &callStack,
    std::map<uint8_t, const llvm::Constant *> *valMap) {
  Function *func = const_cast<Function *>(func0);

  if (DEBUG_RUST_API) {
    llvm::outs() << "SOLANA traverseFunction: " << func->getName() << "\n";
  }

  // simulating call stack
  if (find(callStack.begin(), callStack.end(), func) != callStack.end()) {
    return;
  }

  // FIXME: unsound heuristic
  auto tid = thread->getTID();
  int count = threadNFuncMap[tid];
  count++;
  if (count % FUNC_COUNT_PROGRESS_THESHOLD == 0) {
    LOG_DEBUG("thread {} traversed {} functions", tid, count);
    if (FUNC_COUNT_BUDGET != -1 && count == FUNC_COUNT_BUDGET) {
      // PEIMING: simply return here, as func has not been pushed into
      // callstack yet
      return;
    }
  }
  threadNFuncMap[tid] = count;
  callStack.push_back(func);

  bool isMacroArrayRefUsedInFunction = false;
  // LOG_DEBUG("thread {} enter func : {}", tid,
  // demangle(func->getName().str()));
  cur_trie = xray::trie::getNode(cur_trie, func);
  for (auto &BB : *func) {
    for (BasicBlock::const_iterator BI = BB.begin(), BE = BB.end(); BI != BE;
         ++BI) {
      // traverse each instruction
      const Instruction *inst = cast<Instruction>(BI);
      if (isa<ReturnInst>(inst)) {
        if (inst->getNumOperands() > 0 && isa<LoadInst>(inst->getOperand(0))) {
          auto v = inst->getOperand(0);
          // FIXME: this can also have problem
          auto load = cast<LoadInst>(v);
          auto handle = load->getPointerOperand();
          if (threadIDValueMap.count(handle)) {
            auto e = threadIDValueMap.at(handle);
            threadIDFunctionMap[func] = e;
            LOG_DEBUG(
                "Thread returned from function. thread={}, func={}, eventID={}",
                *handle, func->getName(), e->getID());
          }
        }

      } else if (isa<llvm::CallBase>(inst)) {
        CallSite CS(inst);

        if (!CS.isIndirectCall()) {
          if (LangModel::isRustAPI(CS.getCalledFunction())) {
            auto targetFuncName = CS.getCalledFunction()->getName();
            if (targetFuncName.startswith("sol.iter.")) {
              continue;
            }

            if (CS.getTargetFunction()->getName().startswith("sol.match.")) {
              if (DEBUG_RUST_API) {
                llvm::outs() << "sol.match. funcx: " << func->getName() << "\n";
              }
              if (func->getName().contains("::process_instruction.") ||
                  func->getName().contains("::process.") ||
                  func->getName().contains("::process_with_fee_constraints.") ||
                  func->getName().contains("::process_with_constraints.") ||
                  func->getName().equals("main")) {
                // create a thread for each attack surface
                for (auto i = 0; i < CS.getNumArgOperands(); i++) {
                  if (auto funStart =
                          llvm::dyn_cast<llvm::CallBase>(CS.getArgOperand(i))) {
                    if (funStart->getCalledFunction()->getName().startswith(
                            "sol.ifTrue.anon.") ||
                        funStart->getCalledFunction()->getName().startswith(
                            "sol.ifFalse.anon."))
                      funStart = llvm::dyn_cast<llvm::CallBase>(
                          funStart->getArgOperand(0));

                    if (DEBUG_RUST_API)
                      llvm::outs()
                          << "sol.match. funStart: " << *funStart << "\n";
                    // skip sol.Ok.0
                    if (funStart->getCalledFunction()->getName().startswith(
                            "sol.Ok."))
                      continue;
                    if (funStart->getCalledFunction()->getName().startswith(
                            "sol.if."))
                      continue;
                    if (funStart->getCalledFunction()->getName().startswith(
                            "sol.Err."))
                      continue;
                    bool added =
                        addThreadStartFunction(funStart->getCalledFunction());
                    if (!added)
                      continue;
                    if (tid == 0) {
                      auto e = graph->createForkEvent(ctx, funStart, tid);
                      thread->addForkSite(e);
                      auto newtid = addNewThread(e);
                    }
                  }
                }
              }
              continue;
            }

            if (LangModel::isRustModelAPI(CS.getCalledFunction())) {
              handleRustModelAPI(ctx, tid, func, inst, thread, CS,
                                 isMacroArrayRefUsedInFunction);
            } else { // !isRustModelAPI(CS.getCalledFunction())
              handleNonRustModelAPI(ctx, tid, func, inst, thread, CS);
            }
          } else { // !isRustAPI
            // skip intrinsic such as llvm.dbg.
            if (!CS.getCalledFunction()->isIntrinsic()) {
              traverseFunctionWrapper(ctx, thread, callStack, inst,
                                      CS.getCalledFunction());
            }
          }
        } else { // CS.isIndirectCall()
          // handling indirect function calls (calling function pointer)
          auto cs = pta->getInDirectCallSite(ctx, inst);
          auto indirectCalls = cs->getResolvedTarget();
          // NOTE: the number of indirect calls being resolved is already
          // limited in PTA if (indirectCalls.size() == 1)
          {
            for (auto fun : indirectCalls) {
              // llvm::outs() << "Unique indirect call: funcName=" <<
              // demangle(fun->getName().str())
              // << " in func: " << demangle(func->getName().str()) << " at " <<
              // getSourceLoc(inst).sig() <<
              // "\n"                            <<
              // getSourceLoc(inst).getSnippet() << "\ninst="
              // << *inst
              // << "\n";

              traverseFunctionWrapper(ctx, thread, callStack, inst, fun);
              exploredIndirectTarget.insert(fun);
            }
          }
        }
      } else {
        // llvm::outs() << "Failed to handle instruction: " << *inst << "\n";
        // unhandled instruction
        continue;
      }
    }
  }
  // LOG_DEBUG("thread {} exit func : {}", tid,
  // demangle(func->getName().str()));
  callStack.pop_back();
  if (cur_trie) {
    cur_trie = cur_trie->parent; // return to its parent
  }
}

void SolanaAnalysisPass::traverseFunctionWrapper(
    const xray::ctx *ctx, StaticThread *thread,
    std::vector<const Function *> &callStack, const Instruction *inst,
    const Function *f, std::map<uint8_t, const Constant *> *valMap) {
  // llvm::outs() << "  traverseFunctionWrapper: " << f->getName() << "\n";

  // find the real target if sol.call
  // first BB of f
  auto &bb = f->getEntryBlock();
  auto call_size = callStack.size();
  if (bb.hasName() && bb.getName() == "sol.call" && bb.size() > 2 &&
      call_size > 0) {
    int idx = call_size - 1;
    auto parentFunc = callStack[idx];
    while (parentFunc->getName().startswith("sol.")) {
      idx--;
      if (idx >= 0)
        parentFunc = callStack[idx];
      else
        break;
    }
    bool hasFoundTarget = false;
    if (funcArgTypesMap.find(parentFunc) != funcArgTypesMap.end()) {
      auto pair = funcArgTypesMap[parentFunc][0];
      if (pair.second.startswith("Context<")) {
        auto ctxStructName = getStructName(pair.second);
        if (DEBUG_RUST_API)
          llvm::outs() << "ctxStructName: " << ctxStructName << "\n";
        for (auto &ii : bb) {
          if (isa<CallBase>(&ii)) {
            CallSite CS(&ii);
            auto targetF = CS.getTargetFunction();
            auto fname = targetF->getName();
            if (!ctxStructName.empty() && fname.contains(ctxStructName)) {
              f = targetF;
              if (DEBUG_RUST_API)
                llvm::outs() << "  target: " << fname << "\n";
              hasFoundTarget = true;
              break;
            }
          }
        }
      }
    }
    if (!hasFoundTarget) {
      auto parentFuncName = parentFunc->getName();
      if (DEBUG_RUST_API)
        llvm::outs() << "parentFuncName: " << parentFuncName << "\n";
      for (auto &ii : bb) {
        if (isa<CallBase>(&ii)) {
          CallSite CS(&ii);
          auto targetF = CS.getTargetFunction();
          auto fname = targetF->getName();
          auto found = fname.find("::");
          if (found != std::string::npos) {
            auto packageName = fname.substr(0, found);
            if (DEBUG_RUST_API)
              llvm::outs() << "  packageName: " << packageName << "\n";
            if (parentFuncName.contains(packageName)) {
              f = targetF;
              if (DEBUG_RUST_API)
                llvm::outs() << "  target: " << fname << "\n";
              break;
            }
          }
        }
      }
    }
  }

  if (find(callStack.begin(), callStack.end(), f) != callStack.end()) {
    // recursive call
    return;
  }

  {
    CallEvent *callEvent = nullptr;
    if (f->getName().startswith("sol.")) // for sol only
    {
      callEvent =
          graph->createCallEvent(ctx, inst, f->getName(), thread->getTID());
      callEventTraces[thread->getTID()].push_back(callEvent);
    }

    traverseFunction(ctx, f, thread, callStack, valMap);
    if (callEvent) {
      callEvent->setEndID(Event::getLargestEventID());
    }
    // NOTE: special handling after the thread-creation wrapper returns
    // check if the callee function has created any thread
    auto forkevent = threadIDFunctionMap[f];
    if (forkevent) {
      // LOG_DEBUG("call inst: {}  - fork event id:  {}", *inst,
      // forkevent->getID());
      // FIXME: buggy? inst could overlap
      threadIDValueMap[inst] = forkevent;
    }
  }
}

void SolanaAnalysisPass::initStructFunctions() {
  // find all functions with
  auto &functionList = thisModule->getFunctionList();

  for (auto &function : functionList) {
    auto func = &function;
    if (!func->isDeclaration() &&
        func->getName().startswith("sol.model.struct.")) {
      auto func_struct_name = func->getName();
      if (DEBUG_RUST_API)
        llvm::outs() << "func_struct_name: " << func_struct_name << "\n";

      bool isAnchor = func_struct_name.startswith("sol.model.struct.anchor.");
      std::vector<std::pair<llvm::StringRef, llvm::StringRef>> fieldTypes;
      for (auto &BB : *func) {
        for (auto &I : BB) {
          if (isa<CallBase>(&I)) {
            xray::CallSite CS(&I);
            if (CS.getCalledFunction()->getName().startswith(
                    "sol.model.struct.field")) {
              auto value1 = CS.getArgOperand(0);
              auto field = LangModel::findGlobalString(value1);
              auto value2 = CS.getArgOperand(1);
              auto type = LangModel::findGlobalString(value2);
              if (DEBUG_RUST_API)
                llvm::outs()
                    << "field account: " << field << " type: " << type << "\n";
              fieldTypes.push_back(std::make_pair(field, type));
            }
          }
        }
      }
      if (isAnchor)
        anchorStructFunctionFieldsMap[func] = fieldTypes;
      else
        normalStructFunctionFieldsMap[func] = fieldTypes;
    }
  }
  // next, consolidate all anchor accounts per each thread
}

void SolanaAnalysisPass::detectUntrustfulAccounts(TID tid) {
  auto curThread = StaticThread::getThreadByTID(tid);
  // TODO now, detect vulnerabilities in each thread
  auto funcName = curThread->startFunc->getName();
  llvm::outs() << "\n**************** attack surface #" << curThread->getTID()
               << ": " << funcName.substr(0, funcName.find_last_of("."))
               << " **************** \n";
  if (DEBUG_RUST_API) {
    for (auto [accountName, e] : curThread->accountsMap) {
      llvm::outs() << "account_x: " << accountName << "\n";
    }
    for (auto [pair, e] : curThread->assertKeyEqualMap) {
      llvm::outs() << "assertKeyEqualMap: " << pair.first
                   << " == " << pair.second << "\n";
    }
    for (auto [account, aliasAccounts] : curThread->accountAliasesMap) {
      llvm::outs() << "account: " << account << " aliases: "
                   << "\n";
      for (auto aliasAccount : aliasAccounts) {
        llvm::outs() << "         " << aliasAccount << "\n";
      }
    }
    for (auto [pair, e] : curThread->assertOwnerEqualMap) {
      llvm::outs() << "assertOwnerEqualMap: " << pair.first
                   << " == " << pair.second << "\n";
    }
    for (auto [pair, e] : curThread->assertOtherEqualMap) {
      llvm::outs() << "assertOtherEqualMap: " << pair.first
                   << " == " << pair.second << "\n";
    }
    for (auto [account, e] : curThread->accountsInvokedMap) {
      llvm::outs() << "accountsInvokedMap: " << account << "\n";
    }
    for (auto [pair, e] : assertKeyNotEqualMap) {
      llvm::outs() << "assertKeyNotEqualMap: " << pair.first
                   << " != " << pair.second << "\n";
    }
  }

  bool isInit = funcName.contains("sol.init");
  if (!isInit && !hasThreadStartInitFunction("sol.init")) {
    isInit = funcName.contains("init");
    if (!isInit && !hasThreadStartInitFunction("init")) {
      isInit = funcName.contains("create");
      if (!isInit && !hasThreadStartInitFunction("create")) {
        // try new_
        isInit = funcName.contains("new_");
      }
    }
  }
  if (isInit) {
    auto size = funcArgTypesMap[curThread->startFunc].size();
    for (auto ArgNo = 0; ArgNo < size; ArgNo++) {
      auto pair = funcArgTypesMap[curThread->startFunc][ArgNo];
      // llvm::outs() << "ArgNo: " << ArgNo << " name: " << pair.first << "
      // type: " << pair.second << "\n";
      if (pair.second.contains("Pubkey"))
        potentialOwnerAccounts.insert(pair.first);
      for (auto [accountName, e] : curThread->accountsMap) {
        if (curThread->isAccountPDAInInstruction(accountName) ||
            curThread->isAccountDiscriminator(accountName)) {
          globalStateAccounts.insert(accountName);
          if (DEBUG_RUST_API)
            llvm::errs() << "==============ADDING globalStateAccounts: "
                         << accountName << "============\n";
        }
      }
    }
  }

  bool isPotentiallyOwnerOnly =
      funcName.contains("create_") || funcName.contains("new_");
  bool isOwnerOnly = curThread->isOnceOnlyOwnerOnlyInstruction ||
                     curThread->isAccountSignerMultiSigPDA() ||
                     curThread->isAccountSignerVerifiedByPDAContains() ||
                     curThread->isPotentiallyOwnerOnlyInstruction(
                         potentialOwnerAccounts, isInit);
  if (DEBUG_RUST_API) {
    llvm::outs() << "isOwnerOnly: " << isOwnerOnly << "\n";
  }

  for (auto [accountName, e] : curThread->accountsMap) {
    if (DEBUG_RUST_API) {
      llvm::outs() << "account: " << accountName << "\n";
    }
    if (!isInit) {
      // checker for untrustful accounts
      // skip initialization func
      bool isUnvalidate =
          false; // curThread->isAccountBorrowDataMut(accountName) ||
      // missing owner check
      // ctx.accounts.authority.key != &token.owner
      if (!isUnvalidate &&
          (curThread->isAccountBorrowData(accountName) &&
           !curThread->isAccountBorrowDataMut(accountName)) &&
          !curThread->isAccountBorrowLamportsMut(accountName) &&
          !isAnchorDataAccount(accountName) && !isAccountPDA(accountName) &&
          !curThread->isAccountKeyValidated(accountName) &&
          !curThread->isAccountInvoked(accountName) &&
          !curThread->isAccountOwnerValidated(accountName) &&
          !curThread->isAccountUsedInSeed(accountName)) {
        // llvm::errs() << "==============VULNERABLE:
        // MissingOwnerCheck!============\n";
        UntrustfulAccount::collect(accountName, e, callEventTraces,
                                   SVE::Type::MISS_OWNER, 10);
        isUnvalidate = true;
      }
      if (!isUnvalidate && curThread->isAccountBorrowData(accountName) &&
          !curThread->isAccountBorrowDataMut(accountName)) {
        if (!curThread->isAccountLamportsUpdated(accountName) &&
            !curThread->isAccountKeyValidated(accountName) &&
            (!curThread->isAccountOwnerValidated(accountName))) {
          // llvm::errs() << "==============VULNERABLE: Account
          // Unvalidated!============\n";
          if (!isOwnerOnly) {
            auto e1 = curThread->borrowDataMap[accountName];
            UntrustfulAccount::collect(accountName, e1, callEventTraces,
                                       SVE::Type::ACCOUNT_UNVALIDATED_BORROWED,
                                       9);
            isUnvalidate = true;
          }
        }
      }
      // TODO: define pattern for missing check for p_asset_mint
      // there are some hidden semantc-level invariants between accounts

      if (LangModel::isAuthorityAccount(accountName)) {
        // checker for missing signer accounts
        if (!isUnvalidate && !curThread->isAccountSigner(accountName) &&
            !curThread->isAccountValidatedBySignerAccountInConstraint(
                accountName) &&
            !isAccountPDA(accountName) &&
            !curThread->isAccountInvokedAsAuthority(accountName) &&
            !curThread->isAccountInvoked(accountName) &&
            !curThread->isAccountBorrowDataMut(accountName)) {
          if (!curThread->isAccountReferencedByAnyOtherAccountInHasOne(
                  accountName) &&
              !curThread->isAccountReferencedByAnyOtherAccountInConstraint(
                  accountName) &&
              !curThread->isAccountKeyValidatedSafe(accountName) &&
              !isAnchorDataAccount(accountName)) {
            bool isValidatedBySigner = false;
            if (accountName.endswith("_info")) {
              auto accountName_x =
                  accountName.substr(0, accountName.find("_info"));
              if (curThread->isAccountValidatedBySignerAccountInConstraint(
                      accountName_x)) {
                isValidatedBySigner = true;
              }
            }
            // heuristic: skip checking signer on "manager_fee"
            if (!isOwnerOnly && !isValidatedBySigner &&
                !accountName.contains("pda") && !accountName.contains("mint") &&
                !accountName.contains("new_") &&
                !accountName.contains("from_") &&
                !accountName.contains("to_") && !accountName.contains("_fee")) {
              UntrustfulAccount::collect(accountName, e, callEventTraces,
                                         SVE::Type::MISS_SIGNER, 10);
              isUnvalidate = true;
            }

            // habitat_owner_token_account.owner is signer
          }
        }
      } else {
        // TODO: define pattern for mut declared but not updated account
        //&& !isAccountPDA(accountName) - this condition removed, because PDA
        // can still be faked...
        if (!isUnvalidate && !isAnchorValidatedAccount(accountName) &&
            !curThread->isAccountLamportsUpdated(accountName) &&
            !curThread->isAccountKeyValidated(accountName) &&
            (!curThread->isAccountOwnerValidated(accountName) &&
             !curThread->isAccountSigner(accountName))) {
          // llvm::errs() << "==============VULNERABLE: Account
          // Unvalidated!============\n";
          if (!isOwnerOnly) {
            if (!curThread->isAccountPDAInInstruction(accountName) &&
                !isGlobalStateAccount(accountName) &&
                !curThread->isAccountInvoked(accountName) &&
                !curThread->isAccountBorrowDataMut(accountName) &&
                !curThread->isAccountDiscriminator(accountName) &&
                !curThread->isAccountDataWritten(accountName) &&
                !curThread->isAccountAnchorClosed(accountName)) {
              // bool skipCreateInstruction = funcName.startswith("Create");
              bool skipOrderAccount = false;
              bool skipCloseAccount = false;
              bool skipMintAccount = false;
              bool skipUserAccount = false;
              if (accountName.contains("order")) {
                // TODO: order_account, skip if contains two mints
                // get anchor account type
                auto type = curThread->getStructAccountType(accountName);
                if (accountTypeContainsMoreThanOneMint(type))
                  skipOrderAccount = true;
              } else if (accountName.contains("close")) {
                if (curThread->startFunc->getName().contains("close"))
                  skipCloseAccount = true;
              } else // if (accountName.contains("mint"))
              {
                auto type = curThread->getStructAccountType(accountName);
                // && curThread->isMintAccountValidated(accountName)
                if (type.contains("Mint"))
                  skipMintAccount = true;
                if (type.contains("TokenAccount") &&
                    accountName.contains("user_"))
                  skipUserAccount = true;
                else if ((accountName.contains("user_") ||
                          accountName.contains("multisig_") ||
                          accountName.contains("market_")) &&
                         curThread->isAccountHasOneContraint(accountName)) {
                  skipUserAccount = true;
                }

                // for non-Anchor account
                // habitat_mint
                if (accountName.endswith("_account_info")) {
                  auto accountName_x =
                      accountName.substr(0, accountName.find("_account_info"));
                  if (curThread->isAccountKeyValidated(accountName_x)) {
                    skipMintAccount = true;
                  }
                }
              }
              if (!isPotentiallyOwnerOnly && !skipOrderAccount &&
                  !skipCloseAccount && !skipMintAccount &&
                  !skipUserAccount && //! isPotentiallyOwnerOnly &&
                  !accountName.contains("new_") &&
                  !accountName.contains("payer") &&
                  !accountName.contains("delegat") &&
                  !accountName.contains("dest") &&
                  !accountName.contains("receiver") &&
                  !accountName.equals("to")) {
                if (!curThread->isAccountReferencedBySignerAccountInConstraint(
                        accountName) &&
                    !curThread->isAccountValidatedBySignerAccountInConstraint(
                        accountName) &&
                    !curThread->isAccountReferencedByAnyOtherAccountInSeeds(
                        accountName) &&
                    !isAccountUsedInSeedsProgramAddress(accountName)) {
                  UntrustfulAccount::collect(
                      accountName, e, callEventTraces,
                      SVE::Type::ACCOUNT_UNVALIDATED_OTHER, 8);
                  isUnvalidate = true;
                }
              }
            } else if (curThread->isAccountInvoked(accountName)) {
              // TODO: check if account is isolated.
              // llvm::errs()
              //     << "==============VULNERABLE: Invoke Account Potentially
              //     Unvalidated!============\n";
              // UntrustfulAccount::collect(e, callEventTraces,
              // SVE::Type::ACCOUNT_UNVALIDATED_OTHER, 5); isUnvalidate = true;
            }
          }
        }
      }
    }
  }

  if (DEBUG_RUST_API) {
    llvm::outs() << "checking bump_seed_canonicalization_insecure\n";
  }
  // check bump_seed_canonicalization_insecure
  for (auto [pair, e] : curThread->accountsBumpMap) {
    auto bumpName = pair.first;
    auto account = pair.second;
    // llvm::outs() << "bumpName: " << bumpName << "\n";
    if (!curThread->isAccountBumpValidated(bumpName) &&
        !curThread->isSignerAccountUsedSeedsOfAccount0(account)) {
      // llvm::errs() << "==============VULNERABLE: bump seed
      // canonicalization!============\n";
      if (!isOwnerOnly) {
        UntrustfulAccount::collect(bumpName, e, callEventTraces,
                                   SVE::Type::BUMP_SEED, 9);
      }
    }
  }
}

void SolanaAnalysisPass::detectAccountsCosplay(const xray::ctx *ctx, TID tid) {
  if (anchorStructFunctionFieldsMap.size() != 0) {
    return;
  }

  // now checking cosplay for non-Anchor program only
  // for each pair of structs, do they overlap?
  for (auto &[func1, fieldTypes1] : normalStructFunctionFieldsMap) {
    for (auto &[func2, fieldTypes2] : normalStructFunctionFieldsMap) {
      if (func1 == func2) {
        continue;
      }

      auto size = fieldTypes1.size();
      auto size2 = fieldTypes2.size();
      if (size > size2) {
        continue;
      }

      bool mayCosplay = true;
      bool hasPubkey = false;
      for (size_t i = 0; i < size; i++) {
        // does not need exact match: e.g., u64 vs f64
        auto type1 = fieldTypes1[i].second;
        auto type2 = fieldTypes2[i].second;
        type1 = type1.substr(1);
        type2 = type2.substr(1);
        if (!type1.equals(type2) || (fieldTypes1[i].first.contains("crimi") ||
                                     fieldTypes2[i].first.contains("crimi"))) {
          mayCosplay = false;
          break;
        }
        // llvm::errs() << "==============type1: " << type1 <<
        // "============\n"; llvm::errs() << "==============type2: " <<
        // type2 << "============\n"; must have a Pubkey field
        if (type1.contains("ubkey")) {
          hasPubkey = true;
        }
      }
      if (mayCosplay && hasPubkey) {
        // report
        auto inst1 = func1->getEntryBlock().getFirstNonPHI();
        auto inst2 = func2->getEntryBlock().getFirstNonPHI();
        auto e1 = graph->createApiReadEvent(ctx, inst1, tid);
        auto e2 = graph->createApiReadEvent(ctx, inst2, tid);
        auto file1 = getSourceLoc(e1->getInst()).getFilename();
        auto file2 = getSourceLoc(e2->getInst()).getFilename();
        if (DEBUG_RUST_API) {
          llvm::errs() << "==============file1: " << file1
                       << " file2: " << file2 << "============\n";
        }
        if (file1 == file2) {
          // llvm::errs() << "==============VULNERABLE: Type
          // Cosplay!============\n";
          if (size == size2) {
            CosplayAccounts::collect(e1, e2, callEventTraces,
                                     SVE::Type::COSPLAY_FULL, 6);
          } else {
            CosplayAccounts::collect(e1, e2, callEventTraces,
                                     SVE::Type::COSPLAY_PARTIAL, 5);
          }
        }
      }
    }
  }
}

StaticThread *SolanaAnalysisPass::forkNewThread(ForkEvent *forkEvent) {
  xray::CallSite forkSite(forkEvent->getInst());
  assert(forkSite.isCallOrInvoke());

  if (LangModel::isRustAPI(forkSite.getTargetFunction())) {
    auto threadEntry = pta->getDirectNode(forkEvent->getContext(),
                                          forkSite.getTargetFunction());
    if (!threadEntry) {
      llvm::outs() << "Failed to find function: "
                   << forkSite.getTargetFunction()->getName() << "\n";
      return nullptr;
    }
    auto thread =
        new StaticThread(threadEntry, forkEvent->getInst(), forkEvent);
    forkEvent->setSpawnedThread(thread);
    llvm::outs() << "Found attack surface #" << thread->getTID() << ": "
                 << forkSite.getTargetFunction()->getName() << "\n";
    return thread;
  }
  // FIXME: For indirect pthread callback, only traverse the first function ptr
  // get resolved eventually this should be fixed by having more precise pta
  // results such as adding context or inlining pthread wrapper
  auto cs =
      pta->getInDirectCallSite(forkEvent->getContext(), forkEvent->getInst());
  auto threadEntry = *cs->getResolvedNode().begin();
  // FIXME: memory leakage here!
  auto thread = new StaticThread(threadEntry, forkEvent->getInst(), forkEvent);
  // this->threadNum++;
  forkEvent->setSpawnedThread(thread);
  return thread;
}

bool SolanaAnalysisPass::runOnModule(llvm::Module &module) {
  thisModule = &module;

  // initialization
  getAnalysis<PointerAnalysisPass<PTA>>().analyze(&module);
  this->pta = getAnalysis<PointerAnalysisPass<PTA>>().getPTA();
  this->tbaa = &getAnalysis<TypeBasedAAWrapperPass>().getResult();

  const CallGraphTy *callGraph = pta->getCallGraph();

  initStructFunctions();

  const CallGraphNodeTy *entryNode = GT::getEntryNode(callGraph);

  // FIXME: memory leakage here! use a smart pointer later
  auto mainThread = new StaticThread(entryNode);

  graph = new ReachGraph(*this);
  xray::trie::TrieNode *rootTrie = nullptr;

  // start traversing with the main thread
  threadList.push(mainThread);

  LOG_INFO("Start Analyzing");
  logger::newPhaseSpinner("Detecting Vulnerabilities");

  LOG_INFO("Start Building SHB");
  auto shb_start = std::chrono::steady_clock::now();

  // stop traversing the program until no new thread is found
  while (!threadList.empty()) {
    StaticThread *curThread = threadList.front();
    LOG_DEBUG("Start travering thread {}, thread start = {}",
              curThread->getTID(), curThread->getStartFunction()->getName());
    threadList.pop();
    threadSet.push_back(curThread);

    if (rootTrie) {
      xray::trie::cleanTrie(rootTrie);
    }
    rootTrie = xray::trie::getNode(nullptr, nullptr);
    cur_trie = rootTrie; // init cur_trie to root

    auto tid = curThread->getTID();
    // pthread_create
    // we record this in the call traces for the ease of users' debugging
    CallEvent *ptCreate = nullptr;
    // thread callback function
    CallEvent *threadCB = nullptr;
    if (tid != 0) {
      auto forkEvent = curThread->getParentEvent();
      // SPDLOG_CRITICAL(*forkEvent->getInst());
      ptCreate = graph->createCallEvent(
          forkEvent->getContext(), forkEvent->getInst(),
          CallSite(forkEvent->getInst()).getCalledFunction()->getName(), tid);
      callEventTraces[tid].push_back(ptCreate);
      // main function has no callsite
      // so we cannot create a callEvent for main
      // we can hard code "main" in the stack trace when tid is 0
      if (curThread->getEntryNode()) {
        threadCB = graph->createCallEvent(
            curThread->getEntryNode()->getContext(), forkEvent->getInst(),
            curThread->getEntryNode(), tid);
      } else {
        threadCB = graph->createCallEvent(
            forkEvent->getContext(), forkEvent->getInst(),
            curThread->getStartFunction()->getName(), tid);
      }
      callEventTraces[tid].push_back(threadCB);
    }

    // make sure for new threads initial lockset is empty
    graph->startTraverseNewThread();
    callStack.clear();
    auto threadArg = StaticThread::getThreadArg(curThread->getTID());
    traverseFunction(
        curThread->getEntryNode() ? curThread->getEntryNode()->getContext()
                                  : curThread->getParentEvent()->getContext(),
        curThread->getStartFunction(), curThread, callStack, threadArg);
    if (threadCB != nullptr) {
      threadCB->setEndID(Event::getLargestEventID());
    }

    if (ptCreate != nullptr) {
      ptCreate->setEndID(Event::getLargestEventID());
    }
  }

  if (threadSet.size() == 1) {
    // did not find any new thread
    detectUntrustfulAccounts(0);
  } else {
    // find all potential owner accounts
    for (auto tid = 1; tid < threadSet.size(); tid++) {
      auto curThread = threadSet[tid];
      if (curThread->isOnceOnlyOwnerOnlyInstruction) {
        for (auto [accountName, e] : curThread->accountsMap) {
          if (accountName != "payer" &&
              curThread->isAccountSigner(accountName)) {
            potentialOwnerAccounts.insert(accountName);
            llvm::outs() << "Found potentialOwnerAccount: " << accountName
                         << "\n";
          }
        }
      }
    }
    for (auto tid = 1; tid < threadSet.size(); tid++) {
      detectUntrustfulAccounts(tid);
    }
  }

  NUM_OF_ATTACK_VECTORS = threadSet.size();

  auto shb_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> shb_elapsed = shb_end - shb_start;
  LOG_DEBUG("Finish Building SHB. time={}s", shb_elapsed.count());
  LOG_DEBUG("Number of threads: {}", StaticThread::getThreadNum());

  auto detectStarted = std::chrono::steady_clock::now();
  detectAccountsCosplay(entryNode->getContext(), 0);
  auto detectDone = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = detectDone - detectStarted;
  LOG_DEBUG("Finished Analysis. time={}s", elapsed.count());

  logger::endPhase();

  outputJSON(CONFIG_OUTPUT_PATH);
  return false;
}

// Thread Management.

// add inter-thread HB edges (fork) for SHB Graph
TID SolanaAnalysisPass::addNewThread(ForkEvent *forkEvent) {
  auto t = forkNewThread(forkEvent);
  if (t) {
    threadList.push(t);
    graph->addThreadForkEdge(forkEvent,
                             forkEvent->getSpawnedThread()->getTID());
    hasFoundThread = true;
    return t->getTID();
  } else
    return 0;
}

bool SolanaAnalysisPass::addThreadStartFunction(const llvm::Function *func) {
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

bool SolanaAnalysisPass::hasThreadStartInitFunction(std::string symbol) {
  for (auto func : threadStartFunctions) {
    if (func->getName().contains(symbol)) {
      if (DEBUG_RUST_API)
        llvm::outs() << "hasThreadStartInitFunction: " << func->getName() << " "
                     << symbol << "\n";
      return true;
    }
  }
  if (DEBUG_RUST_API)
    llvm::outs() << "hasThreadStartInitFunction false: " << symbol << "\n";
  return false;
}

// Accounts.

bool SolanaAnalysisPass::accountTypeContainsMoreThanOneMint(
    llvm::StringRef struct_name) const {
  auto num_mints = 0;
  std::string func_struct_name = "sol.model.struct.anchor." + struct_name.str();
  auto func_struct = thisModule->getFunction(func_struct_name);
  if (func_struct) {
    const auto &fieldTypes = anchorStructFunctionFieldsMap.at(func_struct);
    for (const auto &pair : fieldTypes) {
      if (pair.first.contains("mint") && pair.second.equals("Pubkey")) {
        num_mints++;
      }
    }
  }
  return num_mints > 1;
}

bool SolanaAnalysisPass::isAnchorStructFunction(
    const llvm::Function *func) const {
  return anchorStructFunctionFieldsMap.find(func) !=
         anchorStructFunctionFieldsMap.end();
}

bool SolanaAnalysisPass::isAnchorTokenProgram(
    llvm::StringRef accountName) const {
  for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
    for (auto pair : fieldTypes) {
      if (pair.first.equals(accountName) &&
          pair.second.equals("Program<'info, Token>"))
        return true;
    }
  }
  return false;
}

bool SolanaAnalysisPass::isAnchorTokenAccount(
    llvm::StringRef accountName) const {
  for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
    for (auto pair : fieldTypes) {
      if (pair.first.equals(accountName) &&
          pair.second.contains("TokenAccount>"))
        return true;
    }
  }
  return false;
}

bool SolanaAnalysisPass::isAnchorValidatedAccount(
    llvm::StringRef accountName) const {
  for (auto [func, fieldTypes] : anchorStructFunctionFieldsMap) {
    for (auto pair : fieldTypes) {
      if (pair.first.equals(accountName) &&
          (pair.second.contains("Program<") || pair.second.contains("Sysvar<")))
        return true;
    }
  }
  return false;
}

bool SolanaAnalysisPass::isAnchorDataAccount(
    llvm::StringRef accountName) const {
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

bool SolanaAnalysisPass::isCompatibleSeeds(llvm::StringRef seed,
                                           llvm::StringRef seed2) const {
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

bool SolanaAnalysisPass::isAccountPDA(llvm::StringRef accountName) const {
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

bool SolanaAnalysisPass::isAccountUsedInSeedsProgramAddress(
    llvm::StringRef accountName) const {
  if (accountsSeedProgramAddressMap.find(accountName) !=
      accountsSeedProgramAddressMap.end()) {
    if (DEBUG_RUST_API)
      llvm::outs() << "isAccountUsedInSeedsProgramAddress: " << accountName
                   << "\n";
    return true;
  }
  if (DEBUG_RUST_API)
    llvm::outs() << "isAccountUsedInSeedsProgramAddress false: " << accountName
                 << "\n";

  return false;
}

bool SolanaAnalysisPass::isAccountKeyNotEqual(
    llvm::StringRef accountName1, llvm::StringRef accountName2) const {
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

char SolanaAnalysisPass::ID = 0;
static RegisterPass<SolanaAnalysisPass>
    RD("Solana Analysis", "Analyze Solana programs with defined rules",
       true, /*CFG only*/
       true /*is analysis*/);
