//
// Created by peiming on 11/15/19.
//

#include <Util/Log.h>
#include <conflib/conflib.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h> // for llvm LLVMContext
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h> // IR reader for bit file
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/Signals.h>   // signal for command line
#include <llvm/Support/SourceMgr.h> // for SMDiagnostic
#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include "CustomAPIRewriters/RustAPIRewriter.h"
#include "PTAModels/GraphBLASModel.h"
#include "Rules/CosplayAccounts.h"
#include "Rules/Races.h"
#include "Rules/UnsafeOperation.h"
#include "Rules/UntrustfulAccount.h"
#include "SVE.h"
#include "SolanaAnalyzorPass.h"

using namespace llvm;
using namespace aser;
using namespace std;

cl::opt<bool> DebugIR("debug-ir",
                      cl::desc("Load from modified.ll directly, no "
                               "preprocessing and openlib rewriting"),
                      cl::init(false));
cl::opt<bool> NoLinkStub("no-stub", cl::desc("Do not link stub files"),
                         cl::init(true));
cl::opt<bool> SkipOverflowChecks("skip-overflow",
                                 cl::desc("skip overflow checks"));

cl::opt<std::string> TargetModulePath(cl::Positional,
                                      cl::desc("path to input bitcode file"));
cl::opt<bool> ConfigDumpIR("dump-ir", cl::desc("Dump the modified ir file"));
cl::opt<bool>
    ConfigNoFilter("no-filter",
                   cl::desc("Turn off the filtering for race report"));
cl::opt<bool>
    ConfigIncludeAtomic("include-atomics",
                        cl::desc("Include races on atomic operations"));
cl::opt<std::string> ConfigOutputPath("o", cl::desc("JSON output path"),
                                      cl::value_desc("path"));
cl::opt<bool> ConfigNoOMP("nomp",
                          cl::desc("Turn off the OpenMP race detection"));
cl::opt<bool>
    ConfigNoAV("no-av", cl::desc("Turn off the atomicity violation detection"));
cl::opt<bool> ConfigNoOV("no-ov",
                         cl::desc("Turn off the order violation detection"));
cl::opt<bool> ConfigCheckIdenticalWrites(
    "check-identical-writes",
    cl::desc("Turn on detecting races bewteen identical writes"),
    cl::init(true));
cl::opt<bool> ConfigNoMissedOMP("no-missed-omp",
                                cl::desc("Do not scan for missed omp regions"));
cl::opt<bool>
    ConfigNoMissAPI("no-miss-match-api",
                    cl::desc("Turn off the miss-match api detection"));
cl::opt<bool> ConfigDebugLog("v", cl::desc("Turn off log to file"));
cl::opt<int> ConfigReportLimit("limit",
                               cl::desc("Max number of races can be reported"),
                               cl::value_desc("number"), cl::init(-1));

cl::opt<bool> ConfigIgnoreReadWriteRaces("ignore-rw",
                                         cl::desc("Ignore read-write races"));
cl::opt<bool> ConfigIgnoreWriteWriteRaces("ignore-ww",
                                          cl::desc("Ignore write-write races"));

cl::opt<bool>
    ConfigNoReportLimit("no-limit",
                        cl::desc("No limit for the number of races reported"));

cl::opt<size_t> MaxIndirectTarget(
    "max-indirect-target", cl::init(2),
    cl::desc("max number of indirect call target that can be resolved by "
             "indirect call"));
cl::opt<bool> ConfigNoFalseAlias("Xno-false-alias",
                                 cl::desc("Turn off checking false alias"));
cl::opt<bool>
    ConfigNoProducerConsumer("Xno-producer-consumer",
                             cl::desc("Turn off checking producer consumer"));
cl::opt<bool> ConfigEntryPointOnce(
    "entry-point-once",
    cl::desc("Create only one thread for each entry point"));
cl::opt<bool>
    ConfigPrintImmediately("Xprint-fast",
                           cl::desc("Print races immediately on terminal"));
cl::opt<bool> ConfigTerminateImmediately(
    "one-race", cl::desc("Print a race and terminate immediately"));

cl::opt<bool> ConfigShowSummary("Xshow-race-summary",
                                cl::desc("show race summary"));
cl::opt<bool> ConfigShowDetail("Xshow-race-detail",
                               cl::desc("show race details"));
cl::opt<bool> ConfigShowAllTerminal(
    "t", cl::desc("show race detail and summary on the terminal"));

cl::opt<size_t> PTAAnonLimit(
    "Xpta-anon-limit", cl::init(10000),
    cl::desc("max number of anonymous abjects to allocate in pointer analysis. "
             "(Use this if "
             "missed omp takes too much memory)"));

bool DEBUG_LOCK; // default is false
bool DEBUG_LOCK_STR;
bool DEBUG_RUST_API;
bool DEBUG_API;

bool DEBUG_CALL_STACK;
bool DEBUG_LOCK_STACK;
bool DEBUG_DUMP_LOCKSET;
bool DEBUG_RACE;
bool DEBUG_THREAD;
bool DEBUG_OMP_RACE;
bool DEBUG_HB;
bool DEBUG_RACE_EVENT; // DEBUG_EVENT is defined on windows already
bool DEBUG_HAPPEN_IN_PARALLEL;
bool DEBUG_PTA;
bool DEBUG_PTA_VERBOSE;
bool USE_MAIN_CALLSTACK_HEURISTIC;
bool CONFIG_CTX_INSENSITIVE_PTA;
bool ENABLE_OLD_OMP_ALIAS_ANALYSIS = true;
bool FORTRAN_IR_MODE = false;
bool OPT_QUICK_CHECK = true;               // default is false
bool OPT_SAME_THREAD_AT_MOST_TWICE = true; // default is true
bool PRINT_IMMEDIATELY = false;
bool TERMINATE_IMMEDIATELY = false;
bool CONFIG_SKIP_SINGLE_THREAD = false;
// if fast is enabled, we will do aggressive performance optimization
cl::opt<bool> CONFIG_FAST_MODE("fast", cl::desc("Use fast detection mode"));
// if fast is enabled, we will do aggressive performance optimization
cl::opt<bool> CONFIG_EXHAUST_MODE("full",
                                  cl::desc("Use exhaustive detection mode"));

cl::opt<bool> ConfigIgnoreRepeatedMainCallStack(
    "skip-repeat-cs", cl::desc("Skip repeated call stack in main thread"));
cl::opt<bool>
    ConfigNoOrigin("no-origin",
                   cl::desc("Use context-insensitive pointer analysis"));
cl::opt<bool> ConfigDebugThread("debug-threads",
                                cl::desc("Turn on debug thread logs"));
cl::opt<bool> ConfigDebugRace("debug-race",
                              cl::desc("Turn on debug race logs"));
cl::opt<bool> ConfigDebugOMPRace("debug-omp-race",
                                 cl::desc("Turn on debug openmp race logs"));
cl::opt<bool> ConfigDebugCS("debug-cs", cl::desc("Turn on debug call stack"));
cl::opt<bool> ConfigShowCallStack("show-call-stack",
                                  cl::desc("Print call stack in the graph"));
cl::opt<bool> ConfigShowLockStack("show-lock-stack",
                                  cl::desc("Print lock stack in the graph"));
cl::opt<bool> ConfigDumpLockSet("debug-dump-lockset",
                                cl::desc("Print lock set in the graph"));

cl::opt<bool> ConfigDebugPTA("debug-pta",
                             cl::desc("Turn on debug pointer analysis"));
cl::opt<bool> ConfigDebugPTAVerbose(
    "debug-pta-verbose",
    cl::desc("Turn on debug pointer analysis verbose mode"));

cl::list<std::string> ConfigDebugDebugDebug("debug-focus", cl::CommaSeparated,
                                            cl::desc("debug focus mode"));

cl::opt<bool> ConfigDebugLock("debug-lock", cl::desc("Turn on debug lock set"));
cl::opt<bool> ConfigDebugLockStr("debug-lock-str",
                                 cl::desc("Turn on debug lock string"));
cl::opt<bool> ConfigDebugRustAPI("debug-sol",
                                 cl::desc("Turn on debug sol api"));

cl::opt<bool> ConfigDebugHB("debug-hb",
                            cl::desc("Turn on debug happens-before"));
cl::opt<bool> ConfigDebugEvent("debug-event",
                               cl::desc("Turn on debug read-write"));
cl::opt<bool>
    ConfigDebugHappenInParallel("debug-hip",
                                cl::desc("Turn on debug happen-in-parallel"));
cl::opt<bool> ConfigIngoreLock("no-lock",
                               cl::desc("Turn off the lockset check"));
cl::opt<bool>
    ConfigNoPathSensitive("no-ps",
                          cl::desc("Turn off path-sensitive analysis"));

cl::opt<bool> ConfigNoOldOMPAliasAnalysis(
    "Xno-old-omp-alias", cl::desc("Turn off the old OMPIndexAlias analysis"));
cl::opt<bool>
    CONFIG_NO_KEYWORD_FILTER("Xno-svfFilter",
                             cl::desc("Turn off keyword filter in SVF pass"));
cl::opt<bool> ConfigFlowFilter("flowFilter", cl::desc("Enable flow filter"));
cl::opt<bool>
    ConfigDisableProgress("no-progress",
                          cl::desc("Does not print spinner progress"));

bool CONFIG_CHECK_UncheckedAccount;

bool CONFIG_NO_OMP;
bool CONFIG_NO_AV;
bool CONFIG_NO_OV;
bool CONFIG_NO_MISSED_OMP;
bool CONFIG_NO_FILTER;
bool CONFIG_CHECK_IDENTICAL_WRITE;
bool CONFIG_INCLUDE_ATOMIC;
bool CONFIG_NO_REPORT_LIMIT;
bool CONFIG_SHOW_SUMMARY;
bool CONFIG_SHOW_DETAIL;
bool CONFIG_IGNORE_LOCK;
bool CONFIG_LOOP_UNROLL;
bool CONFIG_NO_FALSE_ALIAS;
bool CONFIG_NO_PRODUCER_CONSUMER;

bool CONFIG_ENTRY_POINT_SINGLE_TIME;
// NOTE: temporary config option, turn off path-sensitive
bool CONFIG_NO_PS;
bool CONFIG_SKIP_CONSTRUCTOR;

int MAX_CALLSTACK_DEPTH;   // -1 (no limit) by default
int SAME_FUNC_BUDGET_SIZE; // keep at most x times per func per thread 10 by
                           // default
int FUNC_COUNT_BUDGET;     // 100,000 by default

bool CONFIG_IGNORE_READ_WRITE_RACES = false;
bool CONFIG_IGNORE_WRITE_WRITE_RACES = false;

// for solana
std::map<llvm::StringRef, const llvm::Function *> FUNC_NAME_MAP;
const llvm::Function *getFunctionFromPartialName(llvm::StringRef partialName) {
  for (auto [name, func] : FUNC_NAME_MAP) {
    if (name.contains(partialName) && !name.contains(".anon."))
      return func;
  }
  return nullptr;
}

const llvm::Function *getFunctionMatchStartEndName(llvm::StringRef startName,
                                                   llvm::StringRef endName) {
  for (auto [name, func] : FUNC_NAME_MAP) {
    if (name.startswith(startName) && name.endswith(endName) &&
        !name.contains(".anon."))
      return func;
  }
  return nullptr;
}

logger::LoggingConfig initLoggingConf() {
  logger::LoggingConfig config;

  config.enableProgress = conflib::Get<bool>("XenableProgress", true);
  if (ConfigDisableProgress) {
    config.enableProgress = false;
  }

  auto conflevel = conflib::Get<std::string>("logger.level", "debug");
  if (conflevel == "trace") {
    config.level = spdlog::level::trace;
  } else if (conflevel == "debug") {
    config.level = spdlog::level::debug;
  } else if (conflevel == "info") {
    config.level = spdlog::level::info;
  } else if (conflevel == "warn") {
    config.level = spdlog::level::warn;
  } else if (conflevel == "error") {
    config.level = spdlog::level::err;
  } else {
    // TODO: Print warning. How to log error about setting up logger?
    config.level = spdlog::level::trace;
  }

  config.enableTerminal = conflib::Get<bool>("logger.toStderr", false);
  config.terminalLevel = config.level;

  config.enableFile = true;
  config.logFolder = conflib::Get<std::string>("logger.logFolder", "./");
  config.logFile = "log.current";
  config.fileLevel = config.level;

  if (ConfigDebugLog) {
    config.enableFile = false;
    config.enableTerminal = true;
    config.enableProgress = false;
  }

  return config;
}

void initRaceDetect() {
  CONFIG_IGNORE_READ_WRITE_RACES =
      ConfigIgnoreReadWriteRaces |
      conflib::Get<bool>("ignoreReadWriteRaces", false);
  CONFIG_IGNORE_WRITE_WRITE_RACES =
      ConfigIgnoreWriteWriteRaces |
      conflib::Get<bool>("ignoreWriteWriteRaces", false);

  auto reportLimit = conflib::Get<int>("raceLimit", -1);
  if (reportLimit != -1) {
    ConfigReportLimit = reportLimit;
  }
  OrderViolation::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  DeadLock::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  UntrustfulAccount::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  UnsafeOperation::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  CosplayAccounts::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
}

unsigned int NUM_OF_FUNCTIONS = 0;
unsigned int NUM_OF_ATTACK_VECTORS = 0;
unsigned int NUM_OF_IR_LINES = 0;
unsigned int TOTAL_SOL_COST = 0;
unsigned int TOTAL_SOL_TIME = 0;
static std::unique_ptr<Module>
loadFile(const std::string &FN, LLVMContext &Context, bool abortOnFail) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Result;
  if (DebugIR) {
    // false to ignore potentially broken IR, our preprocessing passes will make
    // the IR unreadable Result = parseIRFile(FN, Err, Context, false);
    Result = parseIRFile(FN, Err, Context);
  } else {
    Result = parseIRFile(FN, Err, Context);
  }

  if (!Result) {
    if (abortOnFail) {
      Err.print("racedetect", llvm::errs());
      abort();
    }

    LOG_ERROR("error loading file: {}", FN);
    return nullptr;
  }
  // count line number
  {
    std::ifstream myfile(FN);

    // new lines will be skipped unless we stop it from happening:
    myfile.unsetf(std::ios_base::skipws);

    // count the newlines with an algorithm specialized for counting:
    unsigned line_count = std::count(std::istream_iterator<char>(myfile),
                                     std::istream_iterator<char>(), '\n');

    // std::cout << "NumOfIRLines: " << line_count << "\n";
    NUM_OF_IR_LINES = line_count;
  }

  return Result;
}

// Method to compare two versions.
// Returns 1 if v2 is smaller, -1
// if v1 is smaller, 0 if equal
int versionCompare(string v1, string v2) {      // v1: the real version: ^0.20.1
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

map<StringRef, StringRef> CARGO_TOML_CONFIG_MAP;
bool hasOverFlowChecks = false;
bool anchorVersionTooOld = false;
bool splVersionTooOld = false;
bool solanaVersionTooOld = false;
void computeCargoTomlConfig(Module *module) {
  // st.class.metadata
  auto f = module->getFunction("sol.model.cargo.toml");
  if (f) {
    for (auto &BB : *f) {
      for (auto &I : BB) {
        if (isa<CallBase>(&I)) {
          aser::CallSite CS(&I);
          if (CS.getNumArgOperands() < 2)
            continue;
          auto v1 = CS.getArgOperand(0);
          auto v2 = CS.getArgOperand(1);

          auto valueName1 = LangModel::findGlobalString(v1);
          auto valueName2 = LangModel::findGlobalString(v2);
          // llvm::outs() << "key: " << valueName1 << "\n";
          // llvm::outs() << "value: " << valueName2 << "\n";
          if (SkipOverflowChecks) {
            hasOverFlowChecks = true;
          } else {
            auto overflow_checks = "profile.release.overflow-checks";
            if (valueName1 == overflow_checks) {
              if (valueName2.contains("1"))
                hasOverFlowChecks = true;
              llvm::outs() << "overflow_checks: " << hasOverFlowChecks << "\n";
            }
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
}

string exec(string command) {
  char buffer[128];
  string result = "";

  // Open pipe to file
  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe) {
    return "popen failed!";
  }

  // read till end of process:
  while (!feof(pipe)) {
    // use buffer to read and add to result
    if (fgets(buffer, 128, pipe) != NULL)
      result += buffer;
  }

  pclose(pipe);
  return result;
}

map<std::string, vector<AccountIDL>> IDL_INSTRUCTION_ACCOUNTS;
void loadIDLInstructionAccounts(std::string api_name, jsoncons::json j) {
  // llvm::outs() << "accounts: " << j.as_string() << "\n";
  if (j.is_array()) {
    // iterate the array
    for (const auto &item : j.array_range()) {
      auto account_name = item["name"].as<std::string>();
      // nested accounts
      if (item.contains("accounts")) {
        auto j2 = item.at("accounts");
        // llvm::outs() << "nested accounts: " << account_name << "\n";
        IDL_INSTRUCTION_ACCOUNTS[api_name].emplace_back(account_name, false,
                                                        false, true);
        loadIDLInstructionAccounts(api_name, j2);
      } else if (item.contains("isMut") && item.contains("isSigner")) {
        auto isMut = item["isMut"].as<bool>();
        auto isSigner = item["isSigner"].as<bool>();
        // llvm::outs() << "account_name: " << account_name << " isMut: " <<
        // isMut << " isSigner: " << isSigner<< "\n";
        IDL_INSTRUCTION_ACCOUNTS[api_name].emplace_back(account_name, isMut,
                                                        isSigner);
      }
    }
  }
}

set<StringRef> SMART_CONTRACT_ADDRESSES;
void computeDeclareIdAddresses(Module *module) {
  // st.class.metadata
  auto f = module->getFunction("sol.model.declare_id.address");
  if (f) {
    for (auto &BB : *f) {
      for (auto &I : BB) {
        if (isa<CallBase>(&I)) {
          aser::CallSite CS(&I);
          if (CS.getNumArgOperands() < 1)
            continue;
          auto v1 = CS.getArgOperand(0);
          auto valueName1 = LangModel::findGlobalString(v1);
          llvm::outs() << "contract address: " << valueName1 << "\n";
          SMART_CONTRACT_ADDRESSES.insert(valueName1);
          if (IDL_INSTRUCTION_ACCOUNTS.empty()) {
            // mainnet only
            // anchor --provider.cluster mainnet idl fetch
            // nosRB8DUV67oLNrL45bo2pFLrmsWPiewe2Lk2DRNYCp -o
            // nosRB8DUV67oLNrL45bo2pFLrmsWPiewe2Lk2DRNYCp.json
            auto address_str = valueName1.str();
            auto path = address_str + "-idl.json";
            if (const char *env_p = std::getenv("CODERRECT_TMPDIR")) {
              path = "/" + path; // unix separator
              path = env_p + path;
            }
            auto cmd = "/usr/bin/anchor --provider.cluster mainnet idl fetch " +
                       address_str + +" -o " + path;
            auto result = exec(cmd);
            // llvm::outs() << "anchor idl : " << result << "\n";
            LOG_INFO("Anchor IDL result: {}", result);
            ifstream ifile;
            ifile.open(path);
            if (ifile) {
              LOG_INFO("Find Anchor IDL File: {}", path);
              auto j0 = jsoncons::json::parse(ifile);
              std::string instructions = "instructions";
              if (j0.contains(instructions)) {
                auto j = j0.at(instructions);
                if (j.is_array()) {
                  // iterate the array
                  for (const auto &item : j.array_range()) {
                    auto api_name = item["name"].as<std::string>();
                    // llvm::outs() << "api_name: " << api_name << "\n";
                    auto j2 = item["accounts"];
                    loadIDLInstructionAccounts(api_name, j2);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

static void createBuilderCallFunction(llvm::IRBuilder<> &builder,
                                      llvm::Function *f) {
  std::vector<llvm::Value *> Args;
  auto it = f->arg_begin();
  auto ie = f->arg_end();

  for (; it != ie; it++) {
    if (it->getType()->isPointerTy()) {
      llvm::AllocaInst *allocaInst = builder.CreateAlloca(
          dyn_cast<PointerType>(it->getType())->getPointerElementType(), 0, "");
      Args.push_back(allocaInst);
    } else {
      llvm::APInt zero(32, 0);
      Args.push_back(
          llvm::Constant::getIntegerValue(builder.getInt32Ty(), zero));
    }
  }
  llvm::ArrayRef<llvm::Value *> argsRef(Args);
  builder.CreateCall(f, argsRef, "");
}

static void createFakeMain(llvm::Module *module) {
  // let's create a fake main func here and add it to the module IR
  // in the fake main, call each entry point func
  llvm::IRBuilder<> builder(module->getContext());
  // create fake main with type int(i32 argc, i8** argv)
  auto functionType = llvm::FunctionType::get(
      builder.getInt32Ty(),
      {builder.getInt32Ty(), builder.getInt8PtrTy()->getPointerTo()}, false);
  llvm::Function *mainFunction = llvm::Function::Create(
      functionType, llvm::Function::ExternalLinkage, "cr_main", module);
  llvm::BasicBlock *entryBB = llvm::BasicBlock::Create(
      module->getContext(), "entrypoint", mainFunction);
  builder.SetInsertPoint(entryBB);

  llvm::Function *realMainFun = module->getFunction("main");
  if (realMainFun && !realMainFun->isDeclaration()) {
    if (realMainFun->getFunctionType() == functionType) {
      // create a call to real main using fake main's argc, argv if possible
      llvm::SmallVector<Value *, 2> args;
      for (auto &arg : mainFunction->args()) {
        args.push_back(&arg);
      }
      builder.CreateCall(realMainFun, args, "");
    } else {
      createBuilderCallFunction(builder, realMainFun);
    }
  }

  // cr_main return
  builder.CreateRet(llvm::ConstantInt::get(builder.getInt32Ty(), 0));
}

int main(int argc, char **argv) {
  // InitLLVM will setup signal handler to print stack trace when the program
  // crashes.
  InitLLVM x(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  // We don't use args in conflib for now
  std::map<std::string, std::string> args;
  conflib::Initialize(args, true);

  auto logConfig = initLoggingConf();
  logger::init(logConfig);

  // TODO: user specified heap api should be used to detect heap wrapper
  // function as well!
  auto heapAPIs =
      conflib::Get<std::vector<std::string>>("heapAllocFunctions", {});

  // Initialize Solana SVEs.
  auto configured =
      conflib::Get<std::map<std::string, std::map<std::string, std::string>>>(
          "solana.sve", {});
  SVE::init(configured);

  MAX_CALLSTACK_DEPTH = conflib::Get<int>("maxCallStackDepth", -1);
  SAME_FUNC_BUDGET_SIZE = conflib::Get<int>("sameFunctionBudget", 10);
  FUNC_COUNT_BUDGET = conflib::Get<int>("functionCountBudget", 20000);

  GraphBLASHeapModel::init(heapAPIs);

  auto enableOMP = conflib::Get<bool>("enableOpenMP", true);
  auto enableAV = conflib::Get<bool>("enableAtomicityViolation", true);
  auto enableOV = conflib::Get<bool>("enableOrderViolation", true);
  auto enableIdenticalWrites =
      conflib::Get<bool>("enableIdenticalWrites", false);
  auto enableLockSet = conflib::Get<bool>("enableLockSet", true);
  auto enableFilter = conflib::Get<bool>("enableFilter", true);
  auto enableLoopUnroll = conflib::Get<bool>("enableLoopUnroll", false);
  auto enableImmediatePrint =
      conflib::Get<bool>("report.enableTerminal", false);
  auto enableShowRaceSummary =
      conflib::Get<bool>("enablePrintRaceSummary", false);
  auto enableShowRaceDetail =
      conflib::Get<bool>("enablePrintRaceDetail", false);

  auto entryPointOnce = conflib::Get<bool>("entryPoint.once", false);

  CONFIG_CHECK_UncheckedAccount =
      conflib::Get<bool>("solana.account.UncheckedAccount", true);

  CONFIG_NO_OMP = ConfigNoOMP | !enableOMP;
  CONFIG_NO_AV = ConfigNoAV | !enableAV;
  CONFIG_NO_OV = ConfigNoOV | !enableOV;
  CONFIG_NO_MISSED_OMP = ConfigNoMissedOMP; // TODO: add conflib check as well
  CONFIG_CHECK_IDENTICAL_WRITE =
      ConfigCheckIdenticalWrites | enableIdenticalWrites;

  CONFIG_INCLUDE_ATOMIC =
      ConfigIncludeAtomic | conflib::Get<bool>("includeAtomics", false);
  CONFIG_NO_FILTER = ConfigNoFilter | !enableFilter;
  CONFIG_NO_REPORT_LIMIT =
      ConfigNoReportLimit | (conflib::Get<int>("raceLimit", -1) < 0);
  CONFIG_IGNORE_LOCK = ConfigIngoreLock | !enableLockSet;
  CONFIG_NO_FALSE_ALIAS = ConfigNoFalseAlias;
  CONFIG_NO_PRODUCER_CONSUMER = ConfigNoProducerConsumer;

  CONFIG_ENTRY_POINT_SINGLE_TIME = ConfigEntryPointOnce | entryPointOnce;

  DEBUG_RACE = ConfigDebugRace;
  DEBUG_THREAD = ConfigDebugThread;
  DEBUG_OMP_RACE = ConfigDebugOMPRace;
  DEBUG_CALL_STACK = ConfigDebugCS | ConfigShowCallStack;
  DEBUG_LOCK_STACK = ConfigShowLockStack;
  DEBUG_DUMP_LOCKSET = ConfigDumpLockSet;
  DEBUG_LOCK = ConfigDebugLock;
  DEBUG_LOCK_STR = ConfigDebugLockStr;
  DEBUG_RUST_API = ConfigDebugRustAPI;
  DEBUG_HB = ConfigDebugHB;
  DEBUG_RACE_EVENT = ConfigDebugEvent;
  DEBUG_HAPPEN_IN_PARALLEL = ConfigDebugHappenInParallel;
  DEBUG_PTA_VERBOSE = ConfigDebugPTAVerbose;
  DEBUG_PTA = ConfigDebugPTA | DEBUG_PTA_VERBOSE;
  USE_MAIN_CALLSTACK_HEURISTIC = ConfigIgnoreRepeatedMainCallStack;
  CONFIG_CTX_INSENSITIVE_PTA = ConfigNoOrigin;
  ENABLE_OLD_OMP_ALIAS_ANALYSIS = !ConfigNoOldOMPAliasAnalysis;

  PRINT_IMMEDIATELY =
      ConfigPrintImmediately | enableImmediatePrint | ConfigShowAllTerminal;
  TERMINATE_IMMEDIATELY = ConfigTerminateImmediately;
  CONFIG_SHOW_SUMMARY = ConfigShowSummary | enableShowRaceSummary |
                        ConfigShowAllTerminal; // show race summary
  CONFIG_SHOW_DETAIL = ConfigShowDetail | enableShowRaceDetail |
                       ConfigShowAllTerminal; // show race details

  // by default, set the pts size to 999
  PTSTrait<PtsTy>::setPTSSizeLimit(9); // set pts size limit to 999
  if (CONFIG_FAST_MODE) {
    CONFIG_CTX_INSENSITIVE_PTA = true;
    MaxIndirectTarget = 1;
  } else if (CONFIG_EXHAUST_MODE) {
    // use origin-sensitive PTA
    CONFIG_CTX_INSENSITIVE_PTA = false;
    // no limit to indirect target
    MaxIndirectTarget = 999;
    // no quick check optimization
    OPT_QUICK_CHECK = false;
    // no same thread limit two optimization
    OPT_SAME_THREAD_AT_MOST_TWICE = false;
    // no filtering??
    // CONFIG_NO_FILTER = true;
  }

  CONFIG_NO_PS = ConfigNoPathSensitive;
  CONFIG_SKIP_CONSTRUCTOR = conflib::Get<bool>("skipConstructors", true);
  CONFIG_SKIP_SINGLE_THREAD = conflib::Get<bool>("skipSingleThreaded", false);

  initRaceDetect();

  LOG_INFO("Loading IR From File: {}", TargetModulePath);
  logger::newPhaseSpinner("Loading IR From File");

  LLVMContext Context;
  // load it now as we need the triple information to determine which stub to
  // link
  auto module = loadFile(TargetModulePath, Context, true);
  module->setModuleIdentifier("coderrect.stub.pid" + std::to_string(getpid()));

  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();

  initializeCore(Registry);
  initializeScalarOpts(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeAggressiveInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);

  LOG_INFO("Running Compiler Optimization Passes");
  logger::newPhaseSpinner("Running Compiler Optimization Passes");

  if (!DebugIR) {
    RustAPIRewriter::rewriteModule(module.get());

    createFakeMain(module.get());
  }

  // Dump IR to file
  if (ConfigDumpIR) {
    std::error_code err;
    llvm::raw_fd_ostream outfile("modified.ll", err, llvm::sys::fs::OF_None);
    if (err) {
      llvm::errs() << "Error dumping IR!\n";
    }

    module->print(outfile, nullptr);
    outfile.close();
  }

  logger::newPhaseSpinner("Running Pointer Analysis");

  llvm::legacy::PassManager analysisPasses;
  analysisPasses.add(new PointerAnalysisPass<PTA>());
  analysisPasses.add(new RaceDetectionPass());

  computeCargoTomlConfig(module.get());
  computeDeclareIdAddresses(module.get());
  analysisPasses.run(*module);

  return 0;
}

static llvm::RegisterPass<PointerAnalysisPass<PTA>>
    PAP("Pointer Analysis Wrapper Pass", "Pointer Analysis Wrapper Pass", true,
        true);