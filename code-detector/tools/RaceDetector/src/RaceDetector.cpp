//
// Created by peiming on 11/15/19.
//

#include <aser/Util/Log.h>
#include <conflib/conflib.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h>  // for llvm LLVMContext
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>  // IR reader for bit file
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/Signals.h>    // signal for command line
#include <llvm/Support/SourceMgr.h>  // for SMDiagnostic
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#ifdef __MINGW32__
#include <libloaderapi.h>
#endif

#include "LinkModules.h"
#include "OpenLib.h"
#include "PTAModels/ExtFunctionManager.h"
#include "PTAModels/GraphBLASModel.h"
#include "RaceDetectionPass.h"
#include "Races.h"
#include "aser/PreProcessing/IRPreProcessor.h"
#include "aser/PreProcessing/Passes/CanonicalizeGEPPass.h"
#include "aser/PreProcessing/Passes/InsertGlobalCtorCallPass.h"
#include "aser/PreProcessing/Passes/LoweringMemCpyPass.h"
#include "aser/PreProcessing/Passes/RemoveASMInstPass.h"
#include "aser/PreProcessing/Passes/RemoveExceptionHandlerPass.h"
#include "aser/PreProcessing/Passes/UnrollThreadCreateLoopPass.h"
#include "aser/PreProcessing/Passes/WrapperFunIdentifyPass.h"

using namespace llvm;
using namespace aser;
using namespace std;

cl::opt<bool> DebugIR("debug-ir",
                      cl::desc("Load from modified.ll directly, no "
                               "preprocessing and openlib rewriting"),
                      cl::init(false));
cl::opt<bool> SkipPreProcess("skip-preprocess",
                             cl::desc("Do not run preprocessing pass"),
                             cl::init(true));
cl::opt<bool> NoLinkStub("no-stub", cl::desc("Do not link stub files"),
                         cl::init(true));
cl::opt<bool> SkipOverflowChecks("skip-overflow",
                                 cl::desc("skip overflow checks"));

cl::opt<std::string> TargetModulePath(cl::Positional,
                                      cl::desc("path to input bitcode file"));
cl::opt<bool> ConfigDumpIR("dump-ir", cl::desc("Dump the modified ir file"));
cl::opt<bool> ConfigNoFilter(
    "no-filter", cl::desc("Turn off the filtering for race report"));
cl::opt<bool> ConfigIncludeAtomic(
    "include-atomics", cl::desc("Include races on atomic operations"));
cl::opt<std::string> ConfigOutputPath("o", cl::desc("JSON output path"),
                                      cl::value_desc("path"));
cl::opt<bool> ConfigNoOMP("nomp",
                          cl::desc("Turn off the OpenMP race detection"));
cl::opt<bool> ConfigNoAV(
    "no-av", cl::desc("Turn off the atomicity violation detection"));
cl::opt<bool> ConfigNoOV("no-ov",
                         cl::desc("Turn off the order violation detection"));
cl::opt<bool> ConfigCheckIdenticalWrites(
    "check-identical-writes",
    cl::desc("Turn on detecting races bewteen identical writes"),
    cl::init(true));
cl::opt<bool> ConfigNoMissedOMP("no-missed-omp",
                                cl::desc("Do not scan for missed omp regions"));
cl::opt<bool> ConfigNoMissAPI(
    "no-miss-match-api", cl::desc("Turn off the miss-match api detection"));
cl::opt<bool> ConfigDebugLog("v", cl::desc("Turn off log to file"));
cl::opt<int> ConfigReportLimit("limit",
                               cl::desc("Max number of races can be reported"),
                               cl::value_desc("number"), cl::init(-1));

cl::opt<bool> ConfigIgnoreReadWriteRaces("ignore-rw",
                                         cl::desc("Ignore read-write races"));
cl::opt<bool> ConfigIgnoreWriteWriteRaces("ignore-ww",
                                          cl::desc("Ignore write-write races"));

cl::opt<bool> ConfigNoReportLimit(
    "no-limit", cl::desc("No limit for the number of races reported"));
cl::opt<bool> ConfigAnalyzeAPI(
    "analyze-api", cl::desc("Analyze all public apis in the library"));
cl::opt<bool> ConfigPrintAPI("print-api",
                             cl::desc("Print all public apis in the library"));
cl::opt<bool> ConfigOpenLibOnce(
    "openlib-once",
    cl::desc("Analyze only once for every public api in the library"));
cl::opt<bool> ConfigOptimalAPI(
    "openlib-optimal",
    cl::desc("Analyze all public synchronization apis in the library"));
cl::opt<bool> ConfigForkAPI(
    "openlib-fork", cl::desc("Analyze all smalltalk fork apis in the library"));

cl::opt<size_t> MaxIndirectTarget(
    "max-indirect-target", cl::init(2),
    cl::desc("max number of indirect call target that can be resolved by "
             "indirect call"));
cl::opt<bool> ConfigLoopUnroll(
    "Xloop-unroll", cl::desc("Turn on unrolling the thread creation loops"));
cl::opt<bool> ConfigNoFalseAlias("Xno-false-alias",
                                 cl::desc("Turn off checking false alias"));
cl::opt<bool> ConfigNoProducerConsumer(
    "Xno-producer-consumer", cl::desc("Turn off checking producer consumer"));
cl::opt<bool> ConfigEntryPointOnce(
    "entry-point-once",
    cl::desc("Create only one thread for each entry point"));
cl::opt<bool> ConfigPrintImmediately(
    "Xprint-fast", cl::desc("Print races immediately on terminal"));
cl::opt<bool> ConfigTerminateImmediately(
    "one-race", cl::desc("Print a race and terminate immediately"));
cl::opt<bool> ConfigPrintCoverage("Xprint-cov",
                                  cl::desc("Print analysis code coverage"));

cl::opt<bool> ConfigIntegrateDynamic(
    "Xdynamic", cl::desc("Store race results for dynamic analysis"));

cl::opt<bool> ConfigShowSummary("Xshow-race-summary",
                                cl::desc("show race summary"));
cl::opt<bool> ConfigShowDetail("Xshow-race-detail",
                               cl::desc("show race details"));
cl::opt<bool> ConfigShowAllTerminal(
    "t", cl::desc("show race detail and summary on the terminal"));

cl::opt<bool> ConfigKeepLockPath("Xkeep-lock-path",
                                 cl::desc("Turn on exploring all lock paths"));

cl::opt<size_t> PTAAnonLimit(
    "Xpta-anon-limit", cl::init(10000),
    cl::desc("max number of anonymous abjects to allocate in pointer analysis. "
             "(Use this if "
             "missed omp takes too much memory)"));
cl::opt<string> SOTERIA_PLAN("plan", cl::desc("soteria plan"));

bool DEBUG_LOCK;  // default is false
bool DEBUG_LOCK_STR;
bool DEBUG_RUST_API;
bool DEBUG_API;

bool DEBUG_CALL_STACK;
bool DEBUG_LOCK_STACK;
bool DEBUG_DUMP_LOCKSET;
bool DEBUG_INDIRECT_CALL;
bool DEBUG_INDIRECT_CALL_ALL;
bool DEBUG_RACE;
bool DEBUG_THREAD;
bool DEBUG_OMP_RACE;
bool DEBUG_HB;
bool DEBUG_RACE_EVENT;  // DEBUG_EVENT is defined on windows already
bool DEBUG_HAPPEN_IN_PARALLEL;
bool DEBUG_PTA;
bool DEBUG_PTA_VERBOSE;
bool USE_MAIN_CALLSTACK_HEURISTIC;
bool CONFIG_CTX_INSENSITIVE_PTA;
bool ENABLE_OLD_OMP_ALIAS_ANALYSIS = true;
bool FORTRAN_IR_MODE = false;
bool OPT_QUICK_CHECK = true;                // default is false
bool OPT_SAME_THREAD_AT_MOST_TWICE = true;  // default is true
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
cl::opt<bool> ConfigNoOrigin(
    "no-origin", cl::desc("Use context-insensitive pointer analysis"));
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

cl::opt<bool> ConfigDebugIndirectCall("debug-indirect",
                                      cl::desc("Turn on debug indirect call"));
cl::opt<bool> ConfigDebugIndirectCallAll(
    "debug-indirect-all", cl::desc("Turn on debug indirect call"));
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
cl::opt<bool> ConfigDebugAPI("debug-api", cl::desc("Turn on debug open API"));

cl::opt<bool> ConfigDebugHB("debug-hb",
                            cl::desc("Turn on debug happens-before"));
cl::opt<bool> ConfigDebugEvent("debug-event",
                               cl::desc("Turn on debug read-write"));
cl::opt<bool> ConfigDebugHappenInParallel(
    "debug-hip", cl::desc("Turn on debug happen-in-parallel"));
cl::opt<bool> ConfigIngoreLock("no-lock",
                               cl::desc("Turn off the lockset check"));
cl::opt<bool> ConfigNoPathSensitive(
    "no-ps", cl::desc("Turn off path-sensitive analysis"));

cl::opt<bool> ConfigNoOldOMPAliasAnalysis(
    "Xno-old-omp-alias", cl::desc("Turn off the old OMPIndexAlias analysis"));
cl::opt<bool> CONFIG_NO_KEYWORD_FILTER(
    "Xno-svfFilter", cl::desc("Turn off keyword filter in SVF pass"));
cl::opt<bool> ConfigFlowFilter("flowFilter", cl::desc("Enable flow filter"));
cl::opt<bool> ConfigDisableProgress(
    "no-progress", cl::desc("Does not print spinner progress"));

bool CONFIG_CHECK_UncheckedAccount;

bool CONFIG_NO_OMP;
bool CONFIG_NO_AV;
bool CONFIG_NO_OV;
bool CONFIG_NO_MISSED_OMP;
bool CONFIG_NO_MISS_MATCH_API;
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
bool CONFIG_EXPLORE_ALL_LOCK_PATHS;
bool CONFIG_INTEGRATE_DYNAMIC;

bool CONFIG_ENTRY_POINT_SINGLE_TIME;
// NOTE: temporary config option, turn off path-sensitive
bool CONFIG_NO_PS;
bool CONFIG_SKIP_CONSTRUCTOR;
bool CONFIG_OPEN_API;
bool CONFIG_OPEN_API_FORK_ONLY = false;
extern cl::opt<bool> CONFIG_USE_FI_MODE;  // for fortran

int MAX_CALLSTACK_DEPTH;    // -1 (no limit) by default
int SAME_FUNC_BUDGET_SIZE;  // keep at most x times per func per thread 10 by
                            // default
int FUNC_COUNT_BUDGET;      // 100,000 by default

bool USE_FAKE_MAIN = true;
// static OriginsSetter<3> OS{"pthread_create", "GrB_Matrix_new",
// "__kmpc_fork_call", "__kmpc_fork_teams"}; using Model =
// GraphBLASModel<KOrigin<1>>;
map<string, string> NONE_PARALLEL_FUNCTIONS;

bool CONFIG_IGNORE_READ_WRITE_RACES = false;
bool CONFIG_IGNORE_WRITE_WRITE_RACES = false;

vector<string> IGNORED_FUN_ALL;
vector<string> IGNORED_VAR_ALL;
vector<string> IGNORED_LOCATION_ALL;
vector<string> LOW_PRIORITY_FILE_NAMES;
vector<string> LOW_PRIORITY_VAR_NAMES;
vector<string> HIGH_PRIORITY_FILE_NAMES;
vector<string> HIGH_PRIORITY_VAR_NAMES;

vector<string> CONFIG_INDIRECT_APIS;
vector<string> CONFIG_MUST_EXPLORE_APIS;
map<string, string> CRITICAL_INDIRECT_TARGETS;

// for solana
std::map<std::string, std::map<std::string, std::string>> SOLANA_SVE_DB;
std::map<llvm::StringRef, const llvm::Function *> FUNC_NAME_MAP;
const llvm::Function *getFunctionFromPartialName(llvm::StringRef partialName) {
  for (auto [name, func] : FUNC_NAME_MAP) {
    if (name.contains(partialName) && !name.contains(".anon.")) return func;
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
// for smalltalk
std::set<std::string> SmallTalkReadAPINames{"st.next", "st.get:", "st.size",
                                            "st.notEmpty", "st.includes:"};
std::set<std::string> SmallTalkWriteAPINames{
    "st.nextPut:", "st.removeKey:", "st.at:put:", "st.removeFirst", "st.add:"};

std::set<std::string> SmallTalkCTRemoteClasses;

vector<string> DEBUG_FOCUS_VEC;

namespace aser {
void rewriteUserSpecifiedAPI(Module *M);
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

openlib::OpenLibConfig initOpenLibConf(Module *module) {
  openlib::OpenLibConfig config;
  config.module = module;
  auto entrys = conflib::Get<vector<jsoncons::json>>("openlib.entryPoints", {});

  for (jsoncons::json &j : entrys) {
    switch (j.kind()) {
      case jsoncons::value_kind::short_string_value:
      case jsoncons::value_kind::long_string_value:
      case jsoncons::value_kind::byte_string_value: {
        string name = j.as<std::string>();
        config.entryPoints.emplace_back(name);
        break;
      }
      default: {
        auto entryMap =
            j.as<std::map<std::string, std::map<std::string, bool>>>();
        for (auto &[name, properties] : entryMap) {
          bool runOnce = false;
          auto onceIt = properties.find("once");
          if (onceIt != properties.end()) {
            runOnce = onceIt->second;
          }

          bool argShared = true;
          auto argSharedIt = properties.find("arg_shared");
          if (argSharedIt != properties.end()) {
            argShared = argSharedIt->second;
          }

          bool isParallel = true;
          auto parallelIt = properties.find("parallel");
          if (parallelIt != properties.end()) {
            isParallel = parallelIt->second;
          }

          config.entryPoints.emplace_back(name, runOnce, argShared, isParallel);
        }
        break;
      }
    }
  }

  auto fork = conflib::Get<bool>("openlib.fork", false);
  auto optimal = conflib::Get<bool>("openlib.optimal", false);
  if (ConfigForkAPI | fork) {
    CONFIG_OPEN_API_FORK_ONLY = true;
    config.mode = 2;
  } else if (ConfigOptimalAPI | optimal)
    config.mode = 1;
  else
    config.mode = 0;
  auto analyzeAPI = conflib::Get<bool>("openlib.analyzeAPI", false);
  config.explorePublicAPIs = config.mode | ConfigAnalyzeAPI |
                             analyzeAPI;  // optimal always explore public apis
  CONFIG_OPEN_API = config.explorePublicAPIs;
  auto printAPI = conflib::Get<bool>("openlib.printAPI", false);
  config.printAPI = ConfigPrintAPI | printAPI;
  auto apiLimit = conflib::Get<uint32_t>("openlib.limit", 999);
  config.apiLimit = apiLimit;
  auto onceOnly = conflib::Get<bool>("openlib.once", false);
  config.onceOnly = ConfigOpenLibOnce | onceOnly;

  return config;
}

void initRaceDetect() {
  NONE_PARALLEL_FUNCTIONS =
      conflib::Get<map<string, string>>("notParallelFunctionPairs", {});

  CONFIG_IGNORE_READ_WRITE_RACES =
      ConfigIgnoreReadWriteRaces |
      conflib::Get<bool>("ignoreReadWriteRaces", false);
  CONFIG_IGNORE_WRITE_WRITE_RACES =
      ConfigIgnoreWriteWriteRaces |
      conflib::Get<bool>("ignoreWriteWriteRaces", false);

  IGNORED_FUN_ALL =
      conflib::Get<std::vector<std::string>>("ignoreRacesInFunctions", {});
  IGNORED_VAR_ALL =
      conflib::Get<std::vector<std::string>>("ignoreRaceVariables", {});
  IGNORED_LOCATION_ALL =
      conflib::Get<std::vector<std::string>>("ignoreRacesAtLocations", {});
  LOW_PRIORITY_FILE_NAMES =
      conflib::Get<std::vector<std::string>>("lowPriorityFiles", {});
  HIGH_PRIORITY_FILE_NAMES =
      conflib::Get<std::vector<std::string>>("highPriorityFiles", {});
  LOW_PRIORITY_VAR_NAMES =
      conflib::Get<std::vector<std::string>>("lowPriorityRaces", {});
  HIGH_PRIORITY_VAR_NAMES =
      conflib::Get<std::vector<std::string>>("highPriorityRaces", {});

  auto reportLimit = conflib::Get<int>("raceLimit", -1);
  if (reportLimit != -1) {
    ConfigReportLimit = reportLimit;
  }
  DataRace::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  OrderViolation::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  DeadLock::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  MismatchedAPI::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  TOCTOU::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  UntrustfulAccount::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  UnSafeOperation::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
  CosplayAccounts::init(ConfigReportLimit, CONFIG_NO_REPORT_LIMIT);
}
unsigned int NUM_OF_FUNCTIONS = 0;
unsigned int NUM_OF_ATTACK_VECTORS = 0;
unsigned int NUM_OF_IR_LINES = 0;
unsigned int TOTAL_SOL_COST = 0;
unsigned int TOTAL_SOL_TIME = 0;
static std::unique_ptr<Module> loadFile(const std::string &FN,
                                        LLVMContext &Context,
                                        bool abortOnFail) {
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

static bool linkFiles(LLVMContext &Context, aser::Linker &L,
                      const std::vector<std::string> &Files, unsigned Flags) {
  // Filter out flags that don't apply to the first file we load.
  unsigned ApplicableFlags = Flags & aser::Linker::Flags::OverrideFromSrc;
  bool isOrignalModule = true;
  for (const auto &File : Files) {
    std::unique_ptr<Module> M = loadFile(File, Context, isOrignalModule);
    isOrignalModule = false;

    if (M.get() == nullptr) {
      LOG_ERROR("fail to load bc file, bc : {}", File);
      continue;
    }
    // Note that when ODR merging types cannot verify input files in here When
    // doing that debug metadata in the src module might already be pointing to
    // the destination.
    if (verifyModule(*M, &errs())) {
      LOG_ERROR("input module is broken, file: {}", File);
      continue;
    }

    bool Err = L.linkInModule(std::move(M), ApplicableFlags);

    if (Err) return false;

    // All linker flags apply to linking of subsequent files.
    ApplicableFlags = Flags;
  }

  return true;
}

set<const Function *> CR_UNExploredFunctions;
const std::set<llvm::StringRef> SKIPPED_APIS{
    "llvm.",  "pthread_", "__kmpc_",         "_mp_",
    ".omp.",  "omp_",     "__clang_",        "_ZNSt",
    "_ZSt",   "_ZNKSt",   "_ZN14coderrect_", "__coderrect_",
    "printf", "je_"};
void addExploredFunction(const Function *f) { CR_UNExploredFunctions.erase(f); }

// Method to compare two versions.
// Returns 1 if v2 is smaller, -1
// if v1 is smaller, 0 if equal
int versionCompare(string v1, string v2) {  // v1: the real version: ^0.20.1
  std::replace(v1.begin(), v1.end(), '*', '0');  // replace all '*' to '0'
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
    if (vnum1 > vnum2) return 1;
    if (vnum2 > vnum1) return -1;
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
          if (CS.getNumArgOperands() < 2) continue;
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
              if (valueName2.contains("1")) hasOverFlowChecks = true;
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
    if (fgets(buffer, 128, pipe) != NULL) result += buffer;
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
          if (CS.getNumArgOperands() < 1) continue;
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
              path = "/" + path;  // unix separator
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

void initUnexploredFunctions(Module *module) {
  auto &functionList = module->getFunctionList();
  for (auto &function : functionList) {
    auto func = &function;
    if (!func->isDeclaration()) {
      bool insert = true;
      // do not count llvm.* pthread_* __kmpc_* __coderrect_ .omp. omp_ std::
      for (auto str : SKIPPED_APIS) {
        if (func->getName().startswith(str)) {
          insert = false;
          break;
        }
      }
      if (insert) {
        CR_UNExploredFunctions.insert(func);
        FUNC_NAME_MAP[func->getName()] = func;
      }
    }
  }
  NUM_OF_FUNCTIONS = CR_UNExploredFunctions.size();
  openlib::computeCandidateAPIs(module, false);
}
void getAllUnexploredFunctionsbyPartialName(
    std::set<const llvm::Function *> &result, StringRef sig) {
  for (auto func : CR_UNExploredFunctions) {
    if (openlib::isInferredPublicAPI(func) && func->getName().contains(sig) &&
        !func->getName().contains(".anon.")) {
      result.insert(func);
    }
  }
}
const llvm::Function *getUnexploredFunctionbyPartialName(StringRef sig) {
  for (auto func : CR_UNExploredFunctions) {
    if (openlib::isInferredPublicAPI(func) && func->getName().contains(sig) &&
        !func->getName().contains(".anon.")) {
      return func;
    }
  }
  return nullptr;
}
void reportUnexploredFunctions() {
  int size = CR_UNExploredFunctions.size();
  if (size > 0) {
    llvm::outs() << "=== unexplored public apis ===\n";
    for (auto func : CR_UNExploredFunctions) {
      if (openlib::isInferredPublicAPI(func))
        llvm::outs() << openlib::getCleanFunctionName(func) << "\n";
    }
  }
  llvm::outs() << "\n============= unexplored functions ==========\n";
  float percent = size * 100.0 / NUM_OF_FUNCTIONS;
  llvm::outs() << "total number of functions: " << NUM_OF_FUNCTIONS
               << "\nunexplored functions: " << size << " ("
               << llvm::format("%.2f%%", percent) << ")\n\n";
}

int main(int argc, char **argv) {
  // llvm::outs() << sizeof(std::pair<NodeID, NodeID>) << ", " <<
  // sizeof(std::pair<void *, void *>); return 1;

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

  CONFIG_MUST_EXPLORE_APIS =
      conflib::Get<std::vector<std::string>>("mustExploreFunctions", {});
  CONFIG_INDIRECT_APIS =
      conflib::Get<std::vector<std::string>>("indirectFunctions", {});
  CRITICAL_INDIRECT_TARGETS =
      conflib::Get<map<string, string>>("criticalIndirectTargets", {});
  if (true) {  // for solana sve
               // initialize SVEIDMap
    std::set<std::string> freeSVEs;
    SVE::addTypeID("1001", SVE::Type::MISS_SIGNER);
    freeSVEs.insert("1001");
    SVE::addTypeID("1002", SVE::Type::MISS_OWNER);
    freeSVEs.insert("1002");
    SVE::addTypeID("1003", SVE::Type::OVERFLOW_ADD);
    freeSVEs.insert("1003");
    SVE::addTypeID("1004", SVE::Type::OVERFLOW_SUB);
    freeSVEs.insert("1004");
    SVE::addTypeID("1005", SVE::Type::OVERFLOW_MUL);
    freeSVEs.insert("1005");
    SVE::addTypeID("1006", SVE::Type::OVERFLOW_DIV);
    freeSVEs.insert("1006");
    SVE::addTypeID("1007", SVE::Type::ACCOUNT_UNVALIDATED_BORROWED);
    SVE::addTypeID("1008", SVE::Type::ACCOUNT_DUPLICATE);
    SVE::addTypeID("1009", SVE::Type::ACCOUNT_CLOSE);
    SVE::addTypeID("1010", SVE::Type::COSPLAY_FULL);
    freeSVEs.insert("1010");
    SVE::addTypeID("1011", SVE::Type::COSPLAY_PARTIAL);
    SVE::addTypeID("1012", SVE::Type::DIV_BY_ZERO);
    freeSVEs.insert("1012");
    SVE::addTypeID("1013", SVE::Type::REINIT);
    SVE::addTypeID("1014", SVE::Type::BUMP_SEED);
    // freeSVEs.insert("1014");
    SVE::addTypeID("1015", SVE::Type::INSECURE_PDA_SHARING);
    SVE::addTypeID("1016", SVE::Type::ARBITRARY_CPI);
    freeSVEs.insert("1016");
    SVE::addTypeID("1017", SVE::Type::MALICIOUS_SIMULATION);
    SVE::addTypeID("1018", SVE::Type::UNSAFE_SYSVAR_API);
    freeSVEs.insert("1018");
    SVE::addTypeID("1019", SVE::Type::ACCOUNT_UNVALIDATED_OTHER);
    SVE::addTypeID("1020", SVE::Type::OUTDATED_DEPENDENCY);
    SVE::addTypeID("1021", SVE::Type::UNSAFE_RUST);
    SVE::addTypeID("1022", SVE::Type::OVERPAY);
    SVE::addTypeID("1023", SVE::Type::STALE_PRICE_FEED);
    SVE::addTypeID("1024", SVE::Type::MISS_INIT_TOKEN_MINT);
    SVE::addTypeID("1025", SVE::Type::MISS_RENT_EXEMPT);
    SVE::addTypeID("1026", SVE::Type::MISS_FREEZE_AUTHORITY);
    SVE::addTypeID("1027", SVE::Type::FLASHLOAN_RISK);
    SVE::addTypeID("1028", SVE::Type::BIDIRECTIONAL_ROUNDING);
    SVE::addTypeID("1029", SVE::Type::CAST_TRUNCATE);
    SVE::addTypeID("1030", SVE::Type::ACCOUNT_UNVALIDATED_PDA);
    SVE::addTypeID("1031", SVE::Type::ACCOUNT_UNVALIDATED_DESTINATION);
    SVE::addTypeID("1032", SVE::Type::ACCOUNT_INCORRECT_AUTHORITY);
    SVE::addTypeID("1033", SVE::Type::INSECURE_INIT_IF_NEEDED);
    freeSVEs.insert("1033");
    SVE::addTypeID("1034", SVE::Type::INSECURE_SPL_TOKEN_CPI);
    SVE::addTypeID("1035", SVE::Type::INSECURE_ASSOCIATED_TOKEN);
    SVE::addTypeID("1036", SVE::Type::INSECURE_ACCOUNT_REALLOC);
    freeSVEs.insert("1036");
    SVE::addTypeID("1037", SVE::Type::PDA_SEEDS_COLLISIONS);
    SVE::addTypeID("2001", SVE::Type::INCORRECT_BREAK_LOGIC);
    SVE::addTypeID("2002", SVE::Type::INCORRECT_CONDITION_CHECK);
    SVE::addTypeID("2003", SVE::Type::EXPONENTIAL_CALCULATION);
    SVE::addTypeID("2004", SVE::Type::INCORRECT_DIVISION_LOGIC);
    SVE::addTypeID("2005", SVE::Type::INCORRECT_TOKEN_CALCULATION);
    SVE::addTypeID("3002", SVE::Type::CRITICAL_REDUNDANT_CODE);
    SVE::addTypeID("3005", SVE::Type::MISS_CPI_RELOAD);
    SVE::addTypeID("3006", SVE::Type::MISS_ACCESS_CONTROL_UNSTAKE);
    SVE::addTypeID("3007", SVE::Type::ORDER_RACE_CONDITION);
    SVE::addTypeID("3008", SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_ADD);
    freeSVEs.insert("3008");
    SVE::addTypeID("3009", SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_MUT);
    freeSVEs.insert("3009");
    SVE::addTypeID("3010", SVE::Type::ACCOUNT_IDL_INCOMPATIBLE_ORDER);
    freeSVEs.insert("3010");
    SVE::addTypeID("10001", SVE::Type::REENTRANCY_ETHER);
    SVE::addTypeID("10002", SVE::Type::ARBITRARY_SEND_ERC20);
    SVE::addTypeID("10003", SVE::Type::SUISIDE_SELFDESTRUCT);
    SVE::addTypeID("20001", SVE::Type::MISS_INIT_UNIQUE_ADMIN_CHECK);
    SVE::addTypeID("20002", SVE::Type::BIT_SHIFT_OVERFLOW);
    SVE::addTypeID("20003", SVE::Type::DIV_PRECISION_LOSS);
    SVE::addTypeID("20004", SVE::Type::VULNERABLE_SIGNED_INTEGER_I128);
    bool isFreePlan = (SOTERIA_PLAN == "free");
    SOLANA_SVE_DB =
        conflib::Get<std::map<std::string, std::map<std::string, std::string>>>(
            "solana.sve", {});
    for (auto &[key, valueMap] : SOLANA_SVE_DB) {
      auto name = valueMap["name"];
      auto description = valueMap["description"];
      auto url = valueMap["url"];
      auto on = valueMap["on"];
      auto free = valueMap["free"];
      if (on == "false" ||
          (isFreePlan &&
           (free != "true" && freeSVEs.find(key) == freeSVEs.end())))
        SVE::addDisabledChecker(key);
      else {
        if (DEBUG_RUST_API)
          llvm::outs() << "ID: " << key << " \n    name: " << name
                       << "\n    description: " << description
                       << "\n    url: " << url << "\n";
      }
    }
    // llvm::outs() << "SOTERIA_PLAN: " << SOTERIA_PLAN << "\n";
  }
  DEBUG_FOCUS_VEC =
      ConfigDebugDebugDebug;  // conflib::Get<std::vector<std::string>>("debugdebugdebug",
                              // {});
  if (DEBUG_FOCUS_VEC.size() > 0) {
    llvm::outs() << "debug focus mode (size " << DEBUG_FOCUS_VEC.size() << "):";
    for (auto elem : DEBUG_FOCUS_VEC) llvm::outs() << " " << elem;
    llvm::outs() << "\n";
  }
  MAX_CALLSTACK_DEPTH = conflib::Get<int>("maxCallStackDepth", -1);
  SAME_FUNC_BUDGET_SIZE = conflib::Get<int>("sameFunctionBudget", 10);
  FUNC_COUNT_BUDGET = conflib::Get<int>("functionCountBudget", 20000);

  GraphBLASHeapModel::init(heapAPIs);
  // PTA will skip looking into those functions
  auto skipFun = conflib::Get<std::vector<std::string>>("skipFunctions", {});
  ExtFunctionsManager::init(skipFun);

  auto enableOMP = conflib::Get<bool>("enableOpenMP", true);
  auto enableAV = conflib::Get<bool>("enableAtomicityViolation", true);
  auto enableOV = conflib::Get<bool>("enableOrderViolation", true);
  auto enableMismatchedAPI = conflib::Get<bool>("enableMismatchedAPI", true);
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
  auto exploreAllLockPaths = conflib::Get<bool>("exploreAllLockPaths", false);

  auto entryPointOnce = conflib::Get<bool>("entryPoint.once", false);

  CONFIG_CHECK_UncheckedAccount =
      conflib::Get<bool>("solana.account.UncheckedAccount", true);

  CONFIG_NO_OMP = ConfigNoOMP | !enableOMP;
  CONFIG_NO_AV = ConfigNoAV | !enableAV;
  CONFIG_NO_OV = ConfigNoOV | !enableOV;
  CONFIG_NO_MISSED_OMP = ConfigNoMissedOMP;  // TODO: add conflib check as well
  CONFIG_NO_MISS_MATCH_API = ConfigNoMissAPI | !enableMismatchedAPI;
  CONFIG_CHECK_IDENTICAL_WRITE =
      ConfigCheckIdenticalWrites | enableIdenticalWrites;

  CONFIG_INCLUDE_ATOMIC =
      ConfigIncludeAtomic | conflib::Get<bool>("includeAtomics", false);
  CONFIG_NO_FILTER = ConfigNoFilter | !enableFilter;
  CONFIG_NO_REPORT_LIMIT =
      ConfigNoReportLimit | (conflib::Get<int>("raceLimit", -1) < 0);
  CONFIG_IGNORE_LOCK = ConfigIngoreLock | !enableLockSet;
  CONFIG_LOOP_UNROLL = ConfigLoopUnroll | enableLoopUnroll;
  CONFIG_NO_FALSE_ALIAS = ConfigNoFalseAlias;
  CONFIG_NO_PRODUCER_CONSUMER = ConfigNoProducerConsumer;

  CONFIG_ENTRY_POINT_SINGLE_TIME = ConfigEntryPointOnce | entryPointOnce;

  DEBUG_RACE = ConfigDebugRace;
  DEBUG_THREAD = ConfigDebugThread;
  DEBUG_OMP_RACE = ConfigDebugOMPRace;
  DEBUG_CALL_STACK = ConfigDebugCS | ConfigShowCallStack;
  DEBUG_LOCK_STACK = ConfigShowLockStack;
  DEBUG_DUMP_LOCKSET = ConfigDumpLockSet;
  DEBUG_INDIRECT_CALL = ConfigDebugIndirectCall || ConfigDebugIndirectCallAll;
  DEBUG_INDIRECT_CALL_ALL = ConfigDebugIndirectCallAll;
  DEBUG_LOCK = ConfigDebugLock;
  DEBUG_LOCK_STR = ConfigDebugLockStr;
  DEBUG_RUST_API = ConfigDebugRustAPI;
  DEBUG_API = ConfigDebugAPI;
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
                        ConfigShowAllTerminal;  // show race summary
  CONFIG_SHOW_DETAIL = ConfigShowDetail | enableShowRaceDetail |
                       ConfigShowAllTerminal;  // show race details

  CONFIG_EXPLORE_ALL_LOCK_PATHS = ConfigKeepLockPath | exploreAllLockPaths;
  CONFIG_INTEGRATE_DYNAMIC = ConfigIntegrateDynamic;
  // by default, set the pts size to 999
  PTSTrait<PtsTy>::setPTSSizeLimit(9);  // set pts size limit to 999
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
  // TODO: this is a temporary fix for excluding `main` function
  // if later we support customized entries, we need to also change this
  for (auto f : skipFun) {
    llvm::Regex pat(f);
    if (pat.match("main")) {
      error(
          "[ABORT] The main function will be excluded by the "
          "\"ignoreFunctions\" in the config file.");
      error("Please check your config file.");
      return 1;
    }
  }

  initRaceDetect();

  LOG_INFO("Loading IR From File: {}", TargetModulePath);
  logger::newPhaseSpinner("Loading IR From File");

  LLVMContext Context;
  // load it now as we need the triple information to determine which stub to
  // link
  auto module = loadFile(TargetModulePath, Context, true);
  module->setModuleIdentifier("coderrect.stub.pid" + std::to_string(getpid()));

  // add -Xuse-fi-model-- this needs to be added to PTA configure before
  // anything else
  if (!module->getFunction("main") && module->getFunction("MAIN_")) {
    CONFIG_USE_FI_MODE =
        true;  // for now fortran, use field-insensitive model by default
    FORTRAN_IR_MODE = true;
  }

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
    // us
    rewriteUserSpecifiedAPI(module.get());
  }

  if (!SkipPreProcess && !DebugIR) {
    // Preprocessing the IR
    IRPreProcessor preProcessor;
    preProcessor.runOnModule(
        *module, [&](llvm::legacy::PassManagerBase &MPM) -> void {
          LMT::addPreProcessingPass(MPM);  // run before inline
        });
  }

  if (!DebugIR) {
    // TODO: probably make main configurable through conflib?
    if (USE_FAKE_MAIN) {
      auto openLibConfig = initOpenLibConf(module.get());
      openlib::createFakeMain(openLibConfig);
    } else {
      llvm::Function *entryFun = module->getFunction("main");
      if (entryFun == nullptr || entryFun->isDeclaration()) {
        llvm::errs() << "Fatal Error: 'main' function cannot be found!\n";
        return 1;
      }
    }
  }

  if (!SkipPreProcess && !DebugIR) {
    llvm::legacy::PassManager passes;
    // for field-sensitive pointer analysis
    // 1. transform getelementptr
    // 2. delete inline asm instruction (change it to Undef Value)
    passes.add(new InsertGlobalCtorCallPass());
    // passes.add(new LoweringMemCpyPass());

    // for race detector
    passes.add(new RemoveASMInstPass());
    passes.add(new RemoveExceptionHandlerPass());
    if (CONFIG_LOOP_UNROLL) {
      // peel once for all 1-level loops containing thread creation
      passes.add(new UnrollThreadCreateLoopPass());
    }
    // has to run after OMP Constant Propagation
    passes.add(new CanonicalizeGEPPass());
    // has to run after gep expansion

    passes.run(*module);

    LOG_INFO("Running Compiler Optimization Passes (Phase II)");
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
  analysisPasses.add(new DominatorTreeWrapperPass());
  analysisPasses.add(new LoopInfoWrapperPass());

  // analysisPasses.add(new BradPass());
  // analysisPasses.add(new ConstructModelPass<LangModel>());
  analysisPasses.add(new PointerAnalysisPass<PTA>());
  analysisPasses.add(new RaceDetectionPass());

  initUnexploredFunctions(module.get());
  computeCargoTomlConfig(module.get());
  computeDeclareIdAddresses(module.get());
  analysisPasses.run(*module);

  if (ConfigPrintCoverage) reportUnexploredFunctions();

  return 0;
}

static llvm::RegisterPass<PointerAnalysisPass<PTA>> PAP(
    "Pointer Analysis Wrapper Pass", "Pointer Analysis Wrapper Pass", true,
    true);
