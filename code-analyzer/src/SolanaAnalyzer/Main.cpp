#include <sys/types.h>
#include <unistd.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <PointerAnalysis/PointerAnalysisPass.h>
#include <Util/Log.h>
#include <conflib/conflib.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h> // for llvm LLVMContext
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h> // IR reader for bit file
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Signals.h>   // signal for command line
#include <llvm/Support/SourceMgr.h> // for SMDiagnostic

#include "AccountIDL.h"
#include "DebugFlags.h"
#include "PTAModels/GraphBLASModel.h"
#include "RustAPIRewriter.h"
#include "SVE.h"
#include "SolanaAnalysisPass.h"

using namespace llvm;
using namespace xray;

cl::opt<std::string> TargetModulePath(cl::Positional,
                                      cl::desc("path to input bitcode file"));
cl::opt<bool> ConfigDumpIR("dump-ir", cl::desc("Dump the modified ir file"));
cl::opt<std::string> ConfigOutputPath("o", cl::desc("JSON output path"),
                                      cl::value_desc("path"));
cl::opt<bool> ConfigDebugLog("v", cl::desc("Turn off log to file"));

cl::opt<int> ConfigReportLimit("limit",
                               cl::desc("Max number of races can be reported"),
                               cl::value_desc("number"), cl::init(-1));

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

cl::opt<bool> ConfigDebugRustAPI("debug-sol",
                                 cl::desc("Turn on debug sol api"));

cl::opt<bool>
    ConfigDisableProgress("no-progress",
                          cl::desc("Does not print spinner progress"));

static logger::LoggingConfig initLoggingConf() {
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

static std::unique_ptr<Module>
loadFile(const std::string &FN, LLVMContext &Context, bool abortOnFail) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Result = parseIRFile(FN, Err, Context);
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
  GraphBLASHeapModel::init(heapAPIs);

  auto enableImmediatePrint =
      conflib::Get<bool>("report.enableTerminal", false);
  auto enableShowRaceSummary =
      conflib::Get<bool>("enablePrintRaceSummary", false);
  auto enableShowRaceDetail =
      conflib::Get<bool>("enablePrintRaceDetail", false);

  FUNC_COUNT_BUDGET = conflib::Get<int>("functionCountBudget", 20000);
  CONFIG_CHECK_UncheckedAccount =
      conflib::Get<bool>("solana.account.UncheckedAccount", true);

  auto reportLimit = conflib::Get<int>("raceLimit", -1);
  if (reportLimit != -1) {
    ConfigReportLimit = reportLimit;
  }

  CONFIG_OUTPUT_PATH = ConfigOutputPath;
  TARGET_MODULE_PATH = TargetModulePath;

  DEBUG_RUST_API = ConfigDebugRustAPI;

  PRINT_IMMEDIATELY =
      ConfigPrintImmediately | enableImmediatePrint | ConfigShowAllTerminal;
  TERMINATE_IMMEDIATELY = ConfigTerminateImmediately;
  CONFIG_SHOW_SUMMARY = ConfigShowSummary | enableShowRaceSummary |
                        ConfigShowAllTerminal; // show race summary
  CONFIG_SHOW_DETAIL = ConfigShowDetail | enableShowRaceDetail |
                       ConfigShowAllTerminal; // show race details

  PTSTrait<PtsTy>::setPTSSizeLimit(9);

  LOG_INFO("Loading IR From File: {}", TargetModulePath);
  logger::newPhaseSpinner("Loading IR From File");

  LLVMContext Context;
  auto module = loadFile(TargetModulePath, Context, true);
  module->setModuleIdentifier("coderrect.stub.pid" + std::to_string(getpid()));

  LOG_INFO("Running Transformation Passes");
  logger::newPhaseSpinner("Running Transformation Passes");

  // Initialize passes, which are required by later passes.
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

  RustAPIRewriter::rewriteModule(module.get());
  createFakeMain(module.get());

  // Dump updated IR to file if requested.
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
  registerPointerAnalysisPass<PTA>();
  analysisPasses.add(new PointerAnalysisPass<PTA>());

  auto analyzer = new SolanaAnalysisPass();
  auto sves = conflib::Get<SVE::Database>("solana.sve", {});
  analyzer->initialize(sves, ConfigReportLimit);

  analysisPasses.add(analyzer);

  computeCargoTomlConfig(module.get());
  computeDeclareIdAddresses(module.get());
  analysisPasses.run(*module);

  return 0;
}
