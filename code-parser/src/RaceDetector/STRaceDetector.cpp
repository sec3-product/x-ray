#include <dirent.h>

#include <any>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h>  // for llvm LLVMContext
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>  // IR reader for bit file
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/Signals.h>    // signal for command line
#include <llvm/Support/SourceMgr.h>  // for SMDiagnostic
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <conflib/conflib.h>
#include <o2/Util/Log.h>
#include <antlr4-runtime.h>
#include <RustLexer.h>
#include <RustParser.h>
#include <toml.hpp>

#include "st/MLIRGen.h"
#include "st/ParserWrapper.h"
#include "st/Passes.h"
#include "st/STParserListener.h"
#include "st/STParserVisitor.h"
#include "ThreadPool.h"

using namespace antlr4;
using namespace antlrcpp;
using namespace llvm;
using namespace o2;
using namespace st;

std::vector<stx::FunctionAST *> processResults;
uint numOfFunctions = 0;

cl::opt<std::string> TargetModulePath(cl::Positional,
                                      cl::desc("path to input bitcode file"));
cl::opt<bool> ConfigDumpIR("dump-ir", cl::desc("Dump the generated IR file"));
cl::opt<bool> ConfigDebugSol("d", cl::desc("single-threaded parser for debug"));
cl::opt<std::string> ConfigOutputFile("o", cl::desc("IR output file name"),
                                      cl::init("t.ll"));
cl::opt<int> NUM_LOW_BOUND(
    "lb", cl::desc("set lower bound of the number of functions"),
    cl::value_desc("number"), cl::init(0));
int LOWER_BOUND_ID = 0;
cl::opt<int> NUM_UP_BOUND(
    "ub", cl::desc("set upper bound of the number of functions"),
    cl::value_desc("number"), cl::init(1000000));

logger::LoggingConfig initLoggingConf() {
  logger::LoggingConfig config;

  config.enableProgress = conflib::Get<bool>("XenableProgress", true);
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

  return config;
}

static std::unique_ptr<Module> loadFile(const std::string &FN,
                                        LLVMContext &Context,
                                        bool abortOnFail) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Result;
  Result = parseIRFile(FN, Err, Context);

  if (!Result) {
    if (abortOnFail) {
      Err.print("racedetect", llvm::errs());
      abort();
    }

    LOG_ERROR("error loading file: {}", FN);
    return nullptr;
  }
  return Result;
}

std::set<const Function *> unExploredFunctions;
const std::set<llvm::StringRef> SKIPPED_APIS{
    "llvm.", "pthread_", "__kmpc_",  ".omp.", "omp_",   "_ZNSt",
    "_ZSt",  "_ZNKSt",   "_ZN14o2_", "__o2_", "printf", "je_"};
void addExploredFunction(const Function *f) { unExploredFunctions.erase(f); }
void initUnexploredFunctions(Module *module) {
  auto &functionList = module->getFunctionList();
  for (auto &function : functionList) {
    auto func = &function;
    if (!func->isDeclaration()) {
      bool insert = true;
      // do not count llvm.* pthread_* __kmpc_* __smalltalk_ .omp. omp_ std::
      for (auto str : SKIPPED_APIS) {
        if (func->getName().startswith(str)) {
          insert = false;
          break;
        }
      }
      if (insert) unExploredFunctions.insert(func);
    }
  }
}

void computeUnexploredFunctions() {
  if (unExploredFunctions.size() > 0) {
    llvm::outs() << "\n============= unexplored function ==========\n";
    for (auto func : unExploredFunctions)
      llvm::outs() << (func->getName()) << "\n";
    // TODO: print potential entry points
    // no caller
  }
}

void initRaceDetect() {
  LOG_INFO("Loading IR From File: {}", TargetModulePath);
  auto module = loadFile(TargetModulePath, *(new LLVMContext()), true);
  // ThreadProfileRewriter::rewriteModule(module.get());

  // Initialize passes
  // PassRegistry &Registry = *PassRegistry::getPassRegistry();

  // initializeCore(Registry);
  // initializeScalarOpts(Registry);
  // initializeIPO(Registry);
  // initializeAnalysis(Registry);
  // initializeTransformUtils(Registry);
  // initializeInstCombine(Registry);
  // initializeAggressiveInstCombine(Registry);
  // initializeInstrumentation(Registry);
  // initializeTarget(Registry);

  llvm::legacy::PassManager analysisPasses;
  // analysisPasses.add(new RaceDetectionPass());
  initUnexploredFunctions(module.get());
  analysisPasses.run(*module);
  computeUnexploredFunctions();
}
// using namespace antlrcpp;

st::ModuleAST moduleAST;

void process(std::string filename, std::string methodname, size_t line,
             std::string s2) {
  std::replace(s2.begin(), s2.end(), '\t', ' ');
  auto *input = new ANTLRInputStream(s2);

  auto *lexer = new RustLexer(input);
  auto *tokens = new CommonTokenStream(lexer);

  tokens->fill();

  auto *parser = new RustParser(tokens);

  // auto *tree = parser->module();
  // llvm::outs() << tree->toStringTree(parser) << "\n";
  // llvm::outs() << "-----------"
  //              << "\n";
  // // TODO: add listener
  // auto *listener = new STParserListener(filename, methodname, line);
  // antlr4::tree::ParseTreeWalker::DEFAULT.walk(listener, tree);
  // auto data = listener->getData();
  // moduleAST.addModule(std::move(data));

  STParserVisitor visitor(filename, methodname, line);
  // std::vector<stx::FunctionAST *> res = visitor.visitCrate(parser->crate());

  std::any any_res = visitor.visitCrate(parser->crate()); 
  std::vector<stx::FunctionAST*> res = std::any_cast<std::vector<stx::FunctionAST*>>(any_res);

  processResults.insert(processResults.end(), res.begin(), res.end());
}

// c++17 only
// using std::filesystem::directory_iterator;
void initParserForFile(stx::ModuleAST *module, std::string &fullPathName,
                       stx::ThreadPool &pool) {
  // for (const auto & entry : directory_iterator(path)){
  //     StringRef name(entry.path());
  llvm::outs() << "  source file: " << fullPathName << "\n";

  StringRef name(fullPathName);
  std::ifstream ws(name.str());
  std::string contents((std::istreambuf_iterator<char>(ws)),
                       std::istreambuf_iterator<char>());

  if (DEBUG_SOL) {
    if (name.endswith(".toml")) {
      llvm::outs() << "=== Test WS Start ===\n";
      llvm::outs() << contents << "\n";
      llvm::outs() << "=== Test WS End ===\n";
    } else if (name.endswith(".rs")) {
      llvm::outs() << "=== Rust Source Start ===\n";
      llvm::outs() << contents << "\n";
      llvm::outs() << "=== Rust Source End ===\n";
    }
  }
  // processResults.emplace_back(
  //                     pool.enqueue([fullPathName, methodName, methodpt] {
  //                       stx::FunctionAST *funcAst = process(
  //                           fullPathName, methodName,
  //                           xmlGetLineNo(methodpt) - 1,
  //                           (char
  //                           *)XML_GET_CONTENT(methodpt->xmlChildrenNode));

  //                       return funcAst;
  //                     }));
  auto found1 = fullPathName.find_last_of("/");
  auto found2 = fullPathName.find_last_of(".");
  auto fnBaseName = fullPathName.substr(found1 + 1, found2 - found1 - 1);
  process(fullPathName, fnBaseName, 0, contents);
}

bool handleRustFile(stx::ModuleAST *module, std::string &fullPathName,
                    stx::ThreadPool &pool) {
  // is it a file?
  FILE *fptr = fopen(fullPathName.c_str(), "r");
  if (fptr == NULL) {
    if (DEBUG_SOL)
      llvm::errs() << "input file does not exist: " << fullPathName << "\n";
    return false;
  }
  // File exists hence close file
  fclose(fptr);
  initParserForFile(module, fullPathName, pool);

  return true;
}

bool handleDiretory(DIR *dir, stx::ModuleAST *module, std::string &fullPathName,
                    stx::ThreadPool &pool) {
  StringRef pathname_stringref(fullPathName);
  if (pathname_stringref.endswith("/src")) {
    // ends with /src
    llvm::outs() << "source path: " << fullPathName << "\n";
  } else if (fullPathName == module->path)
    llvm::outs() << "path: " << fullPathName << "\n";
  else {
    llvm::outs() << "subpath: " << fullPathName << "\n";
    // if subpath contains Cargo.toml, skip
    struct dirent *entry;
    DIR *dir2 = opendir(fullPathName.c_str());
    while ((entry = readdir(dir2)) != NULL) {
      StringRef dname(entry->d_name);
      // llvm::outs() << "dname: " << dname << "\n";
      if (dname.equals("Xargo.toml")) return true;
    }
    closedir(dir2);
  }
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    StringRef dname(entry->d_name);
    if (DEBUG_SOL) llvm::outs() << "dname: " << dname << "\n";
    // std::transform(dname.begin(), dname.end(), dname.begin(), ::tolower);
    // llvm::outs() << "fullPathName: " << fullPathName << "\n";
    std::string fullPathName2 =
        fullPathName + "/" + entry->d_name;  // realpath(dname.data(), NULL);
    // llvm::outs() << "fullPathName2: " << fullPathName2 << "\n";

    DIR *dir2 = opendir(fullPathName2.c_str());
    if (dir2 == NULL) {
      if (dname.endswith(".rs"))
        handleRustFile(module, fullPathName2, pool);
      else if (dname.endswith("argo.toml")) {
        module->path_config = fullPathName2;
        auto data = toml::parse(fullPathName2);
        if (data.contains("profile") &&
            data.at("profile").count("release") != 0) {
          const toml::value answer1 =
              toml::get<toml::table>(data).at("profile");
          const toml::value answer2 =
              toml::get<toml::table>(answer1).at("release");
          if (answer2.count("overflow-checks")) {
            // auto line = answer2.location().line();
            // auto column = answer2.location().column();
            const auto hasOverFlowCheck =
                toml::find<bool>(data, "profile", "release", "overflow-checks");
            llvm::outs() << dname << ": hasOverFlowCheck: " << hasOverFlowCheck
                         << "\n";
            //<< " line: " << line << " column: " << column << "\n";
            if (hasOverFlowCheck)
              module->configMap["profile.release.overflow-checks"] = "1";

            // const toml::value overFlowCheck =
            //     toml::get<toml::table>(answer2).at("overflow-checks");
            // auto hasOverFlowCheck =
            //     toml::get_or(overFlowCheck, false);  // this works
          }
        }

        if (data.contains("dependencies")) {
          // The following version values have two valid forms:
          //
          //   - simple string
          // anchor-lang = "0.18.2"
          //
          //   - detailed table format
          // anchor-lang = { version = "0.18.2", features = ["derive"] }
          //
          if (data.at("dependencies").count("spl-token") != 0) {
            const auto spl_token =
                toml::find(data, "dependencies", "spl-token");
            std::string version = "";
            if (spl_token.is_string()) {
              version = toml::get_or(spl_token, "");
            } else if (spl_token.is_table()) {
              if (spl_token.contains("version")) {
                version = toml::get_or(spl_token.at("version"), "");
              }
            }
            // auto features =
            //     toml::get_or(spl_token.at("default-features"), false);
            // llvm::outs() << dname << ": spl-token version: " << version
            //              << " default-features: " << features << "\n";
            if (!version.empty()) {
              module->configMap["dependencies.spl-token.version"] = version;
            }
          }

          if (data.at("dependencies").count("anchor-lang") != 0) {
            const auto anchor_lang =
                toml::find(data, "dependencies", "anchor-lang");
            std::string version = "";
            if (anchor_lang.is_string()) {
              version = toml::get_or(anchor_lang, "");
            } else if (anchor_lang.is_table()) {
              if (anchor_lang.contains("version")) {
                version = toml::get_or(anchor_lang.at("version"), "");
              }
            }
            if (!version.empty()) {
              module->configMap["dependencies.anchor-lang.version"] = version;
            }
          }

          if (data.at("dependencies").count("anchor-spl") != 0) {
            const auto anchor_spl =
                toml::find(data, "dependencies", "anchor-spl");
            std::string version = "";
            if (anchor_spl.is_string()) {
              version = toml::get_or(anchor_spl, "");
            } else if (anchor_spl.is_table()) {
              if (anchor_spl.contains("version")) {
                version = toml::get_or(anchor_spl.at("version"), "");
              }
            }
            if (!version.empty()) {
              module->configMap["dependencies.anchor-spl.version"] = version;
            }
          }
          if (data.at("dependencies").count("solana-program") != 0) {
            const auto solana_program =
                toml::find(data, "dependencies", "solana-program");
            std::string version = "";
            if (solana_program.is_string()) {
              version = toml::get_or(solana_program, "");
            } else if (solana_program.is_table()) {
              if (solana_program.contains("version")) {
                version = toml::get_or(solana_program.at("version"), "");
              }
            }
            if (!version.empty()) {
              module->configMap["dependencies.solana_program.version"] = version;
            }
          }
        }
      }
    } else if (!dname.startswith(".") && !dname.equals("tests") &&
               !dname.equals("js") && !dname.equals("cli") &&
               !dname.equals("logs") && !dname.equals("target") &&
               !dname.equals("debug") && !dname.equals("migrations") &&
               !dname.equals("services") &&
               !dname.equals("proptest-regressions")) {
      // nested diretory
      handleDiretory(dir2, module, fullPathName2, pool);
      closedir(dir2);
    }
  }
  return true;
}

bool initParser(stx::ModuleAST *module) {
  int COUNT = std::thread::hardware_concurrency();
  if (DEBUG_SOL) COUNT = 1;  // make sure no races
  stx::ThreadPool pool(COUNT);
  LOWER_BOUND_ID = NUM_LOW_BOUND;
  StringRef path(TargetModulePath);
  std::string fullPathName = realpath(path.data(), NULL);
  module->path = fullPathName;

  DIR *dir = opendir(path.data());
  if (dir == NULL) {
    if (path.endswith(".rs")) handleRustFile(module, fullPathName, pool);
  } else {
    handleDiretory(dir, module, fullPathName, pool);
    closedir(dir);
  }
  for (auto functionAst : processResults) {
    module->addFunctionAST(functionAst);
  }
  return true;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    if (DEBUG_SOL) llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  //   auto optPipeline = mlir::makeOptimizingTransformer(
  //       /*optLevel=*/ 0, /*sizeLevel=*/0,
  //       /*targetMachine=*/nullptr);
  //   if (auto err = optPipeline(llvmModule.get())) {
  //     llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
  //     return -1;
  //   }
  if (ConfigDumpIR || ConfigOutputFile != "t.ll") {
    std::error_code err;
    llvm::raw_fd_ostream outfile(ConfigOutputFile, err, llvm::sys::fs::OF_None);
    if (err) {
      if (DEBUG_SOL) llvm::errs() << "Error dumping IR!\n";
    }

    llvmModule->print(outfile, nullptr);
    outfile.close();
    std::string fullPathName = realpath(ConfigOutputFile.c_str(), NULL);
    llvm::outs() << "IR file: " << fullPathName << "\n";
  }

  if (DEBUG_SOL) llvm::errs() << *llvmModule << "\n";
  return 0;
}

void initLLVMIR(stx::ModuleAST *moduleAST) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  //   //from parse tree to moduleAST
  //   //TODO: add parseModule
  //   ParserWrapper wrapper(parser);
  //   std::unique_ptr<st::ModuleAST> moduleAST = wrapper.parseModule();
  // module = mlirGen(context, moduleAST);

  module = stx::mlirGenFull(context, *moduleAST);

  // if (auto temp =
  // module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("anonymousFun1")){
  //                 llvm::outs() << "try to remove redefinition of symbol
  //                 anonymousFun1\n";
  // }

  // if (false)
  {
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the
    // pipeline.
    applyPassManagerCLOptions(pm);

    // Partially lower the toy dialect with a few cleanups afterwards.
    // pm.addPass(mlir::st::createLowerToAffinePass());

    // mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    // optPM.addPass(mlir::createCanonicalizerPass());
    // optPM.addPass(mlir::createCSEPass());

    // Finish lowering the toy IR to the LLVM dialect.
    pm.addPass(mlir::st::createLowerToLLVMPass());
    if (mlir::failed(pm.run(*module))) {
      if (DEBUG_SOL) llvm::errs() << "Errors in createLowerToLLVMPass.\n";
    }
  }

  dumpLLVMIR(*module);
}

int testThreadPool() {
  stx::ThreadPool pool(4);
  std::vector<std::future<int>> results;

  for (int i = 0; i < 8; ++i) {
    results.emplace_back(pool.enqueue([i] {
      std::cout << "hello " << i << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "world " << i << std::endl;
      return i * i;
    }));
  }

  for (auto &&result : results) std::cout << result.get() << ' ';
  std::cout << std::endl;

  return 0;
}

int main(int argc, char **argv) {
  // testThreadPool();
  // llvm::outs() << sizeof(std::pair<NodeID, NodeID>) << ", " <<
  // sizeof(std::pair<void *, void *>); return 1;

  // InitLLVM will setup signal handler to print stack trace when the program
  // crashes.
  InitLLVM x(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);
  DEBUG_SOL = ConfigDebugSol;
  // We don't use args in conflib for now
  std::map<std::string, std::string> args;
  conflib::Initialize(args, true);

  auto logConfig = initLoggingConf();
  logger::init(logConfig);

  stx::ModuleAST *module = new stx::ModuleAST();
  bool success = initParser(module);
  if (success) initLLVMIR(module);
  // initRaceDetect();
  return 0;
}
