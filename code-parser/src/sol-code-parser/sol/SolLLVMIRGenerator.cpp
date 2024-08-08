#include "sol/SolLLVMIRGenerator.h"

#include <any>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LLVMContext.h> // for llvm LLVMContext
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h> // IR reader for bit file
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <RustLexer.h>
#include <RustParser.h>
#include <antlr4-runtime.h>
#include <toml.hpp>

#include "sol/MLIRGen.h"
#include "sol/Passes.h"
#include "sol/SolParserVisitor.h"

using namespace antlr4;
using namespace antlrcpp;
using namespace llvm;

uint numOfFunctions = 0;

// CLI options.
cl::opt<std::string> TargetModulePath(cl::Positional,
                                      cl::desc("path to input bitcode file"));
cl::opt<bool> ConfigDumpIR("dump-ir", cl::desc("Dump the generated IR file"));
cl::opt<bool> ConfigDebugSol("d", cl::desc("single-threaded parser for debug"));
cl::opt<std::string> ConfigOutputFile("o", cl::desc("IR output file name"),
                                      cl::init("t.ll"));

cl::opt<int>
    NUM_LOW_BOUND("lb", cl::desc("set lower bound of the number of functions"),
                  cl::value_desc("number"), cl::init(0));
int LOWER_BOUND_ID = 0;
cl::opt<int>
    NUM_UP_BOUND("ub", cl::desc("set upper bound of the number of functions"),
                 cl::value_desc("number"), cl::init(1000000));

namespace sol {

SolLLVMIRGenerator::SolLLVMIRGenerator(int argc, char **argv) {
  // InitLLVM will setup signal handler to print stack trace when the program
  // crashes.
  llvm::InitLLVM x(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);
  DEBUG_SOL = ConfigDebugSol;
}

std::vector<sol::FunctionAST *>
SolLLVMIRGenerator::process(const std::string &filename,
                            const std::string &method, size_t line,
                            std::string contents) {
  std::replace(contents.begin(), contents.end(), '\t', ' ');

  // Process file contents via ANTLR lexer.
  auto *input = new ANTLRInputStream(contents);
  auto *lexer = new RustLexer(input);
  auto *tokens = new CommonTokenStream(lexer);
  tokens->fill();

  // Parse the tokens.
  auto *parser = new RustParser(tokens);
  SolParserVisitor visitor(filename, method, line);
  auto any_res = visitor.visitCrate(parser->crate());
  return std::any_cast<std::vector<sol::FunctionAST *>>(std::move(any_res));
}

bool SolLLVMIRGenerator::handleRustFile(const std::string &full_path) {
  std::ifstream fptr(full_path);
  if (!fptr) {
    if (DEBUG_SOL) {
      llvm::errs() << "input file does not exist: " << full_path << "\n";
    }
    return false;
  }

  std::ifstream ws(full_path);
  std::string contents((std::istreambuf_iterator<char>(ws)),
                       std::istreambuf_iterator<char>());

  auto path = std::filesystem::path(full_path);
  auto base_name = path.stem().string();
  auto res = process(full_path, base_name, 0, contents);
  processResults.insert(processResults.end(),
                        std::make_move_iterator(res.begin()),
                        std::make_move_iterator(res.end()));
  return true;
}

bool SolLLVMIRGenerator::handleDirectory(
    sol::ModuleAST *mod, const std::filesystem::path &dir_path) {
  std::string dir_path_str(dir_path.string());
  llvm::StringRef pathname(dir_path_str);
  if (pathname.endswith("/src")) {
    llvm::outs() << "source path: " << dir_path << "\n";
  } else if (pathname == mod->path) {
    llvm::outs() << "path: " << dir_path << "\n";
  } else {
    llvm::outs() << "subpath: " << dir_path << "\n";
    // if subpath contains Xargo.toml, skip
    for (const auto &entry : std::filesystem::directory_iterator(dir_path)) {
      if (entry.is_regular_file() && entry.path().filename() == "Xargo.toml") {
        return true;
      }
    }
  }

  for (const auto &entry : std::filesystem::directory_iterator(dir_path)) {
    std::string entry_str(entry.path().string());
    llvm::StringRef dname(entry_str);
    if (DEBUG_SOL)
      llvm::outs() << "dname: " << dname << "\n";

    if (entry.is_directory()) {
      if (!dname.equals("tests") && !dname.equals("js") &&
          !dname.equals("cli") && !dname.equals("logs") &&
          !dname.equals("target") && !dname.equals("debug") &&
          !dname.equals("migrations") && !dname.equals("services") &&
          !dname.equals("proptest-regressions")) {
        // nested diretory
        handleDirectory(mod, entry.path());
      }
      continue;
    }

    if (entry.is_regular_file()) {
      if (dname.endswith(".rs")) {
        handleRustFile(entry_str);
      } else if (dname.endswith("argo.toml")) {
        mod->path_config = entry_str;
        auto data = toml::parse(entry_str);
        if (data.contains("profile") &&
            data.at("profile").count("release") != 0) {
          const toml::value answer1 =
              toml::get<toml::table>(data).at("profile");
          const toml::value answer2 =
              toml::get<toml::table>(answer1).at("release");
          if (answer2.count("overflow-checks")) {
            const auto hasOverFlowCheck =
                toml::find<bool>(data, "profile", "release", "overflow-checks");
            llvm::outs() << dname << ": hasOverFlowCheck: " << hasOverFlowCheck
                         << "\n";
            if (hasOverFlowCheck) {
              mod->configMap["profile.release.overflow-checks"] = "1";
            }
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
              mod->configMap["dependencies.spl-token.version"] = version;
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
              mod->configMap["dependencies.anchor-lang.version"] = version;
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
              mod->configMap["dependencies.anchor-spl.version"] = version;
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
              mod->configMap["dependencies.solana_program.version"] = version;
            }
          }
        }
      }
    }
  }
  return true;
}

bool SolLLVMIRGenerator::generateAST(sol::ModuleAST *mod) {
  LOWER_BOUND_ID = NUM_LOW_BOUND;

  llvm::StringRef path(TargetModulePath);
  std::string full_path = std::filesystem::canonical(path.str()).string();
  mod->path = full_path;

  if (std::filesystem::is_regular_file(full_path)) {
    if (path.endswith(".rs")) {
      handleRustFile(full_path);
    }
  } else if (std::filesystem::is_directory(full_path)) {
    handleDirectory(mod, full_path);
  }
  for (auto functionAst : processResults) {
    mod->addFunctionAST(functionAst);
  }
  return true;
}

static int dumpLLVMIR(mlir::ModuleOp mod) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*mod->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
  if (!llvmModule) {
    if (DEBUG_SOL)
      llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  if (ConfigDumpIR || ConfigOutputFile != "t.ll") {
    std::error_code err;
    llvm::raw_fd_ostream outfile(ConfigOutputFile, err, llvm::sys::fs::OF_None);
    if (err) {
      if (DEBUG_SOL) {
        llvm::errs() << "Error dumping IR!\n";
      }
    }

    llvmModule->print(outfile, nullptr);
    outfile.close();
    std::string fullPathName = realpath(ConfigOutputFile.c_str(), NULL);
    llvm::outs() << "IR file: " << fullPathName << "\n";
  }

  if (DEBUG_SOL) {
    llvm::errs() << *llvmModule << "\n";
  }
  return 0;
}

void SolLLVMIRGenerator::generateLLVMIR(sol::ModuleAST *moduleAST) {
  // Register any command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> mod = sol::mlirGenFull(context, *moduleAST);

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  // Finish lowering the IR to the LLVM dialect.
  pm.addPass(mlir::sol::createLowerToLLVMPass());
  if (mlir::failed(pm.run(*mod))) {
    if (DEBUG_SOL) {
      llvm::errs() << "Errors in createLowerToLLVMPass.\n";
    }
  }

  dumpLLVMIR(*mod);
}

void SolLLVMIRGenerator::run() {
  sol::ModuleAST mod;
  bool success = generateAST(&mod);
  if (success) {
    generateLLVMIR(&mod);
  }
}

}; // namespace sol
