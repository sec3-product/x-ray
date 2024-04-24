
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h>  // for llvm LLVMContext
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>  // IR reader for bit file
#include <llvm/Support/Signals.h>    // singal for command line
#include <llvm/Support/SourceMgr.h>  // for SMDiagnostic
#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include "aser/PointerAnalysis/Context/KOrigin.h"
#include "aser/PointerAnalysis/Context/NoCtx.h"
#include "aser/PointerAnalysis/Context/KCallSite.h"
#include "aser/PointerAnalysis/Context/HybridCtx.h"

#include "aser/PointerAnalysis/Models/LanguageModel/DefaultLangModel/DefaultLangModel.h"
#include "aser/PointerAnalysis/Models/MemoryModel/FieldSensitive/FSMemModel.h"
#include "aser/PointerAnalysis/PTAVerificationPass.h"
#include "aser/PointerAnalysis/PointerAnalysisPass.h"
#include "aser/PointerAnalysis/Solver/PartialUpdateSolver.h"
#include "aser/PreProcessing/Passes/InsertGlobalCtorCallPass.h"
#include "conflib/conflib.h"

using namespace aser;
using namespace llvm;
using namespace std;

static cl::opt<std::string> TargetModulePath(cl::Positional, cl::desc("path to input bitcode file"));

static OriginsSetter<1> OS {"origin"};

using Model = DefaultLangModel<NoCtx, FSMemModel<NoCtx>>;
using PTASolver = PartialUpdateSolver<Model>;

int main(int argc, char** argv) {
    cl::ParseCommandLineOptions(argc, argv);
    llvm::legacy::PassManager passes;

    logger::LoggingConfig config;
    config.enableFile = false;
    config.enableTerminal = true;
    config.level = spdlog::level::info;
    logger::init(config);

    SMDiagnostic Err;
    auto context = new LLVMContext();
    auto module = parseIRFile(TargetModulePath, Err, *context);

    if (!module) {
        Err.print(argv[0], errs());
        return 1;
    }

    passes.add(new CanonicalizeGEPPass(true));
    passes.add(new LoweringMemCpyPass());
    passes.add(new RemoveExceptionHandlerPass());

    passes.add(new InsertGlobalCtorCallPass());
    passes.add(new PointerAnalysisPass<PTASolver>());
    passes.add(new PTAVerificationPass<PTASolver>());

    passes.run(*module);

    // Dump IR to file
    //if (ConfigDumpIR) {

//    std::error_code err;
//    llvm::raw_fd_ostream outfile("modified.ll", err, llvm::sys::fs::F_None);
//    if (err) {
//        llvm::errs() << "Error dumping IR!\n";
//    }
//
//    module->print(outfile, nullptr);
//    outfile.close();

    return 0;
}

static llvm::RegisterPass<PointerAnalysisPass<PTASolver>>
    PAP("Pointer Analysis Wrapper Pass", "Pointer Analysis Wrapper Pass", true, true);
