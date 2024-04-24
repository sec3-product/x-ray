#include <boost/algorithm/string.hpp>
#include <conflib/conflib.h>
#include <dirent.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h> // for llvm LLVMContext
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h> // IR reader for bit file
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/Signals.h>   // signal for command line
#include <llvm/Support/SourceMgr.h> // for SMDiagnostic
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <o2/Util/Log.h>

#include "SmalltalkLexer.h"
#include "SmalltalkParser.h"
#include "antlr4-runtime.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

using namespace antlr4;
using namespace antlrcpp;
using namespace llvm;
using namespace o2;
#include <atomic>

#include "ThreadPool.h"
#include "st/LinkModules.h"
#include "st/MLIRGen.h"
#include "st/ParserWrapper.h"
#include "st/Passes.h"
#include "st/STParserListener.h"
#include "st/STParserVisitor.h"

using namespace st;

std::vector<std::future<stx::FunctionAST *>> processResults;
std::vector<std::string> IRFiles;

std::atomic<uint> numOfFunctions{0};

cl::opt<std::string> TargetModulePath(cl::Positional,
                                      cl::desc("path to input bitcode file"));
cl::opt<bool> ConfigDumpIR("dump-ir", cl::desc("Dump the generated IR file"),
                           cl::init(true));

cl::opt<std::string> ConfigOutputFile("o", cl::desc("IR output file name"),
                                      cl::init("t.ll"));
cl::opt<bool> DebugParsing(
    "debug-parse", cl::desc("Print parser immediate results for debug"));

cl::opt<int> NUM_LOW_BOUND(
    "lb", cl::desc("set lower bound of the number of functions"),
    cl::value_desc("number"), cl::init(0));
int LOWER_BOUND_ID = 0;
cl::opt<int> NUM_UP_BOUND(
    "ub", cl::desc("set upper bound of the number of functions"),
    cl::value_desc("number"), cl::init(1000000));

cl::opt<int> MAX_FUNCTION_LENGTH("max-loc",
                                 cl::desc("set max loc of a function"),
                                 cl::value_desc("number"), cl::init(1000));
cl::opt<int> MAX_FUNCTION_SIZE("max-size",
                               cl::desc("set max size of a function"),
                               cl::value_desc("number"), cl::init(1000));

logger::LoggingConfig initLoggingConf()
{
  logger::LoggingConfig config;

  config.enableProgress = conflib::Get<bool>("XenableProgress", true);
  auto conflevel = conflib::Get<std::string>("logger.level", "debug");
  if (conflevel == "trace")
  {
    config.level = spdlog::level::trace;
  }
  else if (conflevel == "debug")
  {
    config.level = spdlog::level::debug;
  }
  else if (conflevel == "info")
  {
    config.level = spdlog::level::info;
  }
  else if (conflevel == "warn")
  {
    config.level = spdlog::level::warn;
  }
  else if (conflevel == "error")
  {
    config.level = spdlog::level::err;
  }
  else
  {
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
                                        bool abortOnFail)
{
  SMDiagnostic Err;
  std::unique_ptr<Module> Result;
  Result = parseIRFile(FN, Err, Context);

  if (!Result)
  {
    if (abortOnFail)
    {
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
    "llvm.", "pthread_", "__kmpc_", ".omp.", "omp_", "_ZNSt",
    "_ZSt", "_ZNKSt", "_ZN14o2_", "__o2_", "printf", "je_"};
void addExploredFunction(const Function *f) { unExploredFunctions.erase(f); }
void initUnexploredFunctions(Module *module)
{
  auto &functionList = module->getFunctionList();
  for (auto &function : functionList)
  {
    auto func = &function;
    if (!func->isDeclaration())
    {
      bool insert = true;
      // do not count llvm.* pthread_* __kmpc_* __smalltalk_ .omp. omp_ std::
      for (auto str : SKIPPED_APIS)
      {
        if (func->getName().startswith(str))
        {
          insert = false;
          break;
        }
      }
      if (insert)
        unExploredFunctions.insert(func);
    }
  }
}
void computeUnexploredFunctions()
{
  if (unExploredFunctions.size() > 0)
  {
    llvm::outs() << "\n============= unexplored function ==========\n";
    for (auto func : unExploredFunctions)
      llvm::outs() << (func->getName()) << "\n";
    // TODO: print potential entry points
    // no caller
  }
}
void initRaceDetect()
{
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
int largest_func_length = 0;
int largest_func_size = 0;

stx::FunctionAST *process(std::string filename, std::string methodname,
                          size_t line, char *s)
{
  // find ":=" in s, and replace it with " :="
  // llvm::outs() << "before s-----------" << s << "\n";
  std::string s2(s);
  /*  auto found = s2.find(":=");
    while (found != string::npos) {
      s2.insert(found, " ");
      found = s2.find(":=", found + 4);
      // llvm::outs() << "after s-----------" << s2 << "\n";
    } */
  auto found = s2.find("''");
  while (found != string::npos)
  {
    s2.replace(found, 2, "'@'");
    found = s2.find("''", found + 2);
    // s = s2.c_str();
    // llvm::outs() << "after s-----------" << s2 << "\n";
  }

  auto *input = new ANTLRInputStream(s2);
  //  auto *input = new ANTLRInputStream(s);

  auto *lexer = new SmalltalkLexer(input);
  auto *tokens = new CommonTokenStream(lexer);

  tokens->fill();

  auto *parser = new SmalltalkParser(tokens);

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
  stx::FunctionAST *res = visitor.visitModule(parser->module());
  return res;
  // return NULL;
}

// c++17 only
// using std::filesystem::directory_iterator;
void initParserForFile(stx::ModuleAST *module, std::string &fullPathName,
                       stx::ThreadPool &pool)
{
  // for (const auto & entry : directory_iterator(path)){
  //     StringRef name(entry.path());
  StringRef name(fullPathName);
  if (name.endswith(".ws"))
  {
    std::ifstream ws(name.str());
    std::string contents((std::istreambuf_iterator<char>(ws)),
                         std::istreambuf_iterator<char>());
    llvm::outs() << "=== Test WS Start ===\n";
    llvm::outs() << contents << "\n";
    llvm::outs() << "=== Test WS End ===\n";
    module->entry_point =
        process(fullPathName, "main", 0, (char *)contents.c_str());
  }
  else if (name.endswith(".st"))
  {
    xmlDocPtr doc;
    xmlNodePtr cur;
    doc = xmlReadFile(name.data(), NULL,
                      XML_PARSE_BIG_LINES | XML_PARSE_RECOVER | XML_PARSE_HUGE);
    if (doc == NULL)
    {
      fprintf(stderr, "Document not parsed successfully. \n");
    }
    cur = xmlDocGetRootElement(doc);

    if (cur == NULL)
    {
      fprintf(stderr, "Empty document or parsing failures\n");
      xmlFreeDoc(doc);
      return; // recover from crash
    }
    //    llvm::outs() << cur->name << "\n";
    if (xmlStrcmp(cur->name, (const xmlChar *)"st-source"))
    {
      fprintf(stderr, "Document of the wrong type.");
      xmlFreeDoc(doc);
    }
    else
    {
      xmlNodePtr nptr = cur->xmlChildrenNode;
      while (nptr->next)
      {
        //            llvm::outs() << nptr->name << "\n";
        if (0 == xmlStrcmp(nptr->name, (const xmlChar *)"class"))
        {
          auto classAST = new ClassAST();

          xmlNodePtr methodpt = nptr->xmlChildrenNode;
          while (methodpt != NULL)
          {
            if (!methodpt->xmlChildrenNode)
            {
              methodpt = methodpt->next;
              continue;
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("name")))
            {
              classAST->class_name =
                  ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              //                        printf("%s\n",((char*)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
              classAST->line = xmlGetLineNo(methodpt); // methodpt->line;
              classAST->fileName = fullPathName;
              cout << "***  Processing Class " << classAST->class_name
                   << " ***\n";
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("environment")))
            {
              classAST->environment =
                  ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              //                        printf("%s\n",((char*)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("super")))
            {
              classAST->super_class =
                  ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              //                        printf("%s\n",((char*)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("private")))
            {
              classAST->privateinfo =
                  ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              //                        printf("%s\n",((char*)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("indexed-type")))
            {
              classAST->indexed_type =
                  ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              //                        printf("%s\n",((char*)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("inst-vars")))
            {
              string tmp = ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              std::stringstream ss(tmp);
              std::string buf;
              while (ss >> buf)
              {
                classAST->inst_vars.insert(buf);
                llvm::outs() << "inst_vars: " << buf << "\n";
              }

              //                        printf("%s\n",((char*)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("class-inst-vars")))
            {
              string tmp = ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              std::stringstream ss(tmp);
              std::string buf;
              while (ss >> buf)
              {
                classAST->class_inst_vars.insert(buf);
                llvm::outs() << "class_inst_vars: " << buf << "\n";
              }
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("imports")))
            {
              classAST->imports =
                  ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              //                        printf("%s\n",((char*)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
            }
            if (!xmlStrcmp(methodpt->name, BAD_CAST("category")))
            {
              classAST->category =
                  ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));
              //                        printf("%s\n",((char*)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
            }
            methodpt = methodpt->next;
          }
          module->addClassAST(classAST);
        }
        else if (0 == xmlStrcmp(nptr->name, (const xmlChar *)"methods"))
        {
          xmlChar *className = nullptr;
          int prevLine = 0;
          {
            // find class id
            xmlNodePtr methodpt = nptr->xmlChildrenNode;
            while (methodpt->next)
            {
              if (0 == xmlStrcmp(methodpt->name, (const xmlChar *)"class-id"))
              {
                className = XML_GET_CONTENT(methodpt->xmlChildrenNode);
                llvm::outs()
                    << "*** find class-id: " << (char *)className << "\n";
              }
              methodpt = methodpt->next;
            }
          }
          xmlNodePtr methodpt = nptr->xmlChildrenNode;
          while (methodpt != NULL)
          {
            if (!xmlStrcmp(methodpt->name, BAD_CAST("body")))
            {
              auto package = className;
              if (!package)
                package = xmlGetProp(methodpt, (const xmlChar *)("package"));

              auto selector =
                  xmlGetProp(methodpt, (const xmlChar *)("selector"));

              // llvm::outs() << "*** test package: " << (char *)package
              //              << " selector: " << (char *)selector;
              // llvm::outs() << "***\n";
              std::string methodName =
                  std::string("st.") + std::string((char *)selector) +
                  std::string("$") + std::string((char *)package);
              // std::string methodName = std::string((char *)package) +
              //                          std::string("$") +
              //                          std::string((char *)selector);
              int curLine = xmlGetLineNo(methodpt);
              llvm::outs() << "filename: " << fullPathName
                           << " method: " << methodName << " line: " << curLine
                           << "\n";

              if (methodpt->xmlChildrenNode == nullptr)
              {
                methodpt = methodpt->next;
                continue;
              }
              int length = curLine - prevLine;
              char *s = (char *)XML_GET_CONTENT(methodpt->xmlChildrenNode);
              std::string s2(s);
              int size = s2.length();
              if (prevLine == 0)
                length = size / 80;
              prevLine = curLine;
              if (DebugParsing)
              {
                if (length > largest_func_length)
                {
                  largest_func_length = length;
                  llvm::outs() << "\n-----------loc: " << length
                               << " size:" << size << "-----------\n";
                  llvm::outs() << "***\n";
                  printf("%s\n",
                         ((char *)XML_GET_CONTENT(methodpt->xmlChildrenNode)));
                  llvm::outs() << "***"
                               << "\n";
                }

                // if (methodpt->xmlChildrenNode)

                // for testing only
                if (largest_func_length > 0)
                {
                  methodpt = methodpt->next;
                  continue;
                }
              }
              if (length > MAX_FUNCTION_LENGTH || size > MAX_FUNCTION_SIZE)
              {
                methodpt = methodpt->next;
                continue;
              }

              // cout << "Testing Line Number:" << methodpt->line << endl;
              //                   cout << "Testing Line Number psvi:" <<
              //                   methodpt->psvi << endl; cout << methodName <<
              //                   endl; cout << "Testing Line Number
              //                   xmlGetLineNo:" << xmlGetLineNo(methodpt) <<
              //                   endl; cout << "Testing Line Number
              //                   xmlSAX2GetLineNumber:" <<
              //                   xmlSAX2GetLineNumber(methodpt) << endl;

              if (methodpt->xmlChildrenNode)
              {
                numOfFunctions++;
                if (numOfFunctions >= NUM_LOW_BOUND &&
                    numOfFunctions <= NUM_UP_BOUND)
                  processResults.emplace_back(
                      pool.enqueue([fullPathName, methodName, methodpt]
                                   {
                        stx::FunctionAST *funcAst = process(
                            fullPathName, methodName,
                            xmlGetLineNo(methodpt) - 1,
                            (char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));

                        return funcAst; }));
                // auto funcAst = process(
                //     fullPathName, methodName, xmlGetLineNo(methodpt) - 1,
                //     (char *)XML_GET_CONTENT(methodpt->xmlChildrenNode));

                // if (funcAst) module->addFunctionAST(funcAst);
              }
            }
            methodpt = methodpt->next;
          }
        }
        nptr = nptr->next;
      }
      //        llvm::outs() << nptr->next->next->next->next->last->prev->name
      //        << "\n";
    }
    /*
     * Cleanup function for the XML library.
     */
    xmlCleanupParser();
    /*
     * this is to debug memory for regression tests
     */
    xmlMemoryDump();
  }
}

int dumpLLVMIR(mlir::ModuleOp module, std::string &OutputFile)
{
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
  {
    llvm::errs() << "Failed to emit LLVM IR\n";
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
  if (ConfigDumpIR)
  {
    std::error_code err;
    llvm::raw_fd_ostream outfile(OutputFile, err, llvm::sys::fs::F_None);
    if (err)
    {
      llvm::errs() << "Error dumping IR!\n";
    }

    llvmModule->print(outfile, nullptr);
    outfile.close();
    llvm::outs() << "IR file: " << OutputFile << "\n";
    IRFiles.push_back(OutputFile);
  }

  llvm::errs() << *llvmModule << "\n";

  return 0;
}

void initLLVMIR(stx::ModuleAST *moduleAST, std::string &OutputFile)
{
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
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
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Partially lower the toy dialect with a few cleanups afterwards.
    // pm.addPass(mlir::st::createLowerToAffinePass());

    // mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    // optPM.addPass(mlir::createCanonicalizerPass());
    // optPM.addPass(mlir::createCSEPass());

    // Finish lowering the toy IR to the LLVM dialect.
    pm.addPass(mlir::st::createLowerToLLVMPass());
    if (mlir::failed(pm.run(*module)))
    {
      llvm::errs() << "Failed to run createLowerToLLVMPass pass \n";
    }
  }
  dumpLLVMIR(*module, OutputFile);
}
int testThreadPool()
{
  stx::ThreadPool pool(4);
  std::vector<std::future<int>> results;

  for (int i = 0; i < 8; ++i)
  {
    results.emplace_back(pool.enqueue([i]
                                      {
      std::cout << "hello " << i << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "world " << i << std::endl;
      return i * i; }));
  }

  for (auto &&result : results)
    std::cout << result.get() << ' ';
  std::cout << std::endl;

  return 0;
}

// This method will cause the rest of a line after the formfeed disappear
/*void stripformfeeds(const char *file_cstr)
{ // Look for the form feeds and replace them because they cause parsing errors
  std::fstream file(file_cstr, std::ios::in);
  if (file.is_open())
  {
    std::string replace = "\x0C"; // form feeds
    std::string replace_with = " ";
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
      std::string::size_type pos = 0;
      while ((pos = line.find(replace, pos)) != std::string::npos)
      {
        line.replace(pos, line.size(), replace_with);
        pos += replace_with.size();
      }
      lines.push_back(line);
    }
    file.close();
    file.open(file_cstr, std::ios::out | std::ios::trunc);
    for (const auto &i : lines)
    {
      file << i << std::endl;
    }
  }
}*/

// This method will not cause the rest of a line after the formfeed disappear
void stripSpecialSymbols(const char *file_cstr)
{ // Look for the special symbols and remove them because they cause parsing errors
  std::fstream file(file_cstr, std::ios::in);
  if (file.is_open())
  {
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
      //boost::erase_all(line, "\x00"); // remove Null, need to be searched by char. instead of string
      line.erase(remove(line.begin(), line.end(), '\0'), line.end()); // remove Null, need to be searched by char. instead of string
      boost::erase_all(line, "\x0C"); // remove Form Feed
      boost::erase_all(line, "\x1b"); // remove Escape
      boost::erase_all(line, "￿"); // remove <ffff>
      boost::erase_all(line, "\x07"); // remove Bell
      boost::erase_all(line, "\x13"); // remove Device Control 3
      boost::erase_all(line, "\x03"); // remove End of Text
      boost::erase_all(line, "\x08"); // remove Back Space

      lines.push_back(line);
    }
    file.close();
    file.open(file_cstr, std::ios::out | std::ios::trunc);
    //file.open("test.txt", std::ios::out | std::ios::trunc);
    for (const auto &i : lines)
    {
      file << i << std::endl;
    }
  }
}

bool handleSmalltalkFile(std::string &fullPathName)
{
  FILE *fptr = fopen(fullPathName.c_str(), "r");
  if (fptr == NULL)
  {
    llvm::errs() << "input file does not exist: " << fullPathName << "\n";
    return false;
  }
  // File exists hence close file
  fclose(fptr);

  // remove special symbols if exists in files
  stripSpecialSymbols(fullPathName.c_str());

  const int COUNT = thread::hardware_concurrency();
  stx::ThreadPool pool(COUNT);
  LOWER_BOUND_ID = NUM_LOW_BOUND;

  stx::ModuleAST *module = new stx::ModuleAST();
  module->path = fullPathName;

  initParserForFile(module, fullPathName, pool);

  for (auto &&result : processResults)
  {
    try
    {
      auto functionAst = result.get();
      if (functionAst)
        module->addFunctionAST(functionAst);
    }
    catch (const std::future_error &e)
    {
      std::cout << "Caught a future_error with code \"" << e.code()
                << "\"\nMessage: \"" << e.what() << "\"\n";
      break;
    }
  }
  llvm::outs() << "numOfFunctions: " << numOfFunctions << "\n";
  auto OutputFile = fullPathName + ".ll";
  initLLVMIR(module, OutputFile);

  processResults.clear();
  numOfFunctions = 0;

  return true;
}
bool linkIRFiles(std::string &OutputFile)
{
  LLVMContext Context;
  auto module = make_unique<Module>(OutputFile, Context);
  st::Linker L(*module);

  for (auto irFile : IRFiles)
  {
    llvm::outs() << "Loading IR From File: " << irFile << "\n";

    std::unique_ptr<Module> M = loadFile(irFile, Context, true);
    if (M.get() == nullptr)
    {
      llvm::outs() << "Failed to load IR file: " << irFile << "\n";
      continue;
    }
    bool Err = L.linkInModule(std::move(M), st::Linker::Flags::OverrideFromSrc);

    if (Err)
      return false;
  }
  std::error_code err;
  llvm::raw_fd_ostream outfile(OutputFile, err, llvm::sys::fs::F_None);
  if (err)
  {
    llvm::errs() << "Error dumping IR: " << OutputFile << "\n";
  }
  module->print(outfile, nullptr);
  outfile.close();
  llvm::outs() << "Linked IR file: " << OutputFile << "\n";
  return true;
}
int main(int argc, char **argv)
{
  // testThreadPool();
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

  // OK, let's do each file separately and then link together

  StringRef path(TargetModulePath);
  std::string fullPathName = realpath(path.data(), NULL);
  llvm::outs() << "full path: " << fullPathName << "\n";

  struct dirent *entry;
  DIR *dir = opendir(path.data());
  if (dir == NULL)
  {
    handleSmalltalkFile(fullPathName);
  }
  else
  {
    llvm::outs() << "path: " << path << "\n";
    while ((entry = readdir(dir)) != NULL)
    {
      StringRef dname(entry->d_name);
      if (!dname.endswith(".ws") && !dname.endswith(".st"))
        continue;
      // only parse .ws and .st files
      std::string fullFilePathName = fullPathName + "/" + dname.str();
      llvm::outs() << "file: " << fullFilePathName << "\n";
      handleSmalltalkFile(fullFilePathName);
    }
    closedir(dir);
  }
  for (auto irFile : IRFiles)
  {
    llvm::outs() << "IR file: " << irFile << "\n";
  }
  if (IRFiles.size() > 1)
  {
    // link all ir files to folder_name.ll
    auto found = fullPathName.find_last_of("/");
    auto folder_name = fullPathName.substr(found + 1);
    std::string linkedPathName = fullPathName + "/" + folder_name + ".ll";
    linkIRFiles(linkedPathName);
  }
  // initRaceDetect();
  return 0;
}
