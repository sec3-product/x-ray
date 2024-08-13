#include <gtest/gtest.h>
#include <llvm/Support/CommandLine.h>

llvm::cl::opt<std::string>
    TargetModulePath(llvm::cl::Positional,
                     llvm::cl::desc("path to input bitcode file"));
llvm::cl::opt<bool> ConfigDumpIR("dump-ir",
                                 llvm::cl::desc("Dump the modified ir file"));
llvm::cl::opt<bool>
    ConfigNoFilter("no-filter",
                   llvm::cl::desc("Turn off the filtering for race report"));
llvm::cl::opt<std::string> ConfigOutputPath("o",
                                            llvm::cl::desc("JSON output path"),
                                            llvm::cl::value_desc("path"));
llvm::cl::opt<bool>
    ConfigNoOMP("nomp", llvm::cl::desc("Turn off the OpenMP race detection"));

int main(int argc, char **argv) {
  // FIXME: This is a temporary initalization for all global config variables
  int fake_argc = 1;
  char *fake_argv[] = {"racedetect"};
  llvm::cl::ParseCommandLineOptions(fake_argc, fake_argv);

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}