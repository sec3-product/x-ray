#include <RDUtil.h>
#include <gtest/gtest.h>

using namespace aser;

extern llvm::cl::opt<std::string> ConfigOutputPath;
extern llvm::cl::opt<std::string> TargetModulePath;

TEST(RDUtilTest, test_rtrim) {
  std::string str = "test  ";
  EXPECT_EQ(rtrim(str), "test");
}