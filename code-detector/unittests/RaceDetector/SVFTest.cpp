#include <ValueFlowAnalysis/SVFPass.h>
#include <gtest/gtest.h>
#include <llvm/IRReader/IRReader.h> // IR reader for bit file
#include <llvm/Support/SourceMgr.h> // for SMDiagnostic

#include <regex>

#include "PTAModels/GraphBLASModel.h"

using namespace aser;
using namespace llvm;
using namespace std;

extern int test_scev_parser(const char *yy_str);

// NOTE: these are old test for string-based scev simplification
// no longer needed
// TEST(SVFTest, test_simplifySCEVExpression1) {
//     string output = "((4 * (sext i32 (({%tid,+,1}<%.preheader> * %54) +
//     {0,+,1}<%53>) to i64))<nsw> + %7)"; string expected_output = "4 * %tid *
//     %54 + %7"; llvm::outs() << "input: " << output << "\n";
//     SVFModel::simplifySCEVExpression(output);
//     llvm::outs() << "output: " << output << "\n";
//     ASSERT_EQ(expected_output, output);
// }

// TEST(SVFTest, test_simplifySCEVExpression2) {
//     string output = "{((4 * (sext i32 %17 to i64))<nsw> + %0),+,4}<nw><%22>";
//     string expected_output = "4 * %17 + %0";
//     llvm::outs() << "input: " << output << "\n";
//     SVFModel::simplifySCEVExpression(output);
//     llvm::outs() << "output: " << output << "\n";
//     ASSERT_EQ(expected_output, output);
// }
// TEST(SVFTest, test_simplifySCEVExpression3) {
//     string key = "<n[s|u]w>";
//     std::regex e(key);
//     string input = "4 * ({%tid,+,1}<nsw><%.preheader> * %54)";
//     string output = std::regex_replace(input, e, "");
//     string expected_output = "4 * ({%tid,+,1}<%.preheader> * %54)";

//     ASSERT_EQ(expected_output, output);
// }

extern std::string SCEV_TEMP_RESULT;

TEST(SVFTest, test_scev_parser1) {
  SCEV_TEMP_RESULT = "4 * ({%tid,+,1}<nsw><%.preheader> * %54)";
  llvm::outs() << "before: " << SCEV_TEMP_RESULT << "\n";
  int output = test_scev_parser(SCEV_TEMP_RESULT.c_str());
  llvm::outs() << "after: " << SCEV_TEMP_RESULT << "\n";
  ASSERT_EQ(0, output);
  ASSERT_EQ("4 * %tid * %54", SCEV_TEMP_RESULT);
}
TEST(SVFTest, test_scev_parser2) {
  SCEV_TEMP_RESULT = "((4 * (sext i32 (({%tid,+,1}<%.preheader> * %54) + "
                     "{0,+,1}<%53>) to i64))<nsw> + %7)";
  llvm::outs() << "before: " << SCEV_TEMP_RESULT << "\n";
  int output = test_scev_parser(SCEV_TEMP_RESULT.c_str());
  llvm::outs() << "after: " << SCEV_TEMP_RESULT << "\n";
  ASSERT_EQ(0, output);
  ASSERT_EQ("(4 * %tid * %54) + %7", SCEV_TEMP_RESULT);
}

TEST(SVFTest, test_scev_parser3) {
  SCEV_TEMP_RESULT = "((4 * (sext i32 (-10 + (2 * {%tid,+,1}<%.lr.ph>)) to "
                     "i64))<nsw> + (2 * {%tid,+,1}<%.lr.ph>))";
  llvm::outs() << "before: " << SCEV_TEMP_RESULT << "\n";
  int output = test_scev_parser(SCEV_TEMP_RESULT.c_str());
  llvm::outs() << "after: " << SCEV_TEMP_RESULT << "\n";
  ASSERT_EQ(0, output);
  ASSERT_EQ("(4 * (-10 + (2 * %tid))) + (2 * %tid)", SCEV_TEMP_RESULT);
}