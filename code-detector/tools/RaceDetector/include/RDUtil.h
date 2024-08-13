#pragma once

#include <string>
#include <utility>
#include <vector>

#include <llvm/IR/Instruction.h>

namespace aser {

// -----------------------------
// |                           |
// |    pure util functions    |
// |                           |
// -----------------------------

void highlight(std::string msg);

void info(std::string msg);
void info(std::string msg, bool newline);

void error(std::string msg);

static const char *ws = " \t\n\r\f\v";

// trim from end of string (right)
inline std::string &rtrim(std::string &s, const char *t = ws) {
  s.erase(s.find_last_not_of(t) + 1);
  return s;
}

// trim from beginning of string (left)
inline std::string &ltrim(std::string &s, const char *t = ws) {
  s.erase(0, s.find_first_not_of(t));
  return s;
}

// trim from both ends of string (right then left)
inline std::string &trim(std::string &s, const char *t = ws) {
  return ltrim(rtrim(s, t), t);
}

inline bool hasNoAliasMD(const llvm::Instruction *inst) {
  auto AAMD = inst->getAAMetadata();
  return AAMD.NoAlias != nullptr;
}

std::string getCurrentTimeStr();

} // namespace aser
