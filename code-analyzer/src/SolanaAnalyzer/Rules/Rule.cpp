#include "Rules/Rule.h"

#include <algorithm>
#include <string>

#include "PTAModels/GraphBLASModel.h"
#include "SourceInfo.h"

namespace xray {

bool isUpper(const std::string &s) {
  return std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return std::isupper(c) || c == '_';
  });
}

bool isAllCapitalOrNumber(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() &&
         (std::isupper(*it) || std::isdigit(*it) || *it == '_'))
    ++it;
  return !s.empty() && it == s.end();
}

bool isAllCapital(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && (std::isupper(*it) || *it == '_'))
    ++it;
  return !s.empty() && it == s.end();
}

bool isNumber(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

bool isConstant(const std::string &s) { return isNumber(s) || isAllCapital(s); }

bool RuleContext::isSafeType(const llvm::Value *value) const {
  if (auto arg = llvm::dyn_cast<llvm::Argument>(value)) {
    auto typ = FuncArgTypesMap[Func][arg->getArgNo()].second;
    if (isUpper(typ.str()) || typ.size() >= 5 || typ.startswith("i") ||
        typ.contains("Dec")) {
      return true;
    }
  }
  return false;
}

bool RuleContext::isSafeVariable(const llvm::Value *value) const {
  if (auto arg = llvm::dyn_cast<llvm::Argument>(value)) {
    auto valueName = FuncArgTypesMap[Func][arg->getArgNo()].first;
    // current_timestamp - self.last_timestamp
    if (valueName.contains("_time")) {
      return true;
    }
  } else {
    auto valueName = LangModel::findGlobalString(value);
    if (valueName.find("as i") != std::string::npos ||
        valueName.find(".len") != std::string::npos ||
        valueName.find("_len") != std::string::npos ||
        valueName.find("length_") != std::string::npos ||
        valueName.find("size") != std::string::npos ||
        valueName.find("steps") != std::string::npos ||
        valueName.find("gas") != std::string::npos ||
        valueName.find("bits") != std::string::npos ||
        valueName.find("shift") != std::string::npos ||
        (valueName.find("pos") != std::string::npos &&
         valueName.find("iter") == std::string::npos)) {
      return true;
    }
  }
  return false;
}

bool RuleContext::hasValueLessMoreThan(const llvm::Value *value,
                                       bool isLess) const {
  auto symbol = isLess ? "<" : ">";
  auto valueLessThan = LangModel::findGlobalString(value).str() + symbol;
  std::string snippet = getSourceLoc(Inst).getSnippet();
  snippet.erase(std::remove(snippet.begin(), snippet.end(), ' '),
                snippet.end());
  if (DEBUG_RUST_API) {
    llvm::outs() << "valueLessThan: " << valueLessThan << "\n";
    llvm::outs() << "snippet: \n" << snippet << "\n";
  }
  return snippet.find(valueLessThan) != std::string::npos;
}

} // namespace xray
