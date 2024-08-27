#include "LogColor.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <llvm/Support/raw_ostream.h>

namespace xray {

void highlight(std::string msg) {
  llvm::outs().changeColor(llvm::raw_ostream::Colors::YELLOW);
  llvm::outs() << msg << "\n"; // add line break
  llvm::outs().resetColor();
}

void info(std::string msg, bool newline) {
  llvm::outs().changeColor(llvm::raw_ostream::Colors::GREEN);
  if (newline) {
    llvm::outs() << msg << "\n";
  } else {
    llvm::outs() << msg;
  }
  llvm::outs().resetColor();
}

void info(std::string msg) { info(msg, true); }

void error(std::string msg) {
  llvm::outs().changeColor(llvm::raw_ostream::Colors::RED);
  llvm::outs() << msg << "\n";
  llvm::outs().resetColor();
}

std::string getCurrentTimeStr() {
  auto t = std::time(nullptr);
  auto local = *std::localtime(&t);

  std::ostringstream tzoss;
  auto offset = std::localtime(&t)->tm_gmtoff / 3600;
  tzoss << "GMT" << (offset >= 0 ? "+" : "") << offset;

  std::ostringstream oss;
  oss << std::put_time(&local, "%a %d %b %Y %T %p") << " " << tzoss.str();
  return oss.str();
}

} // namespace xray
