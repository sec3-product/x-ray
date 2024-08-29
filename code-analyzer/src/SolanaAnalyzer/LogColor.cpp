#include "LogColor.h"

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

} // namespace xray
