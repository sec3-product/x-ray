#include "LogColor.h"

#include <ctime>
#include <fstream>
#include <sstream>
#include <string>

#include "llvm/Support/raw_ostream.h"

/* --------------------------------

        log util functions

----------------------------------- */

void xray::highlight(std::string msg) {
  llvm::outs().changeColor(llvm::raw_ostream::Colors::YELLOW);
  llvm::outs() << msg << "\n"; // add line break
  llvm::outs().resetColor();
}

void xray::info(std::string msg, bool newline) {
  llvm::outs().changeColor(llvm::raw_ostream::Colors::GREEN);
  if (newline) {
    llvm::outs() << msg << "\n";
  } else {
    llvm::outs() << msg;
  }
  llvm::outs().resetColor();
}

void xray::error(std::string msg) {
  llvm::outs().changeColor(llvm::raw_ostream::Colors::RED);
  llvm::outs() << msg << "\n";
  llvm::outs().resetColor();
}

void xray::info(std::string msg) { info(msg, true); }

// TODO: add time zone info if needed
std::string xray::getCurrentTimeStr() {
  char buf[80];
  std::time_t t = std::time(nullptr);
  auto local = std::localtime(&t);
  std::strftime(buf, sizeof(buf), "%a %d %b %Y %T %p", local);
  std::string str(buf);
  return str;
}
