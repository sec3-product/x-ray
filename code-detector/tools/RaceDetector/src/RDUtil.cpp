#include "RDUtil.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "SourceInfo.h"
#include "aser/Util/Demangler.h"

using namespace aser;
using namespace llvm;

/* --------------------------------

        pure util functions

----------------------------------- */

void aser::highlight(std::string msg) {
  llvm::outs().changeColor(raw_ostream::Colors::YELLOW);
  llvm::outs() << msg << "\n"; // add line break
  llvm::outs().resetColor();
}

void aser::info(std::string msg, bool newline) {
  llvm::outs().changeColor(raw_ostream::Colors::GREEN);
  if (newline) {
    llvm::outs() << msg << "\n";
  } else {
    llvm::outs() << msg;
  }
  llvm::outs().resetColor();
}

void aser::error(std::string msg) {
  llvm::outs().changeColor(raw_ostream::Colors::RED);
  llvm::outs() << msg << "\n";
  llvm::outs().resetColor();
}

void aser::info(std::string msg) { info(msg, true); }

// TODO: add time zone info if needed
std::string aser::getCurrentTimeStr() {
  char buf[80];
  std::time_t t = std::time(nullptr);
  auto local = std::localtime(&t);
  std::strftime(buf, sizeof(buf), "%a %d %b %Y %T %p", local);
  std::string str(buf);
  return str;
}

