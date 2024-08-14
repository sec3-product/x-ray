#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Local.h>
#include <nlohmann/json.hpp>

#include "Graph/Event.h"

namespace aser {

using CallingCtx = std::pair<std::vector<CallEvent *>, TID>;
using json = nlohmann::json;

class SourceInfo {
private:
  const llvm::Value *val;
  // DEFAULT:
  //  line: 0 (actual line starts from 1)
  //  col: 0 (actual col starts from 1)
  //  filename: ""
  //  directory: ""
  //  snippet: ""
  //  sourceLine: ""
  //  name: ""
  unsigned line;
  unsigned col;
  // this should be a relative path
  std::string filename;
  std::string dir;
  std::string snippet;
  std::string sourceLine;
  std::vector<std::string> st;
  // OPTIONAL:
  // the two fields belwo are only for shared object
  std::string name;
  std::string tyStr;
  // the specific field that's shared if the shared object is a class/struct
  std::string accessPath;
  bool inaccurate;
  bool iswrite = false;

public:
  SourceInfo();
  SourceInfo(const llvm::Value *V, unsigned L, unsigned C, std::string FN,
             std::string D, std::string name, bool inaccurate,
             std::string tyStr);
  inline unsigned getLine() const { return this->line; };
  inline void setCorrectLine(unsigned line) { this->line = line; };
  inline unsigned getCol() const { return this->col; };
  inline std::string getFilename() const { return this->filename; };
  inline std::string getDir() const { return this->dir; };
  inline std::string getName() const { return this->name; };
  inline std::string getType() const { return this->tyStr; }
  inline const llvm::Value *getValue() const { return this->val; };
  inline std::string getSourceLine() const { return this->sourceLine; };
  inline void setCorrectSourceLine(std::string line) {
    this->sourceLine = line;
  };

  inline std::string getSnippet() const { return this->snippet; };
  inline std::string getAccessPath() const { return this->accessPath; };
  inline const std::vector<std::string> &getStackTrace() const { return st; };
  inline void setStackTrace(std::vector<std::string> &st) { this->st = st; };
  inline void setAccessPath(const std::string &accessPath) {
    this->accessPath = accessPath;
  };

  bool isInAccurate() { return inaccurate; }
  bool isGlobalValue() const {
    return val && llvm::isa<llvm::GlobalValue>(val);
  }
  void setWrite() { iswrite = true; }
  bool isWrite() const { return iswrite; }

  std::string overview() const;

  // return a source-level signature for the memory access
  // used for filtering redundant races
  std::string sig() const;

  // return the string we want to print out to the terminal
  std::string str();

  inline bool operator<(SourceInfo &ma) const {
    std::string path1 = this->dir + "/" + this->filename;
    std::string path2 = ma.dir + "/" + ma.filename;
    int cond = path1.compare(path2);

    if (cond < 0) {
      return true;
    } else if (cond == 0) {
      if (this->line < ma.line) {
        return true;
      } else if (this->line == ma.line) {
        return (this->col < ma.col);
      }
    }
    return false;
  }

  inline bool operator==(SourceInfo &ma) const {
    std::string path1 = this->dir + "/" + this->filename;
    std::string path2 = ma.dir + "/" + ma.filename;
    int cond = path1.compare(path2);

    if (cond == 0 && (this->line == ma.line) && (this->col == ma.col)) {
      return true;
    }
    return false;
  }
};

// implicitly used by nlohmann/json
inline void to_json(json &j, const SourceInfo &si) {
  j = json{{"line", si.getLine()},
           {"col", si.getCol()},
           {"filename", si.getFilename()},
           {"dir", si.getDir()},
           {"sourceLine", si.getSourceLine()},
           {"snippet", si.getSnippet()},
           {"stacktrace", si.getStackTrace()}};
};

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

// ----------------------------------
// |                                |
// |    report related functions    |
// |                                |
// ----------------------------------

// get the raw source code line
std::string getRawSourceLine(std::string directory, std::string filename,
                             unsigned line);

// get the soure code line for terminal output (add line number info etc.)
std::string getSourceLine(std::string directory, std::string filename,
                          unsigned line);
std::string getSourceLinesForSoteria(SourceInfo &src, unsigned range);
void getSourceLinesForSoteriaAnchorAccount(SourceInfo &src,
                                           std::vector<std::string> &lines);

// very likely the "col" is useless
std::string getCodeSnippet(std::string directory, std::string filename,
                           unsigned line);
std::string getCodeSnippet(std::string directory, std::string filename,
                           unsigned line, unsigned col);
std::string getCodeSnippet(std::string directory, std::string filename,
                           unsigned line, unsigned col, unsigned length);
std::string getCodeSnippet(std::string directory, std::string filename,
                           unsigned line, unsigned col, unsigned above,
                           unsigned below);
inline std::string getCodeSnippet(SourceInfo &src) {
  return getCodeSnippet(src.getDir(), src.getFilename(), src.getLine(),
                        src.getCol());
}
inline std::string getCodeSnippet(SourceInfo &src, unsigned length) {
  return getCodeSnippet(src.getDir(), src.getFilename(), src.getLine(),
                        src.getCol(), length);
}
inline std::string getCodeSnippet(SourceInfo &src, unsigned above,
                                  unsigned below) {
  return getCodeSnippet(src.getDir(), src.getFilename(), src.getLine(),
                        src.getCol(), above, below);
}

// NOTE: For a store instruction such as "x = i"
// LLVM's API will output the column number for this instruction as 3, i.e. the
// position of for "=" While the accurate column number we expect is 1 (the
// position of "x") One solution is to get the position of
// StoreInst::getPointerOperand() but this will not always work. A work-around
// here is to look back from the position of "=" until we hit the indent.
// CAREFUL: this function should only work for "StoreInst"
unsigned getAccurateCol(std::string directory, std::string filename,
                        unsigned line, unsigned col, bool isStore);

std::string getRaceSrcSig(SourceInfo &srcInfo1, SourceInfo &srcInfo2);

std::string getRaceFileSig(SourceInfo &srcInfo1, SourceInfo &srcInfo2);

std::string getRaceMethodSig(Event *e1, Event *e2, const ObjTy *obj);

std::string getRaceRawLineSig(SourceInfo &srcInfo1, SourceInfo &srcInfo2);
std::string getRaceRawLineSig(const SourceInfo &srcInfo);

SourceInfo getSourceLoc(const llvm::Value *val);
void findCorrectSourceInfo(SourceInfo &srcInfo1, std::string &varName);
void tryCorrectSourceInfo(SourceInfo &srcInfo1, SourceInfo &srcInfo2,
                          SourceInfo &sharedObjLoc);

bool filterStrPattern(std::string src);

// `ignoreRaceInFun` should be a set of function name that defined by users
// through the config file we will filter out a race if the racy memory access
// is within that function
// TODO: later on we can adopt more sophisticated logic here
bool customizedFilter(Event *e1, Event *e2, std::vector<std::string> &st1,
                      std::vector<std::string> &st2,
                      std::vector<std::string> &ignoreRaceInFun);

bool customizedOMPFilter(Event *e1, Event *e2, std::vector<std::string> &st1,
                         std::vector<std::string> &st2,
                         const std::vector<std::string> &callingCtx,
                         std::vector<std::string> &ignoreRaceInFun);

bool customizedFilterSoteriaIgnoreFullSymbol(const Event *e);
bool customizedFilterSoteriaIgnoreSymbol(const Event *e,
                                         const std::string symbol);
bool customizedFilterIgnoreLocations(const Event *e1, const Event *e2);

int customizedPriorityAdjust(int P, std::string name, SourceInfo &srcInfo1,
                             SourceInfo &srcInfo2,
                             std::vector<std::string> &st1,
                             std::vector<std::string> &st2);

std::vector<CallEvent *>
getCallEventStack(const Event *e,
                  std::map<TID, std::vector<CallEvent *>> &callEventTraces);

// recursively collect call event stack, until we reach the main function
void getCallEventStackUntilMain(
    const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces,
    std::vector<CallEvent *> &result);

// recursively collect call event stack, until we reach the main function
std::vector<CallEvent *> getCallEventStackUntilMain(
    const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces);

// NOTE: this function is specifically used for OpenMP debugging info
// the parameter `callingCtx` is the incomplete callEventTrace for a specific
// thread usually this is the callEventTrace parent thread of a OpenMP region
// Since the callEventTrace is incomplete, there's a good feature we can
// leverage: all the function that haven't returned does not have a EndID
std::vector<std::string> getCallingCtx(CallingCtx &callingCtx, bool isCpp);

std::vector<std::string>
getStackTrace(const Event *e,
              std::map<TID, std::vector<CallEvent *>> &callEventTraces);
std::vector<std::string>
getStackTrace(const Event *e,
              std::map<TID, std::vector<CallEvent *>> &callEventTraces,
              bool isCpp);

void printStackTrace(const Event *e,
                     std::map<TID, std::vector<CallEvent *>> &callEventTraces);
void printStackTrace(const Event *e,
                     std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                     bool isCpp);

void printStackTrace(const std::vector<std::string> &stackTrace);
void printCallEventStackTrace(std::vector<CallEvent *> &st);

void printSrcInfo(SourceInfo &srcInfo, TID tid);

} // namespace aser
