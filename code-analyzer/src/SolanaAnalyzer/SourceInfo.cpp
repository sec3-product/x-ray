#include "SourceInfo.h"

#include <sstream>
#include <string>
#include <vector>

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>

#include "Graph/Event.h"
#include "LogColor.h"

using namespace xray;

xray::SourceInfo::SourceInfo()
    : line(0), col(0), filename(""), dir(""), inaccurate(false), tyStr("") {}

xray::SourceInfo::SourceInfo(const llvm::Value *V, unsigned L, unsigned C,
                             std::string FN, std::string D, std::string N,
                             bool inaccurate, std::string tyStr)
    : val(V), line(L), col(C), filename(FN), dir(D), name(N),
      inaccurate(inaccurate), tyStr(tyStr) {
  // FIXME: potential performance issue?
  // read source file twice
  this->sourceLine = xray::getSourceLine(this->dir, this->filename, this->line);
  this->snippet = getCodeSnippet(this->dir, this->filename, this->line);
}

std::string xray::SourceInfo::str() {
  std::stringstream ss;
  ss << this->filename << "@" << this->line << ":" << this->col << "\n";
  ss << "Code snippet: "
     << "\n";
  // code snippet should already have a "\n" at the end
  ss << this->getSnippet();

  return ss.str();
}

std::string xray::SourceInfo::overview() const {
  std::stringstream ss;
  ss << "line " << this->line << ", column " << this->col << " in "
     << this->filename;

  return ss.str();
}

std::string xray::SourceInfo::sig() const {
  std::stringstream ss;
  ss << this->dir << "/" << this->filename << "@" << this->line << ":"
     << this->col;

  return ss.str();
}

/* ------------------------------------

      report related util functions

 ------------------------------------ */

std::string xray::getRawSourceLine(std::string directory, std::string filename,
                                   unsigned line) {
  // line number should be an integer greater or equal to 1
  if (line == 0)
    return "";

  std::string absPath;
  if (filename.front() ==
      '/') // filename is already absolute path, by old flang IR
    absPath = filename;
  else
    absPath = directory + "/" + filename;
  unsigned rest = line;

  std::ifstream input(absPath);
  std::stringstream raw;
  std::string ln;

  for (; std::getline(input, ln);) {
    --rest;
    if (rest == 0) {
      break;
    }
  }
  return ln;
}

void xray::getSourceLinesForSoteriaAnchorAccount(
    SourceInfo &srcInfo, std::vector<std::string> &out) {
  auto snippet = srcInfo.getSnippet();
  std::stringstream ss(snippet);
  std::string s;
  while (std::getline(ss, s, '\n')) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s.front() == '>')
      break;
    if (s.find("#[soteria(ignore") != std::string::npos)
      out.push_back(s);
    else if (s.find("/// check") != std::string::npos &&
             (s.find("cpi") != std::string::npos ||
              s.find("checked by") != std::string::npos))
      out.push_back(s);
    else if (s.find("|    pub ") != std::string::npos) {
      out.clear();
    }
  }
  // for (auto line : out) llvm::outs() <<
  // "getSourceLinesForSoteriaAnchorAccount line: " << line << "\n";
}

std::string xray::getSourceLinesForSoteria(SourceInfo &srcInfo,
                                           unsigned range) {
  std::vector<std::string> out;
  auto snippet = srcInfo.getSnippet();
  std::stringstream ss(snippet);
  std::string s;
  int index = -1;
  while (std::getline(ss, s, '\n')) {
    out.push_back(s);
    index++;
    if (s.front() == '>')
      break;
  }
  if (index < 0 || index > out.size() - 1)
    return "";
  auto res = out[index];
  if (index >= range) {
    res = out[index - range];
  }
  // llvm::outs() << "#[soteria(ignore)]: \n" << res << "\n";
  return res;
}

std::string xray::getSourceLine(std::string directory, std::string filename,
                                unsigned line) {
  std::stringstream raw;
  raw << " " << line << "|" << getRawSourceLine(directory, filename, line)
      << "\n";
  return raw.str();
}

std::string xray::getCodeSnippet(std::string directory, std::string filename,
                                 unsigned line) {
  return getCodeSnippet(directory, filename, line, 0);
}

// For now, always treat "col" as 0
// NOTE: by default the code snippet we show is 5 lines
// TODO: later we can highlight variable names based on col
std::string xray::getCodeSnippet(std::string directory, std::string filename,
                                 unsigned line, unsigned col) {
  return getCodeSnippet(directory, filename, line, col, 13);
}

// `length` --- number of lines for the code snippet, must be an odd number
std::string xray::getCodeSnippet(std::string directory, std::string filename,
                                 unsigned line, unsigned col, unsigned length) {
  assert(length % 2 == 1 && "length should be an odd number");
  // the span for the code snippet on each direction
  // e.g. if the length of the snippet is 5 lines
  // the span should be 2 (1 souce line in the middle, 2 lines before and after)
  unsigned span = (length - 1) / 2;
  return getCodeSnippet(directory, filename, line, col, span, span);
}

// `above` --- number of lines above the target line
// `below` --- number of lines below the target line
std::string xray::getCodeSnippet(std::string directory, std::string filename,
                                 unsigned line, unsigned col, unsigned above,
                                 unsigned below) {
  assert(above >= 0 && below >= 0);
  // source code file path
  std::string absPath;
  if (filename.front() ==
      '/') // filename is already absolute path, by old flang IR
    absPath = filename;
  else
    absPath = directory + "/" + filename;

  unsigned begin_line = line - above; // make sure positive or zero
  unsigned end_line = line + below;   // make sure positive or zero
  unsigned cur_line = 0;

  std::stringstream raw;
  std::ifstream input(absPath);

  // llvm::outs() << "absPath: " << absPath << " line: " << line << " above: "
  // << above << " below: " << below <<"\n";
  for (std::string ln; std::getline(input, ln, '\n');) {
    cur_line++;
    // llvm::outs() << "cur_line: " << cur_line << " ln: " << ln << "\n";
    if (cur_line >= begin_line) {
      if (cur_line == line) {
        raw << ">";
      } else {
        raw << " ";
      }
      raw << (cur_line) << "|" << ln << "\n";
    }

    if (cur_line >= end_line)
      break;
  }
  return raw.str();
}

unsigned xray::getAccurateCol(std::string directory, std::string filename,
                              unsigned line, unsigned col, bool isStore) {
  if (line == 0 || col == 0)
    return 0; // this condition is unclear
  if (isStore) {
    std::string srcLine = getRawSourceLine(directory, filename, line);
    // Example: "  x = 1"
    // the given col number is 5
    // the expected result is 3
    // we do it by looking backward
    bool found = false;
    for (col--; col > 0; col--) {
      auto c = srcLine[col - 1];
      if (isspace(c)) {
        if (found) {
          return col + 1;
        }
      } else {
        found = true;
      }
    }
  } else {
    // Since the accurate col number of loads
    // cannot be obtained by looking forwards.
    // Besides, some times llvm give correct col number of loads
    // therefore we directly return.
    return col;
  }
  return 0;
}

// Refer to the complete implementation below
// https://github.com/SVF-tools/SVF/blob/master/lib/Util/SVFUtil.cpp#L345
xray::SourceInfo xray::getSourceLoc(const Value *val) {
  if (val == NULL)
    return SourceInfo();

  unsigned line = 0;
  unsigned col = 0;
  bool inaccurate = false;
  std::string filename;
  std::string directory;
  // variable name
  // for now this will only be set for AllocaInst
  // so that we can know the shared variable name (if it's a stack variable)
  std::string name;
  std::string tyStr = "";

  if (const Instruction *inst = llvm::dyn_cast<Instruction>(val)) {
    // llvm::outs() << "getSourceLoc inst: " << *inst << "\n";

    if (DILocation *loc = inst->getDebugLoc()) {
      line = loc->getLine();
      // this col number is inaccurate
      col = loc->getColumn();
      filename = loc->getFilename().str();
      directory = loc->getDirectory().str();
      // get the accurate col number
      col =
          getAccurateCol(directory, filename, line, col, isa<StoreInst>(inst));

      llvm::SmallVector<DbgValueInst *, 4> dbgVals;
      findDbgValues(dbgVals, const_cast<llvm::Instruction *>(inst));
      for (auto &&DVI : dbgVals) {
        llvm::DIVariable *DIVar =
            llvm::cast<llvm::DIVariable>(DVI->getVariable());
        // line = DIVar->getLine();
        // col = 0;
        // filename = DIVar->getFilename().str();
        // directory = DIVar->getDirectory().str();
        name = DIVar->getName().str();
        auto ty = DIVar->getType();
        auto baseTyStr = ty->getName();
        tyStr = baseTyStr.str();
        // FIXME: not every case is handled for now,
        // but we should be able to recover the type from the source line in
        // most cases, so fix this if we encounter new corner cases.
        if (ty->getMetadataID() == llvm::DIType::DIDerivedTypeKind) {
          auto derivedTy = llvm::cast<llvm::DIDerivedType>(ty);
          auto baseTy = derivedTy->getBaseType();
          if (baseTy == nullptr) {
            baseTyStr = "void";
          } else {
            baseTyStr = baseTy->getName();
          }
          if (ty->getTag() == llvm::dwarf::DW_TAG_pointer_type) {
            tyStr = llvm::formatv("{0}*", baseTyStr).str();
          } else if (ty->getTag() == llvm::dwarf::DW_TAG_reference_type) {
            tyStr = llvm::formatv("{0}&", baseTyStr).str();
          }
        }
        break;
      }

      // llvm::outs() << "old directory: " << directory << " filename: " <<
      // filename << " line: " << line
      //              << " col: " << col << " old col:" << loc->getColumn() <<
      //              "\n";

      // //%13 = load i32, i32* @tscEmbedded, align 4, !dbg !35253, !tbaa !35251
      // // why file name and line number are both wrong, why?

      // if (DILocation *Loc = inst->getDebugLoc()) {
      //     unsigned Line = Loc->getLine();
      //     unsigned Col = Loc->getColumn();
      //     StringRef File = Loc->getFilename();
      //     StringRef Dir = Loc->getDirectory();
      //     bool ImplicitCode = Loc->isImplicitCode();
      //     llvm::outs() << "new directory: " << Dir << " filename: " << File
      //     << " line: " << Line
      //                  << " col: " << Col << " ImplicitCode: " <<
      //                  ImplicitCode << "\n";
      //     if (auto Loc2 = Loc->getInlinedAtScope()) {
      //         // JEFF
      //         llvm::outs() << "getInlinedAtScope: " << Dir << " filename: "
      //         << File << " line: " << Line
      //                      << " col: " << Col << " ImplicitCode: " <<
      //                      ImplicitCode << "\n";
      //     }
      // }
    } else if (isa<AllocaInst>(inst)) {
      // TODO: there must be other insts than AllocaInst
      // that can be a shared variable
      for (DbgInfoIntrinsic *DII :
           FindDbgAddrUses(const_cast<Instruction *>(inst))) {
        if (llvm::DbgDeclareInst *DDI =
                llvm::dyn_cast<llvm::DbgDeclareInst>(DII)) {
          llvm::DIVariable *DIVar =
              llvm::cast<llvm::DIVariable>(DDI->getVariable());
          line = DIVar->getLine();
          col = 0;
          filename = DIVar->getFilename().str();
          directory = DIVar->getDirectory().str();
          name = DIVar->getName().str();

          // llvm::outs() << "shared variable directory: " << directory << "
          // filename: " << filename
          //              << " line: " << line << " col: " << col << "\n";
          break;
        }
      }

      {
        // possible, we get here when the shared var is from external call
        // eg. %nthreads_326 = alloca i32, align 4
        // NTHREADS = OMP_GET_NUM_THREADS()
        // llvm::outs() << "shared variable DbgInfoIntrinsic: " << *DII << "\n";
      }
    }
    if (line == 0) {
      if (auto SI = dyn_cast<StoreInst>(inst)) {
        if (SI->getValueOperand()->hasName() &&
            SI->getValueOperand()->getName().equals("spec.store.select")) {
          // to overcome debug information break by simplifyCFG
          if (auto op = dyn_cast<SelectInst>(SI->getValueOperand())) {
            if (DILocation *loc = op->getDebugLoc()) {
              line = loc->getLine();
              // this col number is inaccurate
              col = loc->getColumn();
              filename = loc->getFilename().str();
              directory = loc->getDirectory().str();
              // get the accurate col number
              col = getAccurateCol(directory, filename, line, col,
                                   isa<StoreInst>(inst));
            }
          }
        }
        // can be due to inlinedAt, debug info broken
        //%var.promoted = load i32, i32* %var, align 4, !tbaa !33
        // store i32 %19, i32* @sum0, align 4, !dbg !83, !tbaa !21
        else if (DILocation *loc = inst->getDebugLoc()) {
          // if (auto Loc2 = loc->getInlinedAt())

          if (auto Loc2 = llvm::dyn_cast<llvm::DISubprogram>(loc->getScope())) {
            line = Loc2->getLine() + 2; // the second line after func decl
            filename = Loc2->getFilename().str();
            directory = Loc2->getDirectory().str();
          } else
          // if (auto scope = loc->getScope())
          {
            if (auto lexicalScope =
                    llvm::dyn_cast<llvm::DILexicalBlock>(loc->getScope())) {
              line =
                  lexicalScope->getLine() + 1; // the first line in the scope {}
              filename = lexicalScope->getFilename().str();
              directory = lexicalScope->getDirectory().str();
            }
          }
        }
      }
    }

    // JEFF: optimization breaks debug info, let's search the next inst, and
    // make sure it contains the share var... todo: this may lead to incorrect
    // line
    if (line == 0) {
      inaccurate = true; // use this to indicate this info needs correction
      // find the next inst that has dbg info
      auto nextInst = inst->getNextNonDebugInstruction(); // getNextNode
      while (nextInst && line == 0) {
        if (DILocation *loc = nextInst->getDebugLoc()) {
          line = loc->getLine();
          // this col number is inaccurate
          // col = loc->getColumn();
          filename = loc->getFilename().str();
          directory = loc->getDirectory().str();
          // get the accurate col number
          // col = getAccurateCol(directory, filename, line, col,
          // isa<StoreInst>(inst));
          break;
        }
        nextInst = nextInst->getNextNonDebugInstruction();
      }
    }

    if (line == 0) {
      // DEBUG
      // llvm::outs() << "tryharder: " << *inst << "\n dir: " << directory << "
      // filename: " << filename << "\n";
    }
  } else if (const GlobalVariable *gvar =
                 llvm::dyn_cast<llvm::GlobalVariable>(val)) {
    // llvm::outs() << "GlobalVariable inst: " << *val << "\n";

    // find the debuggin information for global variables
    llvm::NamedMDNode *CU_Nodes =
        gvar->getParent()->getNamedMetadata("llvm.dbg.cu");
    if (CU_Nodes) {
      // iterate over all the !DICompileUnit
      // each of whom has a field called globals to indicate all the debugging
      // info for global variables
      for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
        llvm::DICompileUnit *CUNode =
            llvm::cast<llvm::DICompileUnit>(CU_Nodes->getOperand(i));
        for (llvm::DIGlobalVariableExpression *GV :
             CUNode->getGlobalVariables()) {
          llvm::DIGlobalVariable *DGV = GV->getVariable();
          // DGV->getLinkageName() is the mangled name
          if (DGV->getName() == gvar->getName() ||
              DGV->getLinkageName() == gvar->getName()) {
            line = DGV->getLine();
            col = 0;
            filename = DGV->getFilename().str();
            directory = DGV->getDirectory().str();
            name = DGV->getName() == gvar->getName()
                       ? DGV->getName().str()
                       : DGV->getLinkageName().str();
            break;
          }
        }
      }
    }

    llvm::SmallVector<llvm::DIGlobalVariableExpression *, 1> GVs;
    gvar->getDebugInfo(GVs);
    for (auto const &gv : GVs) {
      auto var = gv->getVariable();
      line = var->getLine();
      col = 0;
      filename = var->getFilename().str();
      directory = var->getDirectory().str();
      name = var->getName().str();
      // llvm::outs() << "source loc value: " << *val << " varName: " << name <<
      // "\n";

      break;
    }

    if (line == 0) {
      // if we get here, probably something like PTA is wrong
      // TODO: skip races involving this case

      // uses of the global var
      // for (const User *U : val->users()) {
      //     if (auto inst = dyn_cast<Instruction>(U))
      //         if (MDNode *N = inst->getMetadata("dbg")) {
      //             // debug
      //             // llvm::outs() << "source loc value: " << *val << " Use: "
      //             << *U << "\n"; llvm::DILocation *loc =
      //             llvm::cast<llvm::DILocation>(N); line = loc->getLine();
      //             // this col number is inaccurate
      //             col = loc->getColumn();
      //             filename = loc->getFilename().str();
      //             directory = loc->getDirectory().str();
      //             break;
      //         }
      // }
    }
    // } else if (auto *gvar = llvm::dyn_cast<llvm::Constant>(val)) {
    //     llvm::outs() << "Constant inst: " << *val << "\n";

    // }  else if (auto *gvar = llvm::dyn_cast<llvm::Operator>(val)) {
    //     llvm::outs() << "Operator inst: " << *val << "\n";

    // } else if (auto *gvar = llvm::dyn_cast<llvm::Argument>(val)) {
    //     llvm::outs() << "Argument inst: " << *val << "\n";

    // } else if (auto *gvar = llvm::dyn_cast<llvm::MetadataAsValue>(val)) {
    //     llvm::outs() << "MetadataAsValue inst: " << *val << "\n";
    // } else  {
    //     llvm::outs() << "DerivedUser inst: " << *val << "\n";
    //(auto *gvar = llvm::dyn_cast<llvm::DerivedUser>(val))
  } else if (auto fun = dyn_cast<Function>(val)) {
    DISubprogram *DI = fun->getSubprogram();
    if (DI != nullptr) {
      line = DI->getLine();
      col = 0;
      filename = DI->getFilename().str();
      directory = DI->getDirectory().str();
      name = "Argument passed to " + DI->getName().str();
    }
  }

  name = demangle(name); // demangle share variable name
  return SourceInfo(val, line, col, filename, directory, name, inaccurate,
                    tyStr);
}

void xray::findCorrectSourceInfo(SourceInfo &srcInfo1, std::string &varName) {
  // let's search line after and before
  for (int k = 1; k <= 3; k++) {
    auto srcLine1 = getRawSourceLine(srcInfo1.getDir(), srcInfo1.getFilename(),
                                     srcInfo1.getLine() + k);
    llvm::outs() << "RawSourceLine+k: " << srcLine1 << " k: " << k << "\n";

    if (srcLine1.find(varName) != std::string::npos) {
      llvm::outs() << "find correct RawSourceLine: " << srcLine1
                   << " shared var: " << varName << "\n";
      srcInfo1.setCorrectLine(srcInfo1.getLine() + k);
      srcInfo1.setCorrectSourceLine(srcLine1);
      return;
    }

    srcLine1 = getRawSourceLine(srcInfo1.getDir(), srcInfo1.getFilename(),
                                srcInfo1.getLine() - k);
    llvm::outs() << "RawSourceLine-k: " << srcLine1 << " k: " << k << "\n";

    if (srcLine1.find(varName) != std::string::npos) {
      llvm::outs() << "find correct RawSourceLine: " << srcLine1
                   << " shared var: " << varName << "\n";
      srcInfo1.setCorrectLine(srcInfo1.getLine() + k);
      srcInfo1.setCorrectSourceLine(srcLine1);
      return;
    }
  }
}

void xray::tryCorrectSourceInfo(SourceInfo &srcInfo1, SourceInfo &srcInfo2,
                                SourceInfo &sharedObjLoc) {
  // handle case either srcInfo1 or srcInfo2 has line 0, but not both
  if (srcInfo1.getLine() == 0 || srcInfo2.getLine() == 0 ||
      sharedObjLoc.getLine() == 0) {
    auto varName = sharedObjLoc.getName();
    auto v1 = srcInfo1.getValue();
    auto v2 = srcInfo2.getValue();
    llvm::outs() << "tryCorrectSourceInfo v1: " << *v1 << " v2: " << *v2
                 << " shared var: " << varName << "\n";

    if (const Instruction *inst = llvm::dyn_cast<Instruction>(v1)) {
    }
    if (const Instruction *inst = llvm::dyn_cast<Instruction>(v2)) {
    }
    if (!varName.empty()) {
      std::string srcLine1 = getRawSourceLine(
          srcInfo1.getDir(), srcInfo1.getFilename(), srcInfo1.getLine());
      std::string srcLine2 = getRawSourceLine(
          srcInfo2.getDir(), srcInfo2.getFilename(), srcInfo2.getLine());
      llvm::outs() << "RawSourceLine1: " << srcLine1
                   << " shared var: " << varName << "\n";
      llvm::outs() << "RawSourceLine2: " << srcLine2
                   << " shared var: " << varName << "\n";
      if (srcLine1.find(varName) == std::string::npos) {
        findCorrectSourceInfo(srcInfo1, varName);
      }
      if (srcLine2.find(varName) == std::string::npos) {
        findCorrectSourceInfo(srcInfo2, varName);
      }
    }
  }
}

// a source-level race pair signature based on source code Information
std::string xray::getRaceSrcSig(SourceInfo &srcInfo1, SourceInfo &srcInfo2) {
  std::stringstream ss;
  auto sig1 = srcInfo1.sig();
  auto sig2 = srcInfo2.sig();
  if (sig1.compare(sig2) <= 0) {
    ss << sig1 << "|" << sig2;
  } else {
    ss << sig2 << "|" << sig1;
  }
  return ss.str();
}

// a file-level race pair signature based on source code information
// used for AGREESIVE filtering
std::string xray::getRaceFileSig(SourceInfo &srcInfo1, SourceInfo &srcInfo2) {
  std::stringstream ss;
  auto f1 = srcInfo1.getFilename();
  auto f2 = srcInfo1.getFilename();
  if (f1.compare(f2) <= 0) {
    ss << f1 << "|" << f2;
  } else {
    ss << f2 << "|" << f1;
  }
  return ss.str();
}

std::string xray::getRaceMethodSig(Event *e1, Event *e2, const ObjTy *obj) {
  auto m1 = e1->getInst()->getFunction()->getName().str();
  auto m2 = e2->getInst()->getFunction()->getName().str();
  std::stringstream ss;
  if (m1.compare(m2) <= 0) {
    ss << m1 << "|" << m2;
  } else {
    ss << m2 << "|" << m1;
  }
  // ADD OBJECT ID such that the same method pair but race on different object
  // will be reported ss << "|" << obj->getObjectID();
  return ss.str();
}

std::string xray::getRaceRawLineSig(SourceInfo &srcInfo1,
                                    SourceInfo &srcInfo2) {
  auto raw1 = getRawSourceLine(srcInfo1.getDir(), srcInfo1.getFilename(),
                               srcInfo1.getLine());
  auto raw2 = getRawSourceLine(srcInfo2.getDir(), srcInfo2.getFilename(),
                               srcInfo2.getLine());
  // trim the left and right side spaces
  // NOTE: this could be better
  // if we have
  // xxx; VERSUS``
  // xxx ;
  // then it will be considered as 2 different command
  raw1 = trim(raw1);
  raw2 = trim(raw2);
  // NOTE: when source code cannot be found
  // return an empty string
  if (raw1.empty() || raw2.empty()) {
    return "";
  }
  std::stringstream ss;
  if (raw1.compare(raw2) <= 0) {
    ss << raw1 << "|" << raw2;
  } else {
    ss << raw2 << "|" << raw1;
  }
  return ss.str();
}

std::string xray::getRaceRawLineSig(const SourceInfo &srcInfo) {
  auto raw = getRawSourceLine(srcInfo.getDir(), srcInfo.getFilename(),
                              srcInfo.getLine());
  return trim(raw);
}

// NOTE: https://regexr.com/4udrd
// NOTE: in PCRE format
static llvm::Regex
    tidPattern("\\[?\\s*(tid|thread_num)\\s*(\\]|([+\\-\\*\\/\\<>=]=?)|$|\\s+)",
               Regex::IgnoreCase);

// FIXME: don't EVER learn from this
bool xray::filterStrPattern(std::string src) {
  if (tidPattern.match(src)) {
    LOG_DEBUG("Source line filtered by string pattern. line={}", src);
    return true;
  }
  return false;
}

bool xray::customizedFilterSoteriaIgnoreSymbol(const Event *e,
                                               const std::string symbol) {
  SourceInfo srcInfo1 = getSourceLoc(e->getInst());
  auto sig0 = "#[soteria(";
  auto sig1 = "ignore_" + symbol;

  // e.g. SKIP #[soteria(ignore_signer...)]
  // auto line = getSourceLinesForSoteria(srcInfo1, 1);
  // if (line.find(sig0) != std::string::npos) {
  //     if (line.find(sig1) != std::string::npos) {
  //         return true;
  //     } else {
  //         auto line2 = getSourceLinesForSoteria(srcInfo1, 2);
  //         if (line2.find(sig0) != std::string::npos && line2.find(sig1) !=
  //         std::string::npos) return true;
  //     }
  // }
  std::vector<std::string> lines;
  getSourceLinesForSoteriaAnchorAccount(srcInfo1, lines);
  for (auto line : lines) {
    if (line.find(sig0) != std::string::npos &&
        line.find(sig1) != std::string::npos) {
      return true;
    }
    // else if (line.find("#[soteria(ignore)]") != std::string::npos) {
    //     return true;
    // }
  }
  return false;
}

bool xray::customizedFilterSoteriaIgnoreFullSymbol(const Event *e) {
  SourceInfo srcInfo = getSourceLoc(e->getInst());

  std::vector<std::string> lines;
  auto snippet = srcInfo.getSnippet();
  std::stringstream ss(snippet);
  std::string s;
  while (std::getline(ss, s, '\n')) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    // if (s.find(">313|") != std::string::npos) {
    //     llvm::outs() << snippet << "\n";
    //     llvm::outs() << lines.size() << "\n";
    // }
    if (s.front() == '>')
      break;
    if (s.find("#[soteria(ignore") != std::string::npos)
      lines.push_back(s);
    else if (s.find("/// check") != std::string::npos &&
             (s.find("cpi") != std::string::npos ||
              s.find("checked by") != std::string::npos))
      lines.push_back(s);
    else if (s.find("    //") != std::string::npos) {
      lines.clear();
    }
  }
  if (lines.size() > 0)
    return true;

  return false;
}

bool xray::customizedFilterIgnoreLocations(const Event *e1, const Event *e2) {
  if (customizedFilterSoteriaIgnoreFullSymbol(e1) ||
      customizedFilterSoteriaIgnoreFullSymbol(e2)) {
    return true;
  }
  return false;
}

int xray::customizedPriorityAdjust(int P, std::string name,
                                   SourceInfo &srcInfo1, SourceInfo &srcInfo2,
                                   std::vector<std::string> &st1,
                                   std::vector<std::string> &st2) {
  std::string file1 = srcInfo1.getFilename();
  std::string file2 = srcInfo2.getFilename();
  std::transform(file1.begin(), file1.end(), file1.begin(), ::tolower);
  std::transform(file2.begin(), file2.end(), file2.begin(), ::tolower);

  // TODO: improve performance for repeately creating glob patterns

  // return if name is empty
  if (name.empty()) {
    return P;
  }
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);

  // if races involving main has low priority
  if ((!st1.empty() && st1.front() == "main") ||
      (!st2.empty() && st2.front() == "main"))
    return 1;

  //"init" reduce priority
  for (auto &m : st1) {
    std::transform(m.begin(), m.end(), m.begin(), ::tolower);
    if (m.find("init") != std::string::npos) {
      P--;
      break;
    }
  }
  for (auto &m : st2) {
    if (m.find("init") != std::string::npos) {
      P--;
      break;
    }
  }

  return P;
}

// JEFF to Yanze: the following filter can be too strong
// FIXME: this API may be simplified
bool xray::customizedFilter(Event *e1, Event *e2, std::vector<std::string> &st1,
                            std::vector<std::string> &st2,
                            std::vector<std::string> &ignoreRaceInFun) {
  for (auto &str : ignoreRaceInFun) {
    llvm::Regex rg(str);
    for (auto &m : st1) {
      if (rg.match(m))
        return true;
    }
    for (auto &m : st2) {
      if (rg.match(m))
        return true;
    }
  }
  return false;
}

bool xray::customizedOMPFilter(Event *e1, Event *e2,
                               std::vector<std::string> &st1,
                               std::vector<std::string> &st2,
                               const std::vector<std::string> &callingCtx,
                               std::vector<std::string> &ignoreRaceInFun) {
  auto filterResult = customizedFilter(e1, e2, st1, st2, ignoreRaceInFun);
  if (filterResult) {
    return true;
  } else {
    for (auto &str : ignoreRaceInFun) {
      llvm::Regex rg(str);
      for (auto &m : callingCtx) {
        if (rg.match(m))
          return true;
      }
    }
  }
  return false;
}

std::vector<std::string> xray::getCallingCtx(CallingCtx &callingCtx,
                                             bool isCpp) {
  std::vector<std::string> result;

  for (CallEvent *ce : callingCtx.first) {
    if (ce->getEndID() == 0) {
      result.push_back(ce->getCallSiteString(isCpp));
    }
  }

  return result;
}

std::vector<std::string>
xray::getStackTrace(const Event *e,
                    std::map<TID, std::vector<CallEvent *>> &callEventTraces) {
  return getStackTrace(e, callEventTraces, false);
}

std::vector<std::string>
xray::getStackTrace(const Event *e,
                    std::map<TID, std::vector<CallEvent *>> &callEventTraces,
                    bool isCpp) {
  std::vector<std::string> result;
  TID tid = e->getTID();
  EventID id = e->getID();

  auto &callTrace = callEventTraces[tid];
  for (CallEvent *ce : callTrace) {
    if (ce->getID() > id)
      break;
    if (ce->getEndID() > id) {
      result.push_back(ce->getCallSiteString(isCpp));
    }
  }
  //! 116743 = !DILocation(line: 96, column: 24, scope: !116721, inlinedAt:
  //! !116726)
  // if inst is inlined, add the CallSiteString
  if (DILocation *Loc = e->getInst()->getDebugLoc()) {
    if (auto Loc2 = Loc->getInlinedAt()) {
      auto inlinedCallStrs = e->getInlinedCallSiteStrings(isCpp);
      if (!inlinedCallStrs.empty()) {
        for (auto csStr : inlinedCallStrs)
          result.push_back(csStr);
      }
      // llvm::outs() << "getInlinedCallSiteString: " << inlinedCallStr << "
      // inst: " << *e->getInst() << "\n";
    }
  }
  return result;
}

std::vector<CallEvent *> xray::getCallEventStack(
    const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces) {
  std::vector<CallEvent *> result;
  TID tid = e->getTID();
  EventID id = e->getID();

  auto &callTrace = callEventTraces[tid];
  for (CallEvent *ce : callTrace) {
    if (ce->getID() > id)
      break;
    if (ce->getEndID() > id || ce->getEndID() == 0) {
      result.push_back(ce);
    }
  }

  return result;
}

void xray::getCallEventStackUntilMain(
    const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces,
    std::vector<CallEvent *> &result) {
  auto ces = getCallEventStack(e, callEventTraces);
  // insert vector to the front
  result.insert(result.begin(), ces.begin(), ces.end());
  auto e1 = StaticThread::getThreadByTID(e->getTID())->getParentEvent();
  if (e1) {
    getCallEventStackUntilMain(e1, callEventTraces, result);
  }
}

std::vector<CallEvent *> xray::getCallEventStackUntilMain(
    const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces) {
  std::vector<CallEvent *> result;
  getCallEventStackUntilMain(e, callEventTraces, result);
  return result;
}

void xray::printStackTrace(
    const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces,
    bool isCpp) {
  std::vector<std::string> st = getStackTrace(e, callEventTraces, isCpp);
  int count = 0;
  for (auto it = st.begin(); it != st.end(); ++it) {
    llvm::outs() << ">>>" << std::string(count * 2, ' ') << *it << "\n";
    count++;
  }
}

void xray::printStackTrace(
    const Event *e, std::map<TID, std::vector<CallEvent *>> &callEventTraces) {
  xray::printStackTrace(e, callEventTraces, false);
}

bool stringStartsWithAny(std::string s, std::vector<std::string> substrs) {
  for (auto const &substr : substrs) {
    // s starts with substr
    if (s.rfind(substr, 0) == 0) {
      return true;
    }
  }
  return false;
}

void xray::printStackTrace(const std::vector<std::string> &stackTrace) {
  int count = 0;
  for (auto it = stackTrace.begin(); it != stackTrace.end(); ++it) {
    auto content = *it;
    if (stringStartsWithAny(content, {"std::_", "std::thread::_", ".coderrect.",
                                      "__coderrect", "coderrect_stub"})) {
      continue;
    }
    llvm::outs() << ">>>" << std::string(count * 2, ' ') << *it << "\n";
    count++;
  }
}

void xray::printCallEventStackTrace(std::vector<CallEvent *> &st) {
  int count = 0;
  for (auto it = st.begin(); it != st.end(); ++it) {
    llvm::outs() << ">>>" << std::string(count * 2, ' ')
                 << (*it)->getCallSiteString(true) << "\n";
    count++;
  }
}
