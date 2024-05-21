//
// Created by peiming on 9/15/20.
//

#ifndef ASER_PTA_THREADAPIREWRITER_H
#define ASER_PTA_THREADAPIREWRITER_H

#include <llvm/ADT/StringRef.h>

#include "CustomAPIRewriters/ThreadProfileRewriter.h"

namespace llvm { class Module; }
namespace aser {

class ThreadAPIRewriter {
public:
    static void rewriteModule(llvm::Module *M, const std::map<std::string, ThreadProfile> &profiles);

    inline static llvm::StringRef getCanonicalizedAPIPrefix() {
        return ".coderrect.thread.create.";
    }

    inline static llvm::StringRef getStandardCThreadCreateAPI() {
        return ".coderrect.thread.create.C";
    }
};

}


#endif  // ASER_PTA_THREADAPIREWRITER_H
