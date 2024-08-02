#ifndef O2_THREADAPIREWRITER_H
#define O2_THREADAPIREWRITER_H

#include <llvm/ADT/StringRef.h>

#include "CustomAPIRewriters/ThreadProfileRewriter.h"

namespace llvm { class Module; }
namespace o2 {

class ThreadAPIRewriter {
public:
    static void rewriteModule(llvm::Module *M, const std::map<std::string, ThreadProfile> &profiles);

    inline static llvm::StringRef getCanonicalizedAPIPrefix() {
        return ".o2.thread.create.";
    }

    inline static llvm::StringRef getStandardCThreadCreateAPI() {
        return ".o2.thread.create.C";
    }
};

}


#endif  // O2_THREADAPIREWRITER_H
