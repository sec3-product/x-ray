//
// Created by peiming on 9/15/20.
//

#ifndef ASER_PTA_HEAPALLOCAPIREWRITER_H
#define ASER_PTA_HEAPALLOCAPIREWRITER_H

#include <llvm/ADT/StringRef.h>

namespace llvm { class Module; }

namespace aser {

class HeapAllocAPIRewriter {
public:
    static void rewriteModule(llvm::Module *M);

    static inline llvm::StringRef getCanonicalizedAPIName() {
        return ".coderrect.allocation.api";
    }
};

}

#endif  // ASER_PTA_HEAPALLOCAPIREWRITER_H
