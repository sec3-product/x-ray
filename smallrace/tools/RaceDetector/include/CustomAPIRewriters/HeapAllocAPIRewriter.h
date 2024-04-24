#ifndef O2_HEAPALLOCAPIREWRITER_H
#define O2_HEAPALLOCAPIREWRITER_H

#include <llvm/ADT/StringRef.h>

namespace llvm { class Module; }

namespace o2 {

class HeapAllocAPIRewriter {
public:
    static void rewriteModule(llvm::Module *M);

    static inline llvm::StringRef getCanonicalizedAPIName() {
        return ".o2.allocation.api";
    }
};

}

#endif  // O2_HEAPALLOCAPIREWRITER_H
