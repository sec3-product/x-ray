#ifndef O2_LOCKUNLOCKREWRITER_H
#define O2_LOCKUNLOCKREWRITER_H

#include <llvm/ADT/StringRef.h>

#include "ThreadProfileRewriter.h"

namespace llvm { class Module; }

namespace o2 {

class LockUnlockRewriter {
public:
    static void rewriteModule(llvm::Module *M, const std::map<std::string, ThreadProfile> &profiles);
    inline static llvm::StringRef getLockAllocateName() {
        return ".o2.lock.allocate";
    }

    inline static llvm::StringRef getCanonicalizedLockName() {
        return ".o2.mutex.lock";
    }

    inline static llvm::StringRef getCanonicalizedUnlockName() {
        return ".o2.mutex.unlock";
    }

    inline static llvm::StringRef getCanonicalizedRdLockName() {
        return ".o2.rdlock.lock";
    }

    inline static llvm::StringRef getCanonicalizedWrLockName() {
        return ".o2.wrlock.lock";
    }

    inline static llvm::StringRef getCanonicalizedRwULockName() {
        return ".o2.rwlock.unlock";
    }
};

}



#endif  // O2_LOCKUNLOCKREWRITER_H
