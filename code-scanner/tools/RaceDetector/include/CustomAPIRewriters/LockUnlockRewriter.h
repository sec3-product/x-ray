//
// Created by peiming on 9/15/20.
//

#ifndef ASER_PTA_LOCKUNLOCKREWRITER_H
#define ASER_PTA_LOCKUNLOCKREWRITER_H

#include <llvm/ADT/StringRef.h>

#include "ThreadProfileRewriter.h"

namespace llvm { class Module; }

namespace aser {

class LockUnlockRewriter {
public:
    static void rewriteModule(llvm::Module *M, const std::map<std::string, ThreadProfile> &profiles);
    inline static llvm::StringRef getLockAllocateName() {
        return ".coderrect.lock.allocate";
    }

    inline static llvm::StringRef getCanonicalizedLockName() {
        return ".coderrect.mutex.lock";
    }

    inline static llvm::StringRef getCanonicalizedUnlockName() {
        return ".coderrect.mutex.unlock";
    }

    inline static llvm::StringRef getCanonicalizedRAIILockName() {
        return ".coderrect.RAII.mutex.lock";
    }

    inline static llvm::StringRef getCanonicalizedRAIIUnlockName() {
        return ".coderrect.RAII.mutex.unlock";
    }

    inline static llvm::StringRef getCanonicalizedRAIIRdLockName() {
        return ".coderrect.RAII.rdlock.lock";
    }

    inline static llvm::StringRef getCanonicalizedRAIIWrLockName() {
        return ".coderrect.RAII.wrlock.lock";
    }

    inline static llvm::StringRef getCanonicalizedRAIIRwUnLockName() {
        return ".coderrect.RAII.rwlock.unlock";
    }

    inline static llvm::StringRef getCanonicalizedRdLockName() {
        return ".coderrect.rdlock.lock";
    }

    inline static llvm::StringRef getCanonicalizedWrLockName() {
        return ".coderrect.wrlock.lock";
    }

    inline static llvm::StringRef getCanonicalizedRwULockName() {
        return ".coderrect.rwlock.unlock";
    }

    inline static llvm::StringRef getCanonicalizedSignalName() {
        return ".coderrect.cv.signal";
    }

    inline static llvm::StringRef getCanonicalizedSemaphoreWaitName() {
        return ".coderrect.semaphore.wait";
    }

    inline static llvm::StringRef getCanonicalizedSemaphorePostName() {
        return ".coderrect.semaphore.post";
    }

    inline static llvm::StringRef getCanonicalizedSignalWaitName() {
        return ".coderrect.cv.signal.wait";
    }

    inline static llvm::StringRef getCanonicalizedSignalBroadcastName() {
        return ".coderrect.cv.signal.broadcast";
    }
};

}



#endif  // ASER_PTA_LOCKUNLOCKREWRITER_H
