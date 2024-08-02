#ifndef O2_THREADPROFILEREWRITER_H
#define O2_THREADPROFILEREWRITER_H

#include <llvm/ADT/StringRef.h>
#include <jsoncons/json.hpp>
#include <utility>

namespace llvm {
class Module;
}

template <class Json, class T>
T Get(const Json &val, const std::string &key, T defaultVal) {
    auto it = val.find(key);
    if (it == val.object_range().end()) {
        return defaultVal;
    }

    return it->value().template as<T>();
}

namespace o2 {

class AtomicAPI {
private:
    std::vector<std::string> readAPIs;
    std::vector<std::string> writeAPIs;
    std::vector<std::string> casAPIs;
    std::vector<std::string> xchgAPIs;

public:
    inline void init(std::map<std::string, std::vector<std::string>> &APIMap) {
        for (std::pair<const std::string, std::vector<std::string>> &item : APIMap) {
            if (item.first == "read") {
                readAPIs = std::move(item.second);
            } else if (item.first == "write") {
                writeAPIs = std::move(item.second);
            } else if (item.first == "cas") {
                casAPIs = std::move(item.second);
            } else if (item.first == "xchg") {
                xchgAPIs = std::move(item.second);
            }
        }
    }

    inline const std::vector<std::string> &getReadAPI() const { return readAPIs; }
    inline const std::vector<std::string> &getWriteAPI() const { return writeAPIs; }
    inline const std::vector<std::string> &getCASAPI() const { return casAPIs; }
    inline const std::vector<std::string> &getXCHGAPI() const { return xchgAPIs; }
};

class LockAllocator {
private:
    int addrTakenArgIdx;
    std::string lockAllocator;

public:
    LockAllocator() : addrTakenArgIdx(-1) {}

    LockAllocator(int addrArg, std::string name)
        : addrTakenArgIdx(addrArg), lockAllocator(std::move(name)) {}

    int getAddrTakenArgIdx() const { return addrTakenArgIdx; }
    const std::string &getLockAllocatorName() const { return lockAllocator; }
};

class ReadWriteLockAPI {
private:
    std::vector<std::string> readLockAPIs;
    std::vector<std::string> writeLockAPIs;
    std::vector<std::string> unlockAPIs;

    LockAllocator allocator;
public:

    template <typename Json>
    inline void init(const Json &val) {
        // join and exit are the same
        readLockAPIs = Get<Json, std::vector<std::string>>(val, "readLock", {});
        writeLockAPIs = Get<Json, std::vector<std::string>>(val, "writeLock", {});
        unlockAPIs = Get<Json, std::vector<std::string>>(val, "unlock", {});

        allocator = Get<Json, LockAllocator>(val, "lockAllocator", LockAllocator());
    }

    const std::vector<std::string> &getReadLockAPIs() const { return readLockAPIs; }
    const std::vector<std::string> &getWriteLockAPIs() const { return writeLockAPIs; }
    const std::vector<std::string> &getUnlockAPIs() const { return unlockAPIs; }

    bool hasLockAllocator() const {
        return allocator.getAddrTakenArgIdx() >= 0;
    }

    const LockAllocator &getLockAllocator() const {
        return allocator;
    }

};

class MutexAPI {
private:
    std::vector<std::string> lockAPIs;
    std::vector<std::string> unlockAPIs;

    LockAllocator allocator;
public:
    template <typename Json>
    inline void init(const Json &val) {
        // join and exit are the same
        lockAPIs = Get<Json, std::vector<std::string>>(val, "lock", {});
        unlockAPIs = Get<Json, std::vector<std::string>>(val, "unlock", {});
        //createAPIs = Get<Json, std::vector<ThreadCreateAPI>>(val, "create", {});
        allocator = Get<Json, LockAllocator>(val, "lockAllocator", LockAllocator());
    }

    const std::vector<std::string> &getLockAPIs() const { return lockAPIs; }
    const std::vector<std::string> &getUnlockAPIs() const { return unlockAPIs; }

    bool hasLockAllocator() const {
        return allocator.getAddrTakenArgIdx() >= 0;
    }

    const LockAllocator &getLockAllocator() const {
        return allocator;
    }

};

class ConditionVariableAPI {
private:
    std::vector<std::string> waitAPIs;
    std::vector<std::string> signalAPIs;
public:
    inline void init(std::map<std::string, std::vector<std::string>> &APIMap) {
        for (std::pair<const std::string, std::vector<std::string>> &item : APIMap) {
            if (item.first == "wait") {
                waitAPIs = std::move(item.second);
            } else if (item.first == "signal") {
                signalAPIs = std::move(item.second);
            }
        }
    }

    const std::vector<std::string> &getWaitAPIs() const { return waitAPIs; }
    const std::vector<std::string> &getSignalAPIs() const { return signalAPIs; }
};

class ThreadCreateAPI {
private:
    int argIdx;
    int entryIdx;
    // used in C, if the callback function in apr library
    // callback(apr_thread_t *handle, void *data); we care about the argument at index 2 (void *data)
    int cbArgIdx;
    // only used in CXX Thread Create API
    // if the function is a non static member function (thus the first argument is *this*)
    bool nonStaticAPI;
    std::string functionName;
public:
    ThreadCreateAPI(std::string name, int entryIdx, int argIdx, int cbArgIdx)
        : functionName(std::move(name)), argIdx(argIdx), entryIdx(entryIdx), cbArgIdx(cbArgIdx), nonStaticAPI(false) {}

    ThreadCreateAPI(std::string name, int entryIdx, int argIdx, bool isNonStatic)
        : functionName(std::move(name)), argIdx(argIdx), entryIdx(entryIdx), cbArgIdx(-1), nonStaticAPI(isNonStatic) {}

    int getArgIdx() const { return argIdx; }
    int getEntryIdx() const { return entryIdx; }
    int getcallBackArgIdx() const { return cbArgIdx; }
    bool isNonStaticAPI() const { return nonStaticAPI; }

    std::string getFunctionName() const {
        if (isCtor() || isDtor()) {
            size_t pos = functionName.rfind("::$");
            return functionName.substr(0, pos);
        }
        return functionName;
    }

    bool isCtor() const {
        return ((llvm::StringRef)functionName).endswith("::$constructor");
    }

    bool isDtor() const {
        return ((llvm::StringRef)functionName).endswith("::$destructor");
    }
};

class ThreadAPI {
private:
    std::vector<ThreadCreateAPI> createAPIs;
    std::vector<std::string> joinAPIs;
    std::vector<std::string> exitAPIs;
public:
    template <typename Json>
    inline void init(const Json &val) {
        // join and exit are the same
        joinAPIs = Get<Json, std::vector<std::string>>(val, "join", {});
        exitAPIs = Get<Json, std::vector<std::string>>(val, "exit", {});
        createAPIs = Get<Json, std::vector<ThreadCreateAPI>>(val, "create", {});
    }

    const std::vector<ThreadCreateAPI> &getCreateAPIs() const { return createAPIs; }
    const std::vector<std::string> &getJoinAPIs() const { return joinAPIs; }
    const std::vector<std::string> &getExitAPIs() const { return exitAPIs; }
};

class ThreadProfile {
private:
    bool CXXProfile;
    AtomicAPI atomicAPI;
    ReadWriteLockAPI rwLockAPI;
    ConditionVariableAPI conVarAPI;
    MutexAPI mutexAPI;
    ThreadAPI threadAPI;

public:
    explicit ThreadProfile(bool isCXXProfile) : CXXProfile(isCXXProfile) {} ;
    inline void initAtomicAPI(std::map<std::string, std::vector<std::string>> &APIMap) { atomicAPI.init(APIMap); }
    inline void initConVarAPI(std::map<std::string, std::vector<std::string>> &APIMap) { conVarAPI.init(APIMap); }

    template <typename Json>
    inline void initRWLockAPI(const Json &val) {
        rwLockAPI.template init(val);
    }

    template <typename Json>
    inline void initMutexAPI(const Json &val) {
        mutexAPI.template init(val);
    }

    template <typename Json>
    inline void initThreadAPI(const Json &val) {
        threadAPI.template init<Json>(val);
    }

    bool isCXXProfile() const { return CXXProfile; }
    const AtomicAPI &getAtomicAPI() const { return atomicAPI; }
    const ReadWriteLockAPI &getReadWriteLockAPI() const { return rwLockAPI; }
    const ConditionVariableAPI &getConditionVariableAPI() const { return conVarAPI; }
    const MutexAPI &getMutexAPI() const { return mutexAPI; }
    const ThreadAPI &getThreadAPI() const { return threadAPI; }
};

class ThreadProfileRewriter {
public:
    static void rewriteModule(llvm::Module *M);
};

}  // namespace o2

namespace jsoncons {

template <class Json>
struct json_type_traits<Json, o2::ThreadProfile> {
    static bool is(const Json &val) noexcept {
        // unsupported
        std::abort();
    }

    static Json to_json(o2::ThreadProfile val) {
        // unsupported
        std::abort();
    }

    static o2::ThreadProfile as(const Json &val) {
        auto isCXXProfile = Get<Json, bool>(val, "CXXLibrary", false);
        o2::ThreadProfile profile(isCXXProfile);

        auto atomicAPIMap = Get<Json, std::map<std::string, std::vector<std::string>>>(val, "atomic", {});
        profile.initAtomicAPI(atomicAPIMap);

        auto conVarAPIMap = Get<Json, std::map<std::string, std::vector<std::string>>>(val, "conditionVariable", {});
        profile.initConVarAPI(conVarAPIMap);

//        auto mutexAPIMap = Get<Json, std::map<std::string, std::vector<std::string>>>(val, "mutex", {});
//        profile.initMutexAPI(mutexAPIMap);
//
//        auto rwLockAPIMap = Get<Json, std::map<std::string, std::vector<std::string>>>(val, "readWriteLock", {});
//        profile.initRWLockAPI(rwLockAPIMap);

        auto it = val.find("mutex");
        if (it != val.object_range().end()) {
            profile.initMutexAPI(it->value());
        }

        it = val.find("readWriteLock");
        if (it != val.object_range().end()) {
            profile.initRWLockAPI(it->value());
        }

        it = val.find("thread");
        if (it != val.object_range().end()) {
            profile.initThreadAPI(it->value());
        }

        // TODO: thread pool support...
        return profile;
    }
};

template <class Json>
struct json_type_traits<Json, o2::ThreadCreateAPI> {
    static bool is(const Json &val) noexcept {
        // unsupported
        std::abort();
    }

    static Json to_json(o2::ThreadProfile val) {
        // unsupported
        std::abort();
    }

    static o2::ThreadCreateAPI as(const Json &val) {
        int argStart = Get<Json, int>(val, "argStartIdx", -1);
        if (argStart < 0) {
            // this is a C thread create API
            std::string entry = Get<Json, std::string>(val, "function", "");
            int entryIdx = Get<Json, int>(val, "entryPoint", -1);
            int argIdx = Get<Json, int>(val, "arg", -1);
            if (entryIdx < 0 || argIdx < 0 || entry.empty()) {
                std::cerr << "wrong format!";
                std::abort();
            }
            int cbArgIdx = Get<Json, int>(val, "callbackArg", 1);
            return o2::ThreadCreateAPI(entry, entryIdx, argIdx, cbArgIdx);
        } else {
            // TODO:
            // this is a C++ thread create API
            std::string entry = Get<Json, std::string>(val, "function", "");
            int entryIdx = Get<Json, int>(val, "entryPoint", -1);
            int argIdx = Get<Json, int>(val, "argStartIdx", -1);
            if (entryIdx < 0 || argIdx < 0 || entry.empty()) {
                std::cerr << "wrong format!";
                std::abort();
            }
            bool isNonStatic = Get<Json, bool>(val, "isNonStaticMethod", true);
            return o2::ThreadCreateAPI(entry, entryIdx, argIdx, isNonStatic);
        }
    }
};

template <class Json>
struct json_type_traits<Json, o2::LockAllocator> {
    static bool is(const Json &val) noexcept {
        // unsupported
        std::abort();
    }

    static Json to_json(o2::ThreadProfile val) {
        // unsupported
        std::abort();
    }

    static o2::LockAllocator as(const Json &val) {
        auto argStart = Get<Json, int>(val, "arg", 0); // 0 means return value catched the address
        auto funName = Get<Json, std::string>(val, "name", "");

        return o2::LockAllocator(argStart, funName);
    }
};

}

#endif  // O2_THREADPROFILEREWRITER_H
