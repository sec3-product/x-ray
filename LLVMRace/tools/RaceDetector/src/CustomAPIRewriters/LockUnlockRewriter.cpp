//
// Created by peiming on 9/15/20.
//

#include "CustomAPIRewriters/LockUnlockRewriter.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <map>

#include "CustomAPIRewriters/ThreadProfileRewriter.h"
#include "aser/Util/Demangler.h"
#include "aser/Util/Log.h"
#include "conflib/conflib.h"

#define CODERRECT_GLOBAL_LOCK ".coderrect.lock"

using namespace llvm;
using namespace aser;
using namespace std;

static Type *lockType = nullptr;
static FunctionType *lockUnlockAPIType = nullptr;
static FunctionType *RAIIlockAPIType = nullptr;

static std::set<Function *> getFunction(Module *M, StringRef demangledName) {
    std::set<Function *> result;
    Demangler demangler;

    for (auto &F : *M) {
        if (!demangler.partialDemangle(F.getName())) {
            StringRef fun = demangler.getFunctionBaseName(nullptr, nullptr);
            if (fun.equals(demangledName)) {
                result.insert(&F);
            }
        } else {
            if (stripNumberPostFix(F.getName()).equals(demangledName)) {
                result.insert(&F);
            }
        }
    }

    return result;
}

static void createCoderrectLockUnlock(Module *M, Function *lockFun, StringRef funName, Argument *lockArg) {
    IRBuilder<> builder(M->getContext());

    bool shouldCreateRet;
    // clear previous implementation
    if (lockFun->isDeclaration() || !lockFun->getEntryBlock().getName().equals(".coderrect.lock.entry")) {
        lockFun->deleteBody();
        auto entryBB = BasicBlock::Create(M->getContext(), ".coderrect.lock.entry", lockFun);
        builder.SetInsertPoint(entryBB);
        shouldCreateRet = true;
    } else {
        // one API is used for multiple purpose
        // on windows, waitForSingleObject
        builder.SetInsertPoint(lockFun->getEntryBlock().getTerminator()->getPrevNode());
        shouldCreateRet = false;
    }

    Value *lockVar = nullptr;

    if (lockArg == nullptr) {
        lockVar = M->getOrInsertGlobal(CODERRECT_GLOBAL_LOCK, lockType);
    } else {
        lockVar = builder.CreateBitCast(lockArg, PointerType::getUnqual(lockType));
    }

    auto coderrectLockFun = M->getOrInsertFunction(funName, lockUnlockAPIType);
    // creating call to
    builder.CreateCall(coderrectLockFun, {lockVar});

    if (shouldCreateRet) {
        if (lockFun->getReturnType() != Type::getVoidTy(M->getContext())) {
            builder.CreateRet(UndefValue::get(lockFun->getReturnType()));
        } else {
            builder.CreateRetVoid();
        }
    }

    // mark the function as always inline
    lockFun->addFnAttr(Attribute::AlwaysInline);
}

static void rewriteRAIILock(Module *M, Function *lockFun, int lockIdx, StringRef rewritedName) {
    Argument *lockArg = nullptr;
    lockIdx --; // the idx on function argument list starting from 0, while in configuration, it start from 1

    if (lockIdx < 0) {
        // for RAII lock, find the first argument after *this* as the parameter
        for (int i = 1; i < lockFun->arg_size(); i++) {
            auto arg = lockFun->getArg(i);
            if (arg->getType()->isPointerTy()) {
                lockArg = arg;
                break;
            }
        }
        if (lockArg == nullptr) {
            // can not find a pointer other than *this*, faithfully use *this* as the lock.
            lockArg = lockFun->getArg(0);
        }
    } else {
        if (lockFun->arg_size() <= lockIdx) {
            // mismatch idx
            return;
        }

        lockArg = lockFun->getArg(lockIdx);
        assert(lockArg->getType()->isPointerTy());
    }

    bool shouldCreateRet;
    IRBuilder<> builder(M->getContext());
    // clear previous implementation
    if (lockFun->isDeclaration() || !lockFun->getEntryBlock().getName().equals(".coderrect.lock.entry")) {
        lockFun->deleteBody();
        auto entryBB = BasicBlock::Create(M->getContext(), ".coderrect.lock.entry", lockFun);
        builder.SetInsertPoint(entryBB);
        shouldCreateRet = true;
    } else {
        // one API is used for multiple purpose
        // on windows, waitForSingleObject
        builder.SetInsertPoint(lockFun->getEntryBlock().getTerminator()->getPrevNode());
        shouldCreateRet = false;
    }

    Value *lockVar = builder.CreateBitCast(lockArg, PointerType::getUnqual(lockType));
    Value *lockHandle = builder.CreateBitCast(lockFun->getArg(0), PointerType::getUnqual(lockType));

    auto coderrectLockFun = M->getOrInsertFunction(rewritedName, RAIIlockAPIType);
    // creating call to
    builder.CreateCall(coderrectLockFun, {lockHandle, lockVar});

    if (shouldCreateRet) {
        if (lockFun->getReturnType() != Type::getVoidTy(M->getContext())) {
            builder.CreateRet(UndefValue::get(lockFun->getReturnType()));
        } else {
            builder.CreateRetVoid();
        }
    }

    // mark the function as always inline
    lockFun->addFnAttr(Attribute::AlwaysInline);
}

static void rewriteLockUnlockAPI(Module *M, Function *lockFun, StringRef funName, int lockIdx = -1) {
    assert(lockType && lockUnlockAPIType);

    // JEFF: counter example in Redis: malloc_mutex_lock(tsdn, &arena->extent_avail_mtx)
    // let's either start from the last index, or ask user to specify the lock index
    Argument *lockArg = nullptr;
    // for (int i = 0; i < lockFun->arg_size(); i++) {
    if (lockIdx > 0) {
        lockIdx --; // the index for LLVM::Function starts with 0
        assert(lockIdx < lockFun->arg_size());
        lockArg = lockFun->getArg(lockIdx);
        // the lock parameter must be a pointer
        assert(lockArg->getType()->isPointerTy());
    } else {
        for (int i = lockFun->arg_size() - 1; i >= 0; i--) {
            Argument *arg = lockFun->arg_begin() + i;
            if (arg->getType()->isPointerTy()) {
                lockArg = arg;
                break;
            }
        }
    }

    createCoderrectLockUnlock(M, lockFun, funName, lockArg);
}

static int stripAndGetLockIdx(string &configName) {
    StringRef originalName = configName;
    auto splited = originalName.rsplit(":");
    if (!splited.second.empty()) {
        int idx;
        if (!splited.second.getAsInteger(10, idx)) {
            // end with ":number"
            // update the configName to remove the idx
            configName = splited.first.str();
            return idx;
        }
    }
    // no idx is provided
    return -1;
}

static void rewriteUserSpecifiedLockAPI(Module *M, const map<string, string> &lockUnlockPairs) {
    lockType = IntegerType::get(M->getContext(), 8);
    lockUnlockAPIType = FunctionType::get(Type::getVoidTy(M->getContext()), {PointerType::getUnqual(lockType)}, false);
    RAIIlockAPIType = FunctionType::get(Type::getVoidTy(M->getContext()),
                                        // two parameters: the first one for *this* -- the lock handle (critical region)
                                        // the second one is for the actual lock
                                        { PointerType::getUnqual(lockType), PointerType::getUnqual(lockType) },
                                        false);

    // TODO: do we need to keep it? or should we use always threadprofile to specify unlock/unlock pairs?
    for (const auto &lockUnlockPair : lockUnlockPairs) {
        string lockFunName = lockUnlockPair.first;
        string unlockFunName = lockUnlockPair.second;

        // strip and the argument index from the user-specified lock unlock pair
        int lockIdx = stripAndGetLockIdx(lockFunName);
        int unlockIdx = stripAndGetLockIdx(unlockFunName);

        auto lockFuns = getFunction(M, lockFunName);
        auto unlockFuns = getFunction(M, unlockFunName);

        if (!lockFuns.empty() && !unlockFuns.empty()) {
            bool isRAIILock = (unlockFunName.front() == '~');
            Demangler demangler;
            for (auto lockFun : lockFuns) {
                LOG_DEBUG("User specified LOCK function found, name={}", lockFun->getName());
                // if this is C++, then the function name are not equal until demangled
                int idx = lockIdx;
                if (lockIdx >= 0 && !stripNumberPostFix(lockFun->getName()).equals(lockFunName)) {
                    if (lockFun->arg_begin()->hasName() && lockFun->arg_begin()->getName().equals("this")) {
                        // since we compile with -fno-discard-value-names, if the first argument's name
                        // is this, we know that this is a non-static function
                        idx ++;
                    }
                }
                if (isRAIILock) {
                    if (!demangler.partialDemangle(lockFun->getName()) && demangler.isCtor()) {
                        // RAII lock must be a constructor, otherwise false match
                        rewriteRAIILock(M, lockFun, idx, LockUnlockRewriter::getCanonicalizedRAIILockName());
                    } else {
                        continue;
                    }
                } else {
                    rewriteLockUnlockAPI(M, lockFun, LockUnlockRewriter::getCanonicalizedLockName(), idx);
                }
            }

            for (auto unlockFun : unlockFuns) {
                LOG_DEBUG("User specified UNLOCK function found, name={}", unlockFun->getName());
                // if this is C++
                int idx = lockIdx;
                if (isRAIILock) {
                    // destructor as unlock, always the first argument (this) as the handle to find the lock
                    idx = 1;
                } else if (unlockIdx >= 0 && !stripNumberPostFix(unlockFun->getName()).equals(lockFunName)) {
                    if (unlockFun->arg_begin()->hasName() && unlockFun->arg_begin()->getName().equals("this")) {
                        // since we compile with -fno-discard-value-names, if the first argument's name
                        // is this, we know that this is a non-static function
                        idx++;
                    }
                }

                StringRef rewritedName = LockUnlockRewriter::getCanonicalizedUnlockName();
                if (isRAIILock) {
                    if (!demangler.partialDemangle(unlockFun->getName()) && demangler.isDtor()) {
                        // RAII lock must be a destructor, otherwise false match
                        rewritedName = LockUnlockRewriter::getCanonicalizedRAIIUnlockName();
                    } else {
                        continue;
                    }
                }
                rewriteLockUnlockAPI(M, unlockFun, rewritedName, idx);
            }
        } else {
            LOG_DEBUG("Did not find lock/unlock API: lock={}, unlock={}", lockFunName, unlockFunName);
        }
    }
}

static bool isCtor(StringRef funName) { return funName.endswith("::$constructor"); }

static bool isDtor(StringRef funName) { return funName.endswith("::$destructor"); }

static StringRef getFunctionName(StringRef funName) {
    if (isCtor(funName) || isDtor(funName)) {
        size_t pos = funName.rfind("::$");
        return funName.substr(0, pos);
    }
    return funName;
}

static StringRef getDeclCtxWithoutTemplate(StringRef funName) {
    size_t pos = funName.find("<");
    if (pos != StringRef::npos) {
        return funName.substr(0, pos);
    }

    return funName;
}

static bool isCXXNameEqual(const string &config, const Function *F) {
    Demangler demangler;
    if (!demangler.partialDemangle(F->getName())) {
        if ((isCtor(config) && demangler.isCtor()) || (isDtor(config) && demangler.isDtor())) {
            // TODO: what about template parameter a::b<>::c<>::d<>, we need to strip all template parameter but keep name
            StringRef funDeclCtx = getDeclCtxWithoutTemplate(demangler.getFunctionDeclContextName(nullptr, nullptr));
            StringRef targetCtx = getFunctionName(config);
            if (funDeclCtx.equals(targetCtx)) {
                return true;
            }
        } else {
            // TODO: what about template parameter
            StringRef funName = demangler.getFunctionName(nullptr, nullptr);
            if (funName.equals(config)) {
                return true;
            }
        }
    }

    return false;
}

static bool rewriteIfMatchAny(const vector<string> &APIs, Function *F, Module *M, StringRef rewriteName, bool isCXX) {
    for (const string &api : APIs) {
        if (isCXX) {
            // CXX need to demangle the function name
            if (isCXXNameEqual(api, F)) {
                rewriteLockUnlockAPI(M, F, rewriteName);
                return true;
            }
        } else if (stripNumberPostFix(F->getName()).equals(api)) {
            // find a customized thread create
            rewriteLockUnlockAPI(M, F, rewriteName);
            return true;
        }
    }

    return false;
}

enum class LockType {
    RDLOCK, // read lock
    WRLOCK, // write lock
    MUTEX,  // simple mutex lock

    UNLOCK, // mutext unlock
    RWUNLOCK, // rwunlock
};

static inline StringRef getLockRewriteName(LockType type, bool isRAII) {
    if (isRAII) {
        switch (type) {
            case LockType::RDLOCK:
                return LockUnlockRewriter::getCanonicalizedRAIIRdLockName();
            case LockType::WRLOCK:
                return LockUnlockRewriter::getCanonicalizedRAIIWrLockName();
            case LockType::MUTEX:
                return LockUnlockRewriter::getCanonicalizedRAIILockName();
            case LockType::UNLOCK:
                return LockUnlockRewriter::getCanonicalizedRAIIUnlockName();
            case LockType::RWUNLOCK:
                return LockUnlockRewriter::getCanonicalizedRAIIRwUnLockName();
        }
    } else {
        switch (type) {
            case LockType::RDLOCK:
                return LockUnlockRewriter::getCanonicalizedRdLockName();
            case LockType::WRLOCK:
                return LockUnlockRewriter::getCanonicalizedWrLockName();
            case LockType::MUTEX:
                return LockUnlockRewriter::getCanonicalizedLockName();
            case LockType::UNLOCK:
                return LockUnlockRewriter::getCanonicalizedUnlockName();
            case LockType::RWUNLOCK:
                return LockUnlockRewriter::getCanonicalizedRwULockName();
        }
    }
}

static bool rewriteLockIfMatchAny(const vector<string> &APIs, Function *F, Module *M, LockType type, bool isCXX) {
    for (const string &api : APIs) {
        if (isCXX) {
            // CXX need to demangle the function name
            if (isCXXNameEqual(api, F)) {
                if (isCtor(api)) {
                    //RAII lock
                    rewriteRAIILock(M, F, 2, getLockRewriteName(type, true));
                } else if (isDtor(api)) {
                    //RAII unlock
                    rewriteLockUnlockAPI(M, F, getLockRewriteName(type, true));
                } else {
                    rewriteLockUnlockAPI(M, F, getLockRewriteName(type, false));
                }
                return true;
            }
        } else if (stripNumberPostFix(F->getName()).equals(api)) {
            // find a customized thread create
            rewriteLockUnlockAPI(M, F, getLockRewriteName(type, false));
            return true;
        }
    }

    return false;
}

static bool rewriteLockAllocatorIfMatched(const LockAllocator &allocator, Function *F, Module *M) {
    if (stripNumberPostFix(F->getName()).equals(allocator.getLockAllocatorName()) ||
        stripNumberPostFix(F->getName()).equals(allocator.getLockAllocatorName() + "A") ||
        stripNumberPostFix(F->getName()).equals(allocator.getLockAllocatorName() + "W")) {
        F->deleteBody();

        IRBuilder<> builder(M->getContext());
        auto entryBB = BasicBlock::Create(M->getContext(), ".coderrect.entry", F);
        builder.SetInsertPoint(entryBB);

        // insert coderrect.lock.allocate()
        auto coderrectLockFun = M->getOrInsertFunction(LockUnlockRewriter::getLockAllocateName(),
                                                       FunctionType::get(builder.getInt8PtrTy(), false));

        auto lockObj = builder.CreateCall(coderrectLockFun);
        int addrIdx = allocator.getAddrTakenArgIdx();
        if (addrIdx != 0) {
            assert(F->arg_size() >= addrIdx);
            Argument *addrArg = F->arg_begin() + addrIdx - 1;
            assert(addrArg->getType()->isPointerTy() && "address taken parameter must be a pointer type!");
            auto addrTaken = builder.CreateBitCast(addrArg, PointerType::getUnqual(builder.getInt8PtrTy()));
            builder.CreateStore(lockObj, addrTaken);

            if (F->getReturnType() != Type::getVoidTy(M->getContext())) {
                builder.CreateRet(UndefValue::get(F->getReturnType()));
            } else {
                builder.CreateRetVoid();
            }
        } else {
            assert(F->getReturnType()->isPointerTy() && "lock allocator must return a pointer!");
            auto objPtr = builder.CreateBitCast(lockObj, F->getReturnType());
            builder.CreateRet(objPtr);
        }

        F->addFnAttr(Attribute::AlwaysInline);
        return true;
    }
    return false;
}

static void rewriteWaitForMultipleObject(llvm::Module *M) {
    auto F = M->getFunction("WaitForMultipleObjects");
    if (F) {
        // if there is a call to WaitForMultipleObject
        // WaitForMultipleObjects(sizeofarray, arrayhandle, waitforany, waittime);
        F->deleteBody();
        // According to JEFF, if wait time is not infinite, ignore it
        F->addFnAttr(Attribute::AlwaysInline);

        IRBuilder<> builder(M->getContext());

        auto entryBB = BasicBlock::Create(M->getContext(), ".coderrect.entry", F);
        auto infiniteBB = BasicBlock::Create(M->getContext(), ".coderrect.wait.infinite", F);
        auto limitedBB = BasicBlock::Create(M->getContext(), ".coderrect.wait.bounded", F);

        builder.SetInsertPoint(entryBB);
        auto result = builder.CreateICmpEQ(F->getArg(3), ConstantInt::get(F->getArg(3)->getType(), -1, true));
        builder.CreateCondBr(result, infiniteBB, limitedBB);

        builder.SetInsertPoint(limitedBB);
        builder.CreateRet(UndefValue::get(F->getReturnType()));

        builder.SetInsertPoint(infiniteBB);
        auto handles = builder.CreateBitCast(builder.CreateLoad(F->getArg(1)), PointerType::getUnqual(lockType));
        auto coderrectLockFun =
            M->getOrInsertFunction(LockUnlockRewriter::getCanonicalizedLockName(), lockUnlockAPIType);
        builder.CreateCall(coderrectLockFun, {handles});

        coderrectLockFun =
            M->getOrInsertFunction(LockUnlockRewriter::getCanonicalizedSignalWaitName(), lockUnlockAPIType);
        builder.CreateCall(coderrectLockFun, {handles});

        coderrectLockFun =
            M->getOrInsertFunction(LockUnlockRewriter::getCanonicalizedSemaphoreWaitName(), lockUnlockAPIType);
        builder.CreateCall(coderrectLockFun, {handles});
        builder.CreateRet(UndefValue::get(F->getReturnType()));
    }
}

void LockUnlockRewriter::rewriteModule(llvm::Module *M, const std::map<std::string, ThreadProfile> &profiles) {
    auto lockUnlockPair = conflib::Get<map<string, string>>("lockUnlockFunctions", {});
    lockUnlockPair.insert({{"_mp_bcs_nest_red", "_mp_ecs_nest_red"}});  // for fortran !$omp atomic
    lockUnlockPair.insert({{"__kmpc_ordered", "__kmpc_end_ordered"}});  // for c/c++/fortran !$omp ordered

    if (!lockUnlockPair.empty()) {
        rewriteUserSpecifiedLockAPI(M, lockUnlockPair);
    }

    // special case
    // WaitForMultipleObjects
    rewriteWaitForMultipleObject(M);

    // handle lock, unlock defined in thread profile

    for (const auto &it : profiles) {
        if (it.first == "pthread") {
            continue;
        }

        for (auto &F : *M) {
            const ThreadProfile &profile = it.second;
            auto lockAPIs = profile.getMutexAPI().getLockAPIs();
            auto unlockAPIs = profile.getMutexAPI().getUnlockAPIs();
            auto rdLockAPI = profile.getReadWriteLockAPI().getReadLockAPIs();
            auto wrLockAPI = profile.getReadWriteLockAPI().getWriteLockAPIs();
            auto rwUnLockAPI = profile.getReadWriteLockAPI().getUnlockAPIs();

            auto signalAPI = profile.getConditionVariableAPI().getSignalAPIs();
            auto signalWaitAPI = profile.getConditionVariableAPI().getWaitAPIs();

            auto semaphoreWaitAPI = profile.getSemaphoreAPI().getWaitAPIs();
            auto semaphorePostAPI = profile.getSemaphoreAPI().getPostAPIs();

            if (profile.getReadWriteLockAPI().hasLockAllocator()) {
                if (rewriteLockAllocatorIfMatched(profile.getReadWriteLockAPI().getLockAllocator(), &F, M)) {
                    continue;
                }
            }

            if (profile.getMutexAPI().hasLockAllocator()) {
                if (rewriteLockAllocatorIfMatched(profile.getMutexAPI().getLockAllocator(), &F, M)) {
                    continue;
                }
            }

            if (profile.getConditionVariableAPI().hasLockAllocator()) {
                if (rewriteLockAllocatorIfMatched(profile.getConditionVariableAPI().getLockAllocator(), &F, M)) {
                    continue;
                }
            }

            if (profile.getSemaphoreAPI().hasLockAllocator()) {
                if (rewriteLockAllocatorIfMatched(profile.getSemaphoreAPI().getLockAllocator(), &F, M)) {
                    continue;
                }
            }

            if (rewriteIfMatchAny(semaphorePostAPI, &F, M, LockUnlockRewriter::getCanonicalizedSemaphorePostName(),
                                  profile.isCXXProfile())) {
                // continue;
            }

            if (rewriteIfMatchAny(semaphoreWaitAPI, &F, M, LockUnlockRewriter::getCanonicalizedSemaphoreWaitName(),
                                  profile.isCXXProfile())) {
                // continue;
            }

            if (rewriteIfMatchAny(signalWaitAPI, &F, M, LockUnlockRewriter::getCanonicalizedSignalWaitName(),
                                  profile.isCXXProfile())) {
                // continue;
            }
            if (rewriteIfMatchAny(signalAPI, &F, M, LockUnlockRewriter::getCanonicalizedSignalName(),
                                  profile.isCXXProfile())) {
                // continue;
            }

            if (rewriteLockIfMatchAny(lockAPIs, &F, M, LockType::MUTEX, profile.isCXXProfile())) {
                // continue;
            }
            if (rewriteLockIfMatchAny(unlockAPIs, &F, M, LockType::UNLOCK, profile.isCXXProfile())) {
                // continue;
            }
            if (rewriteLockIfMatchAny(rdLockAPI, &F, M, LockType::RDLOCK, profile.isCXXProfile())) {
                // continue;
            }
            if (rewriteLockIfMatchAny(wrLockAPI, &F, M, LockType::WRLOCK, profile.isCXXProfile())) {
                // continue;
            }
            if (rewriteLockIfMatchAny(rwUnLockAPI, &F, M, LockType::RWUNLOCK, profile.isCXXProfile())) {
                // continue;
            }
        }
    }
}
