#include "CustomAPIRewriters/LockUnlockRewriter.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <map>

#include "CustomAPIRewriters/ThreadProfileRewriter.h"
#include "o2/Util/Demangler.h"
#include "o2/Util/Log.h"
#include "conflib/conflib.h"

#define O2_GLOBAL_LOCK ".o2.lock"

using namespace llvm;
using namespace o2;
using namespace std;

static Type *lockType = nullptr;
static FunctionType *lockUnlockAPIType = nullptr;

static std::set<Function *> getFunction(Module *M, StringRef mangledName) {
    std::set<Function *> result;
    Demangler demangler;

    mangledName = stripNumberPostFix(mangledName);

    for (auto &F : *M) {
        if (!demangler.partialDemangle(F.getName())) {
            StringRef fun = demangler.getFunctionBaseName(nullptr, nullptr);
            if (fun.equals(mangledName)) {
                result.insert(&F);
            }
        } else {
            if (stripNumberPostFix(F.getName()).equals(mangledName)) {
                result.insert(&F);
            }
        }
    }

    return result;
}

static void createO2LockUnlock(Module *M, Function *lockFun, StringRef funName, Argument *lockArg) {
    IRBuilder<> builder(M->getContext());

    // clear previous implementation
    lockFun->deleteBody();
    auto entryBB = BasicBlock::Create(M->getContext(), "o2.lock", lockFun);
    builder.SetInsertPoint(entryBB);
    Value *lockVar = nullptr;

    if (lockArg == nullptr) {
        lockVar = M->getOrInsertGlobal(O2_GLOBAL_LOCK, lockType);
    } else {
        lockVar = builder.CreateBitCast(lockArg, PointerType::getUnqual(lockType));
    }

    auto o2LockFun = M->getOrInsertFunction(funName, lockUnlockAPIType);
    // creating call to
    builder.CreateCall(o2LockFun, {lockVar});
    if (lockFun->getReturnType() != Type::getVoidTy(M->getContext())) {
        builder.CreateRet(UndefValue::get(lockFun->getReturnType()));
    } else {
        builder.CreateRetVoid();
    }

    // mark the function as always inline
    lockFun->addFnAttr(Attribute::AlwaysInline);
}

static void rewriteLockUnlockAPI(Module *M, Function *lockFun, StringRef funName) {
    assert(lockType && lockUnlockAPIType);

    // JEFF: counter example in Redis: malloc_mutex_lock(tsdn, &arena->extent_avail_mtx)
    // let's either start from the last index, or ask user to specify the lock index
    Argument *lockArg = nullptr;
    // for (int i = 0; i < lockFun->arg_size(); i++) {
    for (int i = lockFun->arg_size() - 1; i >= 0; i--) {
        Argument *arg = lockFun->arg_begin() + i;
        if (arg->getType()->isPointerTy()) {
            lockArg = arg;
            break;
        }
    }

    createO2LockUnlock(M, lockFun, funName, lockArg);
}

static void rewriteUserSpecifiedLockAPI(Module *M, const map<string, string> &lockUnlockPairs) {
    lockType = IntegerType::get(M->getContext(), 8);
    lockUnlockAPIType = FunctionType::get(Type::getVoidTy(M->getContext()), {PointerType::getUnqual(lockType)}, false);

    // TODO: do we need to keep it? or should we use always threadprofile to specify unlock/unlock pairs?
    for (const auto &lockUnlockPair : lockUnlockPairs) {
        const string &lockFunName = lockUnlockPair.first;
        const string &unlockFunName = lockUnlockPair.second;

        auto lockFuns = getFunction(M, lockFunName);
        auto unlockFuns = getFunction(M, unlockFunName);

        if (!lockFuns.empty() && !unlockFuns.empty()) {
            for (auto lockFun : lockFuns) {
                LOG_DEBUG("User specified LOCK function found, name={}", lockFun->getName());
                rewriteLockUnlockAPI(M, lockFun, LockUnlockRewriter::getCanonicalizedLockName());
            }

            for (auto unlockFun : unlockFuns) {
                LOG_DEBUG("User specified UNLOCK function found, name={}", unlockFun->getName());
                rewriteLockUnlockAPI(M, unlockFun, LockUnlockRewriter::getCanonicalizedUnlockName());
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

static bool isCXXNameEqual(const string &config, const Function *F) {
    Demangler demangler;
    if (!demangler.partialDemangle(F->getName())) {
        if ((isCtor(config) && demangler.isCtor()) || (isDtor(config) && demangler.isDtor())) {
            // TODO: what about template parameter
            StringRef funDeclCtx = demangler.getFunctionDeclContextName(nullptr, nullptr);
            if (funDeclCtx.equals(config)) {
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

static bool rewriteLockAllocatorIfMatched(const LockAllocator &allocator, Function *F, Module *M) {
    if (stripNumberPostFix(F->getName()).equals(allocator.getLockAllocatorName())) {
        F->deleteBody();

        IRBuilder<> builder(M->getContext());
        auto entryBB = BasicBlock::Create(M->getContext(), "o2.entry", F);
        builder.SetInsertPoint(entryBB);

        // insert o2.lock.allocate()
        auto o2LockFun = M->getOrInsertFunction(LockUnlockRewriter::getLockAllocateName(),
                                                       FunctionType::get(builder.getInt8PtrTy(), false));

        auto lockObj = builder.CreateCall(o2LockFun);
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

void LockUnlockRewriter::rewriteModule(llvm::Module *M, const std::map<std::string, ThreadProfile> &profiles) {
    auto lockUnlockPair = conflib::Get<map<string, string>>("lockUnlockFunctions", {});
    lockUnlockPair.insert({{"_mp_bcs_nest_red", "_mp_ecs_nest_red"}});  // for fortran !$omp atomic
    lockUnlockPair.insert({{"__kmpc_ordered", "__kmpc_end_ordered"}});  // for c/c++/fortran !$omp ordered

    if (!lockUnlockPair.empty()) {
        rewriteUserSpecifiedLockAPI(M, lockUnlockPair);
    }

    // handle lock, unlock defined in thread profile
    for (auto &F : *M) {
        // find the boost::thread::ctor
        for (const auto &it : profiles) {
            if (it.first == "pthread") {
                continue;
            }

            const ThreadProfile &profile = it.second;
            auto lockAPIs = profile.getMutexAPI().getLockAPIs();
            auto unlockAPIs = profile.getMutexAPI().getUnlockAPIs();
            auto rdLockAPI = profile.getReadWriteLockAPI().getReadLockAPIs();
            auto wrLockAPI = profile.getReadWriteLockAPI().getWriteLockAPIs();
            auto rwUnLockAPI = profile.getReadWriteLockAPI().getUnlockAPIs();

            if (rewriteIfMatchAny(lockAPIs, &F, M, LockUnlockRewriter::getCanonicalizedLockName(),
                                  profile.isCXXProfile())) {
                continue;
            }
            if (rewriteIfMatchAny(unlockAPIs, &F, M, LockUnlockRewriter::getCanonicalizedUnlockName(),
                                  profile.isCXXProfile())) {
                continue;
            }
            if (rewriteIfMatchAny(rdLockAPI, &F, M, LockUnlockRewriter::getCanonicalizedRdLockName(),
                                  profile.isCXXProfile())) {
                continue;
            }
            if (rewriteIfMatchAny(wrLockAPI, &F, M, LockUnlockRewriter::getCanonicalizedWrLockName(),
                                  profile.isCXXProfile())) {
                continue;
            }
            if (rewriteIfMatchAny(rwUnLockAPI, &F, M, LockUnlockRewriter::getCanonicalizedRwULockName(),
                                  profile.isCXXProfile())) {
                continue;
            }

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
        }
    }
}
