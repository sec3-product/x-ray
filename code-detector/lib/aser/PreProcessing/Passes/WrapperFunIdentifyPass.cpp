//
// Created by peiming on 2/10/20.
//

#include "aser/PreProcessing/Passes/WrapperFunIdentifyPass.h"

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <set>
#include <string>

#include "aser/Util/Log.h"

#define MAX_WRAP_INSTRUCTION_COUNT 100

// TODO: the number can be tweaked
#define MAX_NUM_LOAD_FOR_GETTER 1
#define MAX_NUM_STORE_FOR_SETTER 1
#define MAX_NUM_INST_FOR_GETTER_SETTER 20
#define MAX_NUM_CALL_FOR_GETTER_SETTER 0
#define MAX_NUM_BB_FOR_GETTER_SETTER 3  // minumal number of basic block for a simple loop

using namespace std;
using namespace llvm;

static bool isItaniumEncoding(const StringRef &MangledName) {
    size_t Pos = MangledName.find_first_not_of('_');
    // A valid Itanium encoding requires 1-4 leading underscores, followed by 'Z'.
    return Pos > 0 && Pos <= 4 && MangledName[Pos] == 'Z';
}

static bool pushNonRecursivePhi(Value *use, SmallVector<Value *, 16> &useChain) {
    for (User *phiUser : use->users()) {
        // we might have loop here! skip recursion
        if (std::find(useChain.begin(), useChain.end(), phiUser) == useChain.end()) {
            useChain.push_back(phiUser);
            return true;
        } else {
            return false;
        }
    }
    // phi has no use means no recursive
    return true;
}

static bool inlineHeapWrapperAPIs(Module &M, set<StringRef> &HeapAPIs) {
    ItaniumPartialDemangler demangler;
    SmallVector<StringRef, 16> workList;

    for (StringRef allocAPI : HeapAPIs) {
        workList.push_back(allocAPI);
    }

    while (!workList.empty()) {
        StringRef allocAPI = workList.pop_back_val();

        Function *allocFun = M.getFunction(allocAPI);
        if (M.getFunction(allocAPI) == nullptr) {
            continue;
        }

        // Here inline all the uses of the allocFun
        for (User *user : allocFun->users()) {
            auto allocSite = dyn_cast<CallBase>(user);
            if (allocSite == nullptr) {
                continue;
            }

            Function *caller = allocSite->getFunction();
            // use some heuristic here to infer whether is a customized heap allocation wrapper

            // 1st, if the function is too complicated, just ignore it
            // right now, only depends on how many instruction we have.
            // TODO: the threshold can be adjusted.
            if (caller->getInstructionCount() > MAX_WRAP_INSTRUCTION_COUNT) {
                continue;
            }

            // 2nd, a simple heuristic, whether the function name contains "alloc"
            // demangler.partialDemangle(caller->getName().begin());
            std::string funBaseName;
            if (isItaniumEncoding(caller->getName())) {
                // partialDemangle returns true if the call fails
                if (demangler.partialDemangle(caller->getName().begin())) continue;
                auto result = demangler.getFunctionBaseName(nullptr, nullptr);
                if (result == nullptr) {
                    // If this function was alloc wrapper we will fail to inline
                    LOG_WARN("Failed to demangle function: {}", caller->getName());
                    continue;
                }
                funBaseName = result;
            } else {
                funBaseName = caller->getName().str();
            }

            StringRef baseNameRef = funBaseName;
            if (!(baseNameRef.contains("alloc") || baseNameRef.contains("Alloc") || baseNameRef.contains("new") ||
                  baseNameRef.contains("New"))) {
                // baseNameRef.contains("create")||
                // baseNameRef.contains("Create"))) {
                continue;
            }

            if (!caller->getReturnType()->isPointerTy()) {
                continue;
            }
            // 3rd, there should be a use-def chain from allocation site to the returned value
            SmallVector<Value *, 16> useChain;
            for (User *allocUser : allocSite->users()) {
                useChain.push_back(allocUser);
            }

            while (!useChain.empty()) {
                Value *use = useChain.pop_back_val();
                if (isa<ReturnInst>(use)) {
                    LOG_TRACE("Wrapper Heap API. name={}", caller->getName());
                    caller->addFnAttr(Attribute::AlwaysInline);
                    if (HeapAPIs.insert(caller->getName()).second) {
                        workList.push_back(caller->getName());
                    }
                    break;
                }

                if (isa<PHINode>(use) || isa<BitCastInst>(use)) {
                    if (!pushNonRecursivePhi(use, useChain)) {
                        break;
                    }
                }
            }
        }
    }
    return false;
}

static bool inlineHeapFreeWrapperAPIs(Module &M, const set<StringRef> &freeAPIs) {
    ItaniumPartialDemangler demangler;

    for (auto it = freeAPIs.begin(), ie = freeAPIs.end(); it != ie; it++) {
        Function *freeFun = M.getFunction(*it);
        if (freeFun == nullptr) {
            continue;
        }

        for (auto user : freeFun->users()) {
            if (auto call = dyn_cast<CallBase>(user)) {
                if (call->getFunction()->getInstructionCount() > MAX_WRAP_INSTRUCTION_COUNT) {
                    continue;
                }

                // call to the free
                if (isa<Argument>(call->getArgOperand(0)->stripInBoundsOffsets())) {
                    // this is a wrapper
                    LOG_TRACE("Wrapper Free API. name={}", call->getFunction()->getName());
                    call->getFunction()->addFnAttr(Attribute::AlwaysInline);
                }
            }
        }
    }

    return false;
}

static bool inlineSimpleFunction(Module &M) {
    bool changed = false;
    for (Function &F : M) {
        if (F.getBasicBlockList().size() > MAX_NUM_BB_FOR_GETTER_SETTER) {
            continue;
        }
        if (F.isDeclaration()) {
            continue;
        }
        if (F.getInstructionCount() > MAX_NUM_INST_FOR_GETTER_SETTER) {
            continue;
        }

        int loadNum = 0;
        int storeNum = 0;
        int callNum = 0;

        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (auto call = dyn_cast<CallBase>(&I)) {
                    if (auto calleeFun = call->getCalledFunction()) {
                        if (calleeFun->getName().startswith("llvm."))
                            // skip llvm intrinsics
                            continue;
                    }
                    callNum++;
                    if (callNum > MAX_NUM_CALL_FOR_GETTER_SETTER) {
                        goto loop_break;
                    }
                } else if (isa<LoadInst>(I)) {
                    loadNum++;
                    if (loadNum > MAX_NUM_LOAD_FOR_GETTER) {
                        goto loop_break;
                    }
                } else if (isa<StoreInst>(I)) {
                    storeNum++;
                    if (storeNum > MAX_NUM_STORE_FOR_SETTER) {
                        goto loop_break;
                    }
                }
            }
        }

        LOG_TRACE("Try to inline Getter/Setter. func={}", F.getName());
        F.addFnAttr(Attribute::AlwaysInline);
        changed = true;
    loop_break:
        continue;
    }

    return changed;
}

// identify all the (potential) wrapper functions around external APIs
static bool inlineSimpleWrappers(llvm::Module &M) {
    vector<Function *> externAPIs;
    for (auto &fun : M) {
        // 1st collect all External APIs
        if (fun.isDeclaration() && !fun.isIntrinsic()) {
            externAPIs.push_back(&fun);
        }
    }

    while (!externAPIs.empty()) {
        Function *candidate = externAPIs.back();
        externAPIs.pop_back();

        auto it = candidate->user_begin();
        auto ie = candidate->user_end();
        for (; it != ie; it++) {
            if (auto call = dyn_cast<CallBase>(*it)) {
                auto wrapper = call->getFunction();

                if (wrapper->getNumUses() == 1 || wrapper->hasFnAttribute(Attribute::AlwaysInline)) {
                    // small opts, no different if the function is only called once
                    continue;
                }
                // a simple (yet pretty common) herusitic
                // the wrapper function's name should contains the wrapped function name
                // Or even better.. use string similarity
                // so that so that pthread_create and CreateThread can be identified?
                // any existing algorithm for the purpose?
                if (!wrapper->getName().contains(candidate->getName())) {
                    // TODO: do we need to demangle here? Probably not.
                    continue;
                }

                // avoid inlining too complicated function, this will harm the performance
                // the number can be adjusted...
                if (wrapper->getBasicBlockList().size() > 10 || wrapper->getInstructionCount() > 50) {
                    continue;
                }
                wrapper->addFnAttr(Attribute::AlwaysInline);
            }
        }
    }
    return false;
}

// This pass mark those wrappers and inline them.
bool WrapperFunIdentifyPass::runOnModule(llvm::Module &M) {
    // the set of default heap allocation APIs.
    set<StringRef> HeapAPIs{"malloc", "calloc", "memalign", "posix_memalign", "_Znwm", "_Znam"};

    bool changed = false;
    changed |= inlineHeapWrapperAPIs(M, HeapAPIs);
    changed |= inlineHeapFreeWrapperAPIs(M, {"free", "_ZdlPv", "_ZdaPv"});
    changed |= inlineSimpleFunction(M);
    changed |= inlineSimpleWrappers(M);

    return changed;
}

char WrapperFunIdentifyPass::ID = 0;
static RegisterPass<WrapperFunIdentifyPass> HWIP("", "Passes that identifies the heap wrapper APIs", true, /*CFG only*/
                                                 false /*is analysis*/);
