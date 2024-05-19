//
// Created by peiming on 3/9/20.
//

//
// Created by peiming on 2/10/20.
//
#include <set>
#include <string>

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Demangle/Demangle.h>

#include "aser/Util/Log.h"
#include "aser/PreProcessing/Passes/StandardHeapAPIRewritePass.h"

using namespace std;
using namespace llvm;

static bool isItaniumEncoding(const StringRef &MangledName) {
    size_t Pos = MangledName.find_first_not_of('_');
    // A valid Itanium encoding requires 1-4 leading underscores, followed by 'Z'.
    return Pos > 0 && Pos <= 4 && MangledName[Pos] == 'Z';
}

static Function *getCallocFunction(Module &M, Type *retType, Type *para1, Type *para2) {
    IRBuilder<> builder(M.getContext());
    Function *calloc = M.getFunction("calloc");

    if (calloc == nullptr) {
        // void* calloc (size_t num, size_t size);
        FunctionCallee tmp = M.getOrInsertFunction("calloc", retType, para1, para2);
        if (auto GA = dyn_cast<GlobalAlias>(tmp.getCallee())) {
            FunctionCallee callocTmp = M.getOrInsertFunction(".calloc.tmp", retType, para2, para2);
            auto aliasee = cast<Function>(GA->getAliasee());
            assert(aliasee->arg_size() == 2);
            aliasee->addFnAttr(Attribute::AlwaysInline);

            aliasee->deleteBody();
            auto entryBB = BasicBlock::Create(M.getContext(), "aser.heap", aliasee);
            builder.SetInsertPoint(entryBB);
            auto call = builder.CreateCall(callocTmp, {aliasee->arg_begin(), aliasee->arg_begin() + 1});
            builder.CreateRet(call);

            GA->replaceAllUsesWith(callocTmp.getCallee());
            GA->eraseFromParent();

            callocTmp.getCallee()->setName("calloc");
            calloc = cast<Function>(callocTmp.getCallee());
        } else {
            calloc = cast<Function>(tmp.getCallee());
        }
        calloc->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
    }

    return calloc;
}

static bool identifyOverridenHeapAPIs(Module &M, set<StringRef> &HeapAPIs) {
    ItaniumPartialDemangler demangler;
    LLVMContext &C = M.getContext();
    IRBuilder<> builder(C);
    Type *i8ptr = PointerType::get(IntegerType::get(C, 8), 0);

    bool changed = false;

    for (Function &F : M) {
        StringRef funName = F.getName();
        if (HeapAPIs.find(F.getName()) != HeapAPIs.end()) continue;

        // handle function tagged with allocsize() attribute
        if (F.hasFnAttribute(Attribute::AllocSize)) {
            if (!F.isDeclaration()) {
                F.deleteBody();
            }

            auto entryBB = BasicBlock::Create(C, "aser.heap", &F);
            builder.SetInsertPoint(entryBB);

            const Attribute& attr = F.getFnAttribute(Attribute::AllocSize);
            auto allocArgs = attr.getAllocSizeArgs();

            Value *elementSize = F.arg_begin() + allocArgs.first;
            Value *elementNum = allocArgs.second.hasValue()
                                ? (Value *) (F.arg_begin() + allocArgs.second.getValue())
                                : (Value *) ConstantInt::get(elementSize->getType(), 1, false);

            Function *calloc = getCallocFunction(M, i8ptr, elementSize->getType(), elementNum->getType());

            Value *args[2] = {elementSize, elementNum};
            CallInst *call = builder.CreateCall(calloc, args);
            builder.CreateRet(call);

            // inline them
            F.addFnAttr(Attribute::AlwaysInline);
        }

        // for C++
        if (!isItaniumEncoding(funName) || demangler.partialDemangle(funName.begin())) continue;
        if (demangler.isCtorOrDtor()) continue;

        // use string to avoid memory leak here
        char *ptr = demangler.getFunctionBaseName(nullptr, nullptr);
        if (ptr == nullptr) continue;

        string baseName = ptr;
        StringRef baseNameRef(baseName);

        // demangler.getFunctionName(nullptr, nullptr)
        // for C++
        // 1st, override the new operator
        if (baseNameRef.startswith("operator new")) {
            LOG_TRACE("Overridden new operator. func={}", funName);

            // simply replace it with original new operator
            FunctionCallee Znwm = M.getOrInsertFunction("_Znwm", i8ptr, F.arg_begin()->getType());
            if (!F.isDeclaration()) {
                // remove the original function body if there is one
                F.deleteBody();
            }

            // FIXME: handle debug information correctly
            auto entryBB = BasicBlock::Create(C, "aser.heap", &F);
            builder.SetInsertPoint(entryBB);

            Value *args[1] = {&(*F.arg_begin())};
            CallInst *call = builder.CreateCall(Znwm, args);
            builder.CreateRet(call);

            // mark the function as always inline
            F.addFnAttr(Attribute::AlwaysInline);
            changed = true;
            continue;
        }

        // 2nd, allocator that implemented throught std::allocator_trait
        string ctxName = demangler.getFunctionDeclContextName(nullptr, nullptr);
        StringRef ctxNameRef(ctxName);

        if (ctxNameRef.startswith("std::allocator_traits<")) {
            if (baseNameRef.startswith("allocate")) {
                LOG_TRACE("Overridden allocator_trait. func={}", funName);
                // two different possible signature
                // 1st: allocate(_Alloc& a, size_type n, const_void_pointer hint)
                // 2nd: allocate(_Alloc& a, size_type n)
                // hint can be ignored for static analysis in the first case.
                if (F.arg_size() < 2) {
                    continue;
                }
                // the second argument, indicate how many arguments to allocate
                Value *num = &*(F.arg_begin() + 1);

                if (!num->getType()->isIntegerTy()) {
                    continue;
                }
                // In this case, the return the type is determined.
                // transform it to a calloc
                Function *calloc = getCallocFunction(M, i8ptr, num->getType(), num->getType());

                // we are sure we have the calloc function here now.
                if (!F.isDeclaration()) {
                    // remove the original function body.
                    F.deleteBody();
                }

                auto entryBB = BasicBlock::Create(C, "aser.heap", &F);
                builder.SetInsertPoint(entryBB);

                uint64_t allocSize = M.getDataLayout().getTypeAllocSize(F.getReturnType()->getPointerElementType());

                Value *args[2] = {num, ConstantInt::get(num->getType(), allocSize)};
                CallInst *call = builder.CreateCall(calloc, args);
                Value *bitCast = builder.CreateBitCast(call, F.getReturnType());
                builder.CreateRet(bitCast);

                F.addFnAttr(Attribute::AlwaysInline);
                changed = true;
                continue;
            }
        }
    }

    return changed;
}


bool StandardHeapAPIRewritePass::runOnModule(llvm::Module &M) {
    // the set of default heap allocation APIs.
    set<StringRef> HeapAPIs{"malloc", "calloc", "memalign", "posix_memalign", "_Znwm", "_Znam"};

    bool changed = identifyOverridenHeapAPIs(M, HeapAPIs);
    return changed;
}

char StandardHeapAPIRewritePass::ID = 0;
static RegisterPass<StandardHeapAPIRewritePass> SHARP("", "Rewrite overrides of malloc-like APIs",
                                                      false, /*CFG only*/false /*is analysis*/);

