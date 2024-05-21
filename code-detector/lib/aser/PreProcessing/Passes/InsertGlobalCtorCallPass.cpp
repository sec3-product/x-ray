//
// Created by peiming on 3/14/20.
//
#include "aser/PreProcessing/Passes/InsertGlobalCtorCallPass.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

using namespace aser;
using namespace llvm;

#define INIT_FUNC_INDEX 1

bool InsertGlobalCtorCallPass::runOnModule(llvm::Module &M) {
    // TODO: make main configurable
    auto mainFun = M.getFunction("cr_main");
    if (mainFun == nullptr || mainFun->isDeclaration()) {
        return false;
    }

    IRBuilder<> builder(&mainFun->getEntryBlock().front());

    // Proc_Register, tmp fix for redisgraph
    Function *tmp = M.getFunction("Proc_Register");
    if (tmp) {
        builder.CreateCall(FunctionCallee(tmp->getFunctionType(), tmp));
    }

    auto ctors = M.getGlobalVariable("llvm.global_ctors");
    if (ctors == nullptr) {
        // no global ctors
        return false;
    }

    // @llvm.global_ctors = [N x { i32, void ()*, i8* }]
    if (ctors->hasInitializer()) {
        const llvm::Constant *initializer = ctors->getInitializer();
        if (initializer->isNullValue() || llvm::isa<llvm::UndefValue>(initializer)) {
            return false;
        }

        // traverse the init array
        auto *initArray = llvm::cast<llvm::ConstantArray>(initializer);
        for (int i = 0; i < initArray->getNumOperands(); i++) {
            llvm::Constant *curCtor = initArray->getOperand(i);
            // the ctor is a structure of type { i32, void ()*, i8* }
            llvm::Constant *init = llvm::cast<llvm::ConstantAggregate>(curCtor)->getOperand(1);
            auto initFun = llvm::cast<Function>(init);
            builder.CreateCall(FunctionCallee(initFun->getFunctionType(), initFun));
        }
    }

    return false;
}

char InsertGlobalCtorCallPass::ID = 0;
static RegisterPass<InsertGlobalCtorCallPass> IGCCP("", "Insert call to global variable constructor before main",
                                                    true, /*CFG only*/
                                                    false /*is analysis*/);
