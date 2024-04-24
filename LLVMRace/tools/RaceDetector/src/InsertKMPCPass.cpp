//
// Created by peiming on 3/5/20.
//

#include "InsertKMPCPass.h"

using namespace llvm;
using namespace std;
using namespace aser;

bool InsertKMPCPass::runOnModule(Module &module) {
    map<Value *, vector<const CallInst *>> regions;

    for (auto &F : module) {
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (isa<CallInst>(&I) || isa<InvokeInst>(&I)) {
                    CallSite cs(&I);
                    // Can return null function
                    auto function = const_cast<Function *>(cs.getCalledFunction());
                    if (function && (function->getName().find("__kmpc_fork_call") != string::npos ||
                                     function->getName().find("__kmpc_fork_teams") != string::npos)) {
                        IRBuilder<> build(&I);
                        vector<Value *> arg_list;

                        for (auto const &arg : cs.args()) {
                            auto val = cast<llvm::Value>(arg);
                            arg_list.push_back(val);
                        }

                        if (isa<CallInst>(&I)) {
                            //auto inst = build.CreateCall(const_cast<Value *>(cs.getCalledValue()), ArrayRef<Value *>(arg_list));
                                                   auto inst = build.CreateCall(FunctionCallee(const_cast<Function*>(cs.getCalledFunction())), ArrayRef<Value *>(arg_list));

		       	} else {
                            auto II = cast<InvokeInst>(&I);
                            //auto inst = build.CreateInvoke(const_cast<Value *>(cs.getCalledValue()),
                            //                               II->getNormalDest(), II->getUnwindDest(),
                            //                               ArrayRef<Value *>(arg_list));
                        auto inst = build.CreateInvoke(FunctionCallee(const_cast<Function*>(cs.getCalledFunction())),
                                                           II->getNormalDest(), II->getUnwindDest(),
                                                           ArrayRef<Value *>(arg_list));
			}
                    }
                }
            }
        }
    }

    return true;
}

char aser::InsertKMPCPass::ID = 0;
static llvm::RegisterPass<aser::InsertKMPCPass> IKP("InsertKMPCPass", "Duplicate kmpc_fork calls for pointer analysis",
                                                    false, /*CFG only*/
                                                    false /*is analysis*/);
