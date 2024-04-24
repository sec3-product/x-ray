//
// Created by peiming on 10/16/20.
//
#include <map>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>

#include "RDUtil.h"
#include "conflib/conflib.h"
#include "aser/Util/Util.h"
#include "aser/Util/Demangler.h"

#include "CustomAPIRewriters/IndirectTargetRewriter.h"

using namespace std;
using namespace llvm;

namespace {

void rewriteIndirectTargets(Module *M, const map<string, vector<string>> &targets) {
    IRBuilder<> builder(M->getContext());

    for (Function &F : *M) {
        for (const pair<string, vector<string>> &entry : targets) {
            // TODO: C++
            int pos = entry.first.find('@');
            auto caller = entry.first.substr(0, pos);
            auto lineNum = entry.first.substr(pos + 1);

            if (aser::stripNumberPostFix(F.getName()).equals(caller)) {
                SmallVector<CallBase *, 4> indirectCallSites;
                // 2nd, find indirect function calls in the function
                for (auto &BB : F) {
                    for (auto &I : BB) {
                        if (auto call = dyn_cast<CallBase>(&I); call && call->isIndirectCall()) {
                            indirectCallSites.push_back(call);
                        }
                    }
                }

                if (indirectCallSites.empty()) {
                    // a miss configuration?
                    continue;
                }

                CallBase *theTarget = nullptr;
                if (indirectCallSites.size() == 1) {
                    // common case
                    // only one indirect callsite, no need to checkout the line number
                    theTarget = indirectCallSites.back();
                } else {
                    for (auto call : indirectCallSites) {
                        if (aser::getSourceLoc(call).getSourceLine() == lineNum) {
                            theTarget = call;
                            break;
                        }
                    }
                }

                if (theTarget == nullptr) {
                    // no line number match
                    // TODO: some heuristic??
                    LOG_ERROR("fail find the indirect callsite at: {}", entry.first);
                }

                builder.SetInsertPoint(theTarget);

                CallInst *replaced = nullptr;
                auto configuredTargets = entry.second;

                for (auto &target: configuredTargets) {
                    Function *targetFunction = M->getFunction(target);
                    if (targetFunction == nullptr) {
                        // we can not find the target function
                        // TODO: should we iterate the module again to see is there NumberPostfix version of the function
                        // TODO: C++ function as the target
                        LOG_ERROR("fail to find the configured target : {}", target);
                        continue;
                    }

                    if (aser::isCompatibleCall(theTarget, targetFunction)) {
                        SmallVector<Value *, 8> args;
                        for (int i = 0; i < theTarget->getNumArgOperands(); i++) {
                            if (i < targetFunction->arg_size()) {
                                Value *actual = theTarget->getArgOperand(i);
                                Value *formal = targetFunction->arg_begin() + i;

                                if (actual->getType() != formal->getType()) {
                                    args.push_back(builder.CreateBitCast(actual, formal->getType()));
                                } else {
                                    args.push_back(actual);
                                }
                            } else {
                                args.push_back(theTarget->getArgOperand(i));
                            }
                        }

                        replaced = builder.CreateCall(targetFunction->getFunctionType(), targetFunction, args, "",
                                                      theTarget->getMetadata(Metadata::MetadataKind::DILocationKind));
                    } else {
                        LOG_ERROR("incompatible type between callsite and target? callsite: {}, type: {}",
                                  theTarget,
                                  targetFunction->getFunctionType());
                    }
                }

                if (replaced != nullptr) {
                    if (replaced->getType() != theTarget->getType()) {
                        theTarget->replaceAllUsesWith(builder.CreateBitCast(replaced, theTarget->getType()));
                    } else if (!replaced->getType()->isVoidTy()){
                        theTarget->replaceAllUsesWith(replaced);
                    }
                }
                // remove the previous target
                theTarget->eraseFromParent();
                break;
            }
        }
    }
}

}

namespace aser {

void IndirectTargetRewriter::rewriteModule(llvm::Module *M) {
    auto indirectTarget = conflib::Get<map<string, vector<string>>>("indirectTargets", {});
    if (indirectTarget.empty()) {
        return;
    }

    // start to rewrite the specified indirect target
    rewriteIndirectTargets(M, indirectTarget);
}

}