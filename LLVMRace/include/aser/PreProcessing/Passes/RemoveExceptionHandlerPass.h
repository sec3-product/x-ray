//
// Created by peiming on 1/10/20.
//

#ifndef ASER_PTA_REMOVEEXCEPTIONHANDLERPASS_H
#define ASER_PTA_REMOVEEXCEPTIONHANDLERPASS_H

#include <llvm/Pass.h>

namespace aser {

class RemoveExceptionHandlerPass : public llvm::FunctionPass {
public:
    static char ID;
    RemoveExceptionHandlerPass() : llvm::FunctionPass(ID) {}

    bool runOnFunction(llvm::Function &F) override;
    bool doInitialization(llvm::Module &M) override;
};

}  // namespace aser

#endif  // ASER_PTA_REMOVEEXCEPTIONHANDLERPASS_H
