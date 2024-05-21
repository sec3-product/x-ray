//
// Created by peiming on 4/15/20.
//

#ifndef ASER_PTA_REMOVEASMINSTPASS_H
#define ASER_PTA_REMOVEASMINSTPASS_H

#include <llvm/Pass.h>

namespace aser {

class RemoveASMInstPass : public llvm::FunctionPass {
public:
    static char ID;
    explicit RemoveASMInstPass() : llvm::FunctionPass(ID){}

    bool runOnFunction(llvm::Function &F) override;
};

}

#endif  // ASER_PTA_REMOVEASMINSTPASS_H
