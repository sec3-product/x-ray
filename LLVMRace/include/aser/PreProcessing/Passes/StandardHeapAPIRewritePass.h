//
// Created by peiming on 3/9/20.
//

#ifndef ASER_PTA_STANDARDHEAPAPIREWRITEPASS_H
#define ASER_PTA_STANDARDHEAPAPIREWRITEPASS_H

#include <llvm/Pass.h>

class StandardHeapAPIRewritePass : public llvm::ModulePass {

public:
    static char ID;
    explicit StandardHeapAPIRewritePass() : ModulePass(ID) {}

    bool runOnModule(llvm::Module &M) override;
};


#endif  // ASER_PTA_STANDARDHEAPAPIREWRITEPASS_H
