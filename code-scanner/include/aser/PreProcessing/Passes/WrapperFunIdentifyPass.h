//
// Created by peiming on 2/10/20.
//

#ifndef ASER_PTA_WRAPPERFUNIDENTIFYPASS_H
#define ASER_PTA_WRAPPERFUNIDENTIFYPASS_H

#include <llvm/Pass.h>

class WrapperFunIdentifyPass : public llvm::ModulePass {

public:
    static char ID;
    explicit WrapperFunIdentifyPass() : ModulePass(ID) {}

    bool runOnModule(llvm::Module &M) override;
};


#endif  // ASER_PTA_WRAPPERFUNIDENTIFYPASS_H
