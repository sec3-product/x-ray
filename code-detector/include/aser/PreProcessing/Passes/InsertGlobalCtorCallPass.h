//
// Created by peiming on 3/14/20.
//

#ifndef ASER_PTA_INSERTGLOBALCTORCALLPASS_H
#define ASER_PTA_INSERTGLOBALCTORCALLPASS_H

#include <llvm/Pass.h>

namespace aser {

class InsertGlobalCtorCallPass : public llvm::ModulePass {
public:
    static char ID;
    explicit InsertGlobalCtorCallPass() : llvm::ModulePass(ID) {}

    bool runOnModule(llvm::Module &M) override;
};

}

#endif  // ASER_PTA_INSERTGLOBALCTORCALLPASS_H
