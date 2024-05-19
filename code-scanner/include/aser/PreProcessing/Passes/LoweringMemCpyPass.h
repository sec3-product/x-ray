//
// Created by peiming on 1/22/20.
//

#ifndef ASER_PTA_LOWERINGMEMCPYPASS_H
#define ASER_PTA_LOWERINGMEMCPYPASS_H

#include <llvm/Pass.h>
#include <llvm/IR/IRBuilder.h>

namespace llvm {

template <typename T, unsigned N> class SmallVector;
class NoFolder;
}

namespace aser {

class LoweringMemCpyPass : public llvm::ModulePass {
private:
    llvm::Type *idxType = nullptr;

    void lowerMemCpyForType(llvm::Type *type, llvm::Value *src, llvm::Value *dst,
                            llvm::SmallVector<llvm::Value *, 5> &idx, llvm::IRBuilder<llvm::NoFolder> &builder);
public:
    static char ID;
    explicit LoweringMemCpyPass() : llvm::ModulePass(ID) {}

    bool runOnModule(llvm::Module &) override;
};

}

#endif  // ASER_PTA_LOWERINGMEMCPYPASS_H
