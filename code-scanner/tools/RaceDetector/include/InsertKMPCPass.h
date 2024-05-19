//
// Created by peiming on 11/15/19.
//

#ifndef LOOPBOUNDS_InsertKMPCPass_H
#define LOOPBOUNDS_InsertKMPCPass_H

#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

#include <vector>

#include "ConstantPropagationPass.h"
#include "aser/PointerAnalysis/Program/CallSite.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
using namespace std;

namespace aser {

class InsertKMPCPass : public ModulePass {
public:
    static char ID;
    explicit InsertKMPCPass() : ModulePass(ID) {}

    bool runOnModule(llvm::Module &module) override;
};

}  // namespace aser

#endif