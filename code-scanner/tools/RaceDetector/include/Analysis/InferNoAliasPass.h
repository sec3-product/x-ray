//
// Created by peiming on 4/17/20.
//

#ifndef ASER_PTA_INFERNOALIASPASS_H
#define ASER_PTA_INFERNOALIASPASS_H

#include <llvm/Pass.h>

namespace aser {

llvm::ModulePass *createInferNoAliasPass();

}

#endif  // ASER_PTA_INFERNOALIASPASS_H
