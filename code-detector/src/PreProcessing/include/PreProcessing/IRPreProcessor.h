//
// Created by peiming on 2/26/20.
//

#ifndef ASER_PTA_IRPREPROCESSOR_H
#define ASER_PTA_IRPREPROCESSOR_H

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>

namespace aser {

// maybe just a PreprocessingModule function is enough, we make it as a class in
// case in the future we might want to configure it.
class IRPreProcessor {
public:
  IRPreProcessor() = default;
  void runOnModule(llvm::Module &M,
                   std::function<void(llvm::legacy::PassManagerBase &)> &&);
};

} // namespace aser

#endif // ASER_PTA_IRPREPROCESSOR_H
