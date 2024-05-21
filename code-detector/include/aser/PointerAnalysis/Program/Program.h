//
// Created by peiming on 8/15/19.
//
#ifndef ASER_PTA_PROGRAMPOINT_H
#define ASER_PTA_PROGRAMPOINT_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

namespace aser {

template <typename ctx>
class ProgramPoint {
private:
    const llvm::Instruction *inst;
    const ctx *context;

public:
    ProgramPoint() = delete;

    ProgramPoint(const llvm::Instruction *inst, const ctx *context) : inst(inst), context(context) {}

    [[nodiscard]] inline const llvm::Instruction *getInstruction() const { return this->inst; }

    [[nodiscard]] inline const ctx *getContext() const { return this->context; }

    [[nodiscard]] inline bool isCallSite() const {
        if (inst == nullptr) {
            return false;
        }
        return llvm::isa<llvm::CallInst>(inst) || llvm::isa<llvm::InvokeInst>(inst);
    }
};

}  // namespace aser
#endif