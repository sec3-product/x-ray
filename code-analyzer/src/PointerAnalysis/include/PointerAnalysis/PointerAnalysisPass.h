#pragma once

#include <memory>

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Pass.h>

template <typename Solver>
class PointerAnalysisPass : public llvm::ImmutablePass {
private:
  std::unique_ptr<Solver> solver; // owner of the solver

public:
  static char ID;
  PointerAnalysisPass() : solver(nullptr), llvm::ImmutablePass(ID) {}

  void analyze(llvm::Module *M, llvm::StringRef entry = "cr_main") {
    if (solver.get() != nullptr) {
      if (solver->getLLVMModule() == M &&
          entry.equals(solver->getEntryName())) {
        return;
      }
    }
    // release previous context
    Solver::CT::release();
    solver.reset(new Solver());
    solver->analyze(M, entry);
  }

  Solver *getPTA() const {
    assert(solver.get() != nullptr &&
           "call analyze() before getting the pta instance");
    return solver.get();
  }

  void release() {
    // release the memory hold by the correct solver
    solver.reset(nullptr);
  }
};

template <typename Solver> char PointerAnalysisPass<Solver>::ID = 0;

template <typename Solver> void registerPointerAnalysisPass() {
  static llvm::RegisterPass<PointerAnalysisPass<Solver>> PAP(
      "Pointer Analysis Wrapper Pass", "Pointer Analysis Wrapper Pass", true,
      true);
}

extern size_t MaxIndirectTarget;
