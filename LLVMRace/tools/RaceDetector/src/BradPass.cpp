#include "BradPass.h"
#include <llvm/Analysis/LoopInfo.h>

#include "RDUtil.h"

using namespace llvm;
using namespace aser;

void BradPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
}
bool BradPass::runOnFunction(llvm::Function &F) {
    this->SE = &this->getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    this->LI = &this->getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    this->DT = &this->getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    return false;
}

bool BradPass::bradtesting(const llvm::GetElementPtrInst *inst) {
    auto loop = LI->getLoopFor(inst->getParent());
    if (!loop) {
        return false;
    }
    auto inductionVar = loop->getInductionVariable(*SE);
    if (!inductionVar) {
        llvm::outs() << "FAILEDI: " << *inst << "\n" << getSourceLoc(inst).getSourceLine();
        return false;
    }
    llvm::outs() << *inst << "\n" << getSourceLoc(inst).getSourceLine();
    llvm::outs() << "I: " << *inductionVar << "\n" << getSourceLoc(cast<Instruction>(inductionVar)).getSourceLine();

    auto bounds = loop->getBounds(*SE);
    if (bounds.hasValue()) {
        auto iVal = &bounds->getInitialIVValue();
        llvm::outs() << *iVal << "\n";
        if (auto sext = dyn_cast<SExtInst>(iVal)) {
            auto op = sext->getOperand(0);
            llvm::outs() << *op << "\n";
            if (auto opi = dyn_cast<Instruction>(op)) {
                iVal = opi;
            } else if (auto arg = dyn_cast<Argument>(op)) {
                iVal = arg;
                llvm::outs() << iVal->getName() << "\n";
            }
        }

        std::string sline = "";
        if (isa<Instruction>(iVal)) {
            sline = getSourceLoc(iVal).getSourceLine();
            llvm::outs() << iVal->getName() << "\n";
        } else {
            sline = iVal->getName().str();
        }

        llvm::outs() << *iVal << "\n" << sline;

        auto keywords = {"first", "last"};
        for (auto const &keyword : keywords) {
            if (sline.find(keyword) != std::string::npos) {
                return true;
            }
        }
    }

    return false;
}

char BradPass::ID = 0;
static RegisterPass<BradPass> AIA("OpenMP Brad Array Index Analysis", "OpenMp Brad Array Index Analysis",
                                  true, /*CFG only*/
                                  true /*is analysis*/);