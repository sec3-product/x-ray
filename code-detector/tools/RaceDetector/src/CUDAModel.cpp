#include "CUDAModel.h"

#include "RDUtil.h"
#include "aser/PointerAnalysis/Program/CallSite.h"

using namespace llvm;

namespace aser {
namespace CUDAModel {

const llvm::Function *getOutlinedFunction(const llvm::Instruction &forkCall) {
    assert(isFork(&forkCall) && "Cannot get outlined function from non omp fork call");

    // Offset to omp outlined funciton in fork call
    int callSiteOffset = 2;

    aser::CallSite CS(&forkCall);
    auto func = llvm::dyn_cast<llvm::Function>(CS.getArgOperand(callSiteOffset)->stripPointerCasts());
    assert(func && "Could not find forked function");

    return func;
}

const std::set<StringRef> ForkNames{
    "__cudaPushCallConfiguration",
};

// Utility helper function
static inline bool containsAny(StringRef src, std::set<StringRef> list) { return list.find(src) != list.end(); }

bool isFork(const StringRef func_name) { return containsAny(func_name, ForkNames); }

bool isFork(const Function *func) { return containsAny(func->getName(), ForkNames); }

bool isFork(const Instruction *inst) {
    if (isa<CallBase>(inst)) {
        aser::CallSite CS(inst);
        return isFork(CS.getCalledFunction());
    }
    return false;
}

bool isGetThreadIdX(const Function *func) { return isGetThreadIdX(func->getName()); }
bool isGetThreadIdX(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.tid.x"; }
bool isGetThreadIdY(const Function *func) { return isGetThreadIdY(func->getName()); }
bool isGetThreadIdY(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.tid.y"; }
bool isGetThreadIdZ(const Function *func) { return isGetThreadIdZ(func->getName()); }
bool isGetThreadIdZ(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.tid.z"; }

bool isGetBlockIdX(const Function *func) { return isGetBlockIdX(func->getName()); }
bool isGetBlockIdX(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.ctaid.x"; }
bool isGetBlockIdY(const Function *func) { return isGetBlockIdY(func->getName()); }
bool isGetBlockIdY(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.ctaid.y"; }
bool isGetBlockIdZ(const Function *func) { return isGetBlockIdZ(func->getName()); }
bool isGetBlockIdZ(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.ctaid.z"; }

bool isGetBlockDimX(const Function *func) { return isGetBlockDimX(func->getName()); }
bool isGetBlockDimX(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.ntid.x"; }
bool isGetBlockDimY(const Function *func) { return isGetBlockDimY(func->getName()); }
bool isGetBlockDimY(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.ntid.y"; }
bool isGetBlockDimZ(const Function *func) { return isGetBlockDimZ(func->getName()); }
bool isGetBlockDimZ(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.ntid.z"; }

bool isGetGridDimX(const Function *func) { return isGetGridDimX(func->getName()); }
bool isGetGridDimX(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.nctaid.x"; }
bool isGetGridDimY(const Function *func) { return isGetGridDimY(func->getName()); }
bool isGetGridDimY(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.nctaid.y"; }
bool isGetGridDimZ(const Function *func) { return isGetGridDimZ(func->getName()); }
bool isGetGridDimZ(const StringRef func_name) { return func_name == "llvm.nvvm.read.ptx.sreg.nctaid.z"; }

bool isAnyCUDACall(const Function *func) { return isAnyCUDACall(func->getName()); }
bool isAnyCUDACall(const StringRef func_name) {
    return func_name.startswith("__cuda") || func_name.startswith("cuda") || func_name.startswith("llvm.nvvm.");
}

enum class Type { None = 0, Fork };

Type getType(const Function *func) {
    if (isFork(func)) {
        return Type::Fork;
    }
    // TODO: addd the others

    return Type::None;
}

}  // namespace CUDAModel
}  // namespace aser
