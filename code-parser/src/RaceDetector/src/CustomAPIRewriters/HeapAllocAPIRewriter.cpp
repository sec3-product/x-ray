#include <vector>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>

#include "conflib/conflib.h"
#include "o2/Util/Demangler.h"

#include "CustomAPIRewriters/HeapAllocAPIRewriter.h"

using namespace std;
using namespace o2;
using namespace llvm;

static void rewriteUserSpecifiedHeapAPI(Module *M, const vector<string> &heapAPIs) {
    Type *allocPtrType = PointerType::get(IntegerType::get(M->getContext(), 8), 0);
    auto allocFun = M->getOrInsertFunction(HeapAllocAPIRewriter::getCanonicalizedAPIName(), allocPtrType);

    llvm::cast<Function>(allocFun.getCallee())->setReturnDoesNotAlias();
    IRBuilder<> builder(M->getContext());
    for (auto &F : *M) {
        string name = stripNumberPostFix(F.getName()).str();
        if (std::find(heapAPIs.begin(), heapAPIs.end(), name) != heapAPIs.end()) {
            F.deleteBody();

            auto entryBB = BasicBlock::Create(M->getContext(), "o2.alloc", &F);
            builder.SetInsertPoint(entryBB);
            Value *ptr = builder.CreateCall(allocFun);
            if (F.getReturnType() != allocFun.getFunctionType()->getReturnType()) {
                ptr = builder.CreateBitCast(ptr, F.getReturnType());
            }
            builder.CreateRet(ptr);

            F.setReturnDoesNotAlias(); // append no alias metadata to it.
            F.addFnAttr(Attribute::AlwaysInline);
        }
    }
}


void HeapAllocAPIRewriter::rewriteModule(llvm::Module *M) {
    auto heapAPIs = conflib::Get<std::vector<std::string>>("heapAllocFunctions", {});
    if (!heapAPIs.empty()) {
        rewriteUserSpecifiedHeapAPI(M, heapAPIs);
    }
}

