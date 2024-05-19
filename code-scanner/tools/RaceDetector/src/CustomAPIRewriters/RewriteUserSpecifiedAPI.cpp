#include "CustomAPIRewriters/IndirectTargetRewriter.h"
#include "CustomAPIRewriters/RustAPIRewriter.h"
#include "CustomAPIRewriters/ThreadProfileRewriter.h"

namespace llvm {
class Module;
}

namespace aser {

// This is run before all optimization passes
void rewriteUserSpecifiedAPI(llvm::Module *M) {
    RustAPIRewriter::rewriteModule(M);

    ThreadProfileRewriter::rewriteModule(M);

    IndirectTargetRewriter::rewriteModule(M);
}

}  // namespace aser