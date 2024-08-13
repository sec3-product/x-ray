#include "CustomAPIRewriters/IndirectTargetRewriter.h"
#include "CustomAPIRewriters/RustAPIRewriter.h"

namespace llvm {
class Module;
}

namespace aser {

// This is run before all optimization passes
void rewriteUserSpecifiedAPI(llvm::Module *M) {
    RustAPIRewriter::rewriteModule(M);

    IndirectTargetRewriter::rewriteModule(M);
}

}  // namespace aser
