#ifndef ASER_PTA_SMALLTALKAPIWRITER_H
#define ASER_PTA_SMALLTALKAPIWRITER_H

namespace llvm {
class Module;
}

namespace aser {

class RustAPIRewriter {
public:
    static void rewriteModule(llvm::Module *M);
};

}  // namespace aser

#endif  // ASER_PTA_SMALLTALKAPIWRITER_H
