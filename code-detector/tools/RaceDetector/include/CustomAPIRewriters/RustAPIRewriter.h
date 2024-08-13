#ifndef ASER_PTA_RUSTAPIWRITER_H
#define ASER_PTA_RUSTAPIWRITER_H

namespace llvm {
class Module;
}

namespace aser {

class RustAPIRewriter {
public:
    static void rewriteModule(llvm::Module *M);
};

}  // namespace aser

#endif  // ASER_PTA_RUSTAPIWRITER_H
