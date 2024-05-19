//
// Created by peiming on 10/16/20.
//

#ifndef ASER_PTA_INDIRECTTARGETREWRITER_H
#define ASER_PTA_INDIRECTTARGETREWRITER_H

namespace llvm { class Module; }

namespace aser {

class IndirectTargetRewriter {
public:
    static void rewriteModule(llvm::Module *M);
};

}

#endif  // ASER_PTA_INDIRECTTARGETREWRITER_H
