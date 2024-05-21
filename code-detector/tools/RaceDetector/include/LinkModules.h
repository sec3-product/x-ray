//
// Created by peiming on 8/14/20.
//

#ifndef ASER_PTA_LINKMODULES_H
#define ASER_PTA_LINKMODULES_H

#include "llvm/ADT/StringSet.h"
#include "llvm/Linker/IRMover.h"

namespace llvm {
    class Module;
    class StructType;
    class Type;
}

// to get rid of exit called inside llvm
// FIXME: when LLVM support configuration when errors happens, delete the file and use LLVM's version
namespace aser {
    /// This class provides the core functionality of linking in LLVM. It keeps a
    /// pointer to the merged module so far. It doesn't take ownership of the
    /// module since it is assumed that the user of this class will want to do
    /// something with it after the linking.
    class Linker {
        llvm::IRMover Mover;

    public:
        enum Flags {
            None = 0,
            OverrideFromSrc = (1 << 0),
            LinkOnlyNeeded = (1 << 1),
        };

        Linker(llvm::Module &M);

        /// Link \p Src into the composite.
        ///
        /// Passing OverrideSymbols as true will have symbols from Src
        /// shadow those in the Dest.
        ///
        /// Passing InternalizeCallback will have the linker call the function with
        /// the new module and a list of global value names to be internalized by the
        /// callback.
        ///
        /// Returns true on error.
        bool linkInModule(std::unique_ptr<llvm::Module> Src, unsigned Flags = Flags::None,
                          std::function<void(llvm::Module &, const llvm::StringSet<> &)>
                          InternalizeCallback = {});

        static bool linkModules(llvm::Module &Dest, std::unique_ptr<llvm::Module> Src,
                                unsigned Flags = Flags::None,
                                std::function<void(llvm::Module &, const llvm::StringSet<> &)>
                                InternalizeCallback = {});
    };
}

#endif //ASER_PTA_LINKMODULES_H
