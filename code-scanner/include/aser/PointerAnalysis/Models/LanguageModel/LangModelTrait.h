//
// Created by peiming on 10/19/19.
//
#ifndef ASER_PTA_LANGMODELTRAIT_H
#define ASER_PTA_LANGMODELTRAIT_H

#include <llvm/IR/LegacyPassManager.h>

namespace aser {

// a language langModel trait
template <typename Model>
struct LangModelTrait {
    // context type
    using CtxTy = typename Model::UnknownTypeError;

    // data structure for points-to set
    using PointsToTy = typename Model::UnknownTypeError;

    // should have following types
    using LangModelTy = typename Model::UnknownTypeError;

    // memory langModel
    using MemModelTy = typename Model::UnknownTypeError;

    // constraint callgraph type
    using ConsGraphTy = typename Model::UnknownTypeError;

    // call callgraph type
    using CallGraphTy = typename Model::UnknownTypeError;

    // should have following functions:

    // add required passes into pass manager
    // static inline void addDependentPasses(PassManager *passes);

    // set analysis usage
    // static void getAnalysisUsage(AnalysisUsage &AU);

    // build initial langModel from a llvm module
    // static inline LangModelTy *buildInitModel(llvm::Module *M, StringRef
    // entry)

    // get constructed call callgraph
    // static inline ConsGraphTy *getConsGraph(LangModelTy *langModel);

    // get constructed constraint callgraph
    // static inline CallGraphTy *getCallGraph(LangModelTy *langModel);

    // indirect call resolved, and the resolved callnode is passed in
    // static inline void resolveCallAt(LangModelTy *langModel,
    // CallGraphTy::CGNodeKind *node)

    // get corresponding pointer nodes that represent v
    // static inline void getNodesForValue(LangModelTy *langModel, const Value
    // *v, std::set<NodeBase *> &result);

    // get the entry function name
    // static inline StringRef getEntryName(LangModelTy *langModel);
};

}  // namespace aser

#endif