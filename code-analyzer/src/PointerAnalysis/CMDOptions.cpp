#include <llvm/Support/CommandLine.h>

using namespace llvm;

cl::opt<bool>
    ConfigPrintConstraintGraph("consgraph",
                               cl::desc("Dump Constraint Graph to dot file"));
cl::opt<bool> ConfigPrintCallGraph("callgraph",
                                   cl::desc("Dump SHB graph to dot file"));
cl::opt<bool>
    ConfigDumpPointsToSet("dump-pts",
                          cl::desc("Dump the Points-to Set of every pointer"));
cl::opt<bool> USE_MEMLAYOUT_FILTERING(
    "Xmemlayout-filtering",
    cl::desc("Use memory layout to filter out incompatible types in "
             "field-sensitive PTA"));
cl::opt<bool> CONFIG_VTABLE_MODE("Xenable-vtable",
                                 cl::desc("model vtable specially"),
                                 cl::init(false));

size_t MaxIndirectCallTarget = 2;
