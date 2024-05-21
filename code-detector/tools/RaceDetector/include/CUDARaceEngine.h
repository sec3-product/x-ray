#ifndef RACEDETECTOR_CUDAENGINE_H
#define RACEDETECTOR_CUDAENGINE_H

#include <llvm/Analysis/TypeBasedAliasAnalysis.h>

#include <chrono>

#include "Graph/Event.h"
#include "Graph/ReachGraph.h"
#include "PTAModels/GraphBLASModel.h"
#include "RaceDetectionPass.h"

// See https://stackoverflow.com/a/495056
// for ways to move implementation out of header file

namespace aser {

class CUDAEngine {
private:
    using CallStack = std::vector<const CallGraphNodeTy *>;
    using InstructionMap = std::map<const llvm::Instruction *const, const llvm::Instruction *const>;

    // Points to Analysis
    PTA &pta;
    // Type-based Alias Analysis
    llvm::TypeBasedAAResult &tbaa;
    // used by tbaa
    llvm::AAQueryInfo aaqi;

    // Dirty way to acess pointer analysis and array index analysis
    RaceDetectionPass &pass;

    // Store start/end of blocks executed by a single thread only
    // Currently these are single and master
    InstructionMap singleBlocks;

    // Used to skip instructions until some point
    const llvm::Instruction *skipUntil;

    // Used to simulate function callstack
    CallStack callStack;

    // for each thread (TID), we record the sync point (NodeID)
    std::map<uint64_t, std::vector<NodeID>> syncData;

    // Used to fill syncdata and hold locksets
    ReachGraph graph;

    // Used to record call stack
    // TODO: OMPEngine may not need a call stack?
    std::map<TID, std::vector<CallEvent *>> callEventTraces;

    // NOTE: Since CUDA race detection are done in place
    // we need the current callstack from global race detector
    // to obtain the calling context of each CUDA kernel func
    CallingCtx &callingCtx;
    const llvm::Instruction *cudaRegion;

    std::map<const ObjTy *const, std::map<uint64_t, std::set<MemAccessEvent *>>> reads;
    std::map<const ObjTy *const, std::map<uint64_t, std::set<MemAccessEvent *>>> writes;
    std::map<const ObjTy *const, std::map<uint64_t, std::set<MemAccessEvent *>>> readsMask;
    std::map<const ObjTy *const, std::map<uint64_t, std::set<MemAccessEvent *>>> writesMask;

    void addRead(TID threadID, MemAccessEvent *event);
    void addWrite(TID threadID, MemAccessEvent *event);

    // Build SHB graph, connectivity engine, and thread trace for this node
    // If a new node is created the curEvent argument is updated to point to the new node
    // input:
    //   callNode - Node to be explored
    //   thread - thread that is "executing" the callNode
    //   curEvent - the most recently created node in the callstack
    void traverse(const CallGraphNodeTy *callNode, StaticThread *thread, Event *curEvent = nullptr);

    // return true if this funciton was able to handle the callnode
    bool visitCUDA(const llvm::Instruction *inst, const ctx *const context, StaticThread *thread);

    std::vector<const ObjTy *> collectShared() const;
    void shrinkShared(std::vector<const ObjTy *> &);
    void detectRace(const CallGraphNodeTy *callNode1, const CallGraphNodeTy *callNode2);
    bool enumerateRacePair(std::set<MemAccessEvent *> wset, std::set<MemAccessEvent *> xset, const ObjTy *obj);
    std::string getRaceSig(const llvm::Instruction *inst1, const llvm::Instruction *inst2);

public:
    CUDAEngine(RaceDetectionPass &pass, CallingCtx &CC)
        : skipUntil(nullptr),
          pass(pass),
          pta(*pass.getAnalysis<PointerAnalysisPass<PTA>>().getPTA()),
          graph(pass),
          tbaa(pass.getTbaa()),
          callingCtx(CC) {}

    void analyze(const llvm::Instruction *const cudaKernelCall, const ctx *const context);
};

class CUDARaceDetect {
private:
    RaceDetectionPass &pass;
    // to memorize the omp_outlined function we have encountered
    std::set<const llvm::Function *> cache;

public:
    CUDARaceDetect(RaceDetectionPass &pass) : pass(pass) {}
    void analyze(const ctx *const context, const llvm::Instruction *const cudaKernelCall, CallingCtx &callingCtx);
};

}  // namespace aser

#endif
