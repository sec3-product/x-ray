#ifndef RACEDETECTOR_OMPENGINE_H
#define RACEDETECTOR_OMPENGINE_H

#include <llvm/Analysis/TypeBasedAliasAnalysis.h>

#include <chrono>

#include "Graph/Event.h"
#include "Graph/ReachGraph.h"
#include "PTAModels/GraphBLASModel.h"
#include "RaceDetectionPass.h"

// See https://stackoverflow.com/a/495056
// for ways to move implementation out of header file

namespace aser {
using CallingCtx = std::pair<std::vector<CallEvent *>, TID>;

class OMPEngine {
private:
    using CallStack = std::vector<const CallGraphNodeTy *>;
    using InstructionMap = std::map<const llvm::Instruction *const, const llvm::Instruction *const>;
    bool skipLoadStore;
    const llvm::Instruction *staticForFiniInst;
    int isMasterOnly;
    int isSingleOnly;
    bool isTheFirstThread = true;
    bool isTeams;
    bool isNestedFork;

    const llvm::Function *ompEntryFunc;

    std::set<const llvm::BasicBlock *> onceOnlyBasicBlocks;
    std::set<const llvm::Instruction *> onceOnlyTaskInstructions;

    // Points to Analysis
    PTA *pta;
    // Type-based Alias Analysis
    llvm::TypeBasedAAResult &tbaa;
    // used by tbaa
    llvm::AAQueryInfo aaqi;

    // Dirty way to acess pointer analysis and array index analysis
    RaceDetectionPass &pass;

    // Store start/end of blocks executed by a single thread only
    // Currently these are single and master
    // InstructionMap singleBlocks;

    // Taskgroup start/end points
    // first item is event id before start of task group
    // second item is joinEvent to join all tasks in group
    std::vector<std::pair<NodeID, const JoinEvent *>> taskgroups;

    // Used to skip instructions until some point
    // currently used for omp master and omp single
    // const llvm::Instruction *skipUntil;

    // Used to simulate function callstack
    CallStack callStack;

    // for each thread (TID), we record the sync point (NodeID)
    std::map<uint64_t, std::vector<NodeID>> syncData;

    // Used to fill syncdata and hold locksets
    ReachGraph graph;

    // Used to record call stack inside a OpenMP region
    std::map<TID, std::vector<CallEvent *>> callEventTraces;

    // NOTE: Since OpenMP race detection are done in place
    // we need the current callstack from global race detector
    // to obtain the calling context of each OpenMP region
    CallingCtx &callingCtx;
    // This is the __kmpc_fork_call for the current OpenMP region
    // the debugging info on this call corresponds to #pragma omp parallel
    const llvm::Instruction *ompRegion;

    std::vector<const ObjTy *> objs;
    llvm::DenseMap<const ObjTy *, unsigned int> objIdxCache;
    std::map<unsigned int, std::map<uint64_t, std::vector<MemAccessEvent *>>> memReads;
    std::map<unsigned int, std::map<uint64_t, std::vector<MemAccessEvent *>>> memWrites;
    std::map<unsigned int, std::map<uint64_t, std::vector<MemAccessEvent *>>> readsMask;
    std::map<unsigned int, std::map<uint64_t, std::vector<MemAccessEvent *>>> writesMask;

    void addOMPReadOrWrite(TID threadID, MemAccessEvent *event, bool isWrite);

    // Build SHB graph, connectivity engine, and thread trace for this node
    // If a new node is created the curEvent argument is updated to point to the new node
    // input:
    //   callNode - Node to be explored
    //   thread - thread that is "executing" the callNode
    //   curEvent - the most recently created node in the callstack
    void traverse(const CallGraphNodeTy *callNode, StaticThread *thread, Event *curEvent = nullptr);

    // return true if this funciton was able to handle the callnode
    bool visitOMP(const llvm::Instruction *inst, const ctx *const context, StaticThread *thread);

    std::vector<unsigned int> collectShared() const;
    void shrinkShared(std::vector<unsigned int> &);
    void detectRace(const CallGraphNodeTy *callNode1, const CallGraphNodeTy *callNode2);
    bool enumerateRacePair(std::vector<MemAccessEvent *> wset, std::vector<MemAccessEvent *> xset, const ObjTy *obj);
    std::string getRaceSig(const llvm::Instruction *inst1, const llvm::Instruction *inst2);

public:
    OMPEngine(RaceDetectionPass &pass, CallingCtx &CC, bool teams, bool nestedFork)
        : pass(pass),
          pta(pass.getAnalysis<PointerAnalysisPass<PTA>>().getPTA()),
          graph(pass),
          tbaa(pass.getTbaa()),
          callingCtx(CC),
          isTeams(teams),
          isNestedFork(nestedFork) {}
    void analyze(const llvm::Instruction *const ompForkCall, const ctx *const context);
};

class OMPRaceDetect {
private:
    RaceDetectionPass &pass;
    // to memorize the omp_outlined function we have encountered
    std::set<const llvm::Function *> cache;
    bool teams = false;
    bool nestedFork = false;
    std::set<const llvm::Instruction *> maxThreadInsts;

public:
    OMPRaceDetect(RaceDetectionPass &pass) : pass(pass) {}
    void analyze(const ctx *const context, const llvm::Instruction *const ompForkCall, CallingCtx &callingCtx);
    inline bool isAlreadyAnalyzed(const llvm::Function *f) const { return cache.find(f) != cache.end(); }
    void setTeams(bool b) { teams = b; }
    bool getTeams() { return teams; }
    void setNestedFork(bool b) { nestedFork = b; }
    void addMaxThreadInstructions(const llvm::Instruction *inst);
};

}  // namespace aser

#endif
