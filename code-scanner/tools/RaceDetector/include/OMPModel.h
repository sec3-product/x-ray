#ifndef RACEDETECTOR_OMPMODEL_H
#define RACEDETECTOR_OMPMODEL_H

#include <set>

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace aser {

// forward declaration
class CallSite;

namespace OMPModel {
bool isGetThreadNum(const llvm::StringRef name);

bool isGetThreadNum(const llvm::Function *func);
bool isGetGlobalThreadNum(const llvm::StringRef name);

bool isGetGlobalThreadNum(const llvm::Function *func);

bool isDispatchNext(const llvm::StringRef name);
bool isDispatchNext(const llvm::Function *func);
bool isDispatchInit(const llvm::StringRef name);
bool isDispatchInit(const llvm::Function *func);

bool isFork(const llvm::StringRef func_name);

bool isFork(const llvm::Function *func);

bool isFork(const llvm::Instruction *inst);
bool isForkTeams(const llvm::Instruction *inst);

bool isPushNumThreads(const llvm::Function *func);
bool isPushNumTeams(const llvm::Function *func);

bool isStaticForInit(const llvm::StringRef name);

bool isStaticForInit(const llvm::Function *func);

bool isStaticForFini(const llvm::StringRef name);

bool isStaticForFini(const llvm::Function *func);

bool isReduce(const llvm::StringRef name);
bool isReduce(const llvm::Function *func);

bool isReduceEnd(const llvm::StringRef name);
bool isReduceEnd(const llvm::Function *func);

bool isBarrier(const llvm::StringRef name);

bool isBarrier(const llvm::Function *func);

bool isSingle(const llvm::StringRef name);

bool isSingle(const llvm::Function *func);

bool isSingleEnd(const llvm::StringRef name);

bool isSingleEnd(const llvm::Function *func);

bool isOrderedStart(const llvm::Function *func);
bool isOrderedEnd(const llvm::Function *func);

bool isCriticalStart(const llvm::StringRef name);

bool isCriticalStart(const llvm::Function *func);

bool isCriticalEnd(const llvm::StringRef name);

bool isCriticalEnd(const llvm::Function *func);

bool isCritical(const llvm::StringRef name);

bool isCritical(const llvm::Function *func);

bool isMaster(const llvm::StringRef name);

bool isMaster(const llvm::Function *func);

bool isMasterEnd(const llvm::StringRef name);

bool isMasterEnd(const llvm::Function *func);

bool isMasterOrSingle(const llvm::StringRef name);

bool isMasterOrSingle(const llvm::Function *func);
bool isMasterEndOrSingleEnd(const llvm::Function *func);

bool isSetLock(const llvm::StringRef name);

bool isSetLock(const llvm::Function *func);

bool isUnsetLock(const llvm::StringRef name);

bool isUnsetLock(const llvm::Function *func);

bool isTask(const llvm::StringRef name);

bool isTask(const llvm::Function *func);

bool isTaskAlloc(const llvm::Function *func);
bool isTaskDepend(const llvm::Function *func);

bool isTaskWait(const llvm::StringRef name);

bool isTaskWait(const llvm::Function *func);

bool isTaskGroupStart(const llvm::StringRef name);

bool isTaskGroupStart(const llvm::Function *func);

bool isTaskGroupEnd(const llvm::StringRef name);

bool isTaskGroupEnd(const llvm::Function *func);

bool isAnyOpenMPCall(const llvm::Function *func);

bool isOmpDebug(const llvm::Function *func);

bool isReadAPI(const llvm::Function *func);
bool isWriteAPI(const llvm::Function *func);
bool isReadWriteAPI(const llvm::Function *func);

bool isGetThreadNum(const llvm::Function *func);
bool isGetMaxThreadsNum(const llvm::Function *func);

// return the callback function that contains the body of the omp task or nullptr if it cannot be found
const llvm::Function *getTaskFunction(const aser::CallSite &taskCallSite);

// Returns a pointer to the first instruction after the master/single region
//  Fails assertion if passes a non master/single CallInstruction
const llvm::Instruction *getEndOfMasterOrSingleSection(const aser::CallSite &CS);

// return a pointer to the first instruction in the block after the
//  openmp reduction code (we assume openmp reduction is race free)
const llvm::Instruction *getEndOfReduce(const aser::CallSite &CS);

const llvm::Instruction *getEndOfOrdered(const aser::CallSite &CS);

// Return a list of lastIter values identified by looking for StaticForInitCalls
// llvm::SmallVector<llvm::Value *, 1> getLastIters(const llvm::Function *F);

// Return a map of instructions where
//   key = branch instruction before the start of a lastiter block
//   value = instruction after the end of a lastiter block
std::map<const llvm::Instruction *const, const llvm::Instruction *const> getlastIterMap(const llvm::Function *F);

// Given an openmp fork call get the corresponding outlined function
const llvm::Function *getOutlinedFunction(const llvm::Instruction &forkCall);

class StaticForInitCall : public llvm::CallInst {
    // https://github.com/llvm/llvm-project/blob/release/9.x/openmp/runtime/src/kmp_sched.cpp#L794
    // @param    loc       Source code location
    // @param    gtid      Global thread id of this thread
    // @param    schedtype  Scheduling type
    // @param    plastiter Pointer to the "last iteration" flag
    // @param    plower    Pointer to the lower bound
    // @param    pupper    Pointer to the upper bound
    // @param    pstride   Pointer to the stride
    // @param    incr      Loop increment
    // @param    chunk     The chunk size

public:
    // Return a pointer to the plastiter argument
    llvm::Value *getPLastIter() const { return getArgOperand(3); }

    // TODO: add other getters as needed
};

}  // namespace OMPModel
}  // namespace aser

#endif
