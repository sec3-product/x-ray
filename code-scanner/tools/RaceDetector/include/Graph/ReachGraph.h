//
// Created by peiming on 11/18/19.
//
#ifndef RACEDETECTOR_REACHGRAPH_H
#define RACEDETECTOR_REACHGRAPH_H

#include <llvm/Analysis/CFG.h>
#include <llvm/IR/Instruction.h>
#include <llvm/Support/DOTGraphTraits.h>
#include <stdint.h>

#include <sstream>

#include "Event.h"
#include "LockGraph.h"
#include "RaceDetectionPass.h"
#include "Races.h"
#include "ReachabilityEngine.h"
#include "StaticThread.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/InitializePasses.h"

extern bool DEBUG_LOCK;  // for debug only
extern bool DEBUG_LOCK_STACK;
extern bool DEBUG_DUMP_LOCKSET;
extern bool CONFIG_NO_MISS_MATCH_API;

namespace aser {

struct UnLockState {
    LocksetManager::ID lastUnLockLocksetId;
    const llvm::Instruction *lastUnlockPtr = nullptr;
    const ctx *lastUnlockPtrContext;
    std::set<MemAccessEvent *> unlockRegionMemAccesses;
};

enum class LockState : uint8_t { Init, Locked, TryLocked, RDLocked, WRLocked, Unlocked };

class ReachGraph {
private:
    RaceDetectionPass &pass;
    // a cached graph connectivity engine
    aser::ReachabilityEngine reachEngine;
    // for each thread (TID), we record the sync point (NodeID)
    std::map<TID, std::vector<EventID>> syncData;
    // for each thread (TID), we record their events for acquiring lock
    // for deadlock detection
    std::map<TID, std::vector<const LockEvent *>> lockData;

    // for data ownership transfer detection
    std::map<TID, std::map<const llvm::Instruction *, std::set<const MemAccessEvent *>>> instr2MemEvents;

    // Map Barrier Instructions to
    //   Map of teamID and last event to read the barrier as part of a team
    // std::map<TID, const BarrierEvent *> barrierMap;
    std::map<const llvm::Instruction *, std::map<TID, const BarrierEvent *>> barrierMap;

    // special state machine funcs containing path-sensitive lock/unlock
    std::set<const llvm::Function *> specialLockStateMachines;

    // Track lockset info
    LocksetManager locksetManager;
    // FIXME:
    // this is a heuristic to fix for contorl-flow related
    // the pattern we want to handle:
    // ```
    // kmpc_critical()
    // if (xxx) {
    //     kmpc_end_critical()
    // }
    // ... do something here
    // kmpc_end_critical()
    // ```
    // NOTE: pthread cannot be handled by this pattern
    // because there will always a load before pthread_mutex_unlock
    // (to read the lock object)
    // FIXME: will this cause some locks to hold forever?

    std::map<std::vector<const ObjTy *>, UnLockState> lockSetUnlockState;
    // tracking rwlock
    // need to maintain a state per rwlock object
    std::map<std::vector<const ObjTy *>, int> rwLockState;

    std::map<const llvm::Value *, const llvm::Value *> reentrantLockMap;
    std::map<const llvm::Value *, int> reentrantLockCounter;

    // wait and signal
    // since there wont be too many of them, we use per thread wait/signal list
    std::map<TID, std::vector<const WaitEvent *>> threadWaitSyncList;
    std::map<TID, std::vector<const SignalEvent *>> threadSignalSyncList;
    std::map<TID, std::vector<const SignalEvent *>> threadBroadcastSyncList;

    bool isMutexArrayType(const ctx *C, const llvm::Value *V) const {
        const llvm::Type *type = pass.pta->getPointedType(C, V);
        if (type) {
            // return true if it is array type

            if (type->isArrayTy()) {
                if (type->getArrayElementType()->isStructTy()) {
                    // lock ptr should always points to pthread_mutex_t, do we need to check the array element type?
                    // is there any convient way to do it other than using name?
                    return type->getArrayElementType()->getStructName().equals("union.pthread_mutex_t");
                }
            }

            // or simply? TODO: pick whichever you like
            // return type->isArrayTy();
        }
        return false;
    }

    bool hasUnlockedAlready(const ctx *C, const llvm::Instruction *I, const llvm::Value *lockPtr) {
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, I, lockPtr, pts);
        if (pts.empty()) return false;

        if (lockSetUnlockState.find(pts) != lockSetUnlockState.end()) {
            if (lockSetUnlockState.at(pts).lastUnlockPtr) return true;
        }

        return false;
    }

    std::map<TID, std::map<std::string, bool>> lockStrStateMap;
    void updateLockStrState(TID tid, const std::string lockName, bool isLock) {
        lockStrStateMap[tid][lockName] = isLock;
    }
    bool isRedundantLock(TID tid, const std::string lockName, const ctx *C, const llvm::Instruction *I,
                         const llvm::Value *lockPtr) {
        if (DEBUG_LOCK) llvm::outs() << "lockPtr name: " << lockName << "\n";
        if (!lockName.empty())
            if (lockStrStateMap[tid].find(lockName) != lockStrStateMap[tid].end()) {
                if (lockStrStateMap[tid][lockName]) {
                    return true;
                }
            }

        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, I, lockPtr, pts);
        if (DEBUG_LOCK) {
            llvm::outs() << "pta size: " << pts.size() << "\n";
            for (auto o : pts) {
                SourceInfo sharedObjLoc = getSourceLoc(o->getValue());
                std::string lockName = sharedObjLoc.getName();
                llvm::outs() << "pt: " << o->getValue() << " " << sharedObjLoc.getName() << "\n";
            }
        }
        if (pts.empty()) return false;

        if (lockSetUnlockState.find(pts) != lockSetUnlockState.end()) {
            if (!lockSetUnlockState.at(pts).lastUnlockPtr) {
                return true;  // if nullptr, then there is a lock before
            } else
                return false;

        } else
            return false;
    }
    void updateUnlockStateUponLockEvent(LockEvent *e, const llvm::Value *lockPtr, bool checkReentrancy) {
        const ctx *C = e->getContext();
        const llvm::Instruction *inst = e->getInst();
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, inst, lockPtr, pts);
        if (pts.empty()) return;

        if (lockSetUnlockState.find(pts) != lockSetUnlockState.end()) {
            UnLockState &state = lockSetUnlockState.at(pts);

            // check reentrant lock?
            // if (checkReentrancy && state.lastUnlockPtr == nullptr)
            //     // if (!isInstructionInLoop(inst))  // or skip if in a while loop
            //     if (DEBUG_LOCK) {
            //         SourceInfo instLoc = getSourceLoc(inst);
            //         llvm::outs() << "\nFound a potential deadlock!\nYou may have missed an unlock before this "
            //                         "reentrant lock call: "
            //                      << instLoc.sig() << "\n";
            //         llvm::outs() << "call depth: " << pass.callStack.size() << "\n";

            //         std::vector<std::string> st;
            //         for (auto f : pass.callStack) {
            //             string csStr = llvm::demangle(f->getName());
            //             // TODO : fix the loc for call instructions
            //             auto loc = e->getInst()->getDebugLoc();
            //             if (loc)
            //                 csStr =
            //                     csStr + " [" + loc->getFilename().str() + ":" + std::to_string(loc->getLine()) + "]";
            //             // llvm::outs() << "pushing " << csStr << "\n";
            //             st.push_back(csStr);
            //         }
            //         int count = 0;
            //         // add a main in the stack trace if the event is in main thread
            //         // this is due to main function has no callsite
            //         for (auto it = st.begin(); it != st.end(); ++it) {
            //             llvm::outs() << ">>>" << std::string(count * 2, ' ') << *it << "\n";
            //             count++;
            //         }
            //         llvm::outs() << "\n";
            //     }

            state.lastUnlockPtr = nullptr;
            state.unlockRegionMemAccesses.clear();  // clear all events
        } else {
            UnLockState state;
            state.lastUnlockPtr = nullptr;
            lockSetUnlockState[pts] = state;
        }
    }
    void updateUnlockStateUponUnLockEvent(const ctx *C, const llvm::Value *lockPtr, const llvm::Instruction *inst) {
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, inst, lockPtr, pts);
        if (pts.empty()) return;
        if (lockSetUnlockState.find(pts) != lockSetUnlockState.end()) {
            UnLockState &state = lockSetUnlockState.at(pts);
            if (state.lastUnlockPtr == nullptr) {  // only clean if it has not been updated
                state.lastUnlockPtr = inst;
                state.lastUnLockLocksetId = locksetManager.getCurrentLocksetId();
                state.lastUnlockPtrContext = C;
                state.unlockRegionMemAccesses.clear();  // clear all events
            }
        }
    }

    void retrofitLocksetIdUponUnlockEvent(const ctx *C, const llvm::Value *lockPtr, const llvm::Instruction *inst) {
        // if(true) return;
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, inst, lockPtr, pts);
        if (pts.empty()) return;
        if (lockSetUnlockState.find(pts) != lockSetUnlockState.end()) {
            UnLockState &state = lockSetUnlockState.at(pts);

            // check if there is a branch between previous unlock and this unlock
            // if not, skip this unlock
            if (state.lastUnlockPtr) {
                // TODO: check call stacks of lock events
                if (inst->getFunction() == state.lastUnlockPtr->getFunction()) {
                    auto lastInst = state.lastUnlockPtr->getNextNonDebugInstruction();
                    bool hasBranch = false;
                    while (lastInst) {
                        // if (DEBUG_LOCK) llvm::outs() << "retrofitLockset inst: " << *lastInst << "\n";
                        if (isa<BranchInst>(lastInst) || isa<SwitchInst>(lastInst)) {
                            hasBranch = true;
                        } else if (lastInst == inst) {
                            if (!hasBranch) {
                                if (DEBUG_LOCK) llvm::outs() << "retrofitLockset skipped: " << *inst << "\n";
                                return;
                            }
                        }

                        lastInst = lastInst->getNextNonDebugInstruction();
                    }
                } else {
                    // debug
                    if (DEBUG_LOCK) {
                        auto sourceInfo1 = getSourceLoc(state.lastUnlockPtr);
                        auto sourceInfo2 = getSourceLoc(inst);
                        llvm::outs() << "retrofit lockset issue (two unlock from different functions)! \n"
                                     << "  last unlock inst at " << sourceInfo1.sig() << "\n"
                                     << sourceInfo1.getSnippet() << "\n"
                                     << "  this unlock inst at " << sourceInfo2.sig() << "\n"
                                     << sourceInfo2.getSnippet() << "\n";  //<< "  lock " << *lockPtr
                    }
                }
            }

            // update all the memory access events' lockset
            for (auto *e : state.unlockRegionMemAccesses) {
                uint32_t old_id = e->getLocksetID();
                uint32_t new_id = 0;
                if (old_id == state.lastUnLockLocksetId) continue;

                if (e->getLocksetID() != 0) {
                    // recover lockset
                    // JEFF note: this is hard to implement correctly w/o path-sensitivity
                    new_id = locksetManager.getRetrofitLockSetID(pts, old_id);
                    // if (DEBUG_LOCK)
                    //     llvm::outs() << "RetrofitLockSetID old_id: " << old_id << " new_id: " << new_id << "\n";
                }
                if (new_id == 0) {
                    new_id = state.lastUnLockLocksetId;
                }

                e->setLocksetId(new_id);
            }
            if (DEBUG_LOCK)
                llvm::outs() << "RetrofitLockSetID : " << state.unlockRegionMemAccesses.size() << " events\n";
            state.unlockRegionMemAccesses.clear();  // clear all events
        }
    }

    void addUnlockRegionMemAccessIfNecessary(MemAccessEvent *e) {
        // track instruction to memory accesses
        instr2MemEvents[e->getTID()][e->getInst()].insert(e);

        // this may be expensive, but we do not know how to avoid it
        for (auto &[pts, state] : lockSetUnlockState) {
            if (state.lastUnlockPtr) {
                state.unlockRegionMemAccesses.insert(e);
            }
        }
    }
    unsigned int lock_stack_level = 0;
    string DEBUG_LOCK_STRING_SPACE = "";
    void resetLockUnlockStack() {
        lock_stack_level = 0;
        DEBUG_LOCK_STRING_SPACE = "";
    }
    void printLockUnlockStack(TID tid, const llvm::Instruction *I, const std::string &lockName,
                              std::vector<const ObjTy *> &pts, bool isLock) {
        if (pts.empty()) {
            // probably something is wrong - skip
            if (isLock)
                llvm::outs() << "!!empty pts: "
                             << " thread " << tid << " lock " << lockName << "(none) level: " << lock_stack_level
                             << "\n"
                             << getSourceLoc(I).str() << "\n";
            else
                llvm::outs() << "!!empty pts: "
                             << " thread " << tid << " unlock " << lockName << "(none) level: " << lock_stack_level
                             << "\n"
                             << getSourceLoc(I).str() << "\n";
            return;
        }
        if (isLock) {
            DEBUG_LOCK_STRING_SPACE += "  ";
            lock_stack_level++;
            if (pts.empty()) {
                llvm::outs() << DEBUG_LOCK_STRING_SPACE << " thread " << tid << " lock " << lockName
                             << "(none) level: " << lock_stack_level << "\n";
            } else
                llvm::outs() << DEBUG_LOCK_STRING_SPACE << " thread " << tid << " lock " << lockName << "("
                             << pts.front()->getValue() << ") level: " << lock_stack_level << "\n";
        } else {
            if (lock_stack_level > 0) {
                if (pts.empty()) {
                    llvm::outs() << DEBUG_LOCK_STRING_SPACE << " thread " << tid << " unlock " << lockName
                                 << "(none) level: " << lock_stack_level << "\n";
                } else
                    llvm::outs() << DEBUG_LOCK_STRING_SPACE << " thread " << tid << " unlock " << lockName << "("
                                 << pts.front()->getValue() << ") level: " << lock_stack_level << "\n";
                DEBUG_LOCK_STRING_SPACE.pop_back();
                DEBUG_LOCK_STRING_SPACE.pop_back();
                lock_stack_level--;
            } else {
                // something is wrong, probably unlock is placed before lock due to while loop
                // e.g. memcached
                // while (1) {
                // ...
                // pthread_mutex_lock(&me->mutex);
                // ...
                //         pthread_mutex_unlock(&me->mutex);
                // }
            }
        }
    }

    void resetLockStateforNewThread() {
        locksetManager.resetCurLocksetAndId();
        lockSetUnlockState.clear();
        resetLockUnlockStack();
    }

public:
    ReachGraph(RaceDetectionPass &pass)
        : pass(pass),
          locksetManager(*(pass.pta)){

          };

    void startTraverseNewThread() { resetLockStateforNewThread(); }

    CallEvent *createCallEvent(const ctx *C, const llvm::Instruction *I, const CallGraphNodeTy *callNode, TID tid) {
        return new CallEvent(C, I, callNode, tid);
    }

    CallEvent *createCallEvent(const ctx *C, const llvm::Instruction *I, llvm::StringRef funName, TID tid) {
        return new CallEvent(C, I, funName, tid);
    }

    ReadEvent *createReadEvent(const ctx *C, const llvm::Instruction *I, TID tid) {
        auto *e = new ReadEvent(C, I, tid, locksetManager.getCurrentLocksetId());
        addUnlockRegionMemAccessIfNecessary(e);
        return e;
    }

    WriteEvent *createWriteEvent(const ctx *C, const llvm::Instruction *I, TID tid) {
        auto *e = new WriteEvent(C, I, tid, locksetManager.getCurrentLocksetId());
        addUnlockRegionMemAccessIfNecessary(e);
        return e;
    }

    // NOTE: the last parameter isCtor is for constructors
    // for value copy like:
    // std::string str1 = str2
    // there's no explicit "read" on str2
    // so we consider the constructor to be both a readEvent and writeEvent
    // and the pointer operand offset for read event should be 1 instead of 0
    // this flag is to ditinguish if this read event is created from a constructor
    ApiReadEvent *createApiReadEvent(const ctx *C, const llvm::Instruction *I, TID tid, bool isCtor = false) {
        auto *e = new ApiReadEvent(C, I, tid, locksetManager.getCurrentLocksetId(), isCtor);
        addUnlockRegionMemAccessIfNecessary(e);
        return e;
    }

    ApiWriteEvent *createApiWriteEvent(const ctx *C, const llvm::Instruction *I, TID tid) {
        auto *e = new ApiWriteEvent(C, I, tid, locksetManager.getCurrentLocksetId());
        addUnlockRegionMemAccessIfNecessary(e);
        return e;
    }

    ForkEvent *createForkEvent(const ctx *C, const llvm::Instruction *I, TID tid) {
        auto e = new ForkEvent(C, I, tid);
        syncData[tid].push_back(e->getID());
        return e;
    }

    JoinEvent *createJoinEvent(const ctx *C, const llvm::Instruction *I, TID tid) {
        auto e = new JoinEvent(C, I, tid);
        syncData[tid].push_back(e->getID());
        return e;
    }

    WaitEvent *createWaitEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *condVarPtr) {
        // get conditional variable ptr

        auto e = new WaitEvent(C, I, tid, condVarPtr);
        syncData[tid].push_back(e->getID());
        threadWaitSyncList[tid].push_back(e);
        return e;
    }

    SignalEvent *createSignalEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *condVarPtr) {
        // get conditional variable ptr

        auto e = new SignalEvent(C, I, tid, condVarPtr);
        syncData[tid].push_back(e->getID());
        threadSignalSyncList[tid].push_back(e);
        return e;
    }

    SignalEvent *createBroadcastEvent(const ctx *C, const llvm::Instruction *I, TID tid,
                                      const llvm::Value *condVarPtr) {
        // get conditional variable ptr

        auto e = new SignalEvent(C, I, tid, condVarPtr);
        syncData[tid].push_back(e->getID());
        threadBroadcastSyncList[tid].push_back(e);
        return e;
    }

    void createRWLockEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *lockPtr) {
        if (DEBUG_LOCK)
            llvm::outs() << "thread " << tid << " rwlock inst at: " << Event::getLargestEventID() << " "
                         << getSourceLoc(I).sig() << "\n";  //<< "  lock " << *lockPtr
        auto lockName = locksetManager.findLockStr(I, lockPtr);
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, I, lockPtr, pts);
        if (isRedundantLock(tid, lockName, C, I, lockPtr)) {
            if (DEBUG_LOCK) llvm::outs() << "skipped probably redundant rwlock!\n";
            if (false) {
                llvm::outs() << "\n\nFound a potential re-entrant lock:\n"
                             << "    You might have missed an unlock before this call, \n"
                             << "    which may cause a deadlock.\n"
                             << getSourceLoc(I).getSnippet() << "\n";
            }
            // pass only one to createLockEvent for redundant lock checking
            int old_value = rwLockState[pts];
            rwLockState[pts] = 0;  // make sure check redundancy check will be triggered
            createLockEvent(C, I, tid, lockPtr, EventType::WRLock);
            rwLockState[pts] = old_value;  // recover
            return;
        }

        // call twice
        createLockEvent(C, I, tid, lockPtr, EventType::WRLock);
        rwLockState[pts]++;
        // llvm::outs() <<"setting rwLockState " << lockName<<"("<<pts.front()->getValue()<<") to: "
        //                          << rwLockState[pts] << "\n";
        createLockEvent(C, I, tid, lockPtr, EventType::WRLock);

        LOG_TRACE("\nthread {} rwlock {}", tid, *lockPtr);
    }

    void createRWUnlockEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *lockPtr) {
        // if (DEBUG_LOCK) llvm::outs() << "thread wr unlock " << *lockPtr << "\n";

        createUnlockEvent(C, I, tid, lockPtr);
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, I, lockPtr, pts);
        // call twice
        if (rwLockState[pts] > 0) {  // there is a previous pthread_rwlock_rwlock call
            createUnlockEvent(C, I, tid, lockPtr);
            rwLockState[pts]--;
            auto lockName = locksetManager.findLockStr(I, lockPtr);
            // llvm::outs() <<"setting rwLockState " << lockName<<"("<<pts.front()->getValue()<<") to: "
            //              << rwLockState[pts] << "\n";
        }

        if (false) {  // this only occurs when bad nested rwlock/rdlock...
            // invariant: rwLockState must be equal or larger than zero
            // it would be bad if there is no corresponding rw lock previously
            llvm::outs() << "\n\nFound a potential misuse of RWLock:\n"
                         << "    This pthread_rwlock_unlock call is unmatched: " << *lockPtr << "\n"
                         << "    You may have forgotten a pthread_rwlock_rwlock call.\n\n";
        }
    }

    void createRDLockEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *lockPtr) {
        if (DEBUG_LOCK)
            llvm::outs() << "thread " << tid << " rdlock inst at: " << Event::getLargestEventID() << " "
                         << getSourceLoc(I).sig() << "\n";  //<< "  lock " << *lockPtr

        auto lockName = locksetManager.findLockStr(I, lockPtr);
        if (isRedundantLock(tid, lockName, C, I, lockPtr)) {
            if (DEBUG_LOCK) llvm::outs() << "skipped probably redundant rdlock!\n";

            if (false) {
                llvm::outs() << "\n\nFound a potential re-entrant lock:\n"
                             << "    You might have missed an unlock before this call, \n"
                             << "    which may cause a deadlock.\n"
                             << getSourceLoc(I).getSnippet() << "\n";
            }
            // return; //pass to createLockEvent
        }

        // call only once
        createLockEvent(C, I, tid, lockPtr, EventType::RDLock);

        locksetManager.addRWLock(C, lockPtr);  // add rd lock only

        LOG_TRACE("thread {} rdlock {}", tid, *lockPtr);
    }

    void createLockEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *lockPtr) {
        createLockEvent(C, I, tid, lockPtr, EventType::Lock);
    }
    void createLockEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *lockPtr,
                         EventType type) {
        if (DEBUG_LOCK)
            llvm::outs() << "thread " << tid << " lock inst at: " << Event::getLargestEventID() << " "
                         << getSourceLoc(I).sig() << "\n";  //<< "  lock " << *lockPtr
        auto lockName = locksetManager.findLockStr(I, lockPtr);
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, I, lockPtr, pts);

        // llvm::outs() <<"rwLockState " << lockName<<"("<<pts.front()->getValue()<<") is: "
        //                          << rwLockState[pts] << "\n";
        if (rwLockState[pts] == 0)  // do not apply for rwlocks
        {
            // if (DEBUG_LOCK) {
            //     std::vector<const ObjTy *> pts;
            //     locksetManager.getLockSet(C, I, lockPtr, pts);
            //     llvm::outs() << "lockPtr name: " << lockName << "\n";
            //     llvm::outs() << "pta size: " << pts.size() << "\n";
            //     for (auto o : pts) {
            //         SourceInfo sharedObjLoc = getSourceLoc(o->getValue());
            //         std::string lockName = sharedObjLoc.getName();
            //         llvm::outs() << "pt: " << o->getValue() << " " << sharedObjLoc.getName() << "\n";
            //     }
            // }
            //	malloc_mutex_lock(TSDN_NULL, &init_lock);
            // if (lockName.find("init_lock") != string::npos)
            //     llvm::outs() << "thread " << tid << " lock inst: " << Event::getLargestEventID()
            //                  << " lockName: " << lockName << " at \n"
            //                  << getSourceLoc(I).getSnippet() << "\n"
            //                  << " lockName: " << lockName << "  lockPtr: " << getSourceLoc(lockPtr).getName() <<
            //                  "\n";

            if (isRedundantLock(tid, lockName, C, I, lockPtr)) {
                if (DEBUG_LOCK) llvm::outs() << "skipped probably redundant lock!\n";
                // we should check re-entrance locks here - otherwise, the extra locks will be in the lockset forever
                // JEFF: for bitcoin, remember this redundant lock, to skip its corresponding unlock later
                if (pts.size() > 0) {
                    auto lockObj = pts[0]->getValue();
                    reentrantLockMap[lockObj] = lockPtr;
                    if (reentrantLockCounter.find(lockObj) == reentrantLockCounter.end())
                        reentrantLockCounter[lockObj] = 0;  // initial value 0
                    auto count = reentrantLockCounter.at(lockObj) + 1;
                    reentrantLockCounter[lockObj] = count;
                    if (DEBUG_LOCK)
                        llvm::outs() << "stored to reentrantLockMap key: " << lockObj << " value: " << lockPtr
                                     << " count:" << count << "\n";
                    if (count == 1) {
                        // only analyze one reentrant lock
                        auto e = new LockEvent(C, I, tid, locksetManager.getCurrentLocksetId(), lockPtr, type);
                        lockData[tid].push_back(e);
                    }
                }

                if (false) {
                    auto srcInfo = getSourceLoc(I);
                    llvm::outs() << "\n\nFound a potential re-entrant lock at " << srcInfo.sig() << ":\n"
                                 << "    You might have missed an unlock before this call, \n"
                                 << "    which may cause a deadlock.\n"
                                 << srcInfo.getSnippet() << "\n";
                }

                return;  // still push lock event, but do not add to lockset

            } else {
            }
        }

        if (DEBUG_LOCK_STACK) printLockUnlockStack(tid, I, lockName, pts, true);

        if (!lockPtr->getType()->isPointerTy() ||
            dyn_cast<PointerType>(lockPtr->getType())->getPointerElementType()->isFunctionTy())
            locksetManager.acquireLock(C, I, nullptr);
        else
            locksetManager.acquireLock(C, I, lockPtr);

        auto e = new LockEvent(C, I, tid, locksetManager.getCurrentLocksetId(), lockPtr, type);
        lockData[tid].push_back(e);

        if (DEBUG_LOCK)
            if (e->getLocksetID() == 0) {
                llvm::outs() << "\n!!! something must be wrong! empty lockset of lock inst: " << getSourceLoc(I).sig()
                             << "\n\n";
            }

        updateUnlockStateUponLockEvent(e, lockPtr, true);
        LOG_TRACE("thread {} lock {}", tid, *lockPtr);

        if (DEBUG_LOCK) printLockSourceLevelInfo(e);
        LangModel::addLockCallStackFunction(pass.callStack);
        // if (e->getInst()->getFunction()->getName().startswith("tcache_arena_dissociate")) {
        //     printLockSourceLevelInfo(e);
        // }
    }

    void createTryLockEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *lockPtr) {
        if (DEBUG_LOCK)
            llvm::outs() << "thread " << tid << " trylock at: " << Event::getLargestEventID() << " "
                         << getSourceLoc(I).sig() << "\n";  //<< "  lock " << *lockPtr

        auto lockName = locksetManager.findLockStr(I, lockPtr);
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, I, lockPtr, pts);
        //	malloc_mutex_lock(TSDN_NULL, &init_lock);
        // 145|malloc_mutex_trylock_final(malloc_mutex_t *mutex) {
        //>146|	return MALLOC_MUTEX_TRYLOCK(mutex);
        // pts and lockName can be empty at the same time!
        if (pts.empty()) return;

        // redis - if we see two consecutive trylock w/ the same pts, skip
        if (isRedundantLock(tid, lockName, C, I, lockPtr)) {
            if (DEBUG_LOCK) llvm::outs() << "skipped probably redundant trylock!\n";

            return;
        }
        updateLockStrState(tid, lockName, true);  // set the state of lockName to true
        if (DEBUG_LOCK_STACK) printLockUnlockStack(tid, I, lockName, pts, true);

        if (!lockPtr->getType()->isPointerTy() ||
            dyn_cast<PointerType>(lockPtr->getType())->getPointerElementType()->isFunctionTy())
            locksetManager.acquireLock(C, I, nullptr);
        else
            locksetManager.acquireLock(C, I, lockPtr);

        auto e = new LockEvent(C, I, tid, locksetManager.getCurrentLocksetId(), lockPtr, EventType::TryLock);
        lockData[tid].push_back(e);

        if (DEBUG_LOCK)
            if (e->getLocksetID() == 0) {
                llvm::outs() << "\n!!! something must be wrong! empty lockset of trylock inst: "
                             << getSourceLoc(I).sig() << "\n\n";
            }

        updateUnlockStateUponLockEvent(e, lockPtr, false);
        LOG_TRACE("thread {} trylock {}", tid, *lockPtr);
        if (DEBUG_LOCK) printLockSourceLevelInfo(e);
        LangModel::addLockCallStackFunction(pass.callStack);
    }

    void createUnlockEvent(const ctx *C, const llvm::Instruction *I, TID tid, const llvm::Value *lockPtr) {
        if (DEBUG_LOCK)
            llvm::outs() << "thread " << tid << " unlock inst: " << Event::getLargestEventID() << " at "
                         << getSourceLoc(I).sig() << "\n";  //<< "  lock " << *lockPtr

        auto lockName = locksetManager.findLockStr(I, lockPtr);
        std::vector<const ObjTy *> pts;
        locksetManager.getLockSet(C, I, lockPtr, pts);
        if (DEBUG_LOCK) {
            llvm::outs() << "lockPtr name: " << lockName << "\n";
            llvm::outs() << "pta size: " << pts.size() << "\n";
            for (auto o : pts) {
                SourceInfo sharedObjLoc = getSourceLoc(o->getValue());
                std::string lockName = sharedObjLoc.getName();
                llvm::outs() << "pt: " << o->getValue() << " " << sharedObjLoc.getName() << "\n";
            }
        }

        // JEFF: for bitcoin, check if it is the first unlock of a reentrant lock, if yes, skip it
        if (pts.size() > 0) {
            auto lockObj = pts[0]->getValue();
            if (reentrantLockMap.find(lockObj) != reentrantLockMap.end()) {
                auto lockPtr_lock = reentrantLockMap.at(lockObj);
                auto count = reentrantLockCounter.at(lockObj);
                if (DEBUG_LOCK)
                    llvm::outs() << "found unlock object in reentrantLockMap key: " << lockObj
                                 << " value: " << lockPtr_lock << " lockPtr: " << lockPtr << " count: " << count
                                 << "\n";
                // if (lockPtr_lock == lockPtr)
                {
                    if (count > 1)
                        reentrantLockCounter[lockObj] = count - 1;
                    else if (count == 1) {
                        reentrantLockMap.erase(lockObj);
                        reentrantLockCounter.erase(lockObj);
                        if (DEBUG_LOCK)
                            llvm::outs() << "erased from reentrantLockMap key: " << lockObj << " value: " << lockPtr
                                         << "\n";
                        // only analyze one reentrant lock
                        auto e =
                            new LockEvent(C, I, tid, locksetManager.getCurrentLocksetId(), lockPtr, EventType::Unlock);
                        lockData[tid].push_back(e);
                    }
                    if (DEBUG_LOCK) llvm::outs() << "skipped this unlock: " << lockObj << " value: " << lockPtr << "\n";
                    return;
                }
            }
        }

        // llvm::outs() <<"rwLockState " << lockName<<"("<<pts.front()->getValue()<<") is: "
        //                          << rwLockState[pts] << "\n";

        if (rwLockState[pts] > 0 || !hasUnlockedAlready(C, I, lockPtr)) {
            // please make sure this is called before releaseLock
            updateUnlockStateUponUnLockEvent(C, lockPtr, I);

            if (DEBUG_LOCK_STACK) printLockUnlockStack(tid, I, lockName, pts, false);

            //	malloc_mutex_unlock(tsdn, &init_lock);
            // if (lockName.find("init_lock") != string::npos)
            //     llvm::outs() << "thread " << tid << " unlock inst: " << Event::getLargestEventID() << " at \n"
            //                  << getSourceLoc(I).getSnippet() << "\n"
            //                  << " lockName: " << lockName << "  lockPtr: " << getSourceLoc(lockPtr).getName() <<
            //                  "\n";

            updateLockStrState(tid, lockName, false);  // set the state of lockName to false

            auto e = new LockEvent(C, I, tid, locksetManager.getCurrentLocksetId(), lockPtr, EventType::Unlock);
            lockData[tid].push_back(e);
            if (!lockPtr->getType()->isPointerTy() ||
                dyn_cast<PointerType>(lockPtr->getType())->getPointerElementType()->isFunctionTy())
                locksetManager.releaseLock(C, I, nullptr);
            else
                locksetManager.releaseLock(C, I, lockPtr);
            LOG_TRACE("thread {} unlock {}", tid, *lockPtr);

            if (DEBUG_LOCK) printLockSourceLevelInfo(e);
            LangModel::addLockCallStackFunction(pass.callStack);
            // if (e->getInst()->getFunction()->getName().startswith("tcache_arena_dissociate")) {
            //     printLockSourceLevelInfo(e);
            // }

        } else {
            retrofitLocksetIdUponUnlockEvent(C, lockPtr, I);

            // if (I->getFunction()->getName().startswith("tcache_arena_dissociate")) {
            //     llvm::outs() << "thread " << tid << " skipped unlock inst: " << *I << " at " << getSourceLoc(I).sig()
            //                  << "\n";  //<< "  lock " << *lockPtr
            // }
        }
    }

    void createBarrierEvent(const ctx *C, const llvm::Instruction *I, TID tid, TID teamId) {
        auto e = new BarrierEvent(C, I, tid, teamId);
        auto teamID = e->getTeamId();

        syncData[tid].push_back(e->getID());

        auto it = barrierMap[I].find(teamId);
        if (it != barrierMap[I].end()) {
            const BarrierEvent *prevEvent = it->second;
            assert(prevEvent->getInst() == e->getInst() &&
                   "Events with same teamID should be on same barrier instruction");

            // Add edge from 1->2 and 2->1
            auto tid1 = e->getTID();
            auto id1 = e->getID();
            auto tid2 = prevEvent->getTID();
            auto id2 = prevEvent->getID();
            reachEngine.addEdge(std::to_string(tid1) + ":" + std::to_string(id1),
                                std::to_string(tid2) + ":" + std::to_string(id2));
            reachEngine.addEdge(std::to_string(tid2) + ":" + std::to_string(id2),
                                std::to_string(tid1) + ":" + std::to_string(id1));

            // For now we don't track barriers for anything else so delete to save memory
            delete prevEvent;
        }

        // Mark this as the most recent event to hit the barrier
        barrierMap[I][teamId] = e;
        return;
    }

    void createTaskWaitEvent(const ctx *C, const llvm::Instruction *I, TID tid) {
        auto e = new BarrierEvent(C, I, tid, tid);  // tid is the team id
        syncData[tid].push_back(e->getID());
    }
    void printLockSourceLevelInfo(const LockEvent *e) {
        auto lockPtr = e->getLockPointer();
        auto I = e->getInst();
        auto func = I->getFunction();

        // if lockPtr is a function type, we use the function id?
        if (!lockPtr) {
            llvm::outs() << "\n!!! lockPtr is null: " << *I << "\n";

        } else {
            std::vector<const ObjTy *> pts;
            pass.pta->getPointsTo(e->getContext(), lockPtr, pts);
            if (pts.empty()) {
                llvm::outs() << "\n!!! pts is empty: " << *I << "\n";

            } else {
                if (pts.size() > 1) {
                    llvm::outs() << "!!! lock ptr is an indirect?  (" << pts.size() << " resolved locks in total)\n";
                    if (pts.size() > 2) {
                        std::vector<const ObjTy *> pts2;
                        // only print the first and last
                        pts2.push_back(pts.front());
                        pts2.push_back(pts.back());
                        pts = pts2;
                    }
                    for (auto o : pts) {
                        SourceInfo sharedObjLoc = getSourceLoc(o->getValue());
                        std::string lockName = sharedObjLoc.getName();
                        llvm::outs() << "pt: " << o->getValue() << " " << sharedObjLoc.getName() << "\n";
                    }
                }
            }
        }
        llvm::outs() << " in func: " << demangle(func->getName().str()) << " at " << getSourceLoc(I).sig() << "\n"
                     << getSourceLoc(I).getSnippet() << "\n";
    }
    // TODO: previously used for openmp race detection
    // we may need this later for barrier on non-openmp program.
    // SHBNode *createBarrierNode(const ctx *C, const llvm::Instruction *I, StaticThread *T, Depth D) {
    //     // Look for existing barrier at this instruction
    //     // TODO - instruction level is not enough.
    //     //     Need callsite or team info

    //     auto node = this->template addNewNode<SHBNode>(C, I, SHBNodeKind::Barrier, T, D);

    //     syncData[T->getID()].push_back(node->getNodeID());

    //     auto it = barrierMap.find(I);
    //     if (it != barrierMap.end()) {
    //         // TODO : kind of messy to add edges here because fork/join edges are added outside of SHBGraph
    //         // Add barrier edge from this node to the last node to hit this barrier instruction
    //         addBarrierEdge(node, it->second);
    //     }
    //     barrierMap[I] = node;

    //     // If more than 2 threads encounter a barrier it is very likely a bug
    //     auto &count = barrierCounter[I];
    //     count++;
    //     if (count > 2) {
    //         SPDLOG_ERROR("{} threads encountered barrier: {}", count, *I);
    //     }

    //     return node;
    // }

    void findAllDeadlocks() {
        // detect multithreaded deadlocks

        // step 1: filter out threads with less than two unique lockset IDs
        std::set<TID> candidateTids;
        for (auto &[tid, locks] : lockData) {
            std::set<int> idSet;
            for (auto e : locks) {
                idSet.insert(e->getLocksetID());
            }
            if (idSet.size() > 1) {
                candidateTids.insert(tid);
            }
        }
        // step 2: graph-based cycle detection
        if (candidateTids.size() > 1) {
            LOG_DEBUG("Constructing lock graph");
            // create a graph, node is e, key is lockid, edge is intra-thread ordered transition over nodes
            // TODO: we may also do it on the fly together with lock/unlock
            // FIXME: memory leak
            aser::lock::LockGraph *lockGraph = new aser::lock::LockGraph(locksetManager.getUniqueLocks());

            for (auto tid : candidateTids) {
                for (auto e : lockData[tid]) {
                    if (e->getType() == EventType::Unlock) continue;  // only consider lock events
                    auto locks = locksetManager.getLockSet(e->getLocksetID());
                    int prevId = -1;
                    for (auto o : locks) {
                        int id = lockGraph->getNodeId(o);
                        if (prevId >= 0 && prevId != id) {
                            LOG_TRACE("adding graph edge {} ==> {}", prevId, id);
                            // because the lockset is represented by a vector
                            // so it reflects the lock acquiring order
                            // an edge from 1 -> 2 means
                            // acquiring lock2 while holding lock1
                            lockGraph->addEdge(prevId, id);
                        }
                        prevId = id;
                    }
                    // add event to the last node id
                    // explaination:
                    // if we have such a program
                    // ```
                    // lock l1 -> lockset{o1}
                    // lock l2 -> lockset{o1,o2}
                    // ```
                    // so when we iterate over lockData
                    // the first round will associate event "lock l1" with o1
                    // the second round will associsate event "lock l2" with o2 (but not o1)
                    // so in this way, we associate the correct lock-acquire event with the correct lock object
                    // (assuming PTA is accurate)
                    if (prevId >= 0) lockGraph->addNodeEvent(prevId, e);
                }
            }

            LOG_DEBUG("Detecting cycles");
            // detecting cycles between lock acquiring
            lockGraph->detectCycles();
            if (lockGraph->isCyclic()) {
                for (auto cycle : lockGraph->getCycles()) {
                    // only two object involed
                    // TODO: what if has imprecise PTA?
                    if (cycle.size() == 2) {
                        LOG_TRACE("Detecting 2-thread deadlocks");
                        std::vector<std::vector<const LockEvent *>> dlTraces(2);
                        std::vector<const ObjTy *> locks;
                        bool isRacy = false;

                        for (auto n : cycle) {
                            // 2 thread accessing the lock
                            if (n->eventsMap.size() == 2) {
                                auto it1 = n->eventsMap.begin();
                                auto events1 = it1->second;
                                auto it2 = ++it1;
                                auto events2 = it2->second;

                                for (auto e1 : events1) {
                                    for (auto e2 : events2) {
                                        // step 3: confirm concurrency
                                        // consider (e1,e2), (e3,e4)
                                        // deadlock if at least one pair of events are racy
                                        if (isRacy || !checkHappensBefore(e1, e2)) {
                                            locks.push_back(n->key);
                                            dlTraces[0].push_back(e1);
                                            dlTraces[1].push_back(e2);
                                            isRacy = true;
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        if (isRacy) {
                            // make sure locks are different
                            if (locks.size() == 2) {
                                if (locks[0]->getValue() != locks[1]->getValue()) {
                                    LOG_DEBUG("Found a 2-thread deadlock!");
                                    DeadLock::collect(locks, dlTraces, 2);
                                }
                            }
                        }
                    } else if (cycle.size() == 3) {
                        LOG_TRACE("Detecting three-thread deadlocks");
                        std::vector<std::vector<const LockEvent *>> dlTraces(3);
                        std::vector<const ObjTy *> locks;
                        std::map<TID, int> idxes;
                        int idx = 0;
                        bool isRacy = false;

                        // iterate over the LockNode in a cycle
                        for (auto n : cycle) {
                            if (n->eventsMap.size() == 2) {
                                auto it1 = n->eventsMap.begin();
                                auto tid1 = it1->first;
                                auto events1 = it1->second;
                                if (!idxes.count(tid1)) {
                                    idxes[tid1] = idx++;
                                }
                                auto it2 = ++it1;
                                auto tid2 = it2->first;
                                auto events2 = it2->second;
                                if (!idxes.count(tid2)) {
                                    idxes[tid2] = idx++;
                                }

                                for (auto e1 : events1) {
                                    for (auto e2 : events2) {
                                        // step 3: confirm concurrency
                                        // consider (e1,e2), (e3,e4) (e5,e6)
                                        // deadlock if at least one pair of events are racy
                                        if (isRacy || !checkHappensBefore(e1, e2)) {
                                            locks.push_back(n->key);
                                            dlTraces[idxes.at(tid1)].push_back(e1);
                                            dlTraces[idxes.at(tid2)].push_back(e2);
                                            isRacy = true;
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        if (isRacy) {
                            if (locks.size() == 3) {
                                if (locks[0]->getValue() != locks[1]->getValue() &&
                                    locks[0]->getValue() != locks[2]->getValue() &&
                                    locks[1]->getValue() != locks[2]->getValue()) {
                                    LOG_DEBUG("Found a 3-thread deadlock!");
                                    DeadLock::collect(locks, dlTraces, 2);
                                }
                            }
                        }

                    } else {
                        // more than three threads are rare
                        LOG_DEBUG("other cases for deadlocks");
                    }
                }

                return;
            }
        }
        // llvm::outs() << "Found no multithreaded deadlocks.\n";
    }

    void checkMismatchLocks() {
        // detect single-thread reentrant locks and missing lock
        for (auto it = lockData.begin(); it != lockData.end(); ++it) {
            std::map<std::vector<const ObjTy *>, LockState> lockSetLockState;
            std::map<std::vector<const ObjTy *>, const LockEvent *> lockSetLastEventMap;

            const std::vector<const LockEvent *> &list = it->second;
            for (unsigned int i = 0; i < list.size(); i++) {
                const LockEvent *e = list[i];
                // debug                tcache_arena_dissociate.3141
                // if (e->getInst()->getFunction()->getName().startswith("tcache_arena_dissociate")) {
                //     printLockSourceLevelInfo(e);
                // }

                std::vector<const ObjTy *> pts;
                {
                    locksetManager.getLockSet(e->getContext(), e->getInst(), e->getLockPointer(), pts);
                    if (pts.empty()) continue;  // we only check lock ptr with non-empty  lockset
                }
                LockState state;
                if (lockSetLockState.find(pts) != lockSetLockState.end())
                    state = lockSetLockState.at(pts);
                else {
                    state = LockState::Init;
                    lockSetLockState[pts] = state;
                }

                switch (e->getType()) {
                    case EventType::Lock:
                        if (state == LockState::Locked) {
                            if (!CONFIG_NO_MISS_MATCH_API)
                                if (!isMutexArrayType(e->getContext(), e->getLockPointer()) &&
                                    !LangModel::isSemaphoreWait(e->getInst())) {
                                    // do not consider sem_wait

                                    // because PTA may return identical locksets for different locks
                                    // we also make sure the pts set size equal
                                    std::vector<const ObjTy *> pts1, pts2;
                                    auto e_last = lockSetLastEventMap[pts];

                                    pass.pta->getPointsTo(e->getContext(), e->getLockPointer(), pts1);
                                    pass.pta->getPointsTo(e_last->getContext(), e_last->getLockPointer(), pts2);
                                    if (pts1 == pts2) {
                                        // skip if the two are from the same function but different branches
                                        // if ((uintptr_t)mutex1 < (uintptr_t)mutex2) {
                                        // 	malloc_mutex_lock(tsdn, mutex1);
                                        // 	malloc_mutex_lock(tsdn, mutex2);
                                        // } else if ((uintptr_t)mutex1 == (uintptr_t)mutex2) {
                                        // 	malloc_mutex_lock(tsdn, mutex1);
                                        // } else {
                                        // 	malloc_mutex_lock(tsdn, mutex2);
                                        // 	malloc_mutex_lock(tsdn, mutex1);
                                        // }

                                        // oh mutex1 and mutex2 return exactly the same lockset!

                                        auto inst1 = e->getInst();
                                        auto inst2 = e_last->getInst();

                                        if (inst1->getFunction() != inst2->getFunction() ||
                                            (isPotentiallyReachable(inst1, inst2) ||
                                             isPotentiallyReachable(inst2, inst1)))
                                            MismatchedAPI::collect(e, e_last, pass.callEventTraces,
                                                                   MismatchedAPI::Type::REENTRANT_LOCK, 2);

                                        // llvm::outs() << "pts1 size: " << pts1.size() << "\n";
                                        // for (auto o : pts1) {
                                        //     SourceInfo sharedObjLoc = getSourceLoc(o->getValue());
                                        //     std::string lockName = sharedObjLoc.getName();
                                        //     llvm::outs()
                                        //         << "pt: " << o->getValue() << " " << sharedObjLoc.getName() << "\n";
                                        // }

                                        // llvm::outs() << "pts2 size: " << pts2.size() << "\n";
                                        // for (auto o : pts2) {
                                        //     SourceInfo sharedObjLoc = getSourceLoc(o->getValue());
                                        //     std::string lockName = sharedObjLoc.getName();
                                        //     llvm::outs()
                                        //         << "pt: " << o->getValue() << " " << sharedObjLoc.getName() << "\n";
                                        // }

                                        if (DEBUG_LOCK) {
                                            llvm::outs() << "reentrant lock:\n";
                                            printLockSourceLevelInfo(e);
                                            llvm::outs() << "previous lock:\n";
                                            printLockSourceLevelInfo(e_last);
                                        }
                                    }
                                }
                        }
                        lockSetLockState[pts] = LockState::Locked;
                        lockSetLastEventMap[pts] = e;
                        break;
                    case EventType::TryLock:
                        lockSetLockState[pts] = LockState::TryLocked;
                        lockSetLastEventMap[pts] = e;
                        break;
                    case EventType::WRLock:
                        if (state == LockState::RDLocked) {
                            if (!CONFIG_NO_MISS_MATCH_API) {
                                auto e_last = lockSetLastEventMap[pts];
                                MismatchedAPI::collect(e, e_last, pass.callEventTraces,
                                                       MismatchedAPI::Type::REENTRANT_LOCK, 2);
                            }
                        }
                        lockSetLockState[pts] = LockState::WRLocked;
                        lockSetLastEventMap[pts] = e;

                        break;
                    case EventType::RDLock:
                        if (state == LockState::WRLocked) {
                            if (!CONFIG_NO_MISS_MATCH_API) {
                                auto e_last = lockSetLastEventMap[pts];
                                MismatchedAPI::collect(e, e_last, pass.callEventTraces,
                                                       MismatchedAPI::Type::REENTRANT_LOCK, 2);
                            }
                        }
                        lockSetLockState[pts] = LockState::RDLocked;
                        lockSetLastEventMap[pts] = e;

                        break;
                    case EventType::Unlock:
                        if (state == LockState::Init) {
                            {
                                // this indicates a similar problem to memcached_state_machine_norace.cpp
                                // this problem can be best addressed by path sensitivity and constraint solving
                                // for now, we exploit heuristics to avoid FP by ignoring races in this func

                                if (DEBUG_LOCK)
                                    llvm::outs() << "unlock is seen before lock in a special while loop. func: "
                                                 << demangle(e->getInst()->getFunction()->getName().str())
                                                 << " event ID: " << e->getID() << " at "
                                                 << getSourceLoc(e->getInst()).sig() << "\n";
                                LOG_DEBUG(
                                    "unlock is seen before lock in a special while loop. func={}, eventID={}, "
                                    "src={}",
                                    demangle(e->getInst()->getFunction()->getName().str()), e->getID(),
                                    getSourceLoc(e->getInst()).sig());
                                // check if the particular function contains lock instruction
                                if (hasLockWithinCallDepth(e, 3)) {
                                    LOG_DEBUG("add function to specialLockStateMachines. func={}",
                                              demangle(e->getInst()->getFunction()->getName().str()));
                                    specialLockStateMachines.insert(e->getInst()->getFunction());
                                }
                            }
                        }
                        lockSetLockState[pts] = LockState::Unlocked;
                        lockSetLastEventMap[pts] = e;
                        break;
                    default:
                        break;
                }
            }

            // detect missing unlock
            if (!CONFIG_NO_MISS_MATCH_API)
                for (auto &[pts, state] : lockSetLockState) {
                    if (state != LockState::Unlocked && state != LockState::Init) {
                        // Jie: You might have forgetten to release the lock 'critical_section_' in the function
                        // Consumer()".
                        if (lockSetLastEventMap.find(pts) != lockSetLastEventMap.end()) {
                            auto *e = lockSetLastEventMap.at(pts);
                            // do not consider trylock
                            if (e->getType() == EventType::TryLock) continue;
                            // do not consider sem_wait
                            if (LangModel::isSemaphoreWait(e->getInst())) continue;

                            if (DEBUG_LOCK)
                                llvm::outs()
                                    << "You might have forgotten to release the lock at "
                                    << getSourceLoc(e->getInst()).sig() << "\n"
                                    << "in function: " << demangle(e->getInst()->getFunction()->getName().str()) << "\n"
                                    << getSourceLoc(e->getInst()).getSnippet() << "\n";
                            // heuristic: if in while loop check lock/unlock balance
                        }
                    }
                }
        }
    }

    // two things to do here
    // 1. construct HB introduced by wait/signal
    // 2. detect missing/missed signal, so the wait will block forever
    std::set<std::string> raceyOneLocations;
    void checkWaitSignalSync() {
        // basic algorithm: for each wait, try to match a signal from another thread
        // the match condition: the wait and signal calls operate on the conditional variables, i.e.,
        // the pointers point to the same object
        for (auto [tid, waitList] : threadWaitSyncList) {
            for (auto *wait : waitList) {
                bool matched = false;
                for (auto [tid2, signalList] : threadSignalSyncList) {
                    if (tid != tid2) {
                        for (auto *signal : signalList) {
                            // do the match
                            if (signal->getCondVarPtr() == wait->getCondVarPtr() ||
                                !locksetManager.hasDifferentPointsToSet(wait->getContext(), wait->getCondVarPtr(),
                                                                        signal->getContext(),
                                                                        signal->getCondVarPtr())) {
                                // add an HB edge from signal to wait
                                std::string signalId = std::to_string(tid2) + ":" + std::to_string(signal->getID());
                                std::string waitId = std::to_string(tid) + ":" + std::to_string(wait->getID());

                                if (!reachEngine.canReach(signalId, waitId) &&
                                    !reachEngine.canReach(waitId,
                                                          signalId)) {  // cannot match if there is an existing
                                                                        // happens-before from signalId to waitId

                                    // TODO: needs more thoughts on HB from signal to wait
                                    reachEngine.addEdge(signalId, waitId);
                                    reachEngine.invalidateCachedResult();

                                    LOG_TRACE("adding signal/wait HB edge: {} --> {}", signalId, waitId);
                                    matched = true;

                                    // TODO: once we match one, this signal cannot be used to match with another
                                }
                            }
                        }
                    }
                }
                // now check threadBroadcastSyncList
                // note: a broadcast signal may match "many" waits
                for (auto [tid2, broadcastList] : threadBroadcastSyncList) {
                    if (tid != tid2) {
                        for (auto *signal : broadcastList) {
                            // do the match
                            if (signal->getCondVarPtr() == wait->getCondVarPtr() ||
                                !locksetManager.hasDifferentPointsToSet(wait->getContext(), wait->getCondVarPtr(),
                                                                        signal->getContext(),
                                                                        signal->getCondVarPtr())) {
                                // add an HB edge from signal to wait
                                std::string signalId = std::to_string(tid) + ":" + std::to_string(signal->getID());
                                std::string waitId = std::to_string(tid2) + ":" + std::to_string(wait->getID());
                                if (!reachEngine.canReach(signalId, waitId) &&
                                    !reachEngine.canReach(waitId,
                                                          signalId)) {  // cannot match if there is an existing
                                                                        // happens-before from signalId to waitId
                                    reachEngine.addEdge(signalId, waitId);
                                    reachEngine.invalidateCachedResult();

                                    LOG_DEBUG("adding broadcast/wait HB edge: {} --> {}", signalId, waitId);
                                    matched = true;
                                }
                            }
                        }
                    }
                }

                // found mismatched signal/wait
                if (!matched) {
                    MismatchedAPI::collect(wait, pass.callEventTraces, MismatchedAPI::Type::MISS_SIGNAL, 2);
                }

                // Check if the wait is contained in a while/for-loop
                // This is a common error when calling pthread_cond_wait() without repeating the check
            }
        }
        // TODO: a better algorithm to detect missed signal: a single signal that can happen before wait
    }

    std::string getInstructionSignature(const llvm::Instruction *inst) {
        aser::CallSite CS(inst);
        if (CS.getCalledFunction()) {
            return demangle(CS.getCalledFunction()->getName().str());
        }
        return demangle(inst->getName().str());
    }

    std::map<const llvm::Function *, bool> lockContainingFuncsCache;

    bool hasLockInstInFunction(const llvm::Function *func) {
        if (lockContainingFuncsCache.find(func) != lockContainingFuncsCache.end())
            return lockContainingFuncsCache.at(func);
        else {
            for (auto &BB : *func) {
                for (BasicBlock::const_iterator BI = BB.begin(), BE = BB.end(); BI != BE; ++BI) {
                    // traverse each instruction
                    const Instruction *inst = dyn_cast<Instruction>(BI);
                    if (isa<CallBase>(inst)) {
                        if (LangModel::isMutexLock(inst) || LangModel::isMutexTryLock(inst) ||
                            LangModel::isRWLock(inst) || LangModel::isRDLock(inst)) {
                            lockContainingFuncsCache[func] = true;
                            return true;
                        }
                    }
                }
            }
            lockContainingFuncsCache[func] = false;
            return false;
        }
    }

    bool hasLockWithinCallDepth(const Event *e, uint8_t depth) {
        std::vector<CallEvent *> callstack = getCallEventStack(e, pass.callEventTraces);

        auto inst = e->getInst();
        if (hasLockInstInFunction(inst->getFunction())) return true;
        uint16_t size = callstack.size();  // size should be small

        if (depth > size) depth = size;

        if (depth > 0) {
            for (uint8_t i = size - 1; i >= 0; i--) {  // TODO: improve perform from size-depth
                inst = callstack[i]->getInst();
                if (hasLockInstInFunction(inst->getFunction())) return true;

                depth--;
                if (depth == 0) break;
            }
        }
        return false;
    }

    void initializeReachEngine() {
        for (auto it = syncData.begin(); it != syncData.end(); ++it) {
            TID tid = it->first;
            const std::vector<EventID> &list = it->second;

            // adding HB-edge to each sync point (and start/end point)
            // before: list of non-connected nodes: s, sync1, sync2, ..., e
            // after list of connected nodes: s -> sync1 -> sync2 -> ... -> e
            std::string lastId = std::to_string(tid) + "s";
            for (EventID sid : list) {
                std::string curId = std::to_string(tid) + ":" + std::to_string(sid);
                reachEngine.addEdge(lastId, curId);
                lastId = curId;
            }
            reachEngine.addEdge(lastId, std::to_string(tid) + "e");
        }

        // DEBUG_LOCK = false;
        if (DEBUG_LOCK || DEBUG_LOCK_STACK || DEBUG_DUMP_LOCKSET) {
            locksetManager.dumpLockset();
            for (auto &[tid, locks] : lockData) {
                llvm::outs() << "tid: " << tid << "\n";
                for (auto e : locks) {
                    llvm::outs() << "       event: " << e->getID() << " locksetId: " << e->getLocksetID() << "\n";
                }
            }
        }

        checkMismatchLocks();  // regardless of the mismatch api flag, this is needed to avoid statement-machine FPs
    }

    // this should be done after the thread fork/join edge
    void detectMisMatchingAPIs() {
        checkWaitSignalSync();
        findAllDeadlocks();
    }

    // add inter-thread fork/join HB
    void addThreadForkEdge(const ForkEvent *e, TID tid) {
        const TID tid1 = e->getTID();
        const TID tid2 = tid;
        EventID id1 = e->getID();
        reachEngine.addEdge(std::to_string(tid1) + ":" + std::to_string(id1), std::to_string(tid2) + "s");
    }

    void addThreadJoinEdge(const JoinEvent *e, TID tid) {
        const TID tid1 = e->getTID();
        const TID tid2 = tid;
        EventID id1 = e->getID();
        reachEngine.addEdge(std::to_string(tid2) + "e", std::to_string(tid1) + ":" + std::to_string(id1));
    }

    std::string findNextOut(const TID tid, const NodeID id) {
        const std::vector<EventID> &list = syncData[tid];
        // SPDLOG_DEBUG("=========");
        // SPDLOG_DEBUG("find next out for node {}", id);
        // SPDLOG_DEBUG("TID {}", tid);
        // for (NodeID x : list) {
        //     SPDLOG_DEBUG(x);
        // }
        for (NodeID sid : list) {
            if (sid == id)
                return std::to_string(tid) + ":" + std::to_string(id);
            else if (sid > id)
                return std::to_string(tid) + ":" + std::to_string(sid);
        }
        return std::to_string(tid) + "e";
    }

    std::string findLastIn(const TID tid, NodeID id) {
        const std::vector<EventID> &list = syncData[tid];
        // SPDLOG_DEBUG("=========");
        // SPDLOG_DEBUG("find last in for node {}", id);
        // SPDLOG_DEBUG("TID {}", tid);
        // for (NodeID x : list) {
        //     SPDLOG_DEBUG(x);
        // }
        for (int i = list.size() - 1; i >= 0; i--) {
            NodeID sid = list[i];
            if (sid == id)
                return std::to_string(tid) + ":" + std::to_string(id);
            else if (sid < id)
                return std::to_string(tid) + ":" + std::to_string(sid);
        }
        return std::to_string(tid) + "s";
    }

    bool hasCommonLockProtectInsts(const TID tid1, const llvm::Instruction *inst1, const TID tid2,
                                   const llvm::Instruction *inst2) {
        auto &memEvents1 = instr2MemEvents[tid1][inst1];
        auto &memEvents2 = instr2MemEvents[tid2][inst2];

        // llvm::outs() << "\nhasCommonLockProtectInsts MEM1 size: " << memEvents1.size() << " MEM2 size: " <<
        // memEvents2.size() << "\n";

        for (auto e1 : memEvents1)
            for (auto e2 : memEvents2)
                if (locksetManager.sharesLock(e1, e2)) return true;
        // return locksetManager.sharesLock(e1, e2);  // only check the first pair

        return false;
    }

    bool hasSyncBetweenEvents(const TID tid, const Event *e1, const Event *e2) {
        EventID id1 = e1->getID();
        EventID id2 = e2->getID();
        if (id1 == id2)  // make sure they are different
            return false;
        else if (id1 > id2) {  // swap if e1 is not before e2
            EventID tmp = id1;
            id1 = id2;
            id2 = tmp;
        }
        // check only lock/unlock here
        const std::vector<const LockEvent *> &list = lockData[tid];
        if (list.size() > 10000) LOG_DEBUG("bi sec for large lock list. size={}", list.size());
        // llvm::outs() << "\nBISEC for large lock list: " << list.size() << "\n";
        uint32_t i = 0, n = list.size();
        while (i < n) {
            int mid = (i + n) / 2;
            EventID id = list[mid]->getID();
            if (id > id1 && id < id2)
                return true;
            else if (mid == 0 || mid == n - 1)
                return false;
            else if (id > id2)
                n = mid;
            else if (id < id1)
                i = mid + 1;
        }

        // for (u_int32_t i = 0; i < list.size(); i++) {
        //     EventID id = list[i]->getID();
        //     if (id > id1 && id < id2) return true;
        // }
        return false;
    }

    bool checkHappensBefore(const Event *e1, const Event *e2) {
        const TID tid1 = e1->getTID();
        const TID tid2 = e2->getTID();
        if (tid1 == tid2) return true;

        EventID id1 = e1->getID();
        EventID id2 = e2->getID();

        std::string st1 = findNextOut(tid1, id1);
        std::string ed2 = findLastIn(tid2, id2);

        if (reachEngine.canReach(st1, ed2))
            return true;
        else {
            std::string st2 = findNextOut(tid2, id2);
            std::string ed1 = findLastIn(tid1, id1);
            if (reachEngine.canReach(st2, ed1)) return true;
        }

        return false;
    }

    bool satisfyStateMachineLock(const MemAccessEvent *e) const {
        if (specialLockStateMachines.find(e->getInst()->getFunction()) != specialLockStateMachines.end())
            return true;
        else
            return false;
    }

    bool sharesLock(const MemAccessEvent *e1, const MemAccessEvent *e2) const {
        // to avoid state machine lock heuristics
        return satisfyStateMachineLock(e1) || satisfyStateMachineLock(e2) || locksetManager.sharesLock(e1, e2);
    }
};  // class ReachGraph
}  // namespace aser

#endif
