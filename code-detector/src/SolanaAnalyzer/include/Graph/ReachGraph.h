#pragma once

#include <stdint.h>

#include <map>
#include <string>
#include <vector>

#include <llvm/IR/Instruction.h>
#include <llvm/InitializePasses.h>

#include "Event.h"
#include "ReachabilityEngine.h"
#include "SolanaAnalysisPass.h"
#include "SourceInfo.h"

namespace aser {

struct UnLockState {
  LocksetManager::ID lastUnLockLocksetId;
  const llvm::Instruction *lastUnlockPtr = nullptr;
  const ctx *lastUnlockPtrContext;
  std::set<MemAccessEvent *> unlockRegionMemAccesses;
};

enum class LockState : uint8_t {
  Init,
  Locked,
  TryLocked,
  RDLocked,
  WRLocked,
  Unlocked
};

class ReachGraph {
private:
  SolanaAnalysisPass &pass;
  // a cached graph connectivity engine
  aser::ReachabilityEngine reachEngine;
  // for each thread (TID), we record the sync point (NodeID)
  std::map<TID, std::vector<EventID>> syncData;

  // for data ownership transfer detection
  std::map<TID, std::map<const llvm::Instruction *,
                         std::set<const MemAccessEvent *>>>
      instr2MemEvents;

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

  void resetLockStateforNewThread() {
    locksetManager.resetCurLocksetAndId();
    lockSetUnlockState.clear();
  }

public:
  ReachGraph(SolanaAnalysisPass &pass)
      : pass(pass), locksetManager(*(pass.pta)){

                    };

  void startTraverseNewThread() { resetLockStateforNewThread(); }

  CallEvent *createCallEvent(const ctx *C, const llvm::Instruction *I,
                             const CallGraphNodeTy *callNode, TID tid) {
    return new CallEvent(C, I, callNode, tid);
  }

  CallEvent *createCallEvent(const ctx *C, const llvm::Instruction *I,
                             llvm::StringRef funName, TID tid) {
    return new CallEvent(C, I, funName, tid);
  }

  ReadEvent *createReadEvent(const ctx *C, const llvm::Instruction *I,
                             TID tid) {
    auto *e = new ReadEvent(C, I, tid, locksetManager.getCurrentLocksetId());
    addUnlockRegionMemAccessIfNecessary(e);
    return e;
  }

  WriteEvent *createWriteEvent(const ctx *C, const llvm::Instruction *I,
                               TID tid) {
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
  ApiReadEvent *createApiReadEvent(const ctx *C, const llvm::Instruction *I,
                                   TID tid, bool isCtor = false) {
    auto *e = new ApiReadEvent(C, I, tid, locksetManager.getCurrentLocksetId(),
                               isCtor);
    addUnlockRegionMemAccessIfNecessary(e);
    return e;
  }

  ApiWriteEvent *createApiWriteEvent(const ctx *C, const llvm::Instruction *I,
                                     TID tid) {
    auto *e =
        new ApiWriteEvent(C, I, tid, locksetManager.getCurrentLocksetId());
    addUnlockRegionMemAccessIfNecessary(e);
    return e;
  }

  ForkEvent *createForkEvent(const ctx *C, const llvm::Instruction *I,
                             TID tid) {
    auto e = new ForkEvent(C, I, tid);
    syncData[tid].push_back(e->getID());
    return e;
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
          llvm::outs() << "!!! lock ptr is an indirect?  (" << pts.size()
                       << " resolved locks in total)\n";
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
            llvm::outs() << "pt: " << o->getValue() << " "
                         << sharedObjLoc.getName() << "\n";
          }
        }
      }
    }
    llvm::outs() << " in func: " << demangle(func->getName().str()) << " at "
                 << getSourceLoc(I).sig() << "\n"
                 << getSourceLoc(I).getSnippet() << "\n";
  }

  // add inter-thread fork/join HB
  void addThreadForkEdge(const ForkEvent *e, TID tid) {
    const TID tid1 = e->getTID();
    const TID tid2 = tid;
    EventID id1 = e->getID();
    reachEngine.addEdge(std::to_string(tid1) + ":" + std::to_string(id1),
                        std::to_string(tid2) + "s");
  }

}; // class ReachGraph

} // namespace aser
