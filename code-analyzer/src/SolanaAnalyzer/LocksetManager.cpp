#include "LocksetManager.h"

#include "Graph/Event.h"
#include "PTAModels/GraphBLASModel.h"
#include "SourceInfo.h"

extern bool DEBUG_LOCK_STR; // for debug only

using namespace aser;

// Set to 1 due to performance issue
static const unsigned int MAX_LOCKS_TO_CONSIDER = 1;

LocksetManager::LocksetManager(PTA &pta) : curLocksetId(0), pta(pta) {
  locksetToIdMap[curLockset] = curLocksetId;
  idTolocksetMap[curLocksetId] = curLockset;
}

void LocksetManager::dumpCurrentLockset(bool isLock, LocksetManager::ID id) {
  std::string type = "lock";
  if (!isLock)
    type = "unlock";

  llvm::outs() << type + " lockset id: " << id << "\n";
  for (auto o : curLockset) {
    if (o->getValue())
      llvm::outs() << "        lock: " << o->getValue() << "\n";
    else {
      llvm::outs() << "lock value is null!\n";
    }
  }
}

LocksetManager::ID
LocksetManager::getRetrofitLockSetID(std::vector<const ObjTy *> pts,
                                     LocksetManager::ID id) {
  std::vector<const ObjTy *> lockset = getLockSet(id);
  std::vector<const ObjTy *> lockset_tmp(lockset);

  // push at most MAX_LOCKS_TO_CONSIDER lock objects into the lockset
  // this may cause FPs, for example:
  // if a the pts for a lock pointer is {A, B}, we will only consider it holds
  // lock A while in some other place, there's a lock pointer whose pts is {B}
  // we will consider them "not sharing lock"
  bool OPT = false;
  if (pts.size() > MAX_LOCKS_TO_CONSIDER) {
    OPT = true;
    if (pts.size() > 10)
      LOG_WARN("Too many locks ({}) in the retrofit points-to set!",
               pts.size());
  }
  for (auto o : pts) {
    lockset_tmp.push_back(o);
    // only hold the first lock if the pts exceeds the MAX
    if (OPT)
      break;
  }

  if (locksetToIdMap.find(lockset_tmp) != locksetToIdMap.end()) {
    return locksetToIdMap.at(lockset_tmp);

  } else {
    // something is wrong if we get here
    // return 0;
    // if it does not exist
    // let's return a new lockset for such a case

    size_t tmpLocksetId = locksetToIdMap.size();
    locksetToIdMap[lockset_tmp] = tmpLocksetId;
    idTolocksetMap[tmpLocksetId] = lockset_tmp;
    return tmpLocksetId;
  }
}

std::vector<const ObjTy *> LocksetManager::getLockSet(LocksetManager::ID id) {
  return idTolocksetMap[id];
}

// TODO: cache I->lockStr
std::string LocksetManager::findLockStr(const llvm::Instruction *I,
                                        const llvm::Value *lockPtr) {
  if (lockStrInstCache.find(I) != lockStrInstCache.end())
    return lockStrInstCache.at(I);

  auto sourceInfo = getSourceLoc(I);
  auto lockStr = sourceInfo.getSourceLine();
  if (DEBUG_LOCK_STR)
    llvm::outs() << "\nsource line: " << lockStr;

  // if (pthread_mutex_trylock(lock) == 0) {
  // skip trylock
  // if (lockStr.find("trylock(") != std::string::npos) return "";

  // find out the lock ptr string in    pthread_mutex_lock(&l->mutex);
  auto found1 = lockStr.find(
      "lock("); // TODO: customized APIs may not work: spin_lock_irqsave
  if (found1 == std::string::npos) {
    // MALLOC_MUTEX_UNLOCK(mutex);
    found1 = lockStr.find("LOCK(");
    if (found1 == std::string::npos) {
      lockStrInstCache[I] = "";
      return "";
    }
  }

  int found2;
  // assert(pthread_mutex_unlock(lock) == 0);
  if (found1 + 4 == lockStr.find("("))
    found2 = lockStr.find_last_of(
        ")"); // malloc_mutex_lock(tsd_tsdn(tsd), &info->mtx);
  else
    found2 = lockStr.find(")", found1);

  if (found2 == std::string::npos) {
    lockStrInstCache[I] = "";
    return "";
  }

  auto lockStr2 = lockStr.substr(found1 + 5, found2 - found1 - 5);

  // TODO:
  // STAT_UL(e);
  // mutex_unlock((pthread_mutex_t *) lock);

  //	malloc_mutex_lock(TSDN_NULL, &init_lock);
  //	malloc_mutex_unlock(tsdn, &init_lock);

  // lockPtr is wrong!!
  // lockStr index: 0
  // lockPtr:   %0 = bitcast %struct.tsdn_s* %tsdn to i8*, !dbg !60485
  // final lockStr: tsdn
  // lockStr:  191|	malloc_mutex_lock(tsdn, &arena->extent_avail_mtx);

  // todo split by ',' pick the correct one
  if (lockStr2.find(",") != std::string::npos) {
    if (DEBUG_LOCK_STR)
      llvm::outs() << "process comma lockStr: " << lockStr << "\n";
    auto callInst = llvm::cast<CallBase>(I);
    CallSite CS(callInst);
    auto func = CS.getTargetFunction();
    if (func) {
      // llvm::outs() << "lock/unlock func name: " << func->getName() << "\n";
      // after lock/unlock API rewriting, we should not rely on the inst anymore
      if (func->getName().startswith(".xray.")) {
        if (DEBUG_LOCK_STR) {
          if (lockPtr == nullptr)
            llvm::outs() << "lockPtr is null.. inst: " << *I << "\n";
          else {
            llvm::outs() << "lockPtr: " << *lockPtr << "\n";
          }
        }
        if (lockPtr != nullptr && lockPtr->hasName()) {
          lockStr2 = lockPtr->getName().str(); // we use the name of lockPtr
          // llvm::outs() << "lock/unlock API rewriting: " << lockStr2 << "\n";
        } else {
          // use the last one
          std::vector<std::string> strings;
          std::istringstream f(lockStr2);
          std::string s;
          while (getline(f, s, ',')) {
            strings.push_back(s);
          }
          lockStr2 = strings.front();
        }
      } else {
        for (int i = CS.getNumArgOperands() - 1; i >= 0; i--) {
          auto call_arg = CS.getArgOperand(i);
          if (call_arg->getType()->isPointerTy()) {
            if (lockPtr == nullptr || call_arg == lockPtr) {
              // llvm::outs() << "lockStr index: " << i << "\n";

              std::vector<std::string> strings;
              std::istringstream f(lockStr2);
              std::string s;
              while (getline(f, s, ',')) {
                strings.push_back(s);
              }
              lockStr2 = strings[i];
              // heuristics: _lock mtx
              break;
            }
          }
        }
      }
    }
  }

  // JEFF: fix use filename:lockStr as cache key
  // strip filename /scheduler.h:newTaskMutex => scheduler:newTaskMutex
  auto fullname = sourceInfo.getFilename();
  if (fullname.length() > 0) {
    size_t lastSlashindex = fullname.find_last_of("/"); // TODO: windows
    if (lastSlashindex == std::string::npos)
      lastSlashindex = -1;
    auto rawname = fullname.substr(lastSlashindex + 1);
    size_t lastDotindex = rawname.find_last_of(".");
    if (lastDotindex == std::string::npos)
      lastDotindex = fullname.length() - 1;
    rawname = rawname.substr(0, lastDotindex);
    lockStr2 = rawname + ":" + lockStr2;
  }
  if (DEBUG_LOCK_STR)
    llvm::outs() << "final lockStr: " << lockStr2 << "\n";
  // LOG_TRACE("found lockStr: {} from {}", lockStr2, lockStr);
  lockStrInstCache[I] = lockStr2;
  return lockStr2;
}

// TODO: cache inst->callInst2
const llvm::CallBase *
LocksetManager::findRAIILockInst(const llvm::Instruction *inst) {
  if (raiiLockInstCache.find(inst) != raiiLockInstCache.end())
    return raiiLockInstCache.at(inst);
  raiiLockInstCache[inst] = nullptr;
  return nullptr;
}

// invariant: returned pts size must be one
void LocksetManager::getLockSet(const ctx *context, const llvm::Instruction *I,
                                const llvm::Value *lockPtr,
                                std::vector<const ObjTy *> &pts) {
  // cache is used to avoid a problem: PTA may return wrong results on the same
  // lock pointer lock/unlock
  auto lockStr = findLockStr(I, lockPtr);
  if (!lockStr.empty())
    if (lockSetCache.find(lockStr) != lockSetCache.end()) {
      pts.push_back(lockSetCache.at(lockStr).front());
      return;
    }

  // get rid of PTA for locks!
  pta.getPointsTo(context, lockPtr, pts);
  if (pts.empty()) {
    // llvm::outs() << "getLockSet empty pta: " << pts.size() << "\n";

    if (!lockStr.empty())
      pta.getPointsToForSpecialLockPtr(context, I, lockStr, lockPtr, pts);
    if (pts.empty()) {
      if (DEBUG_LOCK_STR)
        llvm::outs() << "empty lockStr empty pta - likely RAII unlock? inst: "
                     << *I << "\n";
      if (!curLockset.empty())
        pts.push_back(curLockset.front());
    }
  } else if (pts.size() > 1) {
    auto o = pts.front(); // use the first one
    pts.clear();
    pts.push_back(o);
  }
  // update cache
  if (!lockStr.empty()) {
    lockSetCache[lockStr] = pts;
    if (DEBUG_LOCK_STR)
      llvm::outs() << "adding pts to cache for lockStr: " << lockStr
                   << " pts: " << pts.front()->getValue() << "\n";
  }

  // update pts->lockStr cache
  lockSetStringsCache[pts].insert(lockStr);
}
void LocksetManager::acquireLock(const ctx *context, const llvm::Instruction *I,
                                 const llvm::Value *lockPtr) {
  std::vector<const ObjTy *> pts;
  // if lockPtr is a function type, we use the function id?
  if (!lockPtr) {
    // add a special object or duplicate a previous one
    // pts.push_back(LangModel::getSpecialNodeForLock(lockPtr));
    // for now reuse an existing lock
    if (!curLockset.empty())
      pts.push_back(curLockset.front());
  }
  // Special handling for lock ptrs with empty pts
  // pta will create an anonymous object based on the lock variable name (and
  // maintain the mapping) NOTE: if the lock variable name changes (assigned to
  // a temp variable), this will fail
  else {
    getLockSet(context, I, lockPtr, pts);

    // pta.getPointsTo(context, lockPtr, pts);
    // if (pts.size() != 1) {
    //     pts.clear();
    //     // use lock string instead of pta first
    //     auto lockStr = findLockStr(I, lockPtr);
    //     if (!lockStr.empty()) pta.getPointsToForSpecialLockPtr(context, I,
    //     lockStr, lockPtr, pts); if (pts.empty())
    //         if (!curLockset.empty()) pts.push_back(curLockset.front());
    // } else {
    //     // llvm::outs() << "successful str unlock for: " <<
    //     pts.front()->getValue() << "\n";
    //     // LOG_TRACE("successful str unlock for: {} ", lockStr);
    // }
  }

  // it makes no sense if a lock pointer points to more than 10 locks
  // FIXME: Yanze: simply ignore this lock/unlock? shouldn't we at least take
  // one lock?
  bool OPT = false;
  if (pts.size() > MAX_LOCKS_TO_CONSIDER) {
    OPT = true;
    if (pts.size() > 10)
      LOG_WARN("Too many locks ({}) in the points-to set: {}!", pts.size(),
               *lockPtr);
  }
  for (auto o : pts) {
    curLockset.push_back(o);
    // only hold the first lock if the pts exceeds the MAX
    if (OPT)
      break;
  }

  if (locksetToIdMap.find(curLockset) != locksetToIdMap.end()) {
    curLocksetId = locksetToIdMap.at(curLockset);
    return;
  }

  // if it does not exist
  curLocksetId = locksetToIdMap.size();
  locksetToIdMap[curLockset] = curLocksetId;
  // FIXME: probably a copy here
  idTolocksetMap[curLocksetId] = curLockset;

  // for debug
  // dumpCurrentLockset(true, curLocksetId);
}

void LocksetManager::releaseLock(const ctx *context, const llvm::Instruction *I,
                                 const llvm::Value *lockPtr) {
  std::vector<const ObjTy *> pts;
  if (!lockPtr) {
    // pts.push_back(LangModel::getSpecialNodeForLock(lockPtr));
    if (!curLockset.empty())
      pts.push_back(curLockset.front());
  } else {
    getLockSet(context, I, lockPtr, pts);
    {
      // pta.getPointsTo(context, lockPtr, pts);
      // if (pts.size() != 1) {
      //     pts.clear();
      //     // use lock string instead of pta first
      //     auto lockStr = findLockStr(I, lockPtr);
      //     if (!lockStr.empty()) pta.getPointsToForSpecialLockPtr(context, I,
      //     lockStr, lockPtr, pts); if (pts.empty())
      //         if (!curLockset.empty()) pts.push_back(curLockset.front());
      // } else {
      //     // llvm::outs() << "successful str unlock for: " <<
      //     pts.front()->getValue() << "\n";
      //     // LOG_TRACE("successful str unlock for: {} ", lockStr);
      // }
    }
  }

  bool OPT = false;
  if (pts.size() > MAX_LOCKS_TO_CONSIDER) {
    OPT = true;
    if (pts.size() > 10)
      LOG_WARN("Too many locks ({}) in the points-to set: {}!", pts.size(),
               *lockPtr);
  }
  for (auto o : pts) {
    // find o in the reverse direction
    auto found = std::find(curLockset.rbegin(), curLockset.rend(), o);
    if (found != curLockset.rend()) {
      curLockset.erase(std::next(found).base());
      // only erase the first lock, if pts exceeds the MAX
      if (OPT)
        break;
    }
  }

  // find the current lockset id
  if (locksetToIdMap.find(curLockset) != locksetToIdMap.end()) {
    curLocksetId = locksetToIdMap.at(curLockset);
    // for debug
    // dumpCurrentLockset(false, curLocksetId);
  } else {
    // there must be a bug, probably due to PTA imprecision
    LOG_ERROR("error in release lock: unlock/lock do not match!");
  }
}

bool LocksetManager::isRWLock(const ObjTy *o) const { return rwLocks.count(o); }

void LocksetManager::addRWLock(const ctx *context, const llvm::Value *lockPtr) {
  std::vector<const ObjTy *> pts;
  pta.getPointsTo(context, lockPtr, pts);

  bool OPT = false;
  if (pts.size() > MAX_LOCKS_TO_CONSIDER) {
    OPT = true; // it makes no sense if a lock pointer points to more than 10
                // locks
    if (pts.size() > 10)
      LOG_WARN("Too many rwlocks ({}) in the points-to set: {}!", pts.size(),
               *lockPtr);
  }
  if (!OPT) // by default
    for (auto o : pts)
      rwLocks.insert(o);
}
bool LocksetManager::hasCommonLocks(
    const MemAccessEvent *e1, const std::vector<const ObjTy *> &lockset1,
    const MemAccessEvent *e2,
    const std::vector<const ObjTy *> &lockset2) const {
  if (hasRWLockButOnlyRD(e1) && hasRWLockButOnlyRD(e2)) {
    // llvm::outs() << "INTERESTING event e1: " << *e1->getInst() << "\n";
    // llvm::outs() << "INTERESTING event e2: " << *e2->getInst() << "\n";
    // there may exist overlapping locks, but both events use rwlock_rd locks
    return false;
  }
  for (auto *lock1 : lockset1) {
    for (auto *lock2 : lockset2) {
      // llvm::outs() << "INTERESTING event e1: " << *e1->getInst() <<" lock1:
      // "<<lock1<<" lock1->getValue():
      // "<<lock1->getValue()<< "\n"; llvm::outs() << "INTERESTING event e2: "
      // << *e2->getInst() <<" lock2:
      // "<<lock2<<" lock2->getValue(): "<<lock2->getValue()<< "\n";

      // Continue if they share a lock
      if (lock1->getValue() == lock2->getValue()) {
        return true;
      }
    }
  }

  // no overlapping locks
  return false;
}

bool LocksetManager::hasRWLockButOnlyRD(const MemAccessEvent *e) const {
  auto id = e->getLocksetID();
  const std::vector<const ObjTy *> &lockset = idTolocksetMap.at(id);
  // check if the lockset contains RWLock
  for (auto *o : lockset) {
    if (!isRWLock(o))
      return false; // has other locks such as mutex

    { // make sure it has the rwlock at least twice
      // count the number
      // llvm::outs() << "INTERESTING event e3: " << *e->getInst() << "  lockset
      // id:" << id << "\n";
      if (std::count(lockset.begin(), lockset.end(), o) > 1) {
        return false;
      }
    }
  }
  return true;
}

std::map<std::pair<LocksetManager::ID, LocksetManager::ID>, bool>
    sharesLockCache;
bool LocksetManager::sharesLock(const MemAccessEvent *e1,
                                const MemAccessEvent *e2) const {
  ID id1 = e1->getLocksetID();
  ID id2 = e2->getLocksetID();

  // llvm::outs() << "DEBUG e1: " << *e1->getInst() << "  tid1:" << e1->getTID()
  // << "  lockset id1:" << id1 <<
  // "\n"; llvm::outs() << "DEBUG e2: " << *e2->getInst() << "  tid2:" <<
  // e2->getTID() << "  lockset id2:" << id2
  // << "\n";

  if (id1 == 0 || id2 == 0) {
    return false;
  } else if (id1 == id2) {
    if (hasRWLockButOnlyRD(e1))
      return false;
    else
      return true;
    // TODO: add to cache

  } else {
    auto it = sharesLockCache.find(std::make_pair(id1, id2));
    if (it != sharesLockCache.end()) {
      return it->second;
    } else {
      bool result = hasCommonLocks(e1, idTolocksetMap.at(id1), e2,
                                   idTolocksetMap.at(id2));
// update the cache
#pragma omp critical(sharesLockCache)
      { sharesLockCache[std::make_pair(id1, id2)] = result; }
      return result;
    }
  }
}

LocksetManager::ID LocksetManager::getCurrentLocksetId() const {
  return curLocksetId;
}
