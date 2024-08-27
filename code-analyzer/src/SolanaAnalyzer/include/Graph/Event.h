#pragma once

#include <stdint.h>

#include <string>
#include <vector>

#include <PointerAnalysis/Program/CallSite.h>
#include <PointerAnalysis/Util/Demangler.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instruction.h>

#include "LocksetManager.h"
#include "StaticThread.h"

namespace xray {
using EventID = uint64_t;
enum class EventType : uint8_t {
  Write,
  APIWrite,
  Read,
  APIRead,
  Call,
  Lock,
  TryLock,
  RDLock,
  WRLock,
  Unlock,
  Barrier,
  Wait,
  Signal /*, Fork, Join, MaybeMore?*/
};

// This is the Base Event class that all other events derive from
// Each Event object consists of:
//   Type - A logical type for this event (Read, Write, etc)
//   ID - A globally unique ID for each event
//   Instruction - the instruction this event represents
//   Context - The context in which the event was observed
// Each logical Type of instruction has its own abstract class that defines its
// interface See below the Event class for details
class Event {
private:
  EventType type;
  TID tid;
  EventID ID;
  const llvm::Instruction *const inst;
  const ctx *const context;
  static EventID ID_counter;

public:
  Event() = delete;
  Event(const llvm::Instruction *const inst, const ctx *const context,
        EventType type, TID tid)
      : inst(inst), type(type), context(context), tid(tid) {
    assert(inst && "Instruction* cannot be null");
    assert(context && "Context cannot be null");
    ID = ID_counter++;
  }

  static EventID getLargestEventID() { return ID_counter; }

  inline EventID getID() const { return ID; }

  inline TID getTID() const { return tid; }

  inline const EventType getType() const { return type; }

  inline const ctx *getContext() const { return context; }

  inline const llvm::Instruction *getInst() const { return inst; }

  inline const std::vector<std::string>
  getInlinedCallSiteStrings(bool isCpp) const {
    std::vector<std::string> csStrs;
    // use xray::demangler as it will strip the number postfix
    // e.g., funName.123 will demangled to funName
    xray::Demangler demangler;

    if (DILocation *Loc = inst->getDebugLoc()) {
      while (auto Loc2 = Loc->getInlinedAt()) {
        // expected format:
        // funName@filename:line

        auto funcName = Loc->getScope()->getName();
        // if(funcName.empty())//if funcName empty, use inst's func name
        //     funcName = inst->getFunction()->getName();
        // csStr = isCpp ? llvm::demangle(funName) : funName;

        std::string csStr;
        if (isCpp) {
          if (demangler.partialDemangle(funcName.str())) {
            // demange error
            funcName = stripNumberPostFix(funcName);
            csStr = llvm::demangle(funcName.str());
          } else {
            auto res = demangler.getFunctionName(nullptr, nullptr);
            if (res)
              csStr = res;
            else
              csStr = llvm::demangle(funcName.str());
          }
        } else {
          // c functions do not need demangling
          csStr = funcName.str();
        }

        if (csStr.empty())
          csStr = "unknown (inlined)";
        csStr = csStr + " [" + Loc2->getFilename().str() + ":" +
                std::to_string(Loc2->getLine()) + "]";
        csStrs.push_back(csStr);

        Loc = Loc2; // set loc for the next nested inline

        // JEFF
        // unsigned Line = Loc2->getLine();
        // unsigned Col = Loc2->getColumn();
        // StringRef File = Loc2->getFilename();
        // StringRef Dir = Loc2->getDirectory();
        // bool ImplicitCode = Loc2->isImplicitCode();
        // llvm::outs() << "InlinedCallSiteString call: " << funcName << " at
        // Dir: " << Dir
        //              << " filename: " << File << " line: " << Line << " col:
        //              " << Col
        //              << " ImplicitCode: " << ImplicitCode << "\n";
      }
    }
    return csStrs;
  }
};

// Each logical Type of instruction has its own abstract class that defines its
// interface Different implementations of that event type may implement that
// interface For example,
//   The ReadEvent class has a virtual method getPointerOperand that returns the
//   llvm::Value the event is reading. The LoadEvent implements the ReadEvent
//   interface and defines this function for a LoadInst. The VecReadEvent is
//   also a read but has a different implementation to support reading from
//   std::vector Anyone can add a new ReadEvent type by inheriting from
//   ReadEvent and implementing its interface

class MemAccessEvent : public Event {
  LocksetManager::ID locksetID;

public:
  explicit MemAccessEvent(const ctx *const context,
                          const llvm::Instruction *const inst, EventType type,
                          TID tid, LocksetManager::ID locksetID)
      : Event(inst, context, type, tid), locksetID(locksetID) {}

  virtual const llvm::Value *getPointerOperand() const = 0;

  inline const LocksetManager::ID getLocksetID() const { return locksetID; }
  inline void setLocksetId(LocksetManager::ID id) { locksetID = id; }
};

/* ==============================================
        Event Implementations Start Here
============================================== */

// the name is confusing here
// the ScalarReadEvent specifically corresponds to "LoadInst" in LLVM
// I guess we use the word "Scalar" constrasting "Vector"
class ReadEvent : public MemAccessEvent {
public:
  explicit ReadEvent(const ctx *const context,
                     const llvm::Instruction *const inst, TID tid,
                     LocksetManager::ID locksetID)
      : MemAccessEvent(context, inst, EventType::Read, tid, locksetID) {}
  const llvm::Value *getPointerOperand() const override {
    return llvm::cast<llvm::LoadInst>(this->getInst())->getPointerOperand();
  };
};

class ApiReadEvent : public MemAccessEvent {
private:
  bool isCtor;

public:
  explicit ApiReadEvent(const ctx *const context,
                        const llvm::Instruction *const inst, TID tid,
                        LocksetManager::ID locksetID, bool isCtor = false)
      : MemAccessEvent(context, inst, EventType::APIRead, tid, locksetID),
        isCtor(isCtor) {}
  const llvm::Value *getPointerOperand() const override {
    // if call or indirect call, return the first parameter
    // if (auto call = llvm::cast<llvm::CallBase>(this->getInst())) return
    // call->getArgOperand(0);
    if (auto call = llvm::dyn_cast<llvm::CallBase>(getInst())) {
      if (auto callee = llvm::dyn_cast<Function>(
              call->getCalledOperand()->stripPointerCasts())) {
        if (callee->getFunction().getName().equals("f90_str_copy_klen")) {
          // %46 = call i32 (i32, i8*, i64, i8*, i64, ...) %45(i32 1, i8* src,
          // i64 100, i8* dst, i64 50), !dbg !44
          return call->getArgOperand(3);
        }
      }
      if (call->getCalledFunction()->getName().equals("st.model.instVar")) {
        //%5 = call i8* @st.model.instVar(i8* %0, i8* getelementptr inbounds
        //([11 x i8], [11 x i8]*
        //@sharedState, i64 0, i64 0)), !dbg !41
        return call->getArgOperand(1);
      }
    }

    if (isCtor) {
      return this->getInst()->getOperand(1);
    }
    return this->getInst()->getOperand(0);
  }
};

class WriteEvent : public MemAccessEvent {
public:
  explicit WriteEvent(const ctx *const context,
                      const llvm::Instruction *const inst, TID tid,
                      LocksetManager::ID locksetID)
      : MemAccessEvent(context, inst, EventType::Write, tid, locksetID) {}
  const llvm::Value *getPointerOperand() const override {
    return llvm::cast<llvm::StoreInst>(this->getInst())->getPointerOperand();
  };
};

class ApiWriteEvent : public MemAccessEvent {
public:
  explicit ApiWriteEvent(const ctx *const context,
                         const llvm::Instruction *const inst, TID tid,
                         LocksetManager::ID locksetID)
      : MemAccessEvent(context, inst, EventType::APIWrite, tid, locksetID) {}
  const llvm::Value *getPointerOperand() const override {
    // if call or indirect call, return the first parameter
    // if (auto call = llvm::cast<llvm::CallBase>(this->getInst())) return
    // call->getArgOperand(0);
    if (auto call = llvm::dyn_cast<llvm::CallBase>(getInst())) {
      if (auto callee = llvm::dyn_cast<Function>(
              call->getCalledOperand()->stripPointerCasts())) {
        if (callee->getFunction().getName().equals("f90_str_copy_klen")) {
          // %46 = call i32 (i32, i8*, i64, i8*, i64, ...) %45(i32 1, i8* src,
          // i64 100, i8* dst, i64 50), !dbg !44
          return call->getArgOperand(1);
        }
      }
    }
    return this->getInst()->getOperand(0);
  }
};

class CallEvent : public Event {
private:
  const CallGraphNodeTy *callNode;
  llvm::StringRef funcName;
  EventID endId;

public:
  // NOTE: the reason we need both `inst` and `callNode`
  // is that for intercepted callsite, the instruction may not directly reflect
  // the function that gets called For example, pthread_create(t, NULL, func),
  // the actual function gets called is `func` However, the `inst` should be
  // used as callsite because for any callgraph node, there will be multiple
  // callsites. NOTE: the context for CallEvent seems not matter for now
  explicit CallEvent(const ctx *const context,
                     const llvm::Instruction *const inst,
                     const CallGraphNodeTy *callNode, TID tid)
      : Event(inst, context, EventType::Call, tid), callNode(callNode),
        endId(0) {}
  // NOTE: some special functions, such as `pthread_create`
  // they don't actually have corresponding call graph nodes
  // since we only use the callNode to find the function name,
  // we can also just pass a function name string to the constructor
  explicit CallEvent(const ctx *const context,
                     const llvm::Instruction *const inst,
                     llvm::StringRef funName, TID tid)
      : Event(inst, context, EventType::Call, tid), callNode(nullptr),
        funcName(funName), endId(0) {}

  const EventID getEndID() const { return endId; }

  const CallGraphNodeTy *getCallNode() const { return callNode; }

  void setEndID(EventID eid) { endId = eid; }

  std::string getCallSiteString(bool isCpp) {
    std::string csStr;
    // TODO: return the full callsite
    // expected format:
    // funName@filename:line
    if (funcName.empty()) {
      assert(callNode != nullptr);
      funcName = this->callNode->getTargetFun()->getName();
    }

    if (isCpp) {
      xray::Demangler demangler;
      if (demangler.partialDemangle(funcName.str())) {
        // demange error
        funcName = stripNumberPostFix(funcName);
        csStr = llvm::demangle(funcName.str());
      } else {
        auto res = demangler.getFunctionName(nullptr, nullptr);
        if (res)
          csStr = res;
        else
          csStr = llvm::demangle(funcName.str());
      }
    } else {
      // c functions do not need demangling
      csStr = funcName.str();
    }

    auto loc = getInst()->getDebugLoc();
    if (loc)
      csStr = csStr + " [" + loc->getFilename().str() + ":" +
              std::to_string(loc->getLine()) + "]";
    return csStr;
  }
  std::string getCallSiteString() { return getCallSiteString(false); }
};

// TODO:
class ForkEvent : public Event {
private:
  StaticThread *spawnedThread;

public:
  explicit ForkEvent(const ctx *const context,
                     const llvm::Instruction *const inst, TID tid)
      : Event(inst, context, EventType::Call, tid) {}

  const StaticThread *getSpawnedThread() const { return spawnedThread; }
  void setSpawnedThread(StaticThread *thread) { this->spawnedThread = thread; }
};

// TODO:
class LockEvent : public Event {
  const LocksetManager::ID locksetID;
  const llvm::Value *lockPtr;

public:
  explicit LockEvent(const ctx *const context,
                     const llvm::Instruction *const inst, TID tid,
                     LocksetManager::ID locksetID, const llvm::Value *lockPtr,
                     EventType type)
      : Event(inst, context, type, tid), lockPtr(lockPtr),
        locksetID(locksetID) {}

  inline const LocksetManager::ID getLocksetID() const { return locksetID; }
  inline const llvm::Value *getLockPointer() const { return lockPtr; }
};

} // namespace xray
