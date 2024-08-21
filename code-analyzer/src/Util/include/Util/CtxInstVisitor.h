//
// Created by peiming on 9/3/19.
//
// Modified from llvm::InstVisitor
// instead of just visiting an instruction, it visit the instruction with
// context
#ifndef ASER_PTA_CTXINSTVISITOR_H
#define ASER_PTA_CTXINSTVISITOR_H

#include "PointerAnalysis/Program/CtxFunction.h"
//#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

// We operate on opaque instruction classes, so forward declare all instruction
// types now...
//
#define DELEGATE(CLASS_TO_VISIT)                                               \
  return static_cast<SubClass *>(this)->visit##CLASS_TO_VISIT(                 \
      static_cast<llvm::CLASS_TO_VISIT &>(I), context)

/// Base class for instruction visitors
///
/// Instruction visitors are used when you want to perform different actions
/// for different kinds of instructions without having to use lots of casts
/// and a big switch statement (in your code, that is).
///
/// To define your own visitor, inherit from this class, specifying your
/// new type for the 'SubClass' template parameter, and "override" visitXXX
/// functions in your class. I say "override" because this class is defined
/// in terms of statically resolved overloading, not virtual functions.
///
/// For example, here is a visitor that counts the number of malloc
/// instructions processed:
///
///  /// Declare the class.  Note that we derive from InstVisitor instantiated
///  /// with _our new subclasses_ type.
///  ///
///  struct CountAllocaVisitor : public InstVisitor<CountAllocaVisitor> {
///    unsigned Count;
///    CountAllocaVisitor() : Count(0) {}
///
///    void visitAllocaInst(AllocaInst &AI) { ++Count; }
///  };
///
///  And this class would be used like this:
///    CountAllocaVisitor CAV;
///    CAV.visit(function);
///    NumAllocas = CAV.Count;
///
/// The defined has 'visit' methods for Instruction, and also for BasicBlock,
/// Function, and Module, which recursively process all contained instructions.
///
/// Note that if you don't implement visitXXX for some instruction type,
/// the visitXXX method for instruction superclass will be invoked. So
/// if instructions are added in the future, they will be automatically
/// supported, if you handle one of their superclasses.
///
/// The optional second template argument specifies the type that instruction
/// visitation functions should return. If you specify this, you *MUST* provide
/// an implementation of visitInstruction though!.
///
/// Note that this class is specifically designed as a template to avoid
/// virtual function call overhead.  Defining and using an InstVisitor is just
/// as efficient as having your own switch statement over the instruction
/// opcode.

template <typename ctx, typename SubClass, typename RetTy = void>
class CtxInstVisitor {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the InstVisitor that you
  // use to visit instructions...
  //

public:
  // Generic visit method - Allow visitation to all instructions in a range
  template <class Iterator>
  void visit(Iterator Start, Iterator End, const ctx *context) {
    while (Start != End)
      static_cast<SubClass *>(this)->visit(*Start++, context);
  }

  void visit(const xray::CtxFunction<ctx> *ctxFun) {
    auto func = const_cast<Function *>(ctxFun->getFunction());
    if (func != nullptr) {
      visit(func->begin(), func->end(), ctxFun->getContext());
    }
  }

  void visit(BasicBlock &BB, const ctx *context) {
    static_cast<SubClass *>(this)->visitBasicBlock(BB);
    visit(BB.begin(), BB.end(), context);
  }

  // Forwarding functions so that the user can visit with pointers AND refs.
  void visit(BasicBlock *BB, const ctx *context) { visit(*BB, context); }
  RetTy visit(Instruction *I, const ctx *context) { return visit(*I, context); }

  // visit - Finally, code to visit an instruction...
  //
  RetTy visit(Instruction &I, const ctx *context) {
    static_assert(std::is_base_of<CtxInstVisitor, SubClass>::value,
                  "Must pass the derived type to this template!");

    switch (I.getOpcode()) {
    default:
      llvm_unreachable("Unknown instruction type encountered!");
      // Build the switch statement using the Instruction.def file...
#define HANDLE_INST(NUM, OPCODE, CLASS)                                        \
  case Instruction::OPCODE:                                                    \
    return static_cast<SubClass *>(this)->visit##OPCODE(                       \
        static_cast<llvm::CLASS &>(I), context);
#include "llvm/IR/Instruction.def"
    }
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular instruction type.
  // The default behavior is to generalize the instruction type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // When visiting a module, function or basic block directly, these methods
  // get called to indicate when transitioning into a new unit.
  //
  void visitFunction(Function &F) {}
  void visitBasicBlock(BasicBlock &BB) {}

  // Define instruction specific visitor functions that can be overridden to
  // handle SPECIFIC instructions.  These functions automatically define
  // visitMul to proxy to visitBinaryOperator for instance in case the user
  // does not need this generality.
  //
  // These functions can also implement fan-out, when a single opcode and
  // instruction have multiple more specific Instruction subclasses. The Call
  // instruction currently supports this. We implement that by redirecting
  // that instruction to a special delegation helper.
#define HANDLE_INST(NUM, OPCODE, CLASS)                                        \
  RetTy visit##OPCODE(llvm::CLASS &I, const ctx *context) {                    \
    if (NUM == llvm::Instruction::Call)                                        \
      return delegateCallInst(I, context);                                     \
    else                                                                       \
      DELEGATE(CLASS);                                                         \
  }
#include "llvm/IR/Instruction.def"

  // Specific Instruction type classes... note that all of the casts are
  // necessary because we use the instruction classes as opaque types...
  //
  // While terminators don't have a distinct type modeling them, we support
  // intercepting them with dedicated a visitor callback.
  RetTy visitReturnInst(ReturnInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }
  RetTy visitBranchInst(BranchInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }
  RetTy visitSwitchInst(SwitchInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }
  RetTy visitIndirectBrInst(IndirectBrInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }
  RetTy visitResumeInst(ResumeInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }
  RetTy visitUnreachableInst(UnreachableInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }
  RetTy visitCleanupReturnInst(CleanupReturnInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }
  RetTy visitCatchReturnInst(CatchReturnInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }
  RetTy visitCatchSwitchInst(CatchSwitchInst &I, const ctx *context) {
    return static_cast<SubClass *>(this)->visitTerminator(I, context);
  }

  //    RetTy visitReturnInst(ReturnInst &I, const ctx *context)            {
  //    DELEGATE(TerminatorInst);} RetTy visitBranchInst(BranchInst &I, const
  //    ctx *context)            { DELEGATE(TerminatorInst);} RetTy
  //    visitSwitchInst(SwitchInst &I, const ctx *context)            {
  //    DELEGATE(TerminatorInst);} RetTy visitIndirectBrInst(IndirectBrInst
  //    &I, const ctx *context)    { DELEGATE(TerminatorInst);} RetTy
  //    visitResumeInst(ResumeInst &I, const ctx *context)            {
  //    DELEGATE(TerminatorInst);} RetTy visitUnreachableInst(UnreachableInst
  //    &I, const ctx *context)  { DELEGATE(TerminatorInst);} RetTy
  //    visitCleanupReturnInst(CleanupReturnInst &I, const ctx *context) {
  //    DELEGATE(TerminatorInst);} RetTy visitCatchReturnInst(CatchReturnInst
  //    &I, const ctx *context)  { DELEGATE(TerminatorInst); } RetTy
  //    visitCatchSwitchInst(CatchSwitchInst &I, const ctx *context)  {
  //    DELEGATE(TerminatorInst);}
  RetTy visitICmpInst(ICmpInst &I, const ctx *context) { DELEGATE(CmpInst); }
  RetTy visitFCmpInst(FCmpInst &I, const ctx *context) { DELEGATE(CmpInst); }
  RetTy visitAllocaInst(AllocaInst &I, const ctx *context) {
    DELEGATE(UnaryInstruction);
  }
  RetTy visitLoadInst(LoadInst &I, const ctx *context) {
    DELEGATE(UnaryInstruction);
  }
  RetTy visitStoreInst(StoreInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitAtomicCmpXchgInst(AtomicCmpXchgInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitAtomicRMWInst(AtomicRMWInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitFenceInst(FenceInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitGetElementPtrInst(GetElementPtrInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitPHINode(PHINode &I, const ctx *context) { DELEGATE(Instruction); }
  RetTy visitTruncInst(TruncInst &I, const ctx *context) { DELEGATE(CastInst); }
  RetTy visitZExtInst(ZExtInst &I, const ctx *context) { DELEGATE(CastInst); }
  RetTy visitSExtInst(SExtInst &I, const ctx *context) { DELEGATE(CastInst); }
  RetTy visitFPTruncInst(FPTruncInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitFPExtInst(FPExtInst &I, const ctx *context) { DELEGATE(CastInst); }
  RetTy visitFPToUIInst(FPToUIInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitFPToSIInst(FPToSIInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitUIToFPInst(UIToFPInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitSIToFPInst(SIToFPInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitPtrToIntInst(PtrToIntInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitIntToPtrInst(IntToPtrInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitBitCastInst(BitCastInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitAddrSpaceCastInst(AddrSpaceCastInst &I, const ctx *context) {
    DELEGATE(CastInst);
  }
  RetTy visitSelectInst(SelectInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitVAArgInst(VAArgInst &I, const ctx *context) {
    DELEGATE(UnaryInstruction);
  }
  RetTy visitExtractElementInst(ExtractElementInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitInsertElementInst(InsertElementInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitShuffleVectorInst(ShuffleVectorInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitExtractValueInst(ExtractValueInst &I, const ctx *context) {
    DELEGATE(UnaryInstruction);
  }
  RetTy visitInsertValueInst(InsertValueInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitLandingPadInst(LandingPadInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitFuncletPadInst(FuncletPadInst &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitCleanupPadInst(CleanupPadInst &I, const ctx *context) {
    DELEGATE(FuncletPadInst);
  }
  RetTy visitCatchPadInst(CatchPadInst &I, const ctx *context) {
    DELEGATE(FuncletPadInst);
  }

  // Handle the special instrinsic instruction classes.
  RetTy visitDbgDeclareInst(DbgDeclareInst &I, const ctx *context) {
    DELEGATE(DbgInfoIntrinsic);
  }
  RetTy visitDbgValueInst(DbgValueInst &I, const ctx *context) {
    DELEGATE(DbgInfoIntrinsic);
  }
  RetTy visitDbgLabelInst(DbgLabelInst &I, const ctx *context) {
    DELEGATE(DbgInfoIntrinsic);
  }
  RetTy visitDbgInfoIntrinsic(DbgInfoIntrinsic &I, const ctx *context) {
    DELEGATE(IntrinsicInst);
  }
  RetTy visitMemSetInst(MemSetInst &I, const ctx *context) {
    DELEGATE(MemIntrinsic);
  }
  RetTy visitMemCpyInst(MemCpyInst &I, const ctx *context) {
    DELEGATE(MemTransferInst);
  }
  RetTy visitMemMoveInst(MemMoveInst &I, const ctx *context) {
    DELEGATE(MemTransferInst);
  }
  RetTy visitMemTransferInst(MemTransferInst &I, const ctx *context) {
    DELEGATE(MemIntrinsic);
  }
  RetTy visitMemIntrinsic(MemIntrinsic &I, const ctx *context) {
    DELEGATE(IntrinsicInst);
  }
  RetTy visitVAStartInst(VAStartInst &I, const ctx *context) {
    DELEGATE(IntrinsicInst);
  }
  RetTy visitVAEndInst(VAEndInst &I, const ctx *context) {
    DELEGATE(IntrinsicInst);
  }
  RetTy visitVACopyInst(VACopyInst &I, const ctx *context) {
    DELEGATE(IntrinsicInst);
  }
  RetTy visitIntrinsicInst(IntrinsicInst &I, const ctx *context) {
    DELEGATE(CallInst);
  }
  RetTy visitCallBrInst(CallBrInst &I, const ctx *context) {
    DELEGATE(CallBase);
  }
  RetTy visitCallInst(CallInst &I, const ctx *context) { DELEGATE(CallBase); }
  RetTy visitInvokeInst(InvokeInst &I, const ctx *context) {
    DELEGATE(CallBase);
  }

  RetTy visitTerminator(Instruction &I, const ctx *context) {
    DELEGATE(Instruction);
  }

  // Next level propagators: If the user does not overload a specific
  // instruction type, they can overload one of these to get the whole class
  // of instructions...
  //
  RetTy visitCastInst(CastInst &I, const ctx *context) {
    DELEGATE(UnaryInstruction);
  }
  RetTy visitUnaryOperator(UnaryOperator &I, const ctx *context) {
    DELEGATE(UnaryInstruction);
  }
  RetTy visitBinaryOperator(BinaryOperator &I, const ctx *context) {
    DELEGATE(Instruction);
  }
  RetTy visitCmpInst(CmpInst &I, const ctx *context) { DELEGATE(Instruction); }
  // RetTy visitTerminatorInst(TerminatorInst &I, const ctx *context)    {
  // DELEGATE(Instruction);}
  RetTy visitUnaryInstruction(UnaryInstruction &I, const ctx *context) {
    DELEGATE(Instruction);
  }

  RetTy visitCallBase(CallBase &I, const ctx *context) {
    if (isa<InvokeInst>(I) || isa<CallBrInst>(I))
      return static_cast<SubClass *>(this)->visitTerminator(I);

    DELEGATE(Instruction);
  }

  // Provide a special visitor for a 'callsite' that visits both calls and
  // invokes. When unimplemented, properly delegates to either the terminator
  // or regular instruction visitor.
  //    RetTy visitCallSite(CallSite CS, const ctx *context) {
  //        assert(CS);
  //        Instruction &I = *CS.getInstruction();
  //        if (CS.isCall())
  //            DELEGATE(Instruction);
  //
  //        assert(CS.isInvoke());
  //        DELEGATE(TerminatorInst);
  //    }

  // If the user wants a 'default' case, they can choose to override this
  // function.  If this function is not overloaded in the user's subclass,
  // then this instruction just gets ignored.
  //
  // Note that you MUST override this function if your return type is not
  // void.
  //
  void visitInstruction(Instruction &I, const ctx *context) {
    // llvm::errs() << "unhandled instruction:" << I.getOpcodeName() <<
    // "\n";
  } // Ignore unhandled instructions

private:
  // Special helper function to delegate to CallInst subclass visitors.
  RetTy delegateCallInst(CallInst &I, const ctx *context) {
    if (const Function *F = I.getCalledFunction()) {
      switch (F->getIntrinsicID()) {
      default:
        DELEGATE(IntrinsicInst);
      case Intrinsic::dbg_declare:
        DELEGATE(DbgDeclareInst);
      case Intrinsic::dbg_value:
        DELEGATE(DbgValueInst);
      case Intrinsic::dbg_label:
        DELEGATE(DbgLabelInst);
      case Intrinsic::memcpy:
        DELEGATE(MemCpyInst);
      case Intrinsic::memmove:
        DELEGATE(MemMoveInst);
      case Intrinsic::memset:
        DELEGATE(MemSetInst);
      case Intrinsic::vastart:
        DELEGATE(VAStartInst);
      case Intrinsic::vaend:
        DELEGATE(VAEndInst);
      case Intrinsic::vacopy:
        DELEGATE(VACopyInst);
      case Intrinsic::not_intrinsic:
        break;
      }
    }
    // util::dbg_os() << I << "\n";
    DELEGATE(CallInst);
  }

  // An overload that will never actually be called, it is used only from dead
  // code in the dispatching from opcodes to instruction subclasses.
  RetTy delegateCallInst(Instruction &I, const ctx *context) {
    llvm_unreachable("delegateCallInst called for non-CallInst");
  }
};

#undef DELEGATE

} // namespace llvm

#endif