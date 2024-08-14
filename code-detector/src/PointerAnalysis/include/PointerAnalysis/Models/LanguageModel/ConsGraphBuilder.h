//
// Created by peiming on 10/22/19.
//
#ifndef ASER_PTA_CONSGRAPHBUILDER_H
#define ASER_PTA_CONSGRAPHBUILDER_H

#include <llvm/ADT/SparseBitVector.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>

#include <unordered_map>

#include "PointerAnalysis/Graph/ConstraintGraph/ConstraintGraph.h"
#include "PointerAnalysis/Models/LanguageModel/PtrNodeManager.h"
#include "PointerAnalysis/Models/MemoryModel/MemModelTrait.h"
#include "PointerAnalysis/Models/MemoryModel/Object.h"
#include "PointerAnalysis/Models/MemoryModel/SpecialObjects/MapObject.h"
#include "PointerAnalysis/Program/CallSite.h"
#include "PointerAnalysis/Program/CtxModule.h"
#include "PointerAnalysis/Program/Pointer.h"
#include "PreProcessing/Passes/RemoveExceptionHandlerPass.h"
#include "Util/CtxInstVisitor.h"
#include "Util/GraphWriter.h"
#include "Util/Log.h"
#include "Util/SingleInstanceOwner.h"

extern bool DEBUG_PTA;
extern bool DEBUG_PTA_VERBOSE;

namespace aser {

#define MODEL static_cast<SubClass *>(this)
#define ALLOCATE(name, ...)                                                    \
  MMT::template allocate##name<PT>(memModel, __VA_ARGS__)

enum class IndirectResolveOption {
  SKIP,       // do not add the resolved function to the callgraph
  WITH_LIMIT, // only add the resolved function to the callgraph iff the
              // indirect call limit is not exceeded.
  CRITICAL,   // make sure the functon are inserted to callgraph, even if the
              // limit has been exceeded.
};

// bool isCompatibleCall(const llvm::Instruction *indirectCall, const
// llvm::Function *target);

// interaction between constraint callgraph are done through the class

// this class only handles pointers in the constraint graph
// while the objects are created and handled by MemoryModel,
// The class is designed to be inherited by a langModel
// CRTP
template <typename ctx, typename MemModel, typename PtsTy, typename SubClass>
class ConsGraphBuilder : public llvm::CtxInstVisitor<ctx, SubClass>,
                         public PtrNodeManager<ctx> {
protected:
  using Self = ConsGraphBuilder<ctx, MemModel, PtsTy, SubClass>;

  using CallGraphTy = CallGraph<ctx>;
  using CGNodeTy = CGNodeBase<ctx>;
  using CallNodeTy = CallGraphNode<ctx>;

  using CT = CtxTrait<ctx>;
  using PT = PTSTrait<PtsTy>;
  using MMT = MemModelTrait<MemModel>;
  using Canonicalizer = typename MMT::Canonicalizer;
  using ObjT = typename MMT::ObjectTy;

  using ConsGraphTy = ConstraintGraph<ctx>;
  using PtrNode = CGPtrNode<ctx>;
  using ObjNode = CGObjNode<ctx, ObjT>;

  std::unique_ptr<CtxModule<ctx>> module; // owner of the ctx module
  std::unique_ptr<ConsGraphTy> consGraph; // owner of the constraint callgraph

  MemModel memModel;

  // ASSUMPTION:
  // 1st. called before new node created
  struct BeforeNewNode {
    Self &self;
    // return true if the function need to be further evaluated
    // F is value in case it is override to indirect call
    inline InterceptResult operator()(const ctx *callerCtx,
                                      const ctx *calleeCtx,
                                      const llvm::Function *F,
                                      const llvm::Instruction *callsite) {
      // return self.initFunction(node->getTargetFun());
      InterceptResult result =
          MMT::interceptFunction(self.memModel, F, callsite);
      if (result.option == InterceptResult::Option::EXPAND_BODY) {
        return static_cast<SubClass &>(self).overrideFunction(
            calleeCtx, callerCtx, F, callsite);
      } else {
        // intercepted by memory model
        return result;
      }
    }
  } beforeNewNode;

  // ASSUMPTION:
  // 1. the caller of the direct call should already be visited
  // 2. each direct call site only visited once
  struct OnNewDirectCall {
    Self &self;
    // return false if current callnode corresponding to an external
    // function
    inline bool operator()(const CallNodeTy *node) {
      return self.initFunction(node->getTargetFun());
    }
  } onNewDirect;

  // ASSUMPTION:
  // 1. the caller of the indirect call should already be visited
  // 2. each indirect call site only visited once
  struct OnNewInDirectCall {
    Self &self;
    inline void operator()(const CallNodeTy *node) {
      return self.initIndirectCall(node->getTargetFunPtr());
    }
  } onNewInDirect;

  // ASSUMPTION:
  // 1. both caller and callee have been initialized by OnNew(In)DirectCall.
  // 2. each call edge is only visited once
  struct OnNewCallEdge {
    Self &self;
    inline void operator()(const CallNodeTy *n1, const CallNodeTy *n2,
                           const llvm::Function *originalTarget,
                           const llvm::Instruction *cs) {
      // from n1 --call--> n2
      if (n2->isIndirectCall()) {
        return;
      }
      self.processCallSite(n1->getTargetFun(), n2->getTargetFun(),
                           originalTarget, cs);
    }
  } onNewEdge;

  bool initFunction(const CtxFunction<ctx> *fun) {
    if (fun->getFunction()->isDeclaration() ||
        fun->getFunction()->isIntrinsic()) {
      return false;
    }

    // 1st, create return Pointer Node
    // dbg_os() << fun->getName() << "\n";
    if (fun->getFunction()->getReturnType()->isPointerTy()) {
      this->createRetNode(fun);
    }
    // 2nd, create Pointer Node for argument
    for (const llvm::Argument &arg : fun->getFunction()->args()) {
      if (arg.getType()->isPointerTy()) {
        this->createPtrNode(fun->getContext(), &arg);
      }
    }
    //        // 3rd, visit function instruction
    //        if (fun->getName().equals("__nv_MAIN__F1L19_1_")) {
    //            llvm::outs();
    //        }
    this->visit(fun);
    return true;
  }

  void initIndirectCall(const InDirectCallSite<ctx> *indirect) {
    PtrNode *funPtrNode =
        getPtrNode(indirect->getContext(), indirect->getValue());
    // mark the ptr node as a indirect function pointer node
    funPtrNode->setIndirectCallNode(indirect->getCallNode());
  }

  // template <typename BeforeNewNodeHook,typename OnNewDirectHook, typename
  // OnNewInDirectHook,typename OnNewEdgeHook>
  bool updateFunPtrs(const llvm::SparseBitVector<> &funPtrs
                              /*,BeforeNewNodeHook beforeNewNodeHook, OnNewDirectHook onNewDirectHook,
                                 OnNewInDirectHook onNewInDirectHook, OnNewEdgeHook onNewEdgeHook*/) {
    // TODO: static assert here to ensure the callback accept the parameters
    if (funPtrs.empty()) {
      return false;
    }
    LOG_DEBUG("PTA resolving indirect function pointers. size={}",
              funPtrs.count());

    bool changed = false;
    size_t beforeResolve = this->getConsGraph()->getNodeNum();

    for (NodeID id : funPtrs) {
      // in case the node is collapsed
      auto funPtrNode = consGraph->getCGNode(id)->getSuperNode();
      assert(funPtrNode->isFunctionPtr());
      NodeID ptrID = funPtrNode->getNodeID();
      // get a copy of the points to,
      // since indirect call resolution will add more elements to the pts
      // and thus corrupt the iterator
      typename PT::PtsTy pointsTo(PT::getPointsTo(ptrID));
      // llvm::outs() << " funcPtr pta size:" << pointsTo.count()
      //              << " graph nodes: " <<
      //              funPtrNode->getIndirectNodes().size() << "\n";

      // TODO: track if the points-to of a funcPtr has changed?
      // TODO: maintain a map from funcPtr to its newly added function object?
      // JEFF: this can be redundant since pointsTo is repeatedly traversed??
      uint8_t count = 0;
      // bool applyLimit = true;

      for (auto it = pointsTo.begin(), eit = pointsTo.end(); it != eit; it++) {
        // count++;
        auto node = consGraph->getObjectNode(*it);
        if (auto objNode = llvm::cast<ObjNode>(node)) {
          auto object = objNode->getObject();
          if (object->isFunction()) {
            // resolved to a function
            // JEFF: WHY A funPtrNode can have SO MANY indirect nodes?
            for (CallGraphNode<ctx> *indirectNode :
                 funPtrNode->getIndirectNodes()) {
              // auto indirectNode =
              // funPtrNode->getIndirectCallNode();
              const llvm::Function *target =
                  llvm::dyn_cast<llvm::Function>(object->getValue());
              auto callsite = indirectNode->getTargetFunPtr()->getCallSite();
              assert(target);

              if (indirectNode->getTargetFunPtr()->isInterceptedCallSite()) {
                if (!static_cast<SubClass &>(*this).isCompatible(callsite,
                                                                 target)) {
                  continue;
                }
              } else if (!isCompatibleCall(callsite, target)) {
                continue;
              }
              // if callsite is something like configured "op->consume(op);"
              // skip check MaxIndirectTarget

              // PEIMING:
              // the following part are moved to language model to decouple
              // pointer analysis with client application the customized
              // language model will determine whether the target is critical or
              // not

              IndirectResolveOption result =
                  static_cast<SubClass *>(this)->onNewIndirectTargetResolvation(
                      target, callsite);
              if (result == IndirectResolveOption::SKIP) {
                continue;
              }

              // if result == CRITICAL, do not apply limit
              bool applyLimit = result == IndirectResolveOption::WITH_LIMIT;
              bool newTarget = indirectNode->getTargetFunPtr()->resolvedTo(
                  target, applyLimit);

              if (newTarget) {
                module->resolveCallTo(indirectNode, target, beforeNewNode,
                                      onNewDirect, onNewInDirect, onNewEdge);

                LOG_TRACE("Resolved Indirect Call. In={}, from={}, to={}",
                          indirectNode->getTargetFunPtr()
                              ->getCallSite()
                              ->getFunction()
                              ->getName(),
                          *indirectNode->getTargetFunPtr()->getCallSite(),
                          target->getName());

                changed = true;
              }
            }
          }
        }
        // if (applyLimit && count >= MaxIndirectTarget) break;
      }
    }
    if (changed) {
      size_t afterResolve = this->getConsGraph()->getNodeNum();
      LOG_DEBUG("PTA Node Stat: Before={}, After={}, New={}", beforeResolve,
                afterResolve, afterResolve - beforeResolve);
    }
    return changed;
  }

  // We can not use const llvm::Type * because llvm::DataLayout does not accept
  // a constant variable!!!
  inline ObjNode *allocHeapObj(const ctx *C, const llvm::Instruction *allocSite,
                               llvm::Type *T) {
    return ALLOCATE(HeapObj, C, allocSite, getLLVMModule()->getDataLayout(), T);
  }

  // TODO: the operations done here should be consistent with
  // PartialUpdateSolver
  void processCallSite(const CtxFunction<ctx> *caller,
                       const CtxFunction<ctx> *callee,
                       const llvm::Function *originalTarget,
                       const llvm::Instruction *callsite) {
    if (MMT::template interceptCallSite<PT>(this->memModel, caller, callee,
                                            callsite)) {
      // special handled by memory model
      return;
    }

    if (MODEL->overrideCallSite(caller, callee, originalTarget, callsite)) {
      return;
    }

    // function does not get expanded (due to interception)
    if (callee->isExtFunction()) {
      return;
    }

    aser::CallSite CS(callsite);
    // must be a call instruction
    assert(CS.isCallOrInvoke());
    // the target function should match
    assert(CS.isIndirectCall() ||
           CS.getTargetFunction() == callee->getFunction() ||
           CS.getTargetFunction()->getName().startswith("st."));
    // if(!CS.isIndirectCall() &&CS.getTargetFunction() !=
    // callee->getFunction())
    //         llvm::outs() <<"!!! mismatch
    //         target:"<<CS.getTargetFunction()->getName()<<" callee:
    //         "<<callee->getFunction()->getName();

    // callsite should be in the caller function
    assert(callsite->getFunction() == caller->getFunction());
    // the rule of context evolution should be obeyed.
    assert(CT::contextEvolve(caller->getContext(), callsite) ==
           callee->getContext());
    // function type should match
    assert(!CS.isIndirectCall() ||
           isCompatibleCall(callsite, callee->getFunction()));

    if (callee->getFunction()->isIntrinsic()) {
      return;
    }

    if (callee->getFunction()->isDeclaration()) {
      if (callee->getFunction()->getName().startswith("__aser_")) {
        // marker function for verifying the result of the pointer analysis
        // e.g., "__aser_no_alias__", "__aser_alias__"
        return;
      }
      LOG_TRACE("unhandled external function. function={}",
                callee->getFunction()->getName().str());
      return;
    }

    // 1st, link the parameter
    auto aIt = CS.arg_begin();      // actual
    auto fIt = callee->arg_begin(); // formal

    // some low-level bitcast does not maintain the type equvilance....
    while (fIt != callee->arg_end() && aIt != CS.arg_end()) {
      const llvm::Value *actual = *aIt;
      const llvm::Argument *formal = &*fIt;
      // at least they should be pointer at the same time
      // seems some fortran code will call mismatch type function :-(
      if (formal->getType()->isPointerTy() ==
          actual->getType()->isPointerTy()) {
        if (actual->getType()->isPointerTy()) {
          CGNodeTy *aNode = this->getPtrNode(caller->getContext(), actual);

          // If the actual arguments passed to the resolved indirect call
          // site might be super nodes
          auto fNode = this->getPtrNode(callee->getContext(), formal);
          // actual argument is assigned to the formal argument.
          this->consGraph->addConstraints(aNode, fNode, Constraints::copy);
        }
      } else {
        LOG_DEBUG("mismatched type in function call. function={}, callsite={}",
                  callee->getFunction()->getName().str(), *callsite);
      }
      aIt++;
      fIt++;
    }

    if (callee->getFunction()->isVarArg()) {
      // TODO: handle var args function
      LOG_TRACE("var arg function not handled. function={}",
                callee->getFunction()->getName().str());
    } else {
      // %2 = bitcast void ()* @__do_softirq to void (i8*)*, !dbg !11316457
      // tail call void %2(i8* null) #246, !dbg !11316457
      // the following assert can be failed in the above example found in
      // linux.bc assert(aIt == CS.arg_end());
    }

    // 2nd, link the return node
    if (auto rv = CS.getReturnedArgOperand()) {
#ifndef NDEBUG
      if (rv->getType()->isPointerTy()) {
        auto src = Canonicalizer::canonicalize(rv);
        auto dst = Canonicalizer::canonicalize(callsite);
        assert(src == dst);
      }
#endif
      return; // no need to handle this case
    }

    if (callee->getFunction()->getReturnType()->isPointerTy() &&
        callsite->getType()->isPointerTy()) {
      auto src = this->getRetNode(callee->getContext(), callee->getFunction());
      auto dst = this->getPtrNode(caller->getContext(), callsite);
      consGraph->addConstraints(src, dst, Constraints::copy);
    }
  }

  inline void addGlobalVariable(const llvm::GlobalVariable &gVar) {
    auto obj = ALLOCATE(GlobalVariable, &gVar, module->getDataLayout());
    auto ptr = createPtrNode(CT::getGlobalCtx(), &gVar);
    // must be adding new node
    this->consGraph->addConstraints(obj, ptr, Constraints::addr_of);
    // PT::insert(ptr->getNodeID(), obj->getNodeID());
  }

  inline void addFunction(const llvm::Function &fun) {
    auto obj = ALLOCATE(Function, &fun);
    auto ptr = createPtrNode(CT::getGlobalCtx(), &fun);
    // must be adding new node
    this->consGraph->addConstraints(obj, ptr, Constraints::addr_of);
    // PT::insert(ptr->getNodeID(), obj->getNodeID());
  }

  void addGlobals() {
    // process globals
    for (auto const &gVar : module->getLLVMModule()->globals()) {
      addGlobalVariable(gVar);
    }
    // add functions
    for (auto const &func : module->getLLVMModule()->functions()) {
      addFunction(func);
    }
  }

  inline CGNodeTy *indexObject(ObjNode *objNode, const llvm::Instruction *idx) {
    auto object = objNode->getObject();
    return MMT::template indexObject<PT>(getMemModel(), object, idx);
  }

  void addLocals() {
    // then, build call graph and build constraint graph along the call
    // graph construction
    module->buildInitCallGraph(beforeNewNode, onNewDirect, onNewInDirect,
                               onNewEdge);
  }

  // [[nodiscard]] inline const CallGraphNode<ctx> *getDirectNode(const ctx *C,
  // const llvm::Function *F) {
  //     return module->getDirectNode(C, F);
  // }

  // [[nodiscard]] inline const CallGraphNode<ctx> *getDirectNodeOrNull(const
  // ctx *C, const llvm::Function *F) {
  //     return module->getDirectNodeOrNull(C, F);
  // }

  // [[nodiscard]] inline const CallGraphNode<ctx> *getInDirectNode(const ctx
  // *C, const llvm::Instruction *I) {
  //     return module->getInDirectNode(C, I);
  // }

  inline const llvm::Module *getLLVMModule() const {
    return this->module->getLLVMModule();
  }

  inline llvm::StringRef getEntryName() const {
    return this->module->getEntryName();
  }

public:
  ConsGraphBuilder(llvm::Module *M, llvm::StringRef entry)
      : beforeNewNode{.self = *this}, onNewDirect{.self = *this},
        onNewInDirect{.self = *this}, onNewEdge{.self = *this}, // callbacks
        consGraph(new ConsGraphTy()), // the constraint graph
        memModel(*consGraph.get(), *this, *M),
        module(
            new CtxModule<ctx>(M, entry)) { // the module represent the programs

    // init the pointer node manager
    PtrNodeManager<ctx>::template init<PT>(consGraph.get(), M->getContext());

    Object<ctx, ObjT>::resetObjectID();
    // null and universal object nodes;
    CGNodeBase<ctx> *nullObj = ALLOCATE(NullObj, module->getLLVMModule());
    CGNodeBase<ctx> *uniObj = ALLOCATE(UniObj, module->getLLVMModule());

    // uncomments this if you want the special nodes to appear in the points to
    // set
    // TODO: add a parameters for this.
    /*
    consGraph->addConstraints(nullObj, nullPtrNode, Constraints::addr_of);
    PT::insert(nullPtrNode->getNodeID(), nullObj->getNodeID());

    consGraph->addConstraints(uniObj, uniPtrNode, Constraints::addr_of);
    PT::insert(uniPtrNode->getNodeID(), uniObj->getNodeID());
    */
  }

  [[nodiscard]] inline const CallGraphNode<ctx> *
  getDirectNode(const ctx *C, const llvm::Function *F) const {
    return module->getDirectNode(C, F);
  }

  [[nodiscard]] inline const CallGraphNode<ctx> *
  getDirectNodeOrNull(const ctx *C, const llvm::Function *F) const {
    return module->getDirectNodeOrNull(C, F);
  }

  [[nodiscard]] inline const CallGraphNode<ctx> *
  getInDirectNode(const ctx *C, const llvm::Instruction *I) const {
    return module->getInDirectNode(C, I);
  }

  friend SubClass;
  friend llvm::CtxInstVisitor<ctx, SubClass>;

  template <typename Key, typename PT> friend class MapObject;

protected:
  inline void constructConsGraph() {
    // first, add global nodes
    addGlobals();

    // then, using memory model trait to initialize globals
    for (const auto &gVar : getLLVMModule()->globals()) {
      MMT::template initializeGlobal<PT>(getMemModel(), &gVar,
                                         getLLVMModule()->getDataLayout());
    }
    // finally, add locals
    addLocals();
  }

  // override in different language langModel
  inline void visitAllocaInst(llvm::AllocaInst &I, const ctx *context) {
    CGNodeBase<ctx> *stackObj =
        ALLOCATE(StackObj, context, &I, module->getDataLayout());
    CGNodeBase<ctx> *stackPtr = createPtrNode(context, &I);

    consGraph->addConstraints(stackObj, stackPtr, Constraints::addr_of);
    // PT::insert(stackPtr->getNodeID(), stackObj->getNodeID());
  }

  // TODO: returned argument attribute
  inline void visitReturnInst(llvm::ReturnInst &I, const ctx *context) {
    if (auto retValue = I.getReturnValue()) { // not ret void
      if (retValue->getType()->isPointerTy()) {
        // simply map it to the return node
        CGPtrNode<ctx> *returnPtr = getPtrNode(context, retValue);
        CGPtrNode<ctx> *returnNode = getRetNode(context, I.getFunction());

        consGraph->addConstraints(returnPtr, returnNode, Constraints::copy);
      }
    }
  }

  inline void visitGetElementPtrInst(llvm::GetElementPtrInst &I,
                                     const ctx *context) {
    if constexpr (!MMT::COLLAPSE_GEP) {
      // field sensitivity
      const llvm::Value *baseValue =
          Canonicalizer::canonicalize(I.getPointerOperand());
      const llvm::Value *gepValue = Canonicalizer::canonicalize(&I);
      if (baseValue != gepValue) {
        CGNodeBase<ctx> *gepNode = getOrCreatePtrNode(context, gepValue);
        CGNodeBase<ctx> *baseNode = getOrCreatePtrNode(context, baseValue);

        consGraph->addConstraints(baseNode, gepNode, Constraints::offset);
      }
    } else {
#ifndef NDEBUG
      // if gep is not handled, then they should collapse to the same
      // value
      const llvm::Value *baseValue =
          Canonicalizer::canonicalize(I.getPointerOperand());
      const llvm::Value *gepValue = Canonicalizer::canonicalize(&I);
      assert(baseValue == gepValue);
#endif
    }
  }

  inline void visitLoadInst(llvm::LoadInst &I, const ctx *context) {
    if (I.getType()->isPointerTy()) {
      // load a pointer from memory
      // *ptr = load **ptr;

      // In redis graph
      // %30 = inttoptr i64 %29 to %struct.Node*, !dbg !183013
      // %31 = getelementptr inbounds %struct.Node, %struct.Node* %30, i64
      // 0, i32 0, !dbg !183013 %32 = load %struct.Entity*,
      // %struct.Entity** %31, align 8, !dbg !183013, !tbaa !124046

      CGPtrNode<ctx> *loadedFrom =
          this->getOrCreatePtrNode(context, I.getPointerOperand());
      CGPtrNode<ctx> *loadedInto = this->getOrCreatePtrNode(context, &I);

      consGraph->addConstraints(loadedFrom, loadedInto, Constraints::load);
    }
  }

  inline void visitStoreInst(llvm::StoreInst &I, const ctx *context) {
    const llvm::Value *operand = I.getValueOperand();
    if (operand->getType()->isPointerTy()) {
      // store a pointer into memory
      // store *src into **dst
#ifndef NDEBUG
      const llvm::Value *src = Canonicalizer::canonicalize(I.getValueOperand());
      const llvm::Value *dst =
          Canonicalizer::canonicalize(I.getPointerOperand());

      if (dst == this->getUniPtr()->getPointer()->getValue() ||
          dst == this->getNullPtr()->getPointer()->getValue()) {
        // store into a universal pointer or null pointer, just ignore it.
        // the most conservative way is to assume it can write to every object
        // but apparently, it is too conservative to be useful

        // FIXME: some pattern can be handled such as ptrtoint (inttoptr ...)
        LOG_TRACE("store into universal/null pointers! inst={}", I);
        return;
      }

      // assert(dst != uniPtrNode->getPointer()->getValue() && "store into
      // universal ptr?"); assert(dst != nullPtrNode->getPointer()->getValue()
      // && "store into null ptr?");

      // it is possible for src and dst to be the same
      // for example, the address of a field can be store into another field
      // assert(src != dst);
#endif
      CGPtrNode<ctx> *srcNode = this->getOrCreatePtrNode(context, operand);
      CGPtrNode<ctx> *dstNode =
          this->getOrCreatePtrNode(context, I.getPointerOperand());
      consGraph->addConstraints(srcNode, dstNode, Constraints::store);
    }
  }

  // call site is handled by onNewEdge()
  inline void visitCallBase(llvm::CallBase &CB, const ctx *context) {
    if (CB.getType()->isPointerTy()) {
      const llvm::Value *retValue = Canonicalizer::canonicalize(&CB);
      if (LLVM_UNLIKELY(retValue != &CB)) {
        // Could improve both accuracy and performance
        // but it seems that no one use the attribute :-(
        return;
      }
      // else create a new PtrNode to catch the returned pointer
      // the　COPY edge will then be added in OnNewCallEdge() where callee
      // is guaranteed to be initialized already
      getOrCreatePtrNode(context, &CB);
    }
  }

  inline void visitBitCastInst(llvm::BitCastInst &I, const ctx *context) {
    if /*constexpr*/ (MMT::COLLAPSE_BITCAST) {
      return;
    } else {
      assert(false); // not handled yet
    }
  }

  // Phi node might access a node that we have not visited yet
  inline void visitPHINode(llvm::PHINode &I, const ctx *context) {
    if (I.getType()->isPointerTy()) {
      auto dst = this->getOrCreatePtrNode(context, &I);
      for (unsigned i = 0, e = I.getNumIncomingValues(); i != e; i++) {
        auto src = this->getOrCreatePtrNode(context, I.getIncomingValue(i));
        this->consGraph->addConstraints(src, dst, Constraints::copy);
      }
    }
  }

  inline void visitMemCpyInst(llvm::MemCpyInst &I, const ctx *context) {
    PtrNode *src = this->getOrCreatePtrNode(context, I.getSource());
    PtrNode *dst = this->getOrCreatePtrNode(context, I.getDest());

    MMT::template handleMemCpy<PT>(memModel, context, &I, src, dst);
  }

  // TODO: link it to the universal point node
  inline void visitIntToPtrInst(llvm::IntToPtrInst &I, const ctx *context) {}

  inline void visitSelectInst(llvm::SelectInst &I, const ctx *context) {
    if (I.getType()->isPointerTy()) {
      // %ptr = select bool, %ptr1, %ptr2
      auto dst = this->getOrCreatePtrNode(context, &I);
      auto src = this->getOrCreatePtrNode(context, I.getOperand(1));
      this->consGraph->addConstraints(src, dst, Constraints::copy);

      src = this->getOrCreatePtrNode(context, I.getOperand(2));
      this->consGraph->addConstraints(src, dst, Constraints::copy);
    }
  }

  // TODO!! need to be done
  inline void visitExtractValueInst(llvm::ExtractValueInst &I,
                                    const ctx *context) {
    if (I.getType()->isPointerTy()) {
      LOG_TRACE("Extract a pointer is not handled! inst={}", I);
      auto dst = this->getOrCreatePtrNode(context, &I);
    }
  }

  inline void visitInsertValueInst(llvm::InsertValueInst &I,
                                   const ctx *context) {}

  // corner cases
  inline void visitAtomicCmpXchgInst(llvm::AtomicCmpXchgInst &I,
                                     const ctx *context) {}
  inline void visitAtomicRMWInst(llvm::AtomicRMWInst &I, const ctx *context) {}
  inline void visitVAArgInst(llvm::VAArgInst &I, const ctx *context) {}

  // vector operations
  inline void visitExtractElementInst(llvm::ExtractElementInst &I,
                                      const ctx *context) {}
  inline void visitInsertElementInst(llvm::InsertElementInst &I,
                                     const ctx *context) {}
  inline void visitShuffleVectorInst(llvm::ShuffleVectorInst &I,
                                     const ctx *context) {}

  // instrinsic instruction classes.
  inline void visitMemSetInst(llvm::MemSetInst &I, const ctx *context) {}
  inline void visitMemMoveInst(llvm::MemMoveInst &I, const ctx *context) {}

  // need to be handled? but no one use it.
  inline void visitVAStartInst(llvm::VAStartInst &I, const ctx *context) {}
  inline void visitVAEndInst(llvm::VAEndInst &I, const ctx *context) {}
  inline void visitVACopyInst(llvm::VACopyInst &I, const ctx *context) {}

  inline void visitFreezeInst(llvm::FreezeInst &I, const ctx *context) {}
  // getters
  inline MemModel &getMemModel() { return memModel; }

  inline ConsGraphTy *getConsGraph() { return this->consGraph.get(); }

  inline const ConsGraphTy *getConsGraph() const {
    return this->consGraph.get();
  }

  inline const CallGraphTy *getCallGraph() const {
    return module->getCallGraph();
  }

  inline PtrNode *createRetNode(const CtxFunction<ctx> *fun) {
    return PtrNodeManager<ctx>::template createRetNode<PT>(fun);
  }

  inline PtrNode *getRetNode(const ctx *C, const llvm::Function *F) {
    return PtrNodeManager<ctx>::getRetNode(C, F);
  }

  // logically exist, but no corresponding llvm::Value
  inline PtrNode *createAnonPtrNode() {
    return PtrNodeManager<ctx>::template createAnonPtrNode<PT>();
  }

  inline PtrNode *createAnonPtrNode(void *tag) {
    return PtrNodeManager<ctx>::template createAnonPtrNode<PT>(tag);
  }

  inline PtrNode *getPtrNode(const ctx *C, const llvm::Value *V) {
    return PtrNodeManager<ctx>::template getPtrNode<Canonicalizer>(C, V);
  }

  inline PtrNode *getPtrNodeOrNull(const ctx *C, const llvm::Value *V) {
    return PtrNodeManager<ctx>::template getPtrNodeOrNull<Canonicalizer>(C, V);
  }

  inline PtrNode *createPtrNode(const ctx *C, const llvm::Value *V) {
    return PtrNodeManager<ctx>::template createPtrNode<Canonicalizer, PT>(C, V);
  }

  inline PtrNode *getOrCreatePtrNode(const ctx *C, const llvm::Value *V) {
    return PtrNodeManager<ctx>::template getOrCreatePtrNode<Canonicalizer, PT>(
        C, V);
  }
};

} // namespace aser

namespace std {

template <typename T1, typename T2> struct hash<pair<T1 *, T2 *>> {
  size_t operator()(const pair<T1 *, T2 *> &p) const {
    llvm::hash_code seed = llvm::hash_value(p.first);
    llvm::hash_code hash = llvm::hash_combine(p.second, seed);
    return hash_value(hash);
  }
};

} // namespace std

#undef MODEL
#undef ALLOCATE

#endif
