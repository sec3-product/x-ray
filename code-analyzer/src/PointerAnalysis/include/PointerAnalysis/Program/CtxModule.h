//
// Created by peiming on 10/30/19.
//
#ifndef ASER_PTA_CTXMODULE_H
#define ASER_PTA_CTXMODULE_H

#include <llvm/IR/Value.h>

#include "PointerAnalysis/Graph/CallGraph.h"
#include "PointerAnalysis/Program/InterceptResult.h"

namespace xray {

// a program module but with context information
template <typename ctx> class CtxModule {
private:
  using CT = CtxTrait<ctx>;
  using CallGraphTy = CallGraph<ctx>;
  using CallNodeTy = CallGraphNode<ctx>;

  using KeyType = std::pair<const ctx *, const llvm::Value *>;
  llvm::DenseMap<KeyType, const CtxFunction<ctx> *> ctxFunMap;
  llvm::DenseMap<KeyType, const InDirectCallSite<ctx> *>
      ctxFunPtrMap; // indirect call sites

  // call callgraph is needed to determine the context for the function
  std::unique_ptr<CallGraphTy> callGraph;

  llvm::StringRef entryName;
  const llvm::Module *llvmModule;

  struct Noop {
    template <typename... Args> inline void operator()(Args &&...) {}
  };
  // TODO: refactoring: Move beforenewnode call back outside the function!
  //  This function should only deals with direct calls!
  template <typename BeforeNewNode, typename OnNewDirectNode,
            typename OnNewInDirect>
  inline std::pair<CallNodeTy *, bool>
  getOrCreateCallNode(const ctx *curCtx, const ctx *preCtx,
                      const llvm::Function *F, const llvm::Instruction *I,
                      BeforeNewNode beforeNewNode, OnNewDirectNode onNewNode,
                      OnNewInDirect onNewInDirect) {
    // check the signature
    // TODO: add it back!
    // std::is_invocable_r is only available on C++17
    /*
    static_assert(std::is_invocable_r<bool, OnNewDirectNode, CallNodeTy
    *>::value, ""); static_assert( std::is_invocable_r<InterceptResult,
    BeforeNewNode, const llvm::Function **, const llvm::Instruction
    *>::value);
    */
    bool isEntryNode = false;
    if (I == nullptr) {
      assert(F->getName().equals(entryName) &&
             "empty call site for non-entry function??");
      isEntryNode = true;
    }
    // the function can be intercepted by the language module and change it
    // to another function e.g., pthread_create(subroutine) might be
    // redirect to subroutine to explore the callgraph of the created thread
    // even if the function is has body, it is not necessary always visit
    // them e.g., a wrapper heap allocation function or memcpy-like
    // function, which we do not care the implementation but know the
    // semantics
    if (isEntryNode) {
      // must be the first node
      assert(ctxFunMap.empty());

      // new node are inserted
      CallNodeTy *callNode = callGraph->createCallNode(curCtx, F, I);
      auto result = ctxFunMap
                        .insert(std::make_pair(
                            std::make_pair(curCtx, F), // ctx + llvm::Function
                            callNode->getTargetFun()))
                        .second; // CtxFunction
      assert(result);

      auto fun = const_cast<CtxFunction<ctx> *>(callNode->getTargetFun());
      // the call node might be an external function
      bool needToExpand = onNewNode(callNode);
      // we should always expand the entry function
      assert(needToExpand);

      // now allow user to intercept the entry function, we did it after all pta
      // node are added so that they are accessible by the user
      beforeNewNode(CT::getInitialCtx(), CT::getInitialCtx(), F, nullptr);

      return std::make_pair(callNode, true);
    }

    InterceptResult interceptResult = beforeNewNode(curCtx, preCtx, F, I);
    if (!interceptResult.redirectTo ||
        interceptResult.option == InterceptResult::Option::IGNORE_FUN) {
      return std::make_pair(nullptr, false);
    }

    if (llvm::isa<llvm::Function>(interceptResult.redirectTo)) {
      // redirect to a function
      F = llvm::dyn_cast<llvm::Function>(interceptResult.redirectTo);

      auto it = ctxFunMap.find(std::make_pair(curCtx, F));
      if (it != ctxFunMap.end()) {
        // already in the call graph, do not need to traverse
        return std::make_pair(it->second->getCallNode(), false);
      }

      // new node are inserted
      CallNodeTy *callNode = callGraph->createCallNode(curCtx, F, I);
      auto result = ctxFunMap
                        .insert(std::make_pair(std::make_pair(curCtx, F),
                                               callNode->getTargetFun()))
                        .second;
      assert(result);

      auto fun = const_cast<CtxFunction<ctx> *>(callNode->getTargetFun());
      bool needToExpand = false;
      if (interceptResult.option == InterceptResult::Option::EXPAND_BODY) {
        needToExpand =
            onNewNode(callNode); // the call node might be an external function
      }
      // an external function (or marked as ONLY_CALLSITE by user)
      if (!needToExpand) {
        fun->markAsExtFunction();
      }
      return std::make_pair(callNode, needToExpand);
    } else {
      // redirect to indirect calls
      const llvm::Value *indirect = interceptResult.redirectTo;
      // must be a function pointer
      assert(llvm::isa<llvm::PointerType>(indirect->getType()));
      // assert(llvm::isa<llvm::FunctionType>(indirect->getType()->getPointerElementType()));
      // use previous ctx as the indirect call has not been resolved yet
      CallNodeTy *callNode =
          getOrCreateInDirectCallNode(preCtx, I, onNewInDirect, indirect).first;
      return std::make_pair(callNode, false);
    }
  }

  template <typename OnNewInDirectNode>
  inline std::pair<CallNodeTy *, bool>
  getOrCreateInDirectCallNode(const ctx *C, const llvm::Instruction *I,
                              OnNewInDirectNode callBack,
                              const llvm::Value *target = nullptr) {
    // check the signature
    // TODO: add it back!
    // static_assert(std::is_invocable<OnNewInDirectNode, CallNodeTy *>::value,
    // "");

    auto it = ctxFunPtrMap.find(std::make_pair(C, I));
    if (it != ctxFunPtrMap.end()) {
      return std::make_pair(it->second->getCallNode(), false);
    }
    // new node
    CallNodeTy *callNode = callGraph->createIndCallNode(C, target, I);
    auto result = ctxFunPtrMap
                      .insert(std::make_pair(std::make_pair(C, I),
                                             callNode->getTargetFunPtr()))
                      .second;
    assert(result);

    callBack(callNode);
    return std::make_pair(callNode, true);
  }

  template <typename OnNewEdge>
  inline void createEdge(CallNodeTy *src, CallNodeTy *dst,
                         const llvm::Function *originalTarget,
                         const llvm::Instruction *inst, OnNewEdge callBack) {
    // if the node is null, simply return
    if (src == nullptr || dst == nullptr) {
      return;
    }
    // check the signature
    // TODO: add it back
    // static_assert(std::is_invocable<OnNewEdge, CallNodeTy *, CallNodeTy *,
    // const llvm::Instruction *>::value,
    // "");

    src->insertEdge(dst, CallEdge(inst));
    // original target might be different than the intercepted target
    callBack(src, dst, originalTarget, inst);
  }

  // from root, expand the call graph
  template <typename BeforeNewNode, typename OnNewDirectNode,
            typename OnNewInDirectNode, typename OnNewEdge>
  inline void expandFromNode(CallNodeTy *root, BeforeNewNode beforeNewNode,
                             OnNewDirectNode onNewDirect,
                             OnNewInDirectNode onNewInDirect,
                             OnNewEdge onNewEdge) {
    std::queue<CallNodeTy *> BFSQueue;
    BFSQueue.push(root);

    while (!BFSQueue.empty()) {
      auto curNode = BFSQueue.front();
      BFSQueue.pop();
      visitCallNode(curNode, BFSQueue, beforeNewNode, onNewDirect,
                    onNewInDirect, onNewEdge);
    }
  }

  // Assumption: the call node has never been visited before
  template <typename BeforeNewNode, typename OnNewDirectNode,
            typename OnNewInDirectNode, typename OnNewEdge>
  void visitCallNode(CallNodeTy *node, std::queue<CallNodeTy *> &BFSQueue,
                     BeforeNewNode beforeNewNode, OnNewDirectNode onNewDirect,
                     OnNewInDirectNode onNewInDirect, OnNewEdge onNewEdge) {
    if (node->isIndirectCall()) {
      return;
    }

    const CtxFunction<ctx> &ctxFun = *node->getTargetFun();
    const ctx *context = ctxFun.getContext();

    // llvm::outs() << "!!! visitCallNode : " << ctxFun.getName() << "\n";

    for (const auto &BB : ctxFun) {
      for (const auto &inst : BB) {
        if (llvm::isa<llvm::CallInst>(inst) ||
            llvm::isa<llvm::InvokeInst>(inst)) {
          xray::CallSite cs(const_cast<llvm::Instruction *>(&inst));

          if (cs.isIndirectCall()) {
            // llvm::outs() << "!!! visitCallNode isIndirectCall: " << inst <<
            // "\n";
            //  take another shot
            auto targetNode =
                visitInDirectCallSite(&inst, context, onNewInDirect);
            createEdge(node, targetNode, nullptr, &inst, onNewEdge);
            // node->insertEdge(targetNode, CallEdge(&inst));
          } else {
            // evolve the context upon new call site
            const ctx *calledCtx = CT::contextEvolve(context, &inst);
            const llvm::Function *fun = cs.getTargetFunction();
            // llvm::outs() << "!!! visitCallNode directCall: " << inst << "
            // func: " << fun->getName() << "\n";

            auto targetNode =
                visitDirectCallSite(calledCtx, context, fun, &inst, BFSQueue,
                                    beforeNewNode, onNewDirect, onNewInDirect);
            // node->insertEdge(targetNode, CallEdge(&inst));
            createEdge(node, targetNode, fun, &inst, onNewEdge);
          }
        }
      }
    }
  }

  template <typename OnNewInDirectNode>
  inline CallNodeTy *visitInDirectCallSite(const llvm::Instruction *callSite,
                                           const ctx *context,
                                           OnNewInDirectNode onNewInDirect) {
    assert(callSite != nullptr);
    // function ptr is in caller's context, the new context are evolved upon
    // the target is resolved
    auto r =
        this->getOrCreateInDirectCallNode(context, callSite, onNewInDirect);
    return r.first;
  }

  template <typename BeforeNewNode, typename OnNewDirectNode,
            typename OnNewInDirect>
  inline CallNodeTy *
  visitDirectCallSite(const ctx *curCtx, const ctx *preCtx,
                      const llvm::Function *fun, const llvm::Instruction *I,
                      std::queue<CallNodeTy *> &BFSQueue,
                      BeforeNewNode beforeNewNode, OnNewDirectNode onNewDirect,
                      OnNewInDirect onNewInDirect) {
    assert(fun != nullptr);
    auto r = this->getOrCreateCallNode(curCtx, preCtx, fun, I, beforeNewNode,
                                       onNewDirect, onNewInDirect);

    bool needToHandle = r.second;
    if (needToHandle) {
      // a new (non-external) function discovered
      assert(r.first != nullptr);
      BFSQueue.push(r.first);
    }

    return r.first;
  }

public:
  CtxModule(const llvm::Module *M, llvm::StringRef entry)
      : callGraph(new CallGraph<ctx>()), llvmModule(M), entryName(entry) {}

  // OnNewNode: call back
  template <typename BeforeNewNode, typename OnNewDirectNode,
            typename OnNewInDirectNode, typename OnNewEdge>
  void
  buildInitCallGraph(BeforeNewNode beforeNewNode, OnNewDirectNode onNewDirect,
                     OnNewInDirectNode onNewIndirect, OnNewEdge onNewEdge) {
    // TODO: add it back
    /*
    static_assert(std::is_invocable_r<bool, OnNewDirectNode, CallNodeTy
    *>::value, ""); static_assert(std::is_invocable_r<void, OnNewInDirectNode,
    CallNodeTy *>::value, ""); static_assert( std::is_invocable_r<void,
    OnNewEdge, CallNodeTy *, CallNodeTy *, const llvm::Instruction *>::value,
    "");
    */

    assert(ctxFunMap.empty() && ctxFunPtrMap.empty());
    const ctx *initialCtx = CT::getInitialCtx();
    llvm::Function *entryFun = llvmModule->getFunction(entryName);
    if (entryFun == nullptr || entryFun->isDeclaration()) {
      llvm::errs() << "Fatal Error: '" << entryName
                   << "' function cannot be found!\n";
      exit(1);
    }
    // here do the initial call callgraph construction
    // main function has not call site
    auto entryNode =
        getOrCreateCallNode(initialCtx, initialCtx, entryFun, nullptr,
                            beforeNewNode, onNewDirect, onNewIndirect)
            .first;
    // CallNode *entryNode = result.first;
    expandFromNode(entryNode, beforeNewNode, onNewDirect, onNewIndirect,
                   onNewEdge);
  }

  template <typename BeforeNewNode, typename OnNewDirectNode,
            typename OnNewInDirectNode, typename OnNewEdge>
  void resolveCallTo(CallGraphNode<ctx> *resolvedNode,
                     const llvm::Function *target, BeforeNewNode beforeNewNode,
                     OnNewDirectNode onNewDirect,
                     OnNewInDirectNode onNewIndirect, OnNewEdge onNewEdge) {
    // TODO: add it back
    /*
    static_assert(std::is_invocable_r<bool, OnNewDirectNode, CallNodeTy
    *>::value, ""); static_assert(std::is_invocable_r<void, OnNewInDirectNode,
    CallNodeTy *>::value, ""); static_assert( std::is_invocable_r<void,
    OnNewEdge, CallNodeTy *, CallNodeTy *, const llvm::Instruction *>::value,
    "");
    */

    // can a indirect call node have multiple caller?
    assert(resolvedNode->predEdgeCount() == 1);

    auto it = resolvedNode->pred_edge_begin();
    auto et = resolvedNode->pred_edge_end();
    for (; it != et; it++) {
      const llvm::Instruction *callInst = it->first.getCallInstruction();
      CallNodeTy *callerNode = it->second;
      const ctx *callerCtx = callerNode->getContext();
      const ctx *calleeCtx =
          CT::contextEvolve(callerNode->getContext(), callInst);

      std::pair<CallNodeTy *, bool> result;
      result = getOrCreateCallNode(calleeCtx, callerCtx, target, callInst,
                                   beforeNewNode, onNewDirect, onNewIndirect);

      if (result.first == nullptr) {
        // the resolved function might then be ignored by the language model
        // or it replaced to another indirect call
        return;
      } else {
        createEdge(callerNode, result.first, target, callInst, onNewEdge);
        if (!result.first->isIndirectCall()) {
          resolvedNode->getTargetFunPtr()->resolvedToNode(result.first);
          if (result.second) {
            // a new node is added to the call graph!
            expandFromNode(result.first, beforeNewNode, onNewDirect,
                           onNewIndirect, onNewEdge);
          }
        }
      }
    }
  }

  [[nodiscard]] inline llvm::StringRef getEntryName() const {
    return this->entryName;
  }

  [[nodiscard]] inline const llvm::Module *getLLVMModule() {
    return this->llvmModule;
  }

  [[nodiscard]] inline const llvm::DataLayout &getDataLayout() {
    return this->llvmModule->getDataLayout();
  }

  [[nodiscard]] inline const CallGraphTy *getCallGraph() {
    return callGraph.get();
  }

  [[nodiscard]] inline const CallGraphNode<ctx> *
  getDirectNode(const ctx *C, const llvm::Function *F) {
    auto it = ctxFunMap.find(std::make_pair(C, F));
    if (it == ctxFunMap.end()) {
      return nullptr;
    }
    // assert(it != ctxFunMap.end());

    return it->second->getCallNode();
  }

  [[nodiscard]] inline const CallGraphNode<ctx> *
  getDirectNodeOrNull(const ctx *C, const llvm::Function *F) {
    auto it = ctxFunMap.find(std::make_pair(C, F));
    if (it == ctxFunMap.end()) {
      return nullptr;
    }

    return it->second->getCallNode();
  }

  [[nodiscard]] inline const CallGraphNode<ctx> *
  getInDirectNode(const ctx *C, const llvm::Instruction *I) {
    auto it = ctxFunPtrMap.find(std::make_pair(C, I));
    // JEFF: this fails on GraphBLAS openmp_demo
    // assert(it != ctxFunPtrMap.end());
    if (it == ctxFunPtrMap.end())
      return nullptr; // JEFF

    return it->second->getCallNode();
  }
};

} // namespace xray

#endif
