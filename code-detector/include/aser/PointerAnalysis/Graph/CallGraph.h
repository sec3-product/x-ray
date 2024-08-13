//
// Created by peiming on 8/14/19.
//
#ifndef ASER_PTA_CALLGRAPH_H
#define ASER_PTA_CALLGRAPH_H

#include <llvm/ADT/GraphTraits.h>
#include <llvm/Support/DOTGraphTraits.h>

#include "aser/PointerAnalysis/Graph/GraphBase/GraphBase.h"
#include "aser/PointerAnalysis/Program/CtxFunction.h"
#include "aser/PointerAnalysis/Program/Program.h"
#include "aser/Util/Util.h"

namespace aser {

template <typename ctx> class CtxModule;

enum class CallKind : uint8_t { Direct = 0, Indirect = 1 };

// caller --instruction--> callee
// should be cheap to copy
class CallEdge {
private:
  const llvm::Instruction *callerSite;

public:
  explicit CallEdge(const llvm::Instruction *callerSite)
      : callerSite(callerSite) {
    assert(llvm::isa<llvm::CallInst>(callerSite) ||
           llvm::isa<llvm::InvokeInst>(callerSite));
  }

  [[nodiscard]] inline const llvm::Instruction *getCallInstruction() const {
    return callerSite;
  }

  inline bool operator<(const CallEdge &rhs) const {
    return this->getCallInstruction() < rhs.getCallInstruction();
  }

  bool operator==(const CallEdge &rhs) const {
    return this->getCallInstruction() == rhs.getCallInstruction();
  }
};

// forward declaration
template <typename ctx> class CallGraph;

/// CRTP
template <typename ctx>
class CallGraphNode : public NodeBase<CallEdge, CallGraphNode<ctx>> {
private:
  using super = NodeBase<CallEdge, CallGraphNode<ctx>>;

  const CallKind kind;
  union U {
    CtxFunction<ctx> fun;
    InDirectCallSite<ctx> funPtr;

    U(const ctx *C, const llvm::Function *F, const llvm::Instruction *I,
      CallGraphNode<ctx> *N)
        : fun(C, F, I, N) {}
    U(const ctx *C, const llvm::Instruction *I, const llvm::Value *V,
      CallGraphNode<ctx> *N)
        : funPtr(C, I, V, N) {}
    ~U(){};
  } target;

  // use with care!!!!
  CallGraphNode(const ctx *C, const llvm::Function *F,
                const llvm::Instruction *I, NodeID id)
      : super(id), target(C, F, I, this), kind(CallKind::Direct) {}

  CallGraphNode(const ctx *C, const llvm::Instruction *I, const llvm::Value *V,
                NodeID id)
      : super(id), target(C, I, V, this), kind(CallKind::Indirect) {
    // V is not neccessarily be the called value of I, as it can be intercepted!
  }

  CallGraphNode(const ctx *C, const llvm::Instruction *I, NodeID id)
      : super(id), target(C, I, nullptr, this), kind(CallKind::Indirect) {}

public:
  ~CallGraphNode() {
    if (kind == CallKind::Indirect) {
      target.funPtr.~InDirectCallSite<ctx>();
    }
  }

  [[nodiscard]] inline bool isIndirectCall() const {
    return this->kind == CallKind::Indirect;
  }

  [[nodiscard]] inline const CtxFunction<ctx> *getTargetFun() const {
    assert(kind == CallKind::Direct);
    return &this->target.fun;
  }

  [[nodiscard]] inline const InDirectCallSite<ctx> *getTargetFunPtr() const {
    assert(kind == CallKind::Indirect);
    return &this->target.funPtr;
  }

  [[nodiscard]] inline InDirectCallSite<ctx> *getTargetFunPtr() {
    assert(kind == CallKind::Indirect);
    return &this->target.funPtr;
  }

  [[nodiscard]] inline const ctx *getContext() const {
    if (kind == CallKind::Direct) {
      return target.fun.getContext();
    } else {
      return target.funPtr.getContext();
    }
  }

  friend typename super::GraphTy;
  friend CallGraph<ctx>;
  friend GraphBase<CallGraphNode<ctx>, CallKind>;
};

template <typename ctx>
class CallGraph : public GraphBase<CallGraphNode<ctx>, CallEdge> {
public:
  using NodeType = CallGraphNode<ctx>;

private:
  // assumption, the CtxFunction passed is a newly created function
  inline NodeType *createCallNode(const ctx *C, const llvm::Function *F,
                                  const llvm::Instruction *I) {
    return this->template addNewNode<NodeType>(C, F, I);
  }

  // assumption, the CtxFunction passed is a newly created function
  inline NodeType *createIndCallNode(const ctx *C, const llvm::Value *V,
                                     const llvm::Instruction *I) {
    return this->template addNewNode<NodeType>(C, I, V);
  }

  // assumption, the CtxFunction passed is a newly created function
  inline NodeType *createIndCallNode(const ctx *C, const llvm::Instruction *I) {
    return this->template addNewNode<NodeType>(C, I);
  }

public:
  CallGraph() = default;
  friend CtxModule<ctx>;
};

} // namespace aser

namespace llvm {

template <typename ctx>
struct GraphTraits<aser::CallGraph<ctx>>
    : public GraphTraits<
          aser::GraphBase<aser::CallGraphNode<ctx>, aser::CallEdge>> {};

template <typename ctx>
struct GraphTraits<const aser::CallGraph<ctx>>
    : public GraphTraits<
          const aser::GraphBase<aser::CallGraphNode<ctx>, aser::CallEdge>> {};

template <typename ctx>
struct DOTGraphTraits<const aser::CallGraph<ctx>>
    : public DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool simple = false)
      : DefaultDOTGraphTraits(simple) {}

  static std::string getGraphName(const aser::CallGraph<ctx> &) {
    return "CallGraph";
  }

  /// Return function name;
  static std::string getNodeLabel(const aser::CallGraphNode<ctx> *node,
                                  const aser::CallGraph<ctx> &graph) {
    std::string str;
    raw_string_ostream os(str);
    os << aser::CtxTrait<ctx>::toString(node->getContext()) << "\n";
    if (node->isIndirectCall()) {
      os << "ID : " << node->getNodeID() << ", ";
      os << "Indirect, resolved to ";
      auto &targets = node->getTargetFunPtr()->getResolvedNode();
      for (auto &target : targets) {
        os << target->getNodeID() << " ,";
      }
    } else {
      os << "ID : " << node->getNodeID() << ", ";
      aser::prettyFunctionPrinter(node->getTargetFun()->getFunction(), os);
      // os << node->getMethod().getFunction()->getName();
    }
    return os.str();
  }
};

} // namespace llvm

#endif