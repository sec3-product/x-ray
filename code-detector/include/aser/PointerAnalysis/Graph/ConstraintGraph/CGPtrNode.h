//
// Created by peiming on 11/2/19.
//
#ifndef ASER_PTA_CGPTRNODE_H
#define ASER_PTA_CGPTRNODE_H

#include <llvm/IR/GlobalValue.h>

#include "CGNodeBase.h"
#include "aser/PointerAnalysis/Context/CtxTrait.h"

namespace aser {

template <typename ctx> class Pointer;
template <typename ctx> class CallGraphNode;
template <typename ctx> class ConstraintGraph;

struct PtrNodeTag {
  virtual std::string toString() = 0;
};

// nodes represent pointers
template <typename ctx> class CGPtrNode : public CGNodeBase<ctx> {
protected:
  using T = Pointer<ctx>;
  using super = CGNodeBase<ctx>;
  bool isAnonmyous;

  union U {
    const T *ptr;
    PtrNodeTag *tag;
  } u;

  // for anonymous ptrnode
  explicit CGPtrNode(NodeID id)
      : super(id, CGNodeKind::PtrNode), isAnonmyous(true) {
    u.tag = nullptr;
  }

  // C++ does allow implicit conversion from void * to const T*, so we are good
  // here
  CGPtrNode(PtrNodeTag *tag, NodeID id)
      : super(id, CGNodeKind::PtrNode), isAnonmyous(true) {
    u.tag = tag;
  }

  CGPtrNode(const T *ptr, NodeID id)
      : super(id, CGNodeKind::PtrNode), isAnonmyous(false) {
    u.ptr = ptr;
  }

public:
  static inline bool classof(const super *node) {
    return node->getType() == CGNodeKind::PtrNode;
  }

  inline bool isAnonNode() const { return isAnonmyous; }

  inline const T *getPointer() const {
    assert(!isAnonmyous);
    return u.ptr;
  }

  inline PtrNodeTag *getTag() const {
    assert(isAnonmyous);
    return u.tag;
  }

  inline const ctx *getContext() const {
    assert(!isAnonmyous);
    return this->getPointer()->getContext();
  }

  [[nodiscard]] std::string toString() const override {
    std::string str;
    llvm::raw_string_ostream os(str);
    if (this->isSuperNode()) {
      os << "SuperNode: \n";
      llvm::dump(this->childNodes, os);
    } else {
      os << super::getNodeID() << "\n";
      if (isAnonmyous) {
        os << "anonymous ptr";
        if (u.tag != nullptr) {
          os << "\n" << u.tag->toString();
        }
      } else if (u.ptr->getValue()->hasName()) {
        if (llvm::isa<llvm::GlobalValue>(u.ptr->getValue())) {
          os << CtxTrait<ctx>::toString(getContext(), true) << "\n";
          os << u.ptr->getValue()->getName();
        } else {
          os << *u.ptr->getValue(); //->getName();
        }
      } else {
        os << CtxTrait<ctx>::toString(getContext(), true) << "\n";
        os << *u.ptr->getValue() << "\n";
      }
    }

    // os << "\n" << getPTASourceLocSnippet(u.ptr->getValue());  // JEFF

    return os.str();
  }

  friend class GraphBase<super, Constraints>;
  friend class ConstraintGraph<ctx>;
};

} // namespace aser

#endif
