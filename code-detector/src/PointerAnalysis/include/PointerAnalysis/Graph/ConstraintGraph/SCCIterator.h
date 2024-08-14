//
// Created by peiming on 9/16/19.
//
#ifndef ASER_PTA_SCCITERATOR_H
#define ASER_PTA_SCCITERATOR_H

#include <llvm/ADT/BitVector.h>
#include <utility>

#include "ConstraintGraph.h"

// #define DEBUG_OUTPUT

namespace aser {

// SCC iterator on constraint graph
// extend from llvm::scc_iterator, specialized for constraint graph
// cons -> which kind of constraints we need to sort on
// reverse -> whether we should DFS the graph in reverse order (so that the scc
// iterator is in topo order)
//            otherwise it will be in reverse topo-order
template <typename ctx, Constraints cons, bool reverse>
class SCCIterator
    : public llvm::iterator_facade_base<SCCIterator<ctx, cons, reverse>,
                                        std::forward_iterator_tag,
                                        const std::vector<CGNodeBase<ctx> *>> {
private:
  using GraphT = ConstraintGraph<ctx>;
  using NodeT = CGNodeBase<ctx>;
  using NodeRef = NodeT *;
  using ChildItTy = typename NodeT::cg_iterator;
  using SccTy = std::vector<NodeRef>;
  using reference = typename SCCIterator::reference;

  /// Element of VisitStack during DFS.
  struct StackElement {
    NodeRef Node;        ///< The current node pointer.
    ChildItTy NextChild; ///< The next child, modified inplace during DFS.
    unsigned MinVisited; ///< Minimum uplink value of all children of Node.

    StackElement(NodeRef Node, const ChildItTy &Child, unsigned Min)
        : Node(Node), NextChild(Child), MinVisited(Min) {}

    bool operator==(const StackElement &Other) const {
      return Node == Other.Node && NextChild == Other.NextChild &&
             MinVisited == Other.MinVisited;
    }
  };

  /// The visit counters used to detect when a complete SCC is on the stack.
  /// visitNum is the global counter.
  unsigned visitNum{};

  /// the number indicates when the node is access in DFS
  /// nodeVisitNumbers are per-node visit numbers, also used as DFS flags.
  llvm::DenseMap<NodeRef, unsigned> nodeVisitNumbers;

  /// Stack holding nodes of the SCC.
  std::vector<NodeRef> SCCNodeStack;

  /// The current SCC, retrieved using operator*().
  SccTy CurrentSCC;

  /// BitVector that tracks the node that haven't been visited
  /// since contraint graph are not always connected.
  llvm::BitVector visitedNode;
  llvm::BitVector trueVisitedNode;

  /// last Root we visited
  int lastRoot{};

  /// DFS stack, Used to maintain the ordering.  The top contains the current
  /// node, the next child to visit, and the minimum uplink value of all child
  std::vector<StackElement> VisitStack;

  /// A single "visit" within the non-recursive DFS traversal.
  void DFSVisitOne(NodeRef N);

  /// The stack-based DFS traversal; defined below.
  void DFSVisitChildren();

  /// Compute the next SCC using the DFS traversal.
  void GetNextSCC();

  const GraphT *consG;

  explicit SCCIterator(const GraphT &G)
      : visitNum(0), visitedNode(G.getNodeNum()),
        trueVisitedNode(G.getNodeNum()), lastRoot(0), consG(&G) {
    NodeRef entryN = G[0];
    DFSVisitOne(entryN);
    GetNextSCC();
  }

  explicit SCCIterator(const GraphT &G, llvm::BitVector workList)
      : visitNum(0), visitedNode(std::move(workList)),
        trueVisitedNode(G.getNodeNum()), lastRoot(0), consG(&G) {
    int first = visitedNode.find_first_unset();
    if (first < 0) {
      return;
    }
    NodeRef entryN = G.getCGNode(first);
    DFSVisitOne(entryN);
    GetNextSCC();
  }

  /// End is when the DFS stack is empty.
  SCCIterator() = default;

  ChildItTy child_begin(NodeRef node) {
    if constexpr (cons == Constraints::copy) {
      if constexpr (reverse) {
        return node->pred_copy_begin();
      } else {
        return node->succ_copy_begin();
      }
    } else if constexpr (cons == Constraints::offset) {
      if constexpr (reverse) {
        return node->pred_offset_begin();
      } else {
        return node->succ_offset_begin();
      }
    } else {
      llvm_unreachable("unsupported constraints");
    }
  }

  ChildItTy child_end(NodeRef node) {
    if constexpr (cons == Constraints::copy) {
      if constexpr (reverse) {
        return node->pred_copy_end();
      } else {
        return node->succ_copy_end();
      }
    } else if constexpr (cons == Constraints::offset) {
      if constexpr (reverse) {
        return node->pred_offset_end();
      } else {
        return node->succ_offset_end();
      }
    } else {
      llvm_unreachable("unsupported constraints");
    }
  }

public:
  static SCCIterator<ctx, cons, reverse> begin(const GraphT &G) {
    return SCCIterator<ctx, cons, reverse>(G);
  }

  static SCCIterator<ctx, cons, reverse>
  begin(const GraphT &G, const llvm::BitVector &workList) {
    return SCCIterator<ctx, cons, reverse>(G, workList);
  }

  static SCCIterator<ctx, cons, reverse> end(const GraphT &) {
    return SCCIterator<ctx, cons, reverse>();
  }

  [[nodiscard]] size_t visitedNodeNum() const {
    return this->visitedNode.size();
  }

  bool operator==(const SCCIterator &x) const {
    return VisitStack == x.VisitStack && CurrentSCC == x.CurrentSCC;
  }

  // step to next connect sub graph
  bool nextSubGraph() {
    // much call 'step' after finishing traversing
    // the current sub graph
    assert(CurrentSCC.empty());

    // we need to check the next
    lastRoot = visitedNode.find_next_unset(lastRoot);

    while (lastRoot > 0) {
      // there are extra node we need to traverse
      NodeRef root = consG->getNode(lastRoot);

      if (!root->hasSuperNode()) {
        DFSVisitOne(consG->getNode(lastRoot));
        GetNextSCC();
        return true;
      } else {
        // skip node that has super nodes.
        lastRoot = visitedNode.find_next_unset(lastRoot);
      }
    }
    // here we finished traverse the graph
    return false;
  }

  SCCIterator &operator++() {
    GetNextSCC();

    if (CurrentSCC.empty()) {
      // go to next sub graph
      nextSubGraph();
    }

    return *this;
  }

  reference operator*() const {
    assert(!CurrentSCC.empty() && "Dereferencing END SCC iterator!");
    return CurrentSCC;
  }

  /// Test if the current SCC has a loop.
  ///
  /// If the SCC has more than one node, this is trivially true.  If not, it
  /// may still contain a loop if the node has an edge back to itself.
  [[nodiscard]] bool hasLoop() const;
};

template <typename ctx, Constraints cons, bool reverse>
void SCCIterator<ctx, cons, reverse>::DFSVisitOne(NodeRef N) {
  ++visitNum;
  nodeVisitNumbers[N] = visitNum;
  SCCNodeStack.push_back(N);

  // only detect scc that connected by copy edges
  VisitStack.push_back(
      StackElement(N, child_begin(N) /*N->pred_copy_begin()*/, visitNum));
  // this should be the first time to visit the node.
  // assert(!visitedNode.test(N->getNodeID()));
  visitedNode.set(N->getNodeID());
  trueVisitedNode.set(N->getNodeID());

#ifdef DEBUG_OUTPUT // Enable if needed when debugging.
  llvm::dbgs() << "TarjanSCC: Node " << N->getNodeID()
               << " : visitNum = " << visitNum << "\n";
#endif
}

template <typename ctx, Constraints cons, bool reverse>
void SCCIterator<ctx, cons, reverse>::DFSVisitChildren() {
  assert(!VisitStack.empty());
  while (
      VisitStack.back().NextChild !=
      child_end(
          VisitStack.back().Node) /*VisitStack.back().Node->pred_copy_end()*/) {
    // TOS has at least one more child so continue DFS
    NodeRef childN = *VisitStack.back().NextChild++;
    auto Visited = nodeVisitNumbers.find(childN);

    if (Visited == nodeVisitNumbers.end()) {
      // this node has never been seen.
      DFSVisitOne(childN);
      continue;
    }
    // node has been visited, update the min value
    unsigned childNum = Visited->second;
    if (VisitStack.back().MinVisited > childNum)
      VisitStack.back().MinVisited = childNum;
  }
}

template <typename ctx, Constraints cons, bool reverse>
void SCCIterator<ctx, cons, reverse>::GetNextSCC() {
  // Prepare to compute the next SCC
  CurrentSCC.clear();
  while (!VisitStack.empty()) {
    DFSVisitChildren();

    // Pop the leaf on top of the VisitStack.
    NodeRef visitingN = VisitStack.back().Node;
    unsigned minVisitNum = VisitStack.back().MinVisited;
    assert(VisitStack.back().NextChild ==
           child_end(visitingN) /*visitingN->pred_copy_end()*/);
    VisitStack.pop_back();

    // Propagate MinVisitNum to parent so we can detect the SCC starting
    // node.
    if (!VisitStack.empty() && VisitStack.back().MinVisited > minVisitNum)
      VisitStack.back().MinVisited = minVisitNum;

#ifdef DEBUG_OUTPUT // Enable if needed when debugging.
    llvm::dbgs() << "TarjanSCC: Popped node " << visitingN->getNodeID()
                 << " : minVisitNum = " << minVisitNum
                 << "; Node visit num = " << nodeVisitNumbers[visitingN]
                 << "\n";
#endif

    if (minVisitNum != nodeVisitNumbers[visitingN])
      continue;

    // A full SCC is on the SCCNodeStack!  It includes all nodes below
    // visitingN on the stack.  Copy those nodes to CurrentSCC,
    // reset their minVisit values, and return (this suspends
    // the DFS traversal till the next ++).
    do {
      CurrentSCC.push_back(SCCNodeStack.back());
      SCCNodeStack.pop_back();
      // reset the visit numbers
      nodeVisitNumbers[CurrentSCC.back()] = ~0U;
    } while (CurrentSCC.back() != visitingN);
    return;
  }
}

template <typename ctx, Constraints cons, bool reverse>
bool SCCIterator<ctx, cons, reverse>::hasLoop() const {
  assert(!CurrentSCC.empty() && "Dereferencing END SCC iterator!");
  return CurrentSCC.size() > 1;
}

/// if traverse node in reverse order, the iterator is in topo-order
/// otherwise, the iterator is in reverse topo-order

/// Construct the begin iterator for a deduced graph type T.
template <typename ctx, Constraints cons, bool reverse = true>
SCCIterator<ctx, cons, reverse> scc_begin(const ConstraintGraph<ctx> &G) {
  return SCCIterator<ctx, cons, reverse>::begin(G);
}

/// Construct the begin iterator for a deduced graph type T.
template <typename ctx, Constraints cons, bool reverse = true>
SCCIterator<ctx, cons, reverse> scc_begin(const ConstraintGraph<ctx> &G,
                                          const llvm::BitVector &workList) {
  return SCCIterator<ctx, cons, reverse>::begin(G, workList);
}

/// Construct the end iterator for a deduced graph type T.
template <typename ctx, Constraints cons, bool reverse = true>
SCCIterator<ctx, cons, reverse> scc_end(const ConstraintGraph<ctx> &G) {
  return SCCIterator<ctx, cons, reverse>::end(G);
}

/// Construct the end iterator for a deduced graph type T.
template <typename ctx, Constraints cons, bool reverse = true>
SCCIterator<ctx, cons, reverse> scc_end(const ConstraintGraph<ctx> &G,
                                        const llvm::BitVector & /*workList*/) {
  return SCCIterator<ctx, cons, reverse>::end(G);
}

} // namespace aser

#endif
