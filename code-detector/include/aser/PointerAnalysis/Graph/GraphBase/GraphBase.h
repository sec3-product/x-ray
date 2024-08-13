//
// Created by peiming on 8/14/19.
//
#ifndef ASER_PTA_GRAPHBASE_H
#define ASER_PTA_GRAPHBASE_H

#include <llvm/ADT/GraphTraits.h>

#include <cassert>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "aser/PointerAnalysis/Graph/NodeID.def"
#include "aser/Util/Iterators.h"

namespace aser {

// forward declaration
template <typename EdgeKind, typename NodeTy> class NodeBase;

template <typename NodeType, typename EdgeKind> class GraphBase {
protected:
  using NodeList = std::vector<std::unique_ptr<NodeType>>;
  NodeList nodes;

public:
  GraphBase() = default;

  using NodeT = NodeType;
  using ConstNodeT = const NodeType;

  using NodePtrT = NodeType *;
  using ConstNodePtrT = const NodeType *;

  using EdgeT = EdgeKind;

  using iterator = UniquePtrIterator<typename NodeList::iterator>;
  using const_iterator = UniquePtrIterator<typename NodeList::const_iterator>;

  template <typename Node, typename... Args> Node *addNewNode(Args &&...args) {
    auto node = new Node(std::forward<Args>(args)..., this->getNodeNum());
    nodes.emplace_back(node);

    assert(node->getNodeID() == this->getNodeNum() - 1);
    node->setGraph(this);
    return node;
  }

  inline NodeType *getNode(NodeID id) const {
    assert(id < nodes.size());
    return this->nodes[id].get();
  }

  iterator begin() { return iterator(nodes.begin()); }
  iterator end() { return iterator(nodes.end()); }
  const_iterator begin() const { return const_iterator(nodes.begin()); }
  const_iterator end() const { return const_iterator(nodes.end()); }

  const_iterator cbegin() const { return const_iterator(nodes.begin()); }
  const_iterator cend() const { return const_iterator(nodes.end()); }

  [[nodiscard]] inline size_t getNodeNum() const { return nodes.size(); }
};

/// CRTP
// CallKind should be cheap to copy
template <typename EdgeKind, typename NodeTy> class NodeBase {
protected:
  // the callgraph that hold current Node
  using GraphTy = GraphBase<NodeTy, EdgeKind>;
  // edge can have different kinds
  using Edge = std::pair<EdgeKind, NodeTy *>;
  // callgraph edges
  using EdgeSet = std::set<Edge>;

  const GraphTy *graph;
  EdgeSet pred, succ;
  const NodeID id;

public:
  inline void setGraph(GraphTy *g) { this->graph = g; }

  using iterator = PairSecondIterator<typename EdgeSet::iterator>;
  using const_iterator = PairSecondIterator<typename EdgeSet::const_iterator>;

  using edge_iterator = typename EdgeSet::iterator;
  using const_edge_iterator = typename EdgeSet::const_iterator;

  inline explicit NodeBase(NodeID id) : id(id), graph(nullptr) {}

  friend class GraphBase<NodeTy, EdgeKind>;
  friend bool operator<(Edge &e1, Edge &e2) {
    if (e1.first == e2.first) {
      return e1.second < e2.second;
    }

    return e1.first < e2.first;
  }

  inline bool insertEdge(NodeTy *node, EdgeKind edgeKind) {
    assert(node != nullptr);

    bool b1 = succ.insert(std::make_pair(edgeKind, node)).second;
    bool b2 =
        node->pred.insert(std::make_pair(edgeKind, static_cast<NodeTy *>(this)))
            .second;
    assert(b1 == b2);
    return b1;
  }

  inline const GraphTy *getGraph() { return this->graph; }

  [[nodiscard]] inline NodeID getNodeID() const { return id; }

  [[nodiscard]] inline size_t predEdgeCount() const { return pred.size(); }
  [[nodiscard]] inline size_t succEdgeCount() const { return succ.size(); }

  inline iterator succ_begin() { return iterator(succ.begin()); }
  inline iterator succ_end() { return iterator(succ.end()); }
  inline const_iterator succ_begin() const {
    return const_iterator(succ.begin());
  }
  inline const_iterator succ_end() const { return const_iterator(succ.end()); }

  inline iterator pred_begin() { return iterator(pred.begin()); }
  inline iterator pred_end() { return iterator(pred.end()); }
  inline const_iterator pred_begin() const {
    return const_iterator(pred.begin());
  }
  inline const_iterator pred_end() const { return const_iterator(pred.end()); }

  inline edge_iterator edge_begin() { return succ.begin(); }
  inline edge_iterator edge_end() { return succ.end(); }
  inline const_edge_iterator edge_begin() const { return succ.begin(); }
  inline const_edge_iterator edge_end() const { return succ.end(); }

  inline edge_iterator pred_edge_begin() { return pred.begin(); }
  inline edge_iterator pred_edge_end() { return pred.end(); }
  inline const_edge_iterator pred_edge_begin() const { return pred.begin(); }
  inline const_edge_iterator pred_edge_end() const { return pred.end(); }
};
} // namespace aser

namespace llvm {

template <typename NodeType, typename EdgeType>
struct GraphTraits<const aser::GraphBase<NodeType, EdgeType>> {
  using GraphType = typename aser::GraphBase<NodeType, EdgeType>;
  // Elements to provide:
  // typedef NodeRef           - Type of Node token in the callgraph, which
  // should
  //                             be cheap to copy.
  using NodeRef = const NodeType *;
  // typedef ChildIteratorType - Type used to iterate over children in
  // callgraph,
  //                             dereference to a NodeRef.
  using ChildIteratorType = typename NodeType::const_iterator;

  static NodeRef getEntryNode(const GraphType *graph) {
    return *(graph->begin());
  }

  // Return iterators that point to the beginning and ending of the child
  // node list for the specified node.
  static ChildIteratorType child_begin(NodeRef node) {
    return node->const_succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->const_succ_end();
  }

  //- dereference to a NodeRef
  // nodes_iterator/begin/end - Allow iteration over all nodes in the
  // callgraph
  using nodes_iterator = typename GraphType::const_iterator;
  static nodes_iterator nodes_begin(const GraphType &G) { return G.begin(); }
  static nodes_iterator nodes_end(const GraphType &G) { return G.end(); }

  // typedef EdgeRef           - Type of Edge token in the callgraph, which
  //                             should be cheap to copy.
  using EdgeRef = std::pair<EdgeType, NodeRef>;
  // typedef ChildEdgeIteratorType - Type used to iterate over children edges
  //                                 in callgraph, dereference to a EdgeRef.
  using ChildEdgeIteratorType = typename NodeType::const_edge_iterator;

  // Return iterators that point to the beginning and ending of the
  // edge list for the given callgraph node.
  static ChildEdgeIteratorType child_edge_begin(NodeRef node) {
    return node->edge_begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef node) {
    return node->edge_end();
  }

  // Return the destination node of an edge.
  static NodeRef edge_dest(EdgeRef edge) { return edge.second; }

  static unsigned size(GraphType *G) { return G->getNodeNum(); }
};

template <typename NodeType, typename EdgeType>
struct GraphTraits<aser::GraphBase<NodeType, EdgeType>> {
  using GraphType = typename aser::GraphBase<NodeType, EdgeType>;
  // Elements to provide:
  // typedef NodeRef           - Type of Node token in the callgraph, which
  // should
  //                             be cheap to copy.
  using NodeRef = NodeType *;
  // typedef ChildIteratorType - Type used to iterate over children in
  // callgraph,
  //                             dereference to a NodeRef.
  using ChildIteratorType = typename NodeType::iterator;

  static NodeRef getEntryNode(GraphType *graph) { return *(graph->begin()); }

  // Return iterators that point to the beginning and ending of the child
  // node list for the specified node.
  static ChildIteratorType child_begin(NodeRef node) {
    return node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succ_end(); }

  //- dereference to a NodeRef
  // nodes_iterator/begin/end - Allow iteration over all nodes in the
  // callgraph
  using nodes_iterator = typename GraphType::iterator;
  static nodes_iterator nodes_begin(GraphType &G) { return G.begin(); }
  static nodes_iterator nodes_end(GraphType &G) { return G.end(); }

  // typedef EdgeRef           - Type of Edge token in the callgraph, which
  // should
  //                             be cheap to copy.
  using EdgeRef = std::pair<EdgeType, NodeType *>;
  // typedef ChildEdgeIteratorType - Type used to iterate over children edges
  // in
  //                             callgraph, dereference to a EdgeRef.
  using ChildEdgeIteratorType = typename NodeType::edge_iterator;

  // Return iterators that point to the beginning and ending of the
  // edge list for the given callgraph node.
  static ChildEdgeIteratorType child_edge_begin(NodeRef node) {
    return node->edge_begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef node) {
    return node->edge_end();
  }

  // Return the destination node of an edge.
  static NodeRef edge_dest(EdgeRef edge) { return edge.second; }

  static unsigned size(GraphType *G) { return G->getNodeNum(); }
};

// inverse iterate the
template <typename NodeType, typename EdgeType>
struct GraphTraits<Inverse<const aser::GraphBase<NodeType, EdgeType>>> {
  using GraphType = const typename aser::GraphBase<NodeType, EdgeType>;
  // Elements to provide:
  // typedef NodeRef           - Type of Node token in the callgraph, which
  // should
  //                             be cheap to copy.
  using NodeRef = const NodeType *;
  // typedef ChildIteratorType - Type used to iterate over children in
  // callgraph,
  //                             dereference to a NodeRef.
  using ChildIteratorType = typename NodeType::const_iterator;

  static NodeRef getEntryNode(GraphType *graph) { return *(graph->begin()); }

  // Return iterators that point to the beginning and ending of the child
  // node list for the specified node.
  static ChildIteratorType child_begin(NodeRef node) {
    return node->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->pred_end(); }

  //- dereference to a NodeRef
  // nodes_iterator/begin/end - Allow iteration over all nodes in the
  // callgraph
  using nodes_iterator = typename GraphType::const_iterator;
  static nodes_iterator nodes_begin(GraphType &G) { return G.begin(); }
  static nodes_iterator nodes_end(GraphType &G) { return G.end(); }

  // typedef EdgeRef           - Type of Edge token in the callgraph, which
  // should
  //                             be cheap to copy.
  using EdgeRef = std::pair<EdgeType, NodeType *>;
  // typedef ChildEdgeIteratorType - Type used to iterate over children edges
  // in
  //                             callgraph, dereference to a EdgeRef.
  using ChildEdgeIteratorType = typename NodeType::const_edge_iterator;

  // Return iterators that point to the beginning and ending of the
  // edge list for the given callgraph node.
  static ChildEdgeIteratorType child_edge_begin(NodeRef node) {
    return node->pred_edge_begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef node) {
    return node->pred_edge_end();
  }

  // Return the destination node of an edge.
  static NodeRef edge_dest(EdgeRef edge) { return edge.second; }

  static unsigned size(GraphType *G) { return G->getNodeNum(); }
};

// inverse iterate the
template <typename NodeType, typename EdgeType>
struct GraphTraits<Inverse<aser::GraphBase<NodeType, EdgeType>>> {
  using GraphType = typename aser::GraphBase<NodeType, EdgeType>;
  // Elements to provide:
  // typedef NodeRef           - Type of Node token in the callgraph, which
  // should
  //                             be cheap to copy.
  using NodeRef = NodeType *;
  // typedef ChildIteratorType - Type used to iterate over children in
  // callgraph,
  //                             dereference to a NodeRef.
  using ChildIteratorType = typename NodeType::iterator;

  static NodeRef getEntryNode(GraphType *graph) { return *(graph->begin()); }

  // Return iterators that point to the beginning and ending of the child
  // node list for the specified node.
  static ChildIteratorType child_begin(NodeRef node) {
    return node->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->pred_end(); }

  //- dereference to a NodeRef
  // nodes_iterator/begin/end - Allow iteration over all nodes in the
  // callgraph
  using nodes_iterator = typename GraphType::iterator;
  static nodes_iterator nodes_begin(GraphType &G) { return G.begin(); }
  static nodes_iterator nodes_end(GraphType &G) { return G.end(); }

  // typedef EdgeRef           - Type of Edge token in the callgraph, which
  // should
  //                             be cheap to copy.
  using EdgeRef = std::pair<EdgeType, NodeType *>;
  // typedef ChildEdgeIteratorType - Type used to iterate over children edges
  // in
  //                             callgraph, dereference to a EdgeRef.
  using ChildEdgeIteratorType = typename NodeType::edge_iterator;

  // Return iterators that point to the beginning and ending of the
  // edge list for the given callgraph node.
  static ChildEdgeIteratorType child_edge_begin(NodeRef node) {
    return node->pred_edge_begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef node) {
    return node->pred_edge_end();
  }

  // Return the destination node of an edge.
  static NodeRef edge_dest(EdgeRef edge) { return edge.second; }

  static unsigned size(GraphType *G) { return G->getNodeNum(); }
};

} // namespace llvm

#endif