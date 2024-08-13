//
// Created by peiming on 11/1/19.
//
#ifndef ASER_PTA_CGNODEBASE_H
#define ASER_PTA_CGNODEBASE_H
#include <llvm/ADT/SparseBitVector.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Local.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "aser/PointerAnalysis/Graph/GraphBase/GraphBase.h"
namespace std {

template <>
struct iterator_traits<typename llvm::SparseBitVector<64>::iterator> {
  typedef forward_iterator_tag iterator_category;
  typedef aser::NodeID value_type;
  typedef ptrdiff_t difference_type;
  typedef const aser::NodeID *pointer;
  typedef const aser::NodeID &reference;
};

} // namespace std

using namespace llvm;

namespace aser {

template <typename ctx> class CallGraphNode;

template <typename ctx> class ConstraintGraph;

template <typename Pts> struct PTSTrait;

template <typename Model> struct LangModelTrait;

// must start from 0 and increase by one!
enum class Constraints : std::uint8_t {
  load = 0,
  store = 1,
  copy = 2,
  addr_of = 3,
  offset = 4,
  special = 5, // user-extended constraints
};

enum class CGNodeKind : uint8_t {
  PtrNode = 0,
  ObjNode = 1,
  SuperNode = 2,
};

template <typename ctx> class CGNodeBase {
protected:
  using Self = CGNodeBase<ctx>;
  using super = NodeBase<Constraints, CGNodeBase>;
  using GraphTy = GraphBase<Self, Constraints>;

  const CGNodeKind type;
  const NodeID id;
  GraphTy *graph;

  Self *superNode; // the super node
  llvm::SparseBitVector<> childNodes;

  bool isImmutable;
#ifdef USE_NODE_ID_FOR_CONSTRAINTS
  // maybe use ID for the constraints
  using SetTy = llvm::SparseBitVector<64>;
#else
  using SetTy = llvm::DenseSet<Self *>;
#endif
  SetTy succCons[6]; // successor constraints
  SetTy predCons[6]; // predecessor constraints

  inline void setGraph(GraphTy *g) { this->graph = g; }

  // the same function ptr might used multiple times
  // call void %2418(i8* %2910, i8* nonnull %2453, i8* nonnull %2451) #10
  // call void %2418(i8* nonnull %2454, i8* nonnull %2453, i8* nonnull %2451)
  using IndirectNodeSet = std::set<CallGraphNode<ctx> *>;
  IndirectNodeSet indirectNodes;

  inline CGNodeBase(NodeID id, CGNodeKind type)
      : id(id), type(type), superNode(nullptr), childNodes{}, indirectNodes{},
        isImmutable(false) {}

private:
  inline bool insertConstraint(Self *node, Constraints edgeKind) {
    // this --edge-> node
    if (edgeKind != Constraints::special && node->isImmutable) {
      // this --load--> node
      // this --store--> node
      // this --copy--> node
      // this --offset--> node
      // all these constraints will update the points-to set of an immutable
      // node
      return false;
    }
    auto src = this->getSuperNode();
    auto dst = node->getSuperNode();

    // assert(!this->hasSuperNode());
    if (src == dst && edgeKind == Constraints::copy) {
      // self-copy does not have any effect.
      return false;
    }

    auto index = static_cast<std::underlying_type<Constraints>::type>(edgeKind);
    assert(index < 6); // only 6 kinds of constraints

#ifdef USE_NODE_ID_FOR_CONSTRAINTS
    // successors
    bool r1 = src->succCons[index].test_and_set(dst->getNodeID()); //.second;
    // predecessor
    bool r2 = dst->predCons[index].test_and_set(src->getNodeID()); //.second;
#else
    // successors
    bool r1 = src->succCons[index].insert(dst).second;
    // predecessor
    // TODO: remove useless constraints
    bool r2 = dst->predCons[index].insert(src).second;
#endif
    assert(r1 == r2);
    return r1;
  }

public:
  // can not be moved and copied
  CGNodeBase(const CGNodeBase<ctx> &) = delete;
  CGNodeBase(CGNodeBase<ctx> &&) = delete;
  CGNodeBase<ctx> &operator=(const CGNodeBase<ctx> &) = delete;
  CGNodeBase<ctx> &operator=(CGNodeBase<ctx> &&) = delete;

  [[nodiscard]] inline bool isSpecialNode() const {
    return this->getNodeID() < NORMAL_NODE_START_ID;
  }

  [[nodiscard]] inline bool isNullObj() const {
    return this->getNodeID() == NULL_OBJ;
  }

  [[nodiscard]] inline bool isUniObj() const {
    return this->getNodeID() == UNI_OBJ;
  }

  [[nodiscard]] inline bool isNullPtr() const {
    return this->getNodeID() == NULL_PTR;
  }

  [[nodiscard]] inline bool isUniPtr() const {
    return this->getNodeID() == UNI_PTR;
  }

  // after setting the flag, no edges shall be added into the node
  inline void setImmutable() { this->isImmutable = true; }

  inline bool isSuperNode() const { return !childNodes.empty(); }

  inline void setSuperNode(Self *node) { this->superNode = node; }

  inline Self *getSuperNode() {
    Self *node = this;
    while (node->superNode != nullptr) {
      node = node->superNode;
    }
    return node;
  }

  // remove all the edges
  inline void clearConstraints() {
#ifdef USE_NODE_ID_FOR_CONSTRAINTS
#define CLEAR_CONSTRAINT(TYPE)                                                 \
  {                                                                            \
    constexpr auto index =                                                     \
        static_cast<std::underlying_type<Constraints>::type>(                  \
            Constraints::TYPE);                                                \
    for (auto it = this->succ_##TYPE##_begin(),                                \
              ie = this->succ_##TYPE##_end();                                  \
         it != ie; it++) {                                                     \
      Self *target = *it;                                                      \
      /*auto iter =*/target->predCons[index].reset(this->getNodeID());         \
      /*assert(iter != target->predCons[index].end());*/                       \
      /*target->predCons[index].erase(iter);*/                                 \
    }                                                                          \
    for (auto it = this->pred_##TYPE##_begin(),                                \
              ie = this->pred_##TYPE##_end();                                  \
         it != ie; it++) {                                                     \
      Self *target = *it;                                                      \
      /*auto iter = */ target->succCons[index].reset(this->getNodeID());       \
      /*assert(iter != target->succCons[index].end());*/                       \
      /*target->succCons[index].erase(iter);*/                                 \
    }                                                                          \
    this->succCons[index].clear();                                             \
    this->predCons[index].clear();                                             \
  }
#else
#define CLEAR_CONSTRAINT(TYPE)                                                 \
  {                                                                            \
    constexpr auto index =                                                     \
        static_cast<std::underlying_type<Constraints>::type>(                  \
            Constraints::TYPE);                                                \
    for (auto it = this->succ_##TYPE##_begin(),                                \
              ie = this->succ_##TYPE##_end();                                  \
         it != ie; it++) {                                                     \
      Self *target = *it;                                                      \
      auto iter = target->predCons[index].find(this);                          \
      assert(iter != target->predCons[index].end());                           \
      target->predCons[index].erase(iter);                                     \
    }                                                                          \
    for (auto it = this->pred_##TYPE##_begin(),                                \
              ie = this->pred_##TYPE##_end();                                  \
         it != ie; it++) {                                                     \
      Self *target = *it;                                                      \
      auto iter = target->succCons[index].find(this);                          \
      assert(iter != target->succCons[index].end());                           \
      target->succCons[index].erase(iter);                                     \
    }                                                                          \
    this->succCons[index].clear();                                             \
    this->predCons[index].clear();                                             \
  }
#endif
    CLEAR_CONSTRAINT(load)
    CLEAR_CONSTRAINT(store)
    CLEAR_CONSTRAINT(copy)
    CLEAR_CONSTRAINT(addr_of)
    CLEAR_CONSTRAINT(offset)
    CLEAR_CONSTRAINT(special)

#undef CLEAR_CONSTRAINT
  }

  [[nodiscard]] inline CGNodeKind getType() const { return type; }

  [[nodiscard]] inline bool hasSuperNode() const {
    return superNode != nullptr;
  }

  inline void setIndirectCallNode(CallGraphNode<ctx> *callNode) {
    // assert(callNode->isIndirectCall() && this->indirectNode == nullptr);
    this->indirectNodes.insert(callNode);
    // this->indirectNode = callNode;
  }

  inline const IndirectNodeSet &getIndirectNodes() const {
    return indirectNodes;
  }

  inline auto indirect_begin() const -> decltype(this->indirectNodes.begin()) {
    return this->indirectNodes.begin();
  }

  inline auto indirect_end() const -> decltype(this->indirectNodes.end()) {
    return this->indirectNodes.end();
  }

  [[nodiscard]] inline bool isFunctionPtr() {
    return !this->indirectNodes.empty();
  }

  [[nodiscard]] inline ConstraintGraph<ctx> *getGraph() {
    return static_cast<ConstraintGraph<ctx> *>(this->graph);
  }

  [[nodiscard]] inline NodeID getNodeID() const { return id; }

  [[nodiscard]] virtual std::string toString() const = 0;
  virtual ~CGNodeBase() = default;

#ifdef USE_NODE_ID_FOR_CONSTRAINTS
  using cg_iterator = NodeIDWrapperIterator<typename SetTy::iterator, GraphTy>;

#define __CONS_ITER__(DIRECTION, KIND, TYPE)                                   \
  [[nodiscard]] inline cg_iterator DIRECTION##_##KIND##_##TYPE() {             \
    constexpr auto index =                                                     \
        static_cast<std::underlying_type<Constraints>::type>(                  \
            Constraints::KIND);                                                \
    static_assert(index < 6, "");                                              \
    return cg_iterator(this->graph, DIRECTION##Cons[index].TYPE());            \
  }
#else
  using cg_iterator = typename SetTy::iterator;

#define __CONS_ITER__(DIRECTION, KIND, TYPE)                                   \
  [[nodiscard]] inline cg_iterator DIRECTION##_##KIND##_##TYPE() {             \
    constexpr auto index =                                                     \
        static_cast<std::underlying_type<Constraints>::type>(                  \
            Constraints::KIND);                                                \
    static_assert(index < 5, "");                                              \
    return DIRECTION##Cons[index].TYPE();                                      \
  }
#endif

#define __BI_CONS_ITER__(KIND, TYPE)                                           \
  __CONS_ITER__(succ, KIND, TYPE)                                              \
  __CONS_ITER__(pred, KIND, TYPE)

#define DEFINE_CONS_ITER(KIND)                                                 \
  __BI_CONS_ITER__(KIND, begin)                                                \
  __BI_CONS_ITER__(KIND, end)

  // succ_load_begin, succ_load_end, pred_load_begin, pred_load_end
  DEFINE_CONS_ITER(load)
  DEFINE_CONS_ITER(store)
  DEFINE_CONS_ITER(copy)
  DEFINE_CONS_ITER(addr_of)
  DEFINE_CONS_ITER(offset)
  DEFINE_CONS_ITER(special)

#undef DEFINE_CONS_ITER
#undef __BI_CONS_ITER__
#undef __CONS_ITER__

#ifdef USE_NODE_ID_FOR_CONSTRAINTS
  // TODO: use LLVM built-in concat interator, they have better implementation
  using id_iterator = ConcatIterator<typename SetTy::iterator, 6, NodeID>;
  using const_id_iterator = ConcatIterator<typename SetTy::iterator, 6, NodeID>;

  using id_edge_iterator =
      ConcatIteratorWithTag<typename SetTy::iterator, 6, Constraints, NodeID>;
  using const_id_edge_iterator =
      ConcatIteratorWithTag<typename SetTy::iterator, 6, Constraints, NodeID>;
#else
  using id_iterator = ConcatIterator<typename SetTy::iterator, 5>;
  using const_id_iterator = ConcatIterator<typename SetTy::const_iterator, 5>;

  using id_edge_iterator =
      ConcatIteratorWithTag<typename SetTy::iterator, 5, Constraints>;
  using const_id_edge_iterator =
      ConcatIteratorWithTag<typename SetTy::const_iterator, 5, Constraints>;
#endif

#ifdef USE_NODE_ID_FOR_CONSTRAINTS
  using iterator = NodeIDWrapperIterator<id_iterator, GraphTy>;
  using const_iterator =
      NodeIDWrapperIterator<const_id_iterator, const GraphTy>;
  using edge_iterator = NodeIDWrapperEdgeIterator<id_edge_iterator, GraphTy>;
  using const_edge_iterator =
      NodeIDWrapperEdgeIterator<const_id_edge_iterator, const GraphTy>;
#else
  using iterator = id_iterator;
  using const_iterator = const_id_iterator;
  using edge_iterator = id_edge_iterator;
  using const_edge_iterator = const_id_edge_iterator;
#endif

#define INIT_ITERATOR(CONTAINER, BEGIN, END)                                   \
  (CONTAINER[5].BEGIN(), CONTAINER[5].END(), CONTAINER[4].BEGIN(),             \
   CONTAINER[4].END(), CONTAINER[3].BEGIN(), CONTAINER[3].END(),               \
   CONTAINER[2].BEGIN(), CONTAINER[2].END(), CONTAINER[1].BEGIN(),             \
   CONTAINER[1].END(), CONTAINER[0].BEGIN(), CONTAINER[0].END())

#ifdef USE_NODE_ID_FOR_CONSTRAINTS
#define NODE_ITERATOR(CONTAINER, BEGIN, END)                                   \
  iterator(this->graph, id_iterator INIT_ITERATOR(CONTAINER, BEGIN, END))

#define CONST_NODE_ITERATOR(CONTAINER, BEGIN, END)                             \
  const_iterator(this->graph,                                                  \
                 const_id_iterator INIT_ITERATOR(CONTAINER, BEGIN, END))

#define EDGE_ITERATOR(CONTAINER, BEGIN, END)                                   \
  edge_iterator(this->graph,                                                   \
                id_edge_iterator INIT_ITERATOR(CONTAINER, BEGIN, END))

#define CONST_EDGE_ITERATOR(CONTAINER, BEGIN, END)                             \
  const_edge_iterator(this->graph, const_id_edge_iterator INIT_ITERATOR(       \
                                       CONTAINER, BEGIN, END))

#else
#define NODE_ITERATOR(CONTAINER, BEGIN, END)                                   \
  id_iterator INIT_ITERATOR(CONTAINER, BEGIN, END)

#define CONST_NODE_ITERATOR(CONTAINER, BEGIN, END)                             \
  const_id_iterator INIT_ITERATOR(CONTAINER, BEGIN, END)

#define EDGE_ITERATOR(CONTAINER, BEGIN, END)                                   \
  id_edge_iterator INIT_ITERATOR(CONTAINER, BEGIN, END)

#define CONST_EDGE_ITERATOR(CONTAINER, BEGIN, END)                             \
  const_id_edge_iterator INIT_ITERATOR(CONTAINER, BEGIN, END)

#endif

  inline iterator succ_begin() { return NODE_ITERATOR(succCons, begin, end); }
  inline iterator succ_end() { return NODE_ITERATOR(succCons, end, end); }
  inline const_iterator succ_begin() const {
    return CONST_NODE_ITERATOR(succCons, begin, end);
  }
  inline const_iterator succ_end() const {
    return CONST_NODE_ITERATOR(succCons, end, end);
  }

  inline iterator pred_begin() { return NODE_ITERATOR(predCons, begin, end); }
  inline iterator pred_end() { return NODE_ITERATOR(predCons, end, end); }
  inline const_iterator pred_begin() const {
    return CONST_NODE_ITERATOR(predCons, begin, end);
  }
  inline const_iterator pred_end() const {
    return CONST_NODE_ITERATOR(predCons, end, end);
  }

  inline edge_iterator succ_edge_begin() {
    return EDGE_ITERATOR(succCons, begin, end);
  }
  inline edge_iterator succ_edge_end() {
    return EDGE_ITERATOR(succCons, end, end);
  }
  inline const_edge_iterator succ_edge_begin() const {
    return CONST_EDGE_ITERATOR(succCons, begin, end);
  }
  inline const_edge_iterator succ_edge_end() const {
    return CONST_EDGE_ITERATOR(succCons, end, end);
  }

  inline edge_iterator pred_edge_begin() {
    return EDGE_ITERATOR(predCons, begin, end);
  }
  inline edge_iterator pred_edge_end() {
    return EDGE_ITERATOR(predCons, end, end);
  }
  inline const_edge_iterator pred_edge_begin() const {
    return CONST_EDGE_ITERATOR(predCons, begin, end);
  }
  inline const_edge_iterator pred_edge_end() const {
    return CONST_EDGE_ITERATOR(predCons, end, end);
  }

  // needed by GraphTrait
  inline edge_iterator edge_begin() { return succ_edge_begin(); }
  inline edge_iterator edge_end() { return succ_edge_end(); }
  inline const_edge_iterator edge_begin() const { return succ_edge_begin(); }
  inline const_edge_iterator edge_end() const { return succ_edge_end(); }

#undef INIT_ITERATOR

  friend class GraphBase<Self, Constraints>;
  friend class ConstraintGraph<ctx>;
};

// `above` --- number of lines above the target line
// `below` --- number of lines below the target line
static std::string getPTACodeSnippet(std::string directory,
                                     std::string filename, unsigned line,
                                     unsigned col, unsigned above,
                                     unsigned below) {
  assert(above >= 0 && below >= 0);
  // source code file path
  std::string absPath = directory + "/" + filename;
  // counter for how many lines we have to go through from line 1
  unsigned rest = line + below;
  // a simple ring-buffer to store before the target line code
  std::vector<std::string> buf(above);
  unsigned idx = 0;

  std::stringstream raw;
  std::ifstream input(absPath);

  for (std::string ln; std::getline(input, ln);) {
    --rest;
    // either hit or had already hit the target line
    // need to construct the output
    if (rest < below + 1) {
      // just hit the target source line
      // append the previous lines to the output
      if (rest == below) {
        // `n` is the number of lines we need to append before target line
        // e.g. if target line is 5, the span is 3
        // then we need to append 3 lines (line 2,3,4)
        // if the target line is 5, the span is 7
        // then we append 4 lines (that's all we have)
        unsigned n = line > above ? above : (line - 1);
        for (unsigned i = 0; i < n; i++) {
          raw << " " << line - above + i << "|"
              << buf[(idx + above - i) % above] << "\n";
        }
        raw << ">";
      }
      // had already hit the target line
      else {
        raw << " ";
      }
      raw << (line + below - rest) << "|" << ln << "\n";
      if (rest < 0) {
        break;
      }
    }
    // haven't hit the target line
    // store the current line into ring-buffer
    if (above > 0) {
      buf[idx] = ln;
      idx = (idx + 1) % above;
    }
  }
  return raw.str();
}

// JEFF: for test debug
static std::string getPTASourceLocSnippet(const Value *val) {
  if (val == NULL)
    return "null";

  unsigned line = 0;
  unsigned col = 0;
  std::string filename;
  std::string directory;
  // variable name
  // for now this will only be set for AllocaInst
  // so that we can know the shared variable name (if it's a stack variable)
  std::string name;

  if (const Instruction *inst = llvm::dyn_cast<Instruction>(val)) {
    if (MDNode *N = inst->getMetadata("dbg")) {
      llvm::DILocation *loc = llvm::cast<llvm::DILocation>(N);
      line = loc->getLine();
      // this col number is inaccurate
      col = loc->getColumn();
      filename = loc->getFilename().str();
      directory = loc->getDirectory().str();
      // get the accurate col number
      // col = getAccurateCol(directory, filename, line, col,
      // isa<StoreInst>(inst));
    } else if (isa<AllocaInst>(inst)) {
      // TODO: there must be other insts than AllocaInst
      // that can be a shared variable
      for (DbgInfoIntrinsic *DII :
           FindDbgAddrUses(const_cast<Instruction *>(inst))) {
        if (llvm::DbgDeclareInst *DDI =
                llvm::dyn_cast<llvm::DbgDeclareInst>(DII)) {
          llvm::DIVariable *DIVar =
              llvm::cast<llvm::DIVariable>(DDI->getVariable());
          line = DIVar->getLine();
          col = 0;
          filename = DIVar->getFilename().str();
          directory = DIVar->getDirectory().str();
          name = DIVar->getName().str();
          break;
        }
      }
    }
  } else if (const GlobalVariable *gvar =
                 llvm::dyn_cast<llvm::GlobalVariable>(val)) {
    // find the debuggin information for global variables
    llvm::NamedMDNode *CU_Nodes =
        gvar->getParent()->getNamedMetadata("llvm.dbg.cu");
    if (CU_Nodes) {
      // iterate over all the !DICompileUnit
      // each of whom has a field called globals to indicate all the debugging
      // info for global variables
      for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
        llvm::DICompileUnit *CUNode =
            llvm::cast<llvm::DICompileUnit>(CU_Nodes->getOperand(i));
        for (llvm::DIGlobalVariableExpression *GV :
             CUNode->getGlobalVariables()) {
          llvm::DIGlobalVariable *DGV = GV->getVariable();
          // DGV->getLinkageName() is the mangled name
          if (DGV->getName() == gvar->getName() ||
              DGV->getLinkageName() == gvar->getName()) {
            line = DGV->getLine();
            col = 0;
            filename = DGV->getFilename().str();
            directory = DGV->getDirectory().str();
            name = DGV->getName() == gvar->getName()
                       ? DGV->getName().str()
                       : DGV->getLinkageName().str();
            break;
          }
        }
      }
    }
  }

  llvm::outs() << "name: " << name << " dir: " << directory
               << " file: " << filename << " line: " << line << "\n";
  auto snippet = getPTACodeSnippet(directory, filename, line, 0, 2, 2);

  return snippet;
  // return SourceInfo(val, line, col, filename, directory, name);
}

} // namespace aser

#endif
