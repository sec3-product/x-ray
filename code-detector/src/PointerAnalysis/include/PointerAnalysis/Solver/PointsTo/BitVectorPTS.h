//
// Created by peiming on 9/9/19.
//
#ifndef ASER_PTA_BITVECTORPTS_H
#define ASER_PTA_BITVECTORPTS_H

#include <llvm/ADT/SparseBitVector.h>

#include <vector>

#include "PointerAnalysis/Solver/PointsTo/PTSTrait.h"

namespace aser {

class BitVectorPTS {
private:
  using TargetID = NodeID;
  // using PtsTy = llvm::SparseBitVector<5012>;
  using PtsTy = llvm::SparseBitVector<>;
  using iterator = PtsTy::iterator;

  static std::vector<PtsTy> ptsVec;

  static uint32_t PTS_SIZE_LIMIT;
  // ptsVec[20] ==> SparseBitVector ==> "010000..."

  static inline void onNewNodeCreation(NodeID id) {
    // should be the same value
    // int ** ptr = (int **) malloc(sizeof(int *)); // o1
    // *ptr = &o2; // ptr
    assert(id == ptsVec.size());
    ptsVec.emplace_back();
    assert(ptsVec.size() == (id + 1));
  }

  static inline void clearAll() { ptsVec.clear(); }

  static inline void setPTSSizeLimit(uint32_t limit) { PTS_SIZE_LIMIT = limit; }

  // get the pts of the corresponding node
  [[nodiscard]] static inline const PtsTy &getPointsTo(NodeID id) {
    assert(id < ptsVec.size());
    return ptsVec[id];
  }

  // union the pts of the nodes
  static inline bool unionWith(NodeID src, NodeID dst) {
    assert(src < ptsVec.size() && dst < ptsVec.size());
    // JEFF: limit to 999
    if (PTS_SIZE_LIMIT != std::numeric_limits<uint32_t>::max()) {
      if (ptsVec[src].count() > PTS_SIZE_LIMIT) { // count() is expensive
        return false;
      }
    }

    bool r = ptsVec[src] |= ptsVec[dst];
    assert(ptsVec[src].find_last() < 0
               ? true
               : ptsVec[src].find_last() < ptsVec.size());
    return r;
  }

  // whether the two pts intersect
  [[nodiscard]] static inline bool intersectWith(NodeID src, NodeID dst) {
    assert(src < ptsVec.size() && dst < ptsVec.size());
    return ptsVec[src].intersects(ptsVec[dst]);
  }

  [[nodiscard]] static inline bool intersectWithNoSpecialNode(NodeID src,
                                                              NodeID dst) {
    assert(src < ptsVec.size() && dst < ptsVec.size());
    auto result = ptsVec[src] & ptsVec[dst];

    for (int i = 0; i < NORMAL_OBJ_START_ID; i++) {
      // remove special node
      result.reset(i);
    }

    return !result.empty();
  }

  // insert a node into the pts
  static inline bool insert(NodeID src, TargetID idx) {
    assert(src < ptsVec.size() && idx < ptsVec.size());
    // JEFF: limit to 999
    if (PTS_SIZE_LIMIT != std::numeric_limits<uint32_t>::max()) {
      if (ptsVec[src].count() > PTS_SIZE_LIMIT) {
        return false;
      }
    }

    // JEFF TODO: check if they have the same type?
    return ptsVec[src].test_and_set(idx);
  }

  // Return true if this has idx as an element
  [[nodiscard]] static inline bool has(NodeID src, TargetID idx) {
    assert(src < ptsVec.size() && idx < ptsVec.size());
    return ptsVec[src].test(idx);
  }

  [[nodiscard]] static inline bool equal(NodeID src, NodeID dst) {
    assert(src < ptsVec.size() && dst < ptsVec.size());
    return ptsVec[src] == ptsVec[dst];
  }

  // Return true if *this is a superset of other
  [[nodiscard]] static inline bool contains(NodeID src, NodeID dst) {
    assert(src < ptsVec.size() && dst < ptsVec.size());
    return ptsVec[src].contains(ptsVec[dst]);
  }

  [[nodiscard]] static inline bool isEmpty(NodeID id) {
    assert(id < ptsVec.size());
    return ptsVec[id].empty();
  }

  [[nodiscard]] static inline iterator begin(NodeID id) {
    assert(id < ptsVec.size());
    assert(*ptsVec[id].begin() < ptsVec.size());
    return ptsVec[id].begin();
  }

  [[nodiscard]] static inline iterator end(NodeID id) {
    assert(id < ptsVec.size());
    return ptsVec[id].end();
  }

  static inline void clear(NodeID id) {
    assert(id < ptsVec.size());
    ptsVec[id].clear();
  }

  static inline size_t count(NodeID id) {
    assert(id < ptsVec.size());
    return ptsVec[id].count();
  }

  static inline const PtsTy &getPointedBy(NodeID id) {
    llvm_unreachable(
        "not supported by BitVectorPTS, use PointedByPts instead ");
  }

  // TODO: simply traverse the whole points-to information to gather the
  // pointed by information
  static inline constexpr bool supportPointedBy() { return false; }

  friend class PTSTrait<BitVectorPTS>;
};

} // namespace aser

DEFINE_PTS_TRAIT(aser::BitVectorPTS)

#endif
