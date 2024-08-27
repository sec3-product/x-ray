#pragma once

#include "PTSTrait.h"
#include "PointerAnalysis/Util/Util.h"

// the pts data structure that also stores pointed by information

namespace xray {

// this can be slow as the pointedBy relation can not be
// updated in constant time.
class PointedByPts {
  using TargetID = NodeID;
  using PtsTy = llvm::SparseBitVector<>;
  using iterator = PtsTy::iterator;

  // points to set
  static std::vector<PtsTy> pointsTo;
  // pointed by set
  static std::vector<PtsTy> pointedBy;

  static uint32_t PTS_SIZE_LIMIT;

  static void clearAll() {
    pointsTo.clear();
    pointedBy.clear();
  }

  static inline void setPTSSizeLimit(uint32_t limit) { PTS_SIZE_LIMIT = limit; }

  static inline void onNewNodeCreation(NodeID id) {
    assert(id == pointsTo.size());
    assert(pointsTo.size() == pointedBy.size());

    pointsTo.emplace_back();
    pointedBy.emplace_back();

    assert(pointsTo.size() == id + 1 && pointedBy.size() == id + 1);
  }

  // union the pts of the nodes
  static inline bool unionWith(NodeID src, NodeID dst) {
    assert(src < pointsTo.size() && dst < pointsTo.size());
    // update the pointed by relation first
    for (NodeID id : pointsTo[dst]) {
      // must be pointed by dst already
      assert(pointedBy[id].test(dst));
      // now can also be pointed by src
      pointedBy[id].set(src);
    }
    return pointsTo[src] |= pointsTo[dst];
  }

  // whether the two pts intersect
  [[nodiscard]] static inline bool intersectWith(NodeID src, NodeID dst) {
    assert(src < pointsTo.size() && dst < pointsTo.size());
    return pointsTo[src].intersects(pointsTo[dst]);
  }

  [[nodiscard]] static inline bool intersectWithNoSpecialNode(NodeID src,
                                                              NodeID dst) {
    assert(src < pointsTo.size() && dst < pointsTo.size());
    auto result = pointsTo[src] & pointsTo[dst];

    for (int i = 0; i < NORMAL_NODE_START_ID; i++) {
      // remove special node
      result.reset(i);
    }

    return !result.empty();
  }

  // insert a node into the pts
  static inline bool insert(NodeID src, TargetID idx) {
    assert(src < pointsTo.size() && idx < pointsTo.size());
    // idx now can be pointed by src
    pointedBy[idx].set(src);
    return pointsTo[src].test_and_set(idx);
  }

  [[nodiscard]] static inline bool equal(NodeID src, NodeID dst) {
    assert(src < pointsTo.size() && dst < pointsTo.size());
    return pointsTo[src] == pointsTo[dst];
  }

  // Return true if this has idx as an element
  [[nodiscard]] static inline bool has(NodeID src, TargetID idx) {
    assert(src < pointsTo.size() && idx < pointsTo.size());
    return pointsTo[src].test(idx);
  }

  // Return true if *this is a superset of other
  [[nodiscard]] static inline bool contains(NodeID src, NodeID dst) {
    assert(src < pointsTo.size() && dst < pointsTo.size());
    return pointsTo[src].contains(pointsTo[dst]);
  }

  [[nodiscard]] static inline bool isEmpty(NodeID id) {
    assert(id < pointsTo.size());
    return pointsTo[id].empty();
  }

  [[nodiscard]] static inline iterator begin(NodeID id) {
    assert(id < pointsTo.size());
    return pointsTo[id].begin();
  }

  [[nodiscard]] static inline iterator end(NodeID id) {
    assert(id < pointsTo.size());
    return pointsTo[id].end();
  }

  static inline void clear(NodeID id) {
    assert(id < pointsTo.size());
    pointsTo[id].clear();
  }

  [[nodiscard]] static inline const PtsTy &getPointedBy(NodeID id) {
    assert(id < pointsTo.size());
    return pointedBy[id];
  }

  [[nodiscard]] static inline const PtsTy &getPointsTo(NodeID id) {
    assert(id < pointsTo.size());
    return pointsTo[id];
  }

  [[nodiscard]] static inline size_t count(NodeID id) {
    assert(id < pointsTo.size());
    return pointsTo[id].count();
  }

  static inline constexpr bool supportPointedBy() { return true; }

  friend class PTSTrait<PointedByPts>;
};

} // namespace xray

DEFINE_PTS_TRAIT(xray::PointedByPts)
