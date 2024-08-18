// Created by Jeff 12/17/2019
#ifndef RD_REACHABILITYENGINE_H
#define RD_REACHABILITYENGINE_H

#include <llvm/ADT/Hashing.h>

#include <map>
#include <queue>
#include <set>

extern bool DEBUG_HB;

namespace aser {

class ReachabilityEngine {
private:
  // this is the internal identity of a node
  using ID = uint64_t;
  std::map<std::string, ID> idMap;
  // for each unreachable node pair, we first calculate their ID pair hash
  // (using hashIDPair)
  // and cache them in this set
  std::set<llvm::hash_code> unreachable;
  // map between a node to a set of nodes it connects to
  std::map<ID, std::set<ID>> edgeSetMap;

  // internal ID in the reachability engine
  ID getId(std::string sid) {
    ID id;
#pragma omp critical(idMap)
    {
      if (idMap.find(sid) == idMap.end()) {
        id = idMap.size() + 1;
        idMap[sid] = id;
      } else {
        id = idMap[sid];
      }
    }
    return id;
  }

  llvm::hash_code hashIDPair(ID id1, ID id2) {
    return llvm::hash_combine(llvm::hash_value(id1), llvm::hash_value(id2));
  }

  bool hasEdge(ID id1, ID id2) {
    if (edgeSetMap.find(id1) == edgeSetMap.end()) {
      return false;
    } else {
      std::set<ID> &s = edgeSetMap[id1];
      return (s.find(id2) != s.end());
    }
  }

public:
  ReachabilityEngine() = default;

  void addEdge(std::string sid1, std::string sid2) {
    // LOG_DEBUG("ReachEngine add edge from={}, to={}", sid1, sid2);
    if (DEBUG_HB)
      llvm::outs() << "ReachEngine add edge: " << sid1 << " -> " << sid2
                   << "\n";
    ID id1 = getId(sid1);
    ID id2 = getId(sid2);
    edgeSetMap[id1].insert(id2); // no race here
  }

  inline void invalidateCachedResult() { unreachable.clear(); }

  bool canReach(std::string sid1, std::string sid2) {
    // LOG_DEBUG("ReachEngine query can reach. from={}, to={}", sid1, sid2);
    if (DEBUG_HB)
      llvm::outs() << "ReachEngine query edge: " << sid1 << " -> " << sid2
                   << "\n";

    ID id1 = getId(sid1);
    ID id2 = getId(sid2);

    // TODO: reimplement this hash function
    llvm::hash_code sig = hashIDPair(id1, id2);
    if (unreachable.find(sig) != unreachable.end()) {
      // LOG_INFO("Result: UNREACHABLE");
      if (DEBUG_HB)
        llvm::outs() << "ReachEngine query edge: " << sid1 << " -> " << sid2
                     << " false\n";
      return false;

    } else if (hasEdge(id1, id2)) {
      // LOG_INFO("Result: REACHABLE");
      if (DEBUG_HB)
        llvm::outs() << "ReachEngine query edge: " << sid1 << " -> " << sid2
                     << " true\n";
      return true;

    } else {
      // DFS
      // visited node
      llvm::SparseBitVector<> visited;
      std::stack<ID> stack;
      // tmp value, the current node we are visiting
      ID curID;

      visited.set(id1);
      stack.push(id1);
      while (!stack.empty()) {
        curID = stack.top();
        stack.pop();

#pragma omp critical(edgeSetMap)
        { // add internal edge
          edgeSetMap[id1].insert(curID);
        }

        if (curID == id2) {
          return true;
        } else {
          if (hasEdge(curID, id2)) {
#pragma omp critical(edgeSetMap)
            { edgeSetMap[id1].insert(id2); }
            return true;
          } else {
            std::set<ID> &s = edgeSetMap[curID];
            for (ID nextID : s) {
              if (visited.test_and_set(nextID) &&
                  unreachable.find(hashIDPair(nextID, id2)) ==
                      unreachable.end())
                stack.push(nextID);
            }
          }
        }
      }
#pragma omp critical(unreachable)
      { unreachable.insert(sig); }
      if (DEBUG_HB)
        llvm::outs() << "ReachEngine query edge: " << sid1 << " -> " << sid2
                     << " false\n";
      // FILE_INFO("Result: UNREACHABLE");
      return false;
    }
  }
};
} // namespace aser

#endif
