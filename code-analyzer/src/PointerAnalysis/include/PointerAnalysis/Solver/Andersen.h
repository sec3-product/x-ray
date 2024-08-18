//
// Created by peiming on 9/17/19.
//
#ifndef ASER_PTA_ANDERSEN_H
#define ASER_PTA_ANDERSEN_H

#include "SolverBase.h"

// a simple andersen solver (no scc detection as well as no topo sort)
namespace aser {

template <typename LangModel>
class Andersen : public SolverBase<LangModel, Andersen<LangModel>> {
private:
  using super = SolverBase<LangModel, Andersen<LangModel>>;
  using ctx = typename super::ctx;

  using CGNodeTy = typename super::CGNodeTy;
  using ConsGraphTy = typename super::ConsGraphTy;
  using WorkListTy = std::queue<CGNodeTy *>;

  using LMT = LangModelTrait<LangModel>;
  using GT = llvm::GraphTraits<ConsGraphTy>;
  using PT = PTSTrait<typename super::PtsTy>;

protected:
  void processNode(CGNodeTy *src, WorkListTy &workList) {
    for (auto it = GT::child_edge_begin(src); it != GT::child_edge_end(src);
         it++) {
      // iterate the edges
      Constraints edgeKind = (*it).first;
      CGNodeTy *dst = (*it).second;
      switch (edgeKind) {
      case Constraints::addr_of:
        super::processAddrOf(src, dst);
        break;
      case Constraints::copy:
        if (super::processCopy(src, dst)) {
          workList.push(dst);
        }
        break;
      case Constraints::load:
        if (super::processLoad(src, dst)) {
          // new copy edge added to dst
          // every ptr in pts(src) --COPY--> dst
          for (auto nit = PT::begin(src->getNodeID()),
                    eit = PT::end(src->getNodeID());
               nit != eit; nit++) {
            workList.push((*super::consGraph)[*nit]);
          }
        }
        break;
      case Constraints::store:
        if (super::processStore(src, dst)) {
          // new copy edges added (from src --> every ptr in
          // pts(dst))
          workList.push(src);
        }
        break;
      case Constraints::offset: {
        super::processOffset(src, dst);
      }
      }
    }
  }

  void runSolver(LangModel &model) {
    WorkListTy workList;
    auto &consGraph = *super::consGraph;

    for (auto it = GT::nodes_begin(consGraph); it != GT::nodes_end(consGraph);
         it++) {
      workList.push(*it);
    }

    while (!workList.empty()) {
      CGNodeTy *curNode = workList.front();
      workList.pop();

      this->processNode(curNode, workList);
    }
  }

  friend super;
};

} // namespace aser

#endif