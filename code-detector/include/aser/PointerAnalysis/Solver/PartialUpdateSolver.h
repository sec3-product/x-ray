//
// Created by peiming on 2/25/20.
//
#ifndef ASER_PTA_PARTIALUPDATESOLVER_H
#define ASER_PTA_PARTIALUPDATESOLVER_H

#include <stack>

#include "SolverBase.h"
#include "aser/PointerAnalysis/Graph/ConstraintGraph/SCCIterator.h"

//#define HASH_EDGE_LIMIT 6700417 // a large enough prime number
#define HASH_EDGE_LIMIT 1000032953

namespace aser {
// just experimental feature for now.
// after resolving the indirect call, do not traverse the whole
// constraint graph again, instead, we only solve the newly resolved
template <typename LangModel>
class PartialUpdateSolver : public SolverBase<LangModel, PartialUpdateSolver<LangModel>> {
public:
    using super = SolverBase<LangModel, PartialUpdateSolver<LangModel>>;
    using Self = PartialUpdateSolver<LangModel>;
    using PT = typename super::PT;                // points to set trait
    using LMT = typename super::LMT;              // language model trait
    using MMT = typename super::MMT;              // memory model trait
    using ctx = typename super::ctx;              // context
    using ObjTy = typename super::ObjTy;          // object type used by memory model
    using PtsTy = typename super::PtsTy;          // points-to set type
    using MemModel = typename super::MemModel;    // memory model used by the PTA
    using CGNodeTy = typename super::CGNodeTy;    // constraint graph node type
    using ObjNodeTy = typename super::ObjNodeTy;  // object constraint graph node type
    using PtrNodeTy = typename super::PtrNodeTy;
    using CallNodeTy = typename super::CallNodeTy;
    using CallGraphTy = typename super::CallGraphTy;  // call graph type
    using ConsGraphTy = typename super::ConsGraphTy;  // constraint graph type

private:
    class CallBack : public ConsGraphTy::OnNewConstraintCallBack {
        size_t nodeNum;
        Self &solver;

    public:
        CallBack(Self &solver, size_t nodeNum) : solver(solver), nodeNum(nodeNum) {}

        void onNewConstraint(CGNodeTy *src, CGNodeTy *dst, Constraints constraint) override {
            switch (constraint) {
                // copy between globals / parameter passing
                case Constraints::copy: {
                    // new constraint need to be handled
                    if (src->getNodeID() < nodeNum) {
                        solver.recordCopyEdge(src, dst);
                    }
                    break;
                }
                case Constraints::offset: {
                    // offset from globals
                    if (src->getNodeID() < nodeNum) {
                        solver.processOffset(src, dst, [&](CGNodeTy *fieldObj, CGNodeTy *ptr) {
#ifdef NO_ADDR_OF_FOR_OFFSET
                            solver.consGraph->addConstraints(fieldObj, ptr, Constraints::addr_of);
#endif
                            auto addrNode = llvm::cast<ObjNodeTy>(fieldObj)->getAddrTakenNode();
                            solver.recordCopyEdge(addrNode, ptr);
                        });
                    }
                    break;
                }
                case Constraints::load: {
                    // load from global
                    if (src->getNodeID() < nodeNum) {
                        solver.processLoad(src, dst,
                                           [&](CGNodeTy *src, CGNodeTy *dst) { solver.recordCopyEdge(src, dst); });
                    }
                    break;
                }
                case Constraints::store: {
                    // store into global
                    if (dst->getNodeID() < nodeNum) {
                        solver.processStore(src, dst,
                                            [&](CGNodeTy *src, CGNodeTy *dst) { solver.recordCopyEdge(src, dst); });
                    }
                    break;
                }
                default:
                    return;
            }
        }
    };

    inline void recordCopyEdge(CGNodeTy *src, CGNodeTy *dst) {
        // src is not a newly added node
        if (src->getNodeID() < copyWorkList.size()) {
            copyWorkList.reset(src->getNodeID());
        }

        if (dst->getNodeID() < targetList.size()) {
            targetList.reset(dst->getNodeID());
        }

        // we need to handle the copy edge
        requiredEdge.set(hashEdge(src, dst));
    }

    // seems like the scc becomes the bottleneck, need to merge large scc
    void processCopySCC(const std::vector<CGNodeTy *> &scc) {
        assert(scc.size() > 1);

        CGNodeTy *superNode = scc.front();
        for (auto nit = ++(scc.begin()), nie = scc.end(); nit != nie; nit++) {
            // if any node in the scc is the target, the scc supernode is the target
            if (!targetList.test((*nit)->getNodeID())) {
                targetList.reset(superNode->getNodeID());
            }

            // merge pts in scc all into front
            super::processCopy(*nit, superNode);
            // clear the points-to set after it is merged into the super node
            PT::clear((*nit)->getNodeID());
        }

        lsWorkList.reset(superNode->getNodeID());

        // changedCopy.set(hashEdge(superNode, superNode));

        // collapse scc to the front node
        super::getConsGraph()->collapseSCCTo(scc, superNode);

        // if there is a function ptr in the scc, update the function ptr
        if (superNode->isFunctionPtr()) {
            this->updateFunPtr(superNode->getNodeID());
        }

        for (auto cit = superNode->succ_copy_begin(), cie = superNode->succ_copy_end(); cit != cie; cit++) {
            if (super::processCopy(superNode, *cit)) {
                // the copy edge changed the pts of src
                // changedCopy.set(hashEdge(superNode, *cit));

                lsWorkList.reset((*cit)->getNodeID());
            }
        }
    }

    // copy worklist
    llvm::BitVector copyWorkList;
    // load/store/offset worklist
    llvm::BitVector lsWorkList;
    // the target node id of the newly added copy edge by load/store/offset
    llvm::BitVector targetList;

    // set of the new added copy edge (identified by the hash value of src/dst)
    llvm::BitVector requiredEdge;

    // llvm::BitVector changedCopy;

public:
    PartialUpdateSolver() : requiredEdge(HASH_EDGE_LIMIT), copyWorkList(), lsWorkList(), targetList() {}

protected:
    inline size_t hashEdge(CGNodeTy *src, CGNodeTy *dst) {
        size_t hashed = llvm::hash_value(std::make_pair<void *, void *>(src, dst));
        return hashed % HASH_EDGE_LIMIT;
    }

    bool shouldProcessCopy(CGNodeTy *src, CGNodeTy *dst) {
        if (!lsWorkList.test(src->getNodeID())) {
            // if the incoming src changed then yes
            return true;
        }
        bool isDstTarget = !targetList.test(dst->getNodeID());
        bool isSrcUnhandled = !copyWorkList.test(src->getNodeID());

        if (isDstTarget && isSrcUnhandled) {
            // whether this is the edge
            return requiredEdge.test(hashEdge(src, dst));
        }

        return false;
    }

    int numOfPTAIterations = 0;
    void runSolver(LangModel &langModel) {
        ConsGraphTy &consGraph = *(super::getConsGraph());

        do {
            std::stack<std::vector<CGNodeTy *>> copySCCStack;

            // first do SCC detection and topo-sort
            // load/store/offset can create new copy constraint to be handled
            auto copy_it = scc_begin<ctx, Constraints::copy, false>(consGraph, copyWorkList);
            auto copy_ie = scc_end<ctx, Constraints::copy, false>(consGraph, copyWorkList);

            for (; copy_it != copy_ie; ++copy_it) {
                copySCCStack.push(*copy_it);
            }

            while (!copySCCStack.empty()) {
                const std::vector<CGNodeTy *> &scc = copySCCStack.top();
                // llvm::outs() << scc.front()->getNodeID() << ",";

                if (scc.size() > 1) {
                    if (DEBUG_PTA) {
                        llvm::outs() << "SCC(" << scc.size() << "): ";  // JEFF
                        for (auto n : scc) {
                            llvm::outs() << n->getNodeID() << ", ";  // JEFF
                        }
                        llvm::outs() << "\n";  // JEFF
                    }

                    processCopySCC(scc);
                } else {
                    CGNodeTy *curNode = scc.front();
                    for (auto cit = curNode->succ_copy_begin(), cie = curNode->succ_copy_end(); cit != cie; cit++) {
                        if (shouldProcessCopy(curNode, *cit)) {
                            if (super::processCopy(curNode, *cit)) {
                                // changedCopy.set(hashEdge(curNode, *cit));
                                lsWorkList.reset((*cit)->getNodeID());
                            }
                        }
                    }
                }
                copySCCStack.pop();
            }

            assert(copySCCStack.size() == 0);

            // set all copy to be already handled
            copyWorkList.set();  // empty the worklist
            targetList.set();

            requiredEdge.reset();

            const size_t prevNodeNum = consGraph.getNodeNum();
            int lastID = lsWorkList.find_first_unset();
            while (lastID >= 0) {
                CGNodeTy *curNode = consGraph.getNode(lastID);

                for (auto it = curNode->pred_store_begin(), ie = curNode->pred_store_end(); it != ie; it++) {
                    super::processStore(*it, curNode, [&](CGNodeTy *src, CGNodeTy *dst) { recordCopyEdge(src, dst); });
                }

                for (auto it = curNode->succ_load_begin(), ie = curNode->succ_load_end(); it != ie; it++) {
                    super::processLoad(curNode, *it, [&](CGNodeTy *src, CGNodeTy *dst) { recordCopyEdge(src, dst); });
                }

                // to handled special constraints
                for (auto it = curNode->succ_special_begin(), ie = curNode->succ_special_end(); it != ie; it++) {
                    super::processSpecial(curNode, *it,
                                          [&](CGNodeTy *src, CGNodeTy *dst) { recordCopyEdge(src, dst); });
                }

#ifndef NO_ADDR_OF_FOR_OFFSET
                for (auto it = curNode->succ_offset_begin(), ie = curNode->succ_offset_end(); it != ie; it++) {
                    super::processOffset(curNode, *it, [&](CGNodeTy *fieldObj, CGNodeTy *ptr) {
                        auto addrNode = llvm::cast<ObjNodeTy>(fieldObj)->getAddrTakenNode();
                        recordCopyEdge(addrNode, ptr);
                    });
                }
#endif
                lastID = lsWorkList.find_next_unset(lastID);
            }

#ifndef NO_ADDR_OF_FOR_OFFSET
            // index field can create new object thus make the constraint graph larger.
            lsWorkList.resize(consGraph.getNodeNum(), true);
            lsWorkList.set();  // mark all lsWorkList as done
#else
            assert(prevNodeNum == consGraph.getNodeNum());
            llvm::BitVector tmpWorklist(lsWorkList);
            lsWorkList.set();  // copy and clear lsWorkList

            lastID = tmpWorklist.find_first_unset();

            while (lastID >= 0) {
                CGNodeTy *curNode = consGraph.getNode(lastID);

                for (auto it = curNode->succ_offset_begin(), ie = curNode->succ_offset_end(); it != ie; it++) {
                    super::processOffset(curNode, *it, [&](CGNodeTy *fieldObj, CGNodeTy *ptr) {
                        assert(ptr->getNodeID() < targetList.size());
                        // targetList.reset(ptr->getNodeID());
                        PT::insert(ptr->getNodeID(), llvm::cast<ObjNodeTy>(fieldObj)->getObjectID());

                        // ensure that ptr is visited by SCCIterator
                        copyWorkList.reset(ptr->getNodeID());
                        // the pts of ptr has been updated, so need to be revisted by load/store
                        lsWorkList.reset(ptr->getNodeID());
                    });
                }
                lastID = tmpWorklist.find_next_unset(lastID);
            }
#endif
            lsWorkList.resize(consGraph.getNodeNum(), true);
            // the newly added node contains address taken node, which need to be revisited again
            // so set the extend bit to 0 (unhandled)
            copyWorkList.resize(consGraph.getNodeNum(), false);
            targetList.resize(consGraph.getNodeNum(), true);

#ifdef RESOLVE_FUNPTR_IMMEDIATELY
            // resolve function pointer at a earlier stage
            size_t prevNum = super::getConsGraph()->getNodeNum();
            CallBack callBack(*this, prevNum);

            super::getConsGraph()->registerCallBack(&callBack);
            super::resolveFunPtrs();
            super::getConsGraph()->unregisterCallBack();

            // extend the worklist, as the consgraph is expanded,
            lsWorkList.resize(super::getConsGraph()->getNodeNum(), false);
            targetList.resize(super::getConsGraph()->getNodeNum(), false);
            copyWorkList.resize(super::getConsGraph()->getNodeNum(), false);
#endif
            // The call to count in this log message is evaluated regardless of log level and are very expensive
            // LOG_TRACE("hash map fill rate, count = {}, size = {}, rate={}", requiredEdge.count(),
            // requiredEdge.size(),
            //           ((float)requiredEdge.count()) / requiredEdge.size());

            LOG_DEBUG("PTA Iteration No: {} - nodes: {}", numOfPTAIterations++, this->getConsGraph()->getNodeNum());
        } while (!copyWorkList.all());
    }

    void solve() {

        // initially, all node need to be traversed.
        copyWorkList.resize(super::getConsGraph()->getNodeNum(), false);
        lsWorkList.resize(super::getConsGraph()->getNodeNum(), false);
        targetList.resize(super::getConsGraph()->getNodeNum(), false);

#ifdef RESOLVE_FUNPTR_IMMEDIATELY
        this->runSolver(*super::getLangModel());
#else

        int pta_round = 0;
        bool reanalyze;
        do {
            if (DEBUG_PTA)
                llvm::outs() << "pta_round: " << (++pta_round) << " nodes: " << super::getConsGraph()->getNodeNum()
                             << "\n";  //<<" edges: "<<super::getConsGraph()->getEdgeNum()<<"\n";
            // after this, the current contraints graph will reach fixed point.
            this->runSolver(*super::getLangModel());

            assert(lsWorkList.all());  // all visited (1)
            assert(targetList.all());
            assert(copyWorkList.all());
            assert(!requiredEdge.any());  // all zero

            // record every constraints added during indirect call resolve
            size_t prevNodeNum = super::getConsGraph()->getNodeNum();
            CallBack callBack(*this, prevNodeNum);

            super::getConsGraph()->registerCallBack(&callBack);
            reanalyze = super::resolveFunPtrs();  // should always return false now
            super::getConsGraph()->unregisterCallBack();

            // extend the worklist, as the consgraph is expanded,
            lsWorkList.resize(super::getConsGraph()->getNodeNum(), false);
            targetList.resize(super::getConsGraph()->getNodeNum(), false);
            copyWorkList.resize(super::getConsGraph()->getNodeNum(), false);
        } while (reanalyze);
#endif
    }
    friend super;
    friend CallBack;
};

// template <typename LangModel>
// char PartialUpdateSolver<LangModel>::ID = 0;
//
// template <typename LangModel>
// static llvm::RegisterPass<PartialUpdateSolver<LangModel>>
//    PUS("PartialUpdateSolver",
//        "partial update solver that only traversal partial "
//        "of the constraint graph when new indirect calls are resolved",
//        true, true);

}  // namespace aser

#endif  // ASER_PTA_PARTIALUPDATESOLVER_H
