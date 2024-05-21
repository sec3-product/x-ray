//
// Created by peiming on 11/2/19.
//
#ifndef ASER_PTA_CGSUPERNODE_H
#define ASER_PTA_CGSUPERNODE_H

// TODO: maybe merge SCC later?

#include "CGObjNode.h"
#include "CGPtrNode.h"

namespace aser {
// nodes represent collapsed SCC
template <typename ctx>
class CGSuperNode : public CGNodeBase<ctx> {
private:
    using super = CGNodeBase<ctx>;
    using NodeList = llvm::SparseBitVector<>;

    NodeList scc;
    NodeList callNodes;

    CGSuperNode(const vector<super *> &scc, NodeID id) : super(id, CGNodeKind::SuperNode) {
        for (super *node : scc) {
            if (node->isFunctionPtr()) {
                callNodes.set(node->getNodeID());
            }
            if (auto superNode = llvm::dyn_cast<CGSuperNode<ctx>>(node)) {
                // merge a super node
                this->scc |= superNode->scc;
                this->callNodes |= superNode->callNodes;
            }
            this->scc.set(node->getNodeID());
        }
    };

public:
    inline void copyOutgoingEdges(super *node) {
        for (auto it = node->pred_edge_begin(), ie = node->pred_edge_end(); it != ie; it++) {
            auto pred = (*it).second;
            auto edgeKind = (*it).first;
            // FIXME: is this correct?
            pred->insertConstraint(this, edgeKind);
        }

        node->setSuperNode(this->getNodeID());
    }

    void clearAndMergeIncomingEdges() {
        for (NodeID nodeId : scc) {
            auto node = this->graph->getNode(nodeId);
            // merge the incoming edges
            for (auto it = node->edge_begin(), ie = node->edge_end(); it != ie; it++) {
                this->insertConstraint((*it).second, (*it).first);
            }
        }

        for (NodeID nodeId : scc) {
            auto node = this->graph->getNode(nodeId);
            node->clearConstraints();
        }

        this->removeConstraint(this, Constraints::copy);
    }

    static inline bool classof(const super *node) { return node->getType() == CGNodeKind::SuperNode; }

    virtual std::string toString() const { return "SuperNode"; }

    friend class GraphBase<super, Constraints>;
    friend class ConstraintGraph<ctx>;
};

}  // namespace aser

#endif