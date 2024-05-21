//
// Created by peiming on 11/2/19.
//
#ifndef ASER_PTA_CGOBJNODE_H
#define ASER_PTA_CGOBJNODE_H

#include "CGNodeBase.h"

namespace aser {

template <typename MemModel>
struct MemModelTrait;

template <typename ctx>
class ConstraintGraph;

// nodes represent objects
template <typename ctx, typename ObjT>
class CGObjNode : public CGNodeBase<ctx> {
private:
    using Self = CGObjNode<ctx, ObjT>;
    // context
    using super = CGNodeBase<ctx>;

    const ObjT *obj;

    CGObjNode(const ObjT *obj, NodeID id) : super(id, CGNodeKind::ObjNode), obj(obj){};

public:
    static inline bool classof(const super *node) { return node->getType() == CGNodeKind::ObjNode; }

    inline CGNodeBase<ctx> *getAddrTakenNode() {
        CGNodeBase<ctx> *addrTakeNode = this->graph->getNode(this->getNodeID() + 1);
        assert(addrTakeNode->getType() == CGNodeKind::PtrNode);

        return addrTakeNode;
    }

    inline const ObjT *getObject() const { return obj; }

    inline NodeID getObjectID() const { return getObject()->getObjectID(); }

    inline constexpr bool isFIObject() const { return obj->isFIObject(); }

    friend class ConstraintGraph<ctx>;

    [[nodiscard]] std::string toString() const {
        std::string str;
        llvm::raw_string_ostream os(str);
        if (this->isSuperNode()) {
            os << "SuperNode: \n";
            llvm::dump(this->childNodes, os);
        } else {
            os << super::getNodeID() << "\n";
            os << obj->toString() << "\n";
        }
        return os.str();
    }

    friend class GraphBase<super, Constraints>;
    friend class ConstraintGraph<ctx>;
};

}  // namespace aser

#endif