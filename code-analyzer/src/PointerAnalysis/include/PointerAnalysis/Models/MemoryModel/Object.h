//
// Created by peiming on 6/17/20.
//

#ifndef ASER_PTA_OBJECT_H
#define ASER_PTA_OBJECT_H

#include "PointerAnalysis/Graph/NodeID.def"

namespace xray {

// forward declaration
template <typename ctx, typename ObjT> class CGObjNode;

using ObjID = NodeID;

template <typename ctx, typename SubClass> class Object {
protected:
  using ObjNode = CGObjNode<ctx, SubClass>;

  bool isImmutable;
  static ObjID CurID;
  // static std::vector<Object<MemModel>*> ObjVec;

  ObjNode *objNode = nullptr;
  ObjID objID;

  Object() : objID(CurID++), isImmutable(false) {}

  // this can only be called internally
  inline void setObjNode(ObjNode *node) {
    assert(objNode == nullptr);
    objNode = node;
    if (isImmutable) {
      node->setImmutable();
    }
  }

public:
  [[nodiscard]] inline ObjNode *getObjNodeOrNull() const { return objNode; }

  [[nodiscard]] inline ObjNode *getObjNode() const {
    assert(objNode != nullptr);
    return objNode;
  }

  [[nodiscard]] ObjID getObjectID() const { return objID; }

  inline void setImmutable() {
    this->isImmutable = true;
    if (objNode != nullptr) {
      objNode->setImmutable();
    }
  }

  static void resetObjectID() { CurID = 0; }

  friend CGObjNode<ctx, SubClass>;
};

template <typename ctx, typename SubClass>
ObjID Object<ctx, SubClass>::CurID = 0;

} // namespace xray
#endif // ASER_PTA_OBJECT_H