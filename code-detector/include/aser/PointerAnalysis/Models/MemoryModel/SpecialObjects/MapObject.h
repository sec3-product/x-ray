//
// Created by peiming on 6/9/20.
//

#ifndef ASER_PTA_MAPOBJECT_H
#define ASER_PTA_MAPOBJECT_H

#include <llvm/ADT/DenseMap.h>

#include "aser/PointerAnalysis/Graph/ConstraintGraph/ConstraintGraph.h"
#include "aser/PointerAnalysis/Graph/ConstraintGraph/CGPtrNode.h"
#include "aser/PointerAnalysis/Graph/ConstraintGraph/CGObjNode.h"


// a special object that models the behavior of a map

namespace aser {

template <typename MemModel>
struct LangModelTrait;

// TODO: right now, the class has lots of aspects to be improved.
// Biggest limitation, the MapObject need to be identified by the users,
// so that map_lookup(map *m), where the map is a pointer and require pointer analysis to resolve the target map
// can not be supported here.
// To Support those feature, solver need to be updated accordingly to support index map objects as well.
// Which can be applied in future extension.

// NOTE: this is a special type of object, which does not appear in the constraint graph
//       it acts as a delegator to maintain and access the object
template <typename KeyT, typename LangModel>
class MapObject {
private:
    using LMT = LangModelTrait<LangModel>;
    using ctx = typename LMT::CtxTy;
    using ObjectTy = typename LMT::ObjectTy;
    using CT = CtxTrait<ctx>;

    using PtrNodeTy = CGPtrNode<ctx>;

    // a static representation of the underlying map
    llvm::DenseMap<KeyT, PtrNodeTy *> theMap;

    // the underlying memory model
    LangModel &model;

    // the object node that can points to any elements in the map
    // used to handle elements inserted with unknown indices
    PtrNodeTy *allElem;

    // points to all the node that are inserted with unknown key
    PtrNodeTy *unknownIndexElem;

    // anonoyous ptr node should never be indexed, just a place holder
    // logically exist, but no corresponding llvm::Value
    inline PtrNodeTy *createAnonPtrNode() {
        return this->model.createAnonPtrNode();
    }

    inline PtrNodeTy *getOrCreateElem(const KeyT &key) {
        ConstraintGraph<ctx> *consGraph = model.getConsGraph();
        auto it = theMap.find(key);
        PtrNodeTy *element;
        if (it == theMap.end()) {
            element = createAnonPtrNode();
            auto result = theMap.try_emplace(key, element);
            assert(result.second);

            consGraph->addConstraints(unknownIndexElem, element, Constraints::copy);
            consGraph->addConstraints(element, allElem, Constraints::copy);
        } else {
            element = it->second;
        }

        return element;
    }

public:
    explicit MapObject(LangModel &model) : model(model),
                                           allElem(createAnonPtrNode()),
                                           unknownIndexElem(createAnonPtrNode()) {}

    // insert the key-val pair into a map
    void insert(const KeyT& key, PtrNodeTy *val) {
        assert(val);
        ConstraintGraph<ctx> *consGraph = model.getConsGraph();
        auto element = getOrCreateElem(key);
        consGraph->addConstraints(val, element, Constraints::copy);
    }

    // insert the key-val pair into a map, while the value of the key can not be
    // statically inferred.
    void insertWithUnknownKey(PtrNodeTy *val) {
        assert(val);

        ConstraintGraph<ctx> *consGraph = model.getConsGraph();
        consGraph->addConstraints(val, unknownIndexElem, Constraints::copy);
    }

    [[nodiscard]]
    PtrNodeTy *getElem(const KeyT &key) {
        return getOrCreateElem(key);
    }

    [[nodiscard]]
    inline PtrNodeTy *getElemWithUnknownKey() {
        return unknownIndexElem;
    }

};

}

#endif  // ASER_PTA_MAPOBJECT_H
