//
// Created by peiming on 8/26/20.
//

// The class represent the vtable pointer stored at the first byte of the object
// it can only points to one specific vtable pointer.
#include <Util/Log.h>

#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/FSObject.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/MemBlock.h"
#include "Util/TypeMetaData.h"

#ifndef ASER_PTA_VTABLEPTR_H
#define ASER_PTA_VTABLEPTR_H

namespace xray {

bool isVTablePtrType(const llvm::Type *type);

inline bool isVTableVariable(const llvm::Value *g) {
  if (g->hasName()) {
    auto name = getDemangledName(g->getName());
    if (name.find("vtable for") != std::string::npos) {
      return true;
    }
  }
  return false;
}

template <typename ctx> class VTablePtr : public FSObject<ctx> {
  using ConsGraph = ConstraintGraph<ctx>;
  using PtrNode = CGPtrNode<ctx>;

  // the type metadata of the object that holds the vtable pointer
  // const llvm::DICompositeType *typeMD;
  std::string vtableName;

public:
  VTablePtr(MemBlock<ctx> *block, size_t pOffset, const llvm::Type *objType)
      : FSObject<ctx>(block, pOffset, ObjectKind::Special), vtableName() {
    // strip array
    objType = stripArray(objType);
    // resolve the type metadata
    auto structType = llvm::cast<StructType>(objType);
    const DICompositeType *typeMD =
        getTypeMetaData(block->getValue(), block->getAllocKind(), structType);

    if (typeMD != nullptr) {
      SmallVector<const DIType *, 8> vptrPath;
      getFieldAccessPath(typeMD, pOffset, vptrPath);

      if (!vptrPath.empty()) {
        auto vptrMD = vptrPath.pop_back_val();
        const DICompositeType *vptrHolderMD = nullptr;

        if (vptrMD->getName().startswith("_vptr$")) {
          // one level up, is the composite type that holds the vptr
          vptrHolderMD = llvm::cast<DICompositeType>(vptrPath.pop_back_val());

          // look up to find the the vptr holder class
          while (!vptrPath.empty()) {
            auto holder = cast<DIDerivedType>(vptrPath.pop_back_val());
            if (holder->getTag() == dwarf::DW_TAG_inheritance) {
              // still need to look up for the child class
              vptrHolderMD =
                  llvm::cast<DICompositeType>(vptrPath.pop_back_val());
            } else {
              break;
            }
          }
        } else {
          LOG_ERROR("The resolved MD information might be incorrect. MD: {}, "
                    "allocSite: {}",
                    *typeMD, *block->getValue());
        }

        if (vptrHolderMD) {
          vtableName = vptrHolderMD->getIdentifier().str();
          if (vtableName.size() > 4) {
            // identifier :
            // _ZT*S*NSt6thread11_State_implINS_8_InvokerISt5tupleIJPFPvP4BaseEP5ChildEEEEEE
            // demanged to typeinfo for XXX
            // vtable :
            // _ZT*V*NSt6thread11_State_implINS_8_InvokerISt5tupleIJPFPvP4BaseEP5ChildEEEEEE
            // demanged to vtable for XXX
            vtableName.replace(3, 1, 1, 'V');
          } else {
            vtableName = "";
            LOG_ERROR("invalid identifier value, MD : {}", *vptrHolderMD);
          }
        }
      }
    } else {
      LOG_DEBUG("fail to resolve MD information. allocSite: {}",
                *block->getValue());
    }
  }

  // const FSObject<ctx> *getVptrObj() const { return &this->vtablePtrObj; }

  // obj is in the pts of (src)
  bool processSpecial(CGNodeBase<ctx> *src,
                      CGNodeBase<ctx> *vptr) const override {
    ConsGraph *consGraph = src->getGraph();

    PtrNode *ptrNode = llvm::cast<PtrNode>(vptr);
    const llvm::Value *vtable =
        ptrNode->getPointer()->getValue()->stripInBoundsConstantOffsets();

    // vptr is the ptrnode that holds the address of the vtable
    // src is the ptrnode that the vtable is store into && obj \in pts(src)
    if (!isVTableVariable(vtable)) {
      return false;
    }

    if (vtableName.empty()) {
      // no type metadata?
      LOG_DEBUG("can not resolve class hierarchy information for {}",
                *this->getAllocSite().getValue());
      // do not do filtering
      return consGraph->addConstraints(vptr, this->getObjNode(),
                                       Constraints::copy);
    }

    // using type metadata to do filtering on the vtable
    if (vtable->getName().equals(vtableName)) {
      // this is the right vtable
      return consGraph->addConstraints(vptr, this->getObjNode(),
                                       Constraints::copy);
    }
    // using
    return false;
  };
};

} // namespace xray
#endif // ASER_PTA_VTABLEPTR_H
