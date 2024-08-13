//
// Created by peiming on 8/26/20.
//

#ifndef ASER_PTA_TYPEMETADATA_H
#define ASER_PTA_TYPEMETADATA_H

#include "aser/PointerAnalysis/Models/MemoryModel/AllocSite.h"
#include <llvm/ADT/SmallVector.h>

namespace llvm {

class DICompositeType;
class DIDerivedType;
class Module;
class StructType;

} // namespace llvm

namespace aser {

llvm::DIType *stripTypeDefDI(llvm::DIType *DI);
llvm::DIType *stripArrayDI(llvm::DIType *DI);
llvm::DIType *stripArrayAndTypeDefDI(llvm::DIType *DI);

void TypeMDinit(const llvm::Module *M);

// only support looking up composite type, scalar type like int, float or
// pointer are not supported
const llvm::DICompositeType *getTypeMetaData(const llvm::StructType *T);
const llvm::DICompositeType *getTypeMetaData(const llvm::Module *M,
                                             const llvm::StructType *T);

// look up the type metadata of the object allocated
const llvm::DICompositeType *getTypeMetaData(const llvm::Value *allocSite,
                                             AllocKind T,
                                             const llvm::Type *allocType);

llvm::SmallVector<llvm::DIDerivedType *, 8>
getNonStaticDataMember(const llvm::DICompositeType *DI);

void getFieldAccessPath(const llvm::DICompositeType *DI, size_t offsetInByte,
                        llvm::SmallVector<const llvm::DIType *, 8> &result);

// getbasetype, but strip off all the typedef
template <typename DebugInfoType>
inline llvm::DIType *getBaseType(DebugInfoType *T) {
  llvm::DIType *baseType = T->getBaseType();
  return stripTypeDefDI(baseType);
}

std::size_t getDISize(llvm::DIDerivedType *T);
} // namespace aser

#endif // ASER_PTA_TYPEMETADATA_H
