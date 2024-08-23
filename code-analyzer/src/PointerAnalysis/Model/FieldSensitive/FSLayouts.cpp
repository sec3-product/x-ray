//
// Created by peiming on 1/3/20.
//
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/Layout/ArrayLayout.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/Layout/MemLayout.h"
#include "PointerAnalysis/Models/MemoryModel/FieldSensitive/MemBlock.h"
#include "Util/Log.h"
#include "Util/TypeMetaData.h"

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

using namespace llvm;
using namespace std;

namespace xray {

static size_t indexBetweenArrays(const std::map<size_t, ArrayLayout *> &arrays,
                                 size_t &pOffset) {
  size_t lOffset = 0;
  size_t curOffset = 0;
  for (auto arrayPair : arrays) {
    // std::map is ordered by key from small value to large value
    size_t arrayOffset = arrayPair.first;
    const ArrayLayout *arrayLayout = arrayPair.second;

    // 1st, may be between two arrays
    if (pOffset <= arrayOffset) {
      lOffset += pOffset - curOffset;
      return lOffset;
    }

    assert(arrayOffset >= curOffset);
    lOffset += arrayOffset - curOffset;
    // 2nd, may be larger than current array
    size_t arrayEnd = arrayOffset + arrayLayout->getArraySize();
    if (pOffset >= arrayOffset + arrayLayout->getArraySize()) {
      // the physical offset bypass the current array
      // accumulate offsets and skip to next array
      lOffset += arrayLayout->getLayoutSize();
      curOffset = arrayEnd;
      continue;
    }

    // 3rd, maybe in the middle of an array. ( start offset < pOffset < start
    // offset + array size)
    size_t relativeOffset = pOffset - arrayOffset;
    size_t result = arrayLayout->indexPhysicalOffset(relativeOffset) + lOffset;
    // relative offset might be shrink when index array
    // i.e., a[0], a[1] is the same in our pointer analysis.
    pOffset = arrayOffset + relativeOffset;
    return result;
  }

  // 4th, after the last array.
  lOffset += pOffset - curOffset;
  return lOffset;
}

size_t MemLayout::indexPhysicalOffset(size_t &pOffset) const {
  if (!this->hasArray()) {
    // fast path
    // if the memory block does not have array ==> physical offset == layout
    // offset
    return pOffset;
  } else if (pOffset >= maxPOffset) {
    // invalid offset (exceed the max physical offset)
    return std::numeric_limits<size_t>::max();
  } else {
    // :-(, now we have to translate the physical offset
    return indexBetweenArrays(this->subArrays, pOffset);
  }
}

// merge a sub-layout into current memory layout
void MemLayout::mergeMemoryLayout(const MemLayout *subLayout, size_t pOffset,
                                  size_t lOffset) {
  for (auto elem : subLayout->elementLayout) {
    elementLayout.set(elem + lOffset);
  }
  for (auto elem : subLayout->pointerLayout) {
    pointerLayout.set(elem + lOffset);
  }
  for (auto elem : subLayout->specialLayout) {
    specialLayout.set(elem + pOffset);
  }

  assert(elementLayout.contains(pointerLayout));

  // TODO: make sure there is no overlapping between arrays
  // merge the array layout
  if (mIsArray) {
    assert(subArrays.size() == 1 && subArrays.begin()->first == 0);
    subArrays.begin()->second->mergeSubArrays(subLayout->subArrays, 0);
  } else {
    for (auto subArray : subLayout->subArrays) {
      subArrays.insert(
          std::make_pair(subArray.first + pOffset, subArray.second));
    }
  }
}

size_t ArrayLayout::indexPhysicalOffset(size_t &pOffset) const {
  if (this->hasSubArrays()) {
    pOffset = pOffset % this->getElementSize();
    return indexBetweenArrays(this->subArrays, pOffset);
  } else {
    assert(pOffset <= this->getArraySize());
    // we do not distinguish elements in the arrays.
    pOffset = pOffset % this->getElementSize();
    return pOffset;
  }
}

static void getPathNameVec(const SmallVector<DIDerivedType *, 8> &members,
                           size_t pOffset, vector<StringRef> &pathVec) {
  for (auto member : members) {
    if (pOffset >= member->getOffsetInBits() &&
        pOffset < member->getOffsetInBits() + getDISize(member)) {
      // this is the member!
      if (!member->getName().empty()) {
        pathVec.push_back(member->getName());
      }

      auto baseType = getBaseType(member); // member->getBaseType();
      while (baseType->getTag() == dwarf::DW_TAG_array_type) {
        // strip off arrays
        baseType =
            getBaseType(cast<DICompositeType>(baseType)); //->getBaseType();
      }

      if (auto CompositebaseType = dyn_cast<DICompositeType>(baseType)) {
        if (baseType->getTag() == dwarf::DW_TAG_structure_type ||
            baseType->getTag() == dwarf::DW_TAG_class_type) {
          getPathNameVec(getNonStaticDataMember(CompositebaseType),
                         pOffset -
                             member->getOffsetInBits(), // update the offset
                         pathVec);
        }
        // unhandled: DW_TAG_enumeration_type, DW_TAG_union_type
        return;
      } else {
        // TODO: inner type can be an array!!
        if (pOffset != member->getOffsetInBits()) {
          // will this happen??
          LOG_WARN("Find a Incompatible DICompositeType MD");
          pathVec.clear();
          return;
        }
      }
      return;
    }
  }
  // pOffset > total size of the object
  // should be unreachable
  pathVec.clear();
  return;
}

// NOTE: this might be expensive! call it with caution
std::string
MemLayout::getFieldAccessPath(const Module *M, size_t pOffset,
                              const llvm::StringRef separator) const {
  if (M == nullptr) {
    return "";
  }

  // strip off all the out arrays
  Type *T = this->type;
  while (isa<ArrayType>(T)) {
    T = T->getArrayElementType();
  }

  // now that all the array are stripped
  if (isa<StructType>(T)) {
    if (cast<StructType>(T)->isLiteral()) {
      return "";
    }
  } else {
    return "";
  }

  // llvm::errs() << "querying: ";
  // here, T is a struct type with name
  auto DI = getTypeMetaData(M, cast<StructType>(T));
  if (DI) {
    pOffset = pOffset * 8; // change byte into bit
    auto members = getNonStaticDataMember(DI);
    // check which field is accessing
    // recursively into the type tree, it should be okay as the type tree will
    // not be too deep
    vector<StringRef> vec;
    getPathNameVec(members, pOffset, vec);

    string result;
    raw_string_ostream os(result);
    for (StringRef path : vec) {
      os << separator << path;
    }
    return os.str();
  }

  return "";
}

} // namespace xray
