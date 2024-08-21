//
// Created by peiming on 12/18/19.
//

#ifndef ASER_PTA_MEMLAYOUT_H
#define ASER_PTA_MEMLAYOUT_H

#include <map>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SparseBitVector.h>
#include <llvm/IR/Module.h>

// forward declaration
namespace llvm {
class Type;
}

namespace xray {

// forward declaration
class MemLayoutManager;
class ArrayLayout;

class MemLayout {
private:
  // By byte. If the bit is set, the offset stores a pointer
  llvm::SparseBitVector<> pointerLayout;
  // By byte. If the bit is set, the offset is valid (not in the middle of a
  // primary type)
  llvm::SparseBitVector<> elementLayout;
  // By byte. If the bit is set, the (physical) offset is for a speical object;
  llvm::SparseBitVector<> specialLayout;

  // seems the test on sparsebitvector can become bottleneck, cache it using
  // bitvector need more memory but faster. or increase the element size of the
  // sparsebitvector
  llvm::BitVector indexableLayout;

  // a map of the subarrays in the Memory Layout
  std::map<size_t, ArrayLayout *> subArrays;

  // logicalIndex[layoutOffset] -> the No. of the field
  std::vector<int> logicalIndex;

  // the type of the memory Layout
  llvm::Type *type;

  // Max Layout offset and Max Physical Offset
  // NOTE: layout offset may differ with physical offset at existence of arrays.
  size_t maxLOffset = 0; // from [0, maxLOffset) are indexable
  size_t maxPOffset = 0; // from [0, maxPOffset) are indexable

  bool cached = false;
  const bool mIsArray;
  explicit MemLayout(llvm::Type *T, bool isArray = false)
      : type(T), mIsArray(isArray) {}

  // should only be called once
  void setMaxOffsets(size_t LOffset, size_t POffset) {
    assert(maxLOffset == 0 && maxPOffset == 0);
    this->maxLOffset = LOffset;
    this->maxPOffset = POffset;
  }

  inline void cacheIntoBitVector() {
    assert(!cached);
    int num = elementLayout.find_last();
    indexableLayout.resize(num + 1);
    logicalIndex.resize(num + 1, -1);

    // cache it into bitvector for faster indexing
    int i = 0;
    for (size_t offset : elementLayout) {
      indexableLayout.set(offset);
      logicalIndex[offset] = i++;
    }
    cached = true;
  }

public:
  MemLayout(const MemLayout &) = delete;
  MemLayout(MemLayout &&) = delete;
  MemLayout &operator=(const MemLayout &) = delete;
  MemLayout &operator=(MemLayout &&) = delete;

  inline bool hasArray() const { return !subArrays.empty(); }

  inline bool isArray() const { return mIsArray; }

  inline const llvm::Type *getType() const { return type; }
  // return -1 is the lOffset can not be indexed
  inline int getLogicalOffset(size_t lOffset) const {
    if (LLVM_UNLIKELY(!cached)) {
      const_cast<MemLayout *>(this)->cacheIntoBitVector();
    }
    if (lOffset >= logicalIndex.size()) {
      return -1;
    }
    return logicalIndex[lOffset];
  }

  inline const llvm::SparseBitVector<> &getSpecialLayout() const {
    return this->specialLayout;
  }

  inline unsigned int getNumIndexableElem() const {
    return this->elementLayout.count();
  }

  inline bool offsetIndexable(size_t lOffset) const {
    if (LLVM_UNLIKELY(!cached)) {
      const_cast<MemLayout *>(this)->cacheIntoBitVector();
    }
    if (lOffset >= indexableLayout.size()) {
      return false;
    }
    return indexableLayout.test(lOffset);
  }

  inline bool offsetIsPtr(size_t lOffset) const {
    return pointerLayout.test(lOffset);
  }

  __attribute__((noinline))
  __attribute__((used)) // for debugging purpose, do not opt out the function
  void
  dump() {
    llvm::dump(elementLayout, llvm::errs());
    llvm::dump(pointerLayout, llvm::errs());
    llvm::dump(specialLayout, llvm::errs());
  }

  inline const std::map<size_t, ArrayLayout *> &getSubArrayMap() const {
    return this->subArrays;
  }

  // insert a subarray layout into the memory layout at certain *physical*
  // offset
  inline void insertSubArray(size_t pOffset, ArrayLayout *subArray) {
    bool result = this->subArrays.emplace(pOffset, subArray).second;
    assert(result && "more than two arrarys at the same offset");
  }

  inline void setElementOffset(size_t offset) {
    // the following assertion can be invalidated by zero-sized object
    // assert(!elementLayout.test(offset) && "element layout already set!");
    elementLayout.set(offset);
  }

  inline void setSpecialOffset(size_t offset) {
    assert(!specialLayout.test(offset) && "special layout already set!");
    specialLayout.set(offset);
  }

  inline void setPointerOffset(size_t offset) {
    assert(!pointerLayout.test(offset) && "pointer layout already set!");
    pointerLayout.set(offset);
    assert(elementLayout.contains(pointerLayout));
  }

  // convert Physical Offset to Layout Offset
  // the passed in physical offset can be changed internally when indexing array
  // elements
  // TODO: maybe cache the result if this becomes the bottleneck
  size_t indexPhysicalOffset(size_t &pOffset) const;
  // merge a sub-layout into current memory layout
  void mergeMemoryLayout(const MemLayout *subLayout, size_t pOffset,
                         size_t lOffset);

  [[nodiscard]] std::string getFieldAccessPath(const llvm::Module *M,
                                               size_t pOffset,
                                               llvm::StringRef separator) const;

  friend MemLayoutManager;
};

} // namespace xray

#endif // ASER_PTA_MEMLAYOUT_H
