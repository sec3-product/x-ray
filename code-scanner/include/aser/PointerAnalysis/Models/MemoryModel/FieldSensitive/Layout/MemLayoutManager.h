//
// Created by peiming on 12/18/19.
//

#ifndef ASER_PTA_MEMLAYOUTMANAGER_H
#define ASER_PTA_MEMLAYOUTMANAGER_H

#include <llvm/ADT/BitVector.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Allocator.h>

#include <unordered_map>

#include "aser/PointerAnalysis/Models/MemoryModel/FieldSensitive/Layout/ArrayLayout.h"
#include "aser/PointerAnalysis/Models/MemoryModel/FieldSensitive/Layout/MemLayout.h"
#include "aser/Util/Log.h"

namespace aser {

class MemLayoutManager {
private:
    using MemLayoutAllocator = llvm::SpecificBumpPtrAllocator<MemLayout>;
    using ArrayLayoutAllocator = llvm::SpecificBumpPtrAllocator<ArrayLayout>;
    using CallBackT = const std::function<bool(llvm::Type *, MemLayout *, size_t&, size_t&)>;

    // owner of memlayout
    MemLayoutAllocator memLayoutAllocator;
    ArrayLayoutAllocator arrayLayoutAllocator;

    //std::unordered_map<std::pair<llvm::Type *, bool>, MemLayout *> memLayoutMap;
    llvm::DenseMap<llvm::Type *, MemLayout *> collapsedLayoutMap;
    llvm::DenseMap<llvm::Type *, MemLayout *> unCollapsedLayoutMap;

    void setElementLayout(llvm::Type *elementType, MemLayout *parentLayout, const llvm::DataLayout &DL, size_t &lOffset,
                          size_t &pOffset, bool collapseArray, CallBackT &callback) {
        assert(!collapseArray ? pOffset == lOffset : pOffset >= lOffset);

        if (callback && callback(elementType, parentLayout, lOffset, pOffset)) {
            return;
        }
        
        size_t typeAllocSize = DL.getTypeAllocSize(elementType);
        switch (elementType->getTypeID()) {
            case llvm::Type::StructTyID:
            case llvm::Type::ArrayTyID: {
                // recursive call, should be fine, the type tree shouldn't be too deep
                // change it if it becomes a problem.
                size_t subPOffset = 0, subLOffset = 0;
                const MemLayout *subLayout
                    = DFSTypeTree(elementType, DL, subLOffset, subPOffset, collapseArray, callback);
                parentLayout->mergeMemoryLayout(subLayout, pOffset, lOffset);
                // forward the logical offset
                lOffset += subLOffset;
                // forward the physical offset
                pOffset += typeAllocSize;
                break;
            }
            case llvm::Type::PointerTyID:
                // if it is pointer, mark the offset
                parentLayout->setElementOffset(lOffset);
                parentLayout->setPointerOffset(lOffset);
                // forward the logical offset
                lOffset += typeAllocSize;
                // forward the physical offset
                pOffset += typeAllocSize;
                break;
            case llvm::Type::FixedVectorTyID:
            case llvm::Type::ScalableVectorTyID: {
            //case llvm::Type::VectorTyID: {
                // simply skip it
                LOG_TRACE("Unhandled Type. type={}", *elementType);
            }
            default: {
                // primitive type
                parentLayout->setElementOffset(lOffset);
                // forward the logical offset
                lOffset += typeAllocSize;
                // forward the physical offset
                pOffset += typeAllocSize;
                break;
            }
        }
    }

    const MemLayout *getCollapsedArrayLayout(llvm::ArrayType *T, const llvm::DataLayout &DL, size_t &lOffset,
                                             size_t &pOffset, CallBackT &callback) {
        assert(lOffset == 0 && pOffset == 0);

        auto layout = new (memLayoutAllocator.Allocate()) MemLayout(T, true);
        // creating a new array layout
        llvm::Type *elementType = T->getElementType();
        size_t numElement = T->getArrayNumElements();
        size_t elementSize = DL.getTypeAllocSize(elementType);

        auto arrayLayout = new (arrayLayoutAllocator.Allocate()) ArrayLayout(numElement, elementSize);
        layout->insertSubArray(0, arrayLayout);

        // set the element layout
        setElementLayout(elementType, layout, DL, lOffset, pOffset, true, callback);
        // forward the physical offset
        if (numElement != std::numeric_limits<size_t>::max()) {
            pOffset += elementSize * (numElement - 1);
        } else {
            pOffset = std::numeric_limits<size_t>::max();
        }
        // layout->insertArrayTuple(numElement, elementSize, 0, layoutOffset);
        layout->setMaxOffsets(lOffset, pOffset);
        arrayLayout->setLayoutSize(lOffset);

        // cache the type layout
        bool result = collapsedLayoutMap.insert(std::make_pair(T, layout)).second;
        assert(result && "creating a exsiting type layout");
        return layout;
    }

    const MemLayout *getNonCollapsedArrayLayout(llvm::ArrayType *T, const llvm::DataLayout &DL, size_t &lOffset,
                                                size_t &pOffset, CallBackT &callback) {
        assert(lOffset == 0 && pOffset == 0);
        auto layout = new (memLayoutAllocator.Allocate()) MemLayout(T, false);
        // creating a new array layout
        llvm::Type *elementType = T->getElementType();
        size_t numElement = T->getArrayNumElements();
        size_t elementSize = DL.getTypeAllocSize(elementType);

        assert(numElement * elementSize == DL.getTypeAllocSize(T));
        assert(numElement != std::numeric_limits<uint64_t>::max() && "Unbounded Array Should be Collapsed!");
        // auto arrayLayout = new (arrayLayoutAllocator.Allocate()) ArrayLayout(numElement, elementSize);
        // layout->insertSubArray(0, arrayLayout);

        for (int i = 0; i < numElement; i++) {
            setElementLayout(elementType, layout, DL, lOffset, pOffset, false, callback);
            assert(lOffset==pOffset && pOffset == (i + 1) * elementSize);
        }
        // layout->insertArrayTuple(numElement, elementSize, 0, layoutOffset);
        assert(pOffset == DL.getTypeAllocSize(T) && lOffset == DL.getTypeAllocSize(T));

        layout->setMaxOffsets(lOffset, pOffset);
        // arrayLayout->setLayoutSize(lOffset);

        // cache the type layout
        bool result = unCollapsedLayoutMap.insert(std::make_pair(T, layout)).second;
        assert(result && "creating a exsiting type layout");
        return layout;

    }
    const MemLayout *getArrayLayout(llvm::ArrayType *T, const llvm::DataLayout &DL, size_t &lOffset, size_t &pOffset,
                                    bool collapseArray, CallBackT &callback) {
        if (collapseArray) {
            return getCollapsedArrayLayout(T, DL, lOffset, pOffset, callback);
        } else {
            return getNonCollapsedArrayLayout(T, DL, lOffset, pOffset, callback);
        }
    }

    const MemLayout *getStructLayout(llvm::StructType *T, const llvm::DataLayout &DL, size_t &lOffset, size_t &pOffset,
                                     bool collapseArray, CallBackT &callback) {
        assert(lOffset == 0 && pOffset == 0);
        // create a new MemLayout
        auto layout = new (memLayoutAllocator.Allocate()) MemLayout(T);
        auto structLayout = DL.getStructLayout(T);

        for (int i = 0; i < T->getNumElements(); i++) {
            // accumulate physical offset
            size_t elemOffset = structLayout->getElementOffset(i);
            assert(elemOffset >= pOffset);

            // get the memory layout of sub elements.
            auto elementType = T->getElementType(i);
            size_t padding = elemOffset - pOffset;
            // adjust the padding in the structure
            lOffset += padding;
            pOffset += padding;

            // recursively build the sub-layout of the field
            this->setElementLayout(elementType, layout, DL, lOffset, pOffset, collapseArray, callback);
        }

        // here pOffset might not equal to the allocation size because of the padding.
        // the memory beyond the last element is used as padding, adjust the layout accordingly
        if (pOffset != DL.getTypeAllocSize(T)) {
            assert(pOffset < DL.getTypeAllocSize(T));

            size_t padding = DL.getTypeAllocSize(T) - pOffset;
            pOffset += padding;
            lOffset += padding;
        }
        // assert(structLayout->hasPadding() ? true: pOffset == DL.getTypeAllocSize(T));
        layout->setMaxOffsets(lOffset, pOffset);
        // cached the result
        bool result = collapseArray
                      ? collapsedLayoutMap.insert(std::make_pair(T, layout)).second
                      : unCollapsedLayoutMap.insert(std::make_pair(T, layout)).second;

        assert(result && "creating a existing type layout");
        return layout;
    }

    // TODO: change it to non-recursive version if the type tree can be too deep
    // layout offset may not be equal to physical offset at the presence of arrays.
    const MemLayout *DFSTypeTree(llvm::Type *T, const llvm::DataLayout &DL, size_t &lOffset, size_t &pOffset,
                                 bool collapseArray, CallBackT &callback) {
        assert(T->isAggregateType());

        // already cached
        auto it = collapseArray
                  ? collapsedLayoutMap.find(T)
                  : unCollapsedLayoutMap.find(T) ;

        if (collapseArray ? it != collapsedLayoutMap.end() : it != unCollapsedLayoutMap.end()) {
            // also update the layout offset and physical offset
            const MemLayout *layout = it->second;
            lOffset += layout->maxLOffset;
            pOffset += layout->maxPOffset;
            return layout;
        }

        if (DL.getTypeAllocSize(T) == 0) {
            auto layout = new (memLayoutAllocator.Allocate()) MemLayout(T, true);
            layout->setMaxOffsets(0, 0);
            layout->setElementOffset(0);

            bool result = collapseArray
                          ? collapsedLayoutMap.insert(std::make_pair(T, layout)).second
                          : unCollapsedLayoutMap.insert(std::make_pair(T, layout)).second;

            assert(result && "creating a existing type layout");
            return layout;
        }

        // dispatch
        if (auto arrayTy = llvm::dyn_cast<llvm::ArrayType>(T)) {
            return getArrayLayout(arrayTy, DL, lOffset, pOffset, collapseArray, callback);
        } else if (auto structTy = llvm::dyn_cast<llvm::StructType>(T)) {
            return getStructLayout(structTy, DL, lOffset, pOffset, collapseArray, callback);
        }

        llvm_unreachable("aggregate type is not array or structure!?");
    }

public:
    const MemLayout *getLayoutForType(llvm::Type *T, const llvm::DataLayout &DL, bool collapseArray=true,
                                      CallBackT &callback = nullptr) {
        // offset index the layout table might be different than the phsical layout of the type
        // as arrays are collapsed.

        size_t lOffset = 0, pOffset = 0;
        return DFSTypeTree(T, DL, lOffset, pOffset, collapseArray, callback);
    }
};

}  // namespace aser

#endif  // ASER_PTA_MEMLAYOUTMANAGER_H
