//
// Created by peiming on 1/3/20.
//

#ifndef ASER_PTA_ARRAYLAYOUT_H
#define ASER_PTA_ARRAYLAYOUT_H

#include <map>
#include <cassert>
#include <limits>

namespace aser {

class MemLayout;
class MemLayoutManager;

class ArrayLayout {
    size_t elementNum;
    size_t elementSize;

    // the number of bytes in memory layout
    size_t layoutSize = 0;
    // the key of the map is the *physical* offset of the subarray
    std::map<size_t, ArrayLayout *> subArrays;

    ArrayLayout(size_t elementNum, size_t elementSize)
        : elementNum(elementNum), elementSize(elementSize) {}

    void mergeSubArrays(const std::map<size_t, ArrayLayout *> &arrays, size_t offset) {
        for (auto subArray : arrays) {
            subArrays.insert(std::make_pair(subArray.first + offset, subArray.second));
        }
    }

    inline void setLayoutSize(size_t size) {
        assert(this->layoutSize == 0);
        this->layoutSize = size;
    }

public:
    [[nodiscard]] inline size_t getArraySize() const {
        if (elementNum == std::numeric_limits<size_t>::max()) {
            return std::numeric_limits<size_t>::max();
        }
        return elementNum * elementSize;
    }

    [[nodiscard]] inline size_t getElementSize() const {
        return elementSize;
    }

    [[nodiscard]] inline size_t getElementNum() const {
        return elementNum;
    }

    [[nodiscard]] inline size_t getLayoutSize() const {
        assert(layoutSize != 0);
        return layoutSize;
    }

    [[nodiscard]] inline bool hasSubArrays() const {
        return !subArrays.empty();
    }

    inline const std::map<size_t, ArrayLayout *> &getSubArrayMap() const {
        return this->subArrays;
    }

    size_t indexPhysicalOffset(size_t &pOffset) const;

    friend MemLayout;
    friend MemLayoutManager;
};

}

#endif  // ASER_PTA_ARRAYLAYOUT_H
