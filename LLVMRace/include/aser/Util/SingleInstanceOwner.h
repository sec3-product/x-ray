//
// Created by peiming on 8/29/19.
//
#ifndef ASER_PTA_SINGLEINSTANCEOWNER_H
#define ASER_PTA_SINGLEINSTANCEOWNER_H

#include <unordered_set>

namespace aser {

template <typename T>
class SingleInstanceOwner {
protected:
    std::unordered_set<T> innerSet;  // TODO: maybe other container is faster?
    using iterator = typename std::unordered_set<T>::iterator;

    // create if does not exist
    // get if already exists
    template <typename... Args>
    inline std::pair<const T *, bool> getOrCreate(Args &&... args) {
        auto r = innerSet.emplace(std::forward<Args>(args)...);
        return std::make_pair(&*r.first, r.second);
    }

    // create or abort
    template <typename... Args>
    inline const T *create(Args &&... args) {
        auto r = innerSet.emplace(std::forward<Args>(args)...);
        if (LLVM_UNLIKELY(!r.second)) {
            llvm_unreachable("Trying to re-create a existing item");
        }
        return &*r.first;
    }

public:
    // get or abort
    template <typename... Args>
    inline const T *get(Args &&... args) const {
        auto r = innerSet.find(T(std::forward<Args>(args)...));
        if (LLVM_UNLIKELY(r == innerSet.end())) {
            llvm_unreachable("Trying to get a non-exist item!");
        }
        return &*r;
    }

    // get or abort
    template <typename... Args>
    inline const T *getOrNull(Args &&... args) const {
        auto r = innerSet.find(T(std::forward<Args>(args)...));
        if (LLVM_UNLIKELY(r == innerSet.end())) {
            return nullptr;
        }
        return &*r;
    }

    // get or abort
    inline const T *getOrNull(const T &t) const {
        auto r = innerSet.find(t);
        if (LLVM_UNLIKELY(r == innerSet.end())) {
            return nullptr;
        }
        return &*r;
    }

    // get or abort
    inline const T *get(const T &t) const {
        auto r = innerSet.find(t);
        if (LLVM_UNLIKELY(r == innerSet.end())) {
            llvm_unreachable("Trying to get a non-exist item!");
        }
        return &*r;
    }
};

}  // namespace aser

#endif