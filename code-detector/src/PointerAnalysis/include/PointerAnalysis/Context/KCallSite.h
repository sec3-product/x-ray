//
// Created by peiming on 11/19/19.
//
#ifndef ASER_PTA_KCALLSITE_H
#define ASER_PTA_KCALLSITE_H

#include <llvm/ADT/Hashing.h>

#include "CtxTrait.h"

#include "PtrRingBuffer.h"
#include "PointerAnalysis/Program/CallSite.h"
#include "Util/SingleInstanceOwner.h"

namespace aser {

template <uint32_t K> class KCallSite {
private:
  using self = KCallSite<K>;
  PtrRingBuffer<const llvm::Instruction, K> ctxBuffer;

public:
  using iterator = typename PtrRingBuffer<const llvm::Instruction, K>::iterator;

  KCallSite() noexcept : ctxBuffer() {}

  KCallSite(const self *prevCtx, const llvm::Instruction *I)
      : ctxBuffer(prevCtx->ctxBuffer) {
    assert(aser::CallSite(I).isCallOrInvoke());
    ctxBuffer.push(I);
  }

  KCallSite(const self &) = delete;
  KCallSite(self &&) = delete;
  KCallSite &operator=(const self &) = delete;
  KCallSite &operator=(self &&) = delete;

  iterator begin() const { return ctxBuffer.begin(); }

  iterator end() const { return ctxBuffer.end(); }

  [[nodiscard]] std::string toString(bool detailed = false) const {
    std::string str;
    llvm::raw_string_ostream os(str);
    if (detailed) {
      os << '<';
      for (const llvm::Instruction *I : ctxBuffer) {
        if (I == nullptr)
          continue;
        os << *I << ";";
      }
      os << '>';
    } else {
      os << '<';
      for (const llvm::Instruction *I : ctxBuffer) {
        if (I == nullptr)
          continue;
        os << I << "->";
      }
      os << '>';
    }
    return os.str();
  };

  [[nodiscard]] const llvm::Instruction *getLast() const {
    const llvm::Instruction *last = nullptr;
    for (const llvm::Instruction *I : ctxBuffer) {
      if (I == nullptr)
        continue;
      last = I;
    }
    return last;
  };

  bool empty() const { return getLast() == nullptr; };

  bool operator==(const self &rhs) const {
    auto it1 = this->begin();
    auto it2 = this->begin();

    auto ie1 = this->end();
    auto ie2 = this->end();

    for (; it1 != ie1; it1++, it2++) {
      if (*it1 != *it2) {
        return false;
      }
    }
    assert(it2 == ie2);
    return true;
  }
};

template <uint32_t K> struct CtxTrait<KCallSite<K>> {
private:
  static const KCallSite<K> initCtx;
  static const KCallSite<K> globCtx;
  static std::unordered_set<KCallSite<K>> ctxSet;

public:
  static const KCallSite<K> *contextEvolve(const KCallSite<K> *prevCtx,
                                           const llvm::Instruction *I) {
    auto result = ctxSet.emplace(prevCtx, I);
    return &*result.first;
  }

  static const KCallSite<K> *getInitialCtx() { return &initCtx; }

  static const KCallSite<K> *getGlobalCtx() { return &globCtx; }

  static std::string toString(const KCallSite<K> *context,
                              bool detailed = false) {
    if (context == &globCtx)
      return "<global>";
    if (context == &initCtx)
      return "<empty>";
    return context->toString(detailed);
  }

  static void release() { ctxSet.clear(); }
};

template <uint32_t K> const KCallSite<K> CtxTrait<KCallSite<K>>::initCtx{};

template <uint32_t K> const KCallSite<K> CtxTrait<KCallSite<K>>::globCtx{};

template <uint32_t K>
std::unordered_set<KCallSite<K>> CtxTrait<KCallSite<K>>::ctxSet{};

} // namespace aser

namespace std {

// only hash context and value
template <uint32_t K> struct hash<aser::KCallSite<K>> {
  size_t operator()(const aser::KCallSite<K> &cs) const {
    llvm::hash_code hash = llvm::hash_combine_range(cs.begin(), cs.end());
    return hash_value(hash);
  }
};

} // namespace std

#endif // ASER_PTA_KCALLSITE_H
