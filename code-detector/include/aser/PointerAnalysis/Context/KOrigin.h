//
// Created by peiming on 11/19/19.
//

#ifndef ASER_PTA_KORIGIN_H
#define ASER_PTA_KORIGIN_H

#include <aser/Util/Log.h>
#include <llvm/ADT/StringSet.h>

#include "CtxTrait.h"
#include "KCallSite.h"

namespace aser {

template <uint32_t K, uint32_t L = 1> struct OriginsSetter;

// L is only useful in hybrid context,
// e.g., when use with <k-callsite + origin>, L=k+1 can make origin more precise

// TODO: support L > 1 to make it more accurate
// L is the length of the callchain that can be used to indentify an origin
template <uint32_t K, uint32_t L = 1> class KOrigin : public KCallSite<K * L> {
private:
  using self = KOrigin<K, L>;
  using super = KCallSite<K * L>;

  static std::function<bool(const self *, const llvm::Instruction *)> callback;

public:
  KOrigin() noexcept : super() {}
  KOrigin(const self *prevCtx, const llvm::Instruction *I)
      : super(prevCtx, I) {}

  static void setOriginRules(
      std::function<bool(const self *, const llvm::Instruction *)> cb) {
    callback = cb;
  }

  KOrigin(const self &) = delete;
  KOrigin(self &&) = delete;
  KOrigin &operator=(const self &) = delete;
  KOrigin &operator=(self &&) = delete;

  friend OriginsSetter<K, L>;
  friend CtxTrait<KOrigin<K, L>>;
};

template <uint32_t K, uint32_t L> struct CtxTrait<KOrigin<K, L>> {
private:
  static const KOrigin<K, L> initCtx;
  static const KOrigin<K, L> globCtx;
  static std::unordered_set<KOrigin<K, L>> ctxSet;

public:
  static const KOrigin<K, L> *contextEvolve(const KOrigin<K, L> *prevCtx,
                                            const llvm::Instruction *I) {
    if constexpr (L == 1) {
      if (KOrigin<K, L>::callback(prevCtx, I)) {
        auto result = ctxSet.emplace(prevCtx, I);
        return &*result.first;
      }
      return prevCtx;
    } else {
      llvm_unreachable("No support yet");
    }
  }

  inline static size_t getNumCtx() { return ctxSet.size(); }

  static const KOrigin<K, L> *getInitialCtx() { return &initCtx; }

  static const KOrigin<K, L> *getGlobalCtx() { return &globCtx; }

  // 3rd, string representation
  static std::string toString(const KOrigin<K, L> *context,
                              bool detailed = false) {
    if (context == &globCtx)
      return "<global>";
    if (context == &initCtx)
      return "<empty>";
    return context->toString(detailed);
  }

  static void release() { ctxSet.clear(); }
};

template <uint32_t K, uint32_t L>
const KOrigin<K, L> CtxTrait<KOrigin<K, L>>::initCtx{};

template <uint32_t K, uint32_t L>
const KOrigin<K, L> CtxTrait<KOrigin<K, L>>::globCtx{};

template <uint32_t K, uint32_t L>
std::unordered_set<KOrigin<K, L>> CtxTrait<KOrigin<K, L>>::ctxSet{};

template <uint32_t K, uint32_t L>
std::function<bool(const KOrigin<K, L> *, const llvm::Instruction *)>
    KOrigin<K, L>::callback =
        [](const KOrigin<K, L> *, const llvm::Instruction *) {
          // by default no function is origin
          return false;
        };

} // namespace aser

namespace std {

// only hash context and value
template <uint32_t K, uint32_t L> struct hash<aser::KOrigin<K, L>> {
  size_t operator()(const aser::KOrigin<K, L> &origin) const {
    return hash<aser::KCallSite<K * L>>()(origin);
  }
};

} // namespace std

#endif // ASER_PTA_KORIGIN_H
