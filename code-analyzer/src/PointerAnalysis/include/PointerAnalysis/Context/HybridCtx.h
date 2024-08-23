//
// Created by peiming on 4/2/20.
//
#ifndef ASER_PTA_HYBRIDCTX_H
#define ASER_PTA_HYBRIDCTX_H

#include <llvm/ADT/STLExtras.h>

#include <tuple>
#include <unordered_set>

#include "CtxTrait.h"

namespace xray {
// although we only have two context (origin, callsite)
// we make it va_args in case of future extension
template <typename... Args> class HybridCtx {
  std::tuple<const Args *...> ctx;

private:
  template <size_t... N>
  static std::tuple<const Args *...>
  evolveInnerContext(const HybridCtx<Args...> *prevCtx,
                     const llvm::Instruction *I, std::index_sequence<N...>) {
    return {CtxTrait<Args>::contextEvolve(std::get<N>(prevCtx->ctx), I)...};
  }

public:
  explicit HybridCtx(const Args *...args) : ctx{args...} {}

  HybridCtx(const HybridCtx<Args...> *prevCtx, const llvm::Instruction *I)
      : ctx(evolveInnerContext(prevCtx, I,
                               std::index_sequence_for<Args...>{})) {}

  const std::tuple<const Args *...> &getContext() const { return ctx; }

  friend CtxTrait<HybridCtx<Args...>>;
  friend std::hash<xray::HybridCtx<Args...>>;
};

// for container operation
template <typename... Args>
bool operator==(const HybridCtx<Args...> &lhs, const HybridCtx<Args...> &rhs) {
  return isTupleEqual(lhs.getContext(), rhs.getContext());
}

// here is the true logic
// basically just delegate it to different CtxTrait
template <typename... Args> struct CtxTrait<HybridCtx<Args...>> {
private:
  static const HybridCtx<Args...> initCtx;
  static const HybridCtx<Args...> globCtx;
  static std::unordered_set<HybridCtx<Args...>> ctxSet;

public:
  static const HybridCtx<Args...> *
  contextEvolve(const HybridCtx<Args...> *prevCtx, const llvm::Instruction *I) {
    auto result = ctxSet.emplace(prevCtx, I);
    return &*result.first;
  }

  static const HybridCtx<Args...> *getInitialCtx() { return &initCtx; }
  static const HybridCtx<Args...> *getGlobalCtx() { return &globCtx; }

  static std::string toString(const HybridCtx<Args...> *context,
                              bool detailed = false) {
    if (context == &globCtx)
      return "<global>";
    if (context == &initCtx)
      return "<empty>";

    return "no support yet"; // context->toString(detailed);
  }

  static void release() { ctxSet.clear(); }
};

template <typename... Args>
const HybridCtx<Args...> CtxTrait<HybridCtx<Args...>>::initCtx{
    CtxTrait<Args>::getInitialCtx()...};

template <typename... Args>
const HybridCtx<Args...> CtxTrait<HybridCtx<Args...>>::globCtx{
    CtxTrait<Args>::getGlobalCtx()...};

template <typename... Args>
std::unordered_set<HybridCtx<Args...>> CtxTrait<HybridCtx<Args...>>::ctxSet{};

} // namespace xray

namespace std {

// only hash context and value
template <typename... Args> struct hash<xray::HybridCtx<Args...>> {
  template <size_t... N>
  size_t hash_tuple(const xray::HybridCtx<Args...> &wrapper,
                    std::index_sequence<N...> sequence) const {
    llvm::hash_code code =
        llvm::hash_combine(((const void *)std::get<N>(wrapper.ctx))...);
    return hash_value(code);
  }

  size_t operator()(const xray::HybridCtx<Args...> &wrapper) const {
    return this->hash_tuple(wrapper, std::index_sequence_for<Args...>{});
  }
};

} // namespace std
#endif // ASER_PTA_HYBRIDCTX_H
