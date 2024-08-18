//
// Created by peiming on 11/8/19.
//
#ifndef ASER_PTA_ITERATORS_H
#define ASER_PTA_ITERATORS_H

#include <iterator>

#include <llvm/ADT/iterator.h>

namespace aser {

template <typename _Tp> struct const_pointer {
  typedef const typename std::remove_pointer<_Tp>::type *type;
};

template <typename WrapperIteratorT,
          typename ValueT = typename std::conditional<
              std::is_const<typename std::remove_pointer<
                  typename WrapperIteratorT::iterator_type>::type>::value,
              typename const_pointer<typename std::iterator_traits<
                  WrapperIteratorT>::value_type::pointer>::type,
              typename std::iterator_traits<
                  WrapperIteratorT>::value_type::pointer>::type>
class UniquePtrIterator
    : public llvm::iterator_adaptor_base<
          UniquePtrIterator<WrapperIteratorT>, WrapperIteratorT,
          typename std::iterator_traits<WrapperIteratorT>::iterator_category,
          ValueT> {
private:
  using BaseT = llvm::iterator_adaptor_base<
      UniquePtrIterator<WrapperIteratorT>, WrapperIteratorT,
      typename std::iterator_traits<WrapperIteratorT>::iterator_category,
      ValueT>;

public:
  UniquePtrIterator() = default;

  explicit UniquePtrIterator(const WrapperIteratorT &i) : BaseT(i) {}
  explicit UniquePtrIterator(WrapperIteratorT &&i) : BaseT(std::move(i)) {}

  ValueT operator*() const { return this->I->get(); }
};

template <typename WrapperIteratorT,
          typename ValueT = typename std::iterator_traits<
              WrapperIteratorT>::value_type::second_type>
class PairSecondIterator
    : public llvm::iterator_adaptor_base<
          PairSecondIterator<WrapperIteratorT>, WrapperIteratorT,
          typename std::iterator_traits<WrapperIteratorT>::iterator_category,
          ValueT> {
private:
  using BaseT = llvm::iterator_adaptor_base<
      PairSecondIterator<WrapperIteratorT>, WrapperIteratorT,
      typename std::iterator_traits<WrapperIteratorT>::iterator_category,
      ValueT>;

public:
  PairSecondIterator() = default;

  explicit PairSecondIterator(const WrapperIteratorT &i) : BaseT(i) {}
  explicit PairSecondIterator(WrapperIteratorT &&i) : BaseT(std::move(i)) {}

  ValueT operator*() const { return this->I->second; }
};

// TODO: this is a bad implementation, just use llvm::concat_iterator
// iterator that concat N iterator together
template <typename Wrapped, int N,
          typename ValueT = typename std::iterator_traits<Wrapped>::value_type>
struct ConcatIterator : public ConcatIterator<Wrapped, N - 1, ValueT> {
  using super = ConcatIterator<Wrapped, N - 1, ValueT>;
  using self = ConcatIterator<Wrapped, N, ValueT>;

  // using ValueT = typename std::iterator_traits<Wrapped>::value_type;
  using ReferenceT = ValueT &;

  Wrapped cur;
  Wrapped end;

  template <typename... Args>
  ConcatIterator(Wrapped cur, Wrapped end, Args &&...args)
      : super(std::forward<Args>(args)...), cur(cur), end(end) {}

  inline self &operator++() {
    if (cur == end) {
      static_cast<super *>(this)->operator++();
    } else {
      cur++;
    }
    return *this;
  };

  self operator++(int) {
    self tmp = *static_cast<self *>(this);
    if (cur == end) {
      ++*static_cast<super *>(this);
    } else {
      ++cur;
    }
    return tmp;
  }

  auto operator*() -> decltype(*cur) const {
    if (cur == end) {
      return static_cast<const super *>(this)->operator*();
    } else {
      return *cur;
    }
  }

  inline bool operator!=(const self &rhs) const {
    return !this->operator==(rhs);
  }

  inline bool operator==(const self &rhs) const {
    return cur == rhs.cur &&
           ((const super *)this)->operator==((const super &)rhs);
  }

  //    auto operator-> () -> decltype(cur.operator->()) {
  //        if (cur == end) {
  //            return static_cast<super *>(this)->operator->();
  //        } else {
  //            return cur.operator->();
  //        }
  //    }
};

template <typename Wrapped, typename ValueT>
struct ConcatIterator<Wrapped, 1, ValueT> {
  using iterator_category = std::forward_iterator_tag;
  using value_type = ValueT;
  using difference_type = std::ptrdiff_t;
  using pointer = ValueT *;
  using reference = ValueT &;

  using self = ConcatIterator<Wrapped, 1, ValueT>;
  using ReferenceT = ValueT &;

  Wrapped cur;
  Wrapped end;

  ConcatIterator(Wrapped cur, Wrapped end) : cur(cur), end(end) {}

  inline self &operator++() {
    cur++;
    return *this;
  };

  self operator++(int) {
    self tmp = *static_cast<self *>(this);
    ++cur;
    return tmp;
  }

  inline ReferenceT operator*() const { return *cur; }

  inline bool operator!=(const self &rhs) const {
    return !this->operator==(rhs);
  }
  inline bool operator==(const self &rhs) const { return cur == rhs.cur; }
};

// ASSUMPTION: E (a enum) and N are convertible
template <typename Wrapped, int N, typename E,
          typename ValueT = typename std::iterator_traits<Wrapped>::value_type>
struct ConcatIteratorWithTag
    : public ConcatIteratorWithTag<Wrapped, N - 1, E, ValueT> {
  using super = ConcatIteratorWithTag<Wrapped, N - 1, E, ValueT>;
  using self = ConcatIteratorWithTag<Wrapped, N, E, ValueT>;

  using ReferenceT = std::pair<E, ValueT>;

  Wrapped cur;
  Wrapped end;

  template <typename... Args>
  ConcatIteratorWithTag(Wrapped cur, Wrapped end, Args &&...args)
      : super(std::forward<Args>(args)...), cur(cur), end(end) {}

  inline self &operator++() {
    if (cur == end) {
      static_cast<super *>(this)->operator++();
    } else {
      cur++;
    }
    return *this;
  };

  self operator++(int) {
    self tmp = *static_cast<self *>(this);
    if (cur == end) {
      ++*static_cast<super *>(this);
    } else {
      ++cur;
    }
    return tmp;
  }

  ReferenceT operator*() const {
    if (cur == end) {
      return static_cast<const super *>(this)->operator*();
    } else {
      return std::make_pair(static_cast<E>(N - 1), *cur);
    }
  }

  inline bool operator!=(const self &rhs) const {
    return !this->operator==(rhs);
  }

  inline bool operator==(const self &rhs) const {
    return cur == rhs.cur &&
           static_cast<const super *>(this)->operator==((const super &)rhs);
  }
};

template <typename Wrapped, typename E, typename ValueT>
struct ConcatIteratorWithTag<Wrapped, 1, E, ValueT> {
  using iterator_category = std::forward_iterator_tag;
  using value_type = const std::pair<E, ValueT>;
  using difference_type = std::ptrdiff_t;
  using pointer = const std::pair<E, ValueT> *;
  using reference = const std::pair<E, ValueT> &;

  using self = ConcatIteratorWithTag<Wrapped, 1, E, ValueT>;
  using ReferenceT = std::pair<E, ValueT>;

  Wrapped cur;
  Wrapped end;

  ConcatIteratorWithTag(Wrapped cur, Wrapped end) : cur(cur), end(end) {}

  inline self &operator++() {
    cur++;
    return *this;
  };

  self operator++(int) const {
    self tmp = *static_cast<const self *>(this);
    ++cur;
    return tmp;
  }

  inline ReferenceT operator*() const {
    return std::make_pair(static_cast<E>(0), *cur);
  }

  inline bool operator!=(const self &rhs) const {
    return !this->operator==(rhs);
  }

  inline bool operator==(const self &rhs) const { return cur == rhs.cur; }
};

template <typename NodeIDIteratorT, typename GraphT,
          typename ValueT = typename std::conditional<
              std::is_const<GraphT>::value,
              /*if true*/ typename GraphT::ConstNodePtrT,
              /*if false*/ typename GraphT::NodePtrT>::type>
class NodeIDWrapperIterator
    : public llvm::iterator_adaptor_base<
          NodeIDWrapperIterator<NodeIDIteratorT, GraphT, ValueT>,
          NodeIDIteratorT,
          typename std::iterator_traits<NodeIDIteratorT>::iterator_category,
          ValueT> {
private:
  using BaseT = llvm::iterator_adaptor_base<
      NodeIDWrapperIterator<NodeIDIteratorT, GraphT>, NodeIDIteratorT,
      typename std::iterator_traits<NodeIDIteratorT>::iterator_category,
      ValueT>;

  GraphT *G;

public:
  explicit NodeIDWrapperIterator(GraphT *G) : G(G) {}

  explicit NodeIDWrapperIterator(GraphT *G, const NodeIDIteratorT &i)
      : G(G), BaseT(i) {}
  explicit NodeIDWrapperIterator(GraphT *G, NodeIDIteratorT &&i)
      : G(G), BaseT(std::move(i)) {}

  ValueT operator*() const {
    NodeID id = *(this->I);
    return G->getNode(id);
  }
};

template <
    typename NodeIDEdgeIteratorT, typename GraphT,
    typename ValueT = typename std::conditional<
        std::is_const<GraphT>::value,
        /*if true*/
        std::pair<typename GraphT::EdgeT, typename GraphT::ConstNodePtrT>,
        /*if false*/
        std::pair<typename GraphT::EdgeT, typename GraphT::NodePtrT>>::type>
class NodeIDWrapperEdgeIterator
    : public llvm::iterator_adaptor_base<
          NodeIDWrapperEdgeIterator<NodeIDEdgeIteratorT, GraphT, ValueT>,
          NodeIDEdgeIteratorT,
          typename std::iterator_traits<NodeIDEdgeIteratorT>::iterator_category,
          ValueT> {
private:
  using BaseT = llvm::iterator_adaptor_base<
      NodeIDWrapperEdgeIterator<NodeIDEdgeIteratorT, GraphT>,
      NodeIDEdgeIteratorT,
      typename std::iterator_traits<NodeIDEdgeIteratorT>::iterator_category,
      ValueT>;

  GraphT *G;

public:
  explicit NodeIDWrapperEdgeIterator() = default;

  explicit NodeIDWrapperEdgeIterator(GraphT *G, const NodeIDEdgeIteratorT &i)
      : G(G), BaseT(i) {}
  explicit NodeIDWrapperEdgeIterator(GraphT *G, NodeIDEdgeIteratorT &&i)
      : G(G), BaseT(std::move(i)) {}

  ValueT operator*() const {
    NodeID id = (*(this->I)).second;
    return std::make_pair((*(this->I)).first, G->getNode(id));
  }
};

} // namespace aser

#endif
