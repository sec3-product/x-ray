//===- ScopedHashTableX.h - A simple scoped hash table -----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an efficient scoped hash table, which is useful for
// things like dominator-based optimizations.  This allows clients to do things
// like this:
//
//  ScopedHashTableX<int, int> HT;
//  {
//    ScopedHashTableXScope<int, int> Scope1(HT);
//    HT.insert(0, 0);
//    HT.insert(1, 1);
//    {
//      ScopedHashTableXScope<int, int> Scope2(HT);
//      HT.insert(0, 42);
//    }
//  }
//
// Looking up the value for "0" in the Scope2 block will return 42.  Looking
// up the value for 0 before 42 is inserted or after Scope2 is popped will
// return 0.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ScopedHashTableX_H
#define LLVM_ADT_ScopedHashTableX_H

#include <cassert>
#include <new>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/AllocatorBase.h"

namespace llvm {

template <typename K, typename V, typename KInfo = DenseMapInfo<K>,
          typename AllocatorTy = MallocAllocator>
class ScopedHashTableX;

template <typename K, typename V>
class ScopedHashTableXVal {
  ScopedHashTableXVal *NextInScope;
  ScopedHashTableXVal *NextForKey;
  K Key;
  V Val;

  ScopedHashTableXVal(const K &key, const V &val) : Key(key), Val(val) {}

 public:
  const K &getKey() const { return Key; }
  const V &getValue() const { return Val; }
  V &getValue() { return Val; }

  ScopedHashTableXVal *getNextForKey() { return NextForKey; }
  const ScopedHashTableXVal *getNextForKey() const { return NextForKey; }
  ScopedHashTableXVal *getNextInScope() { return NextInScope; }

  template <typename AllocatorTy>
  static ScopedHashTableXVal *Create(ScopedHashTableXVal *nextInScope,
                                     ScopedHashTableXVal *nextForKey,
                                     const K &key, const V &val,
                                     AllocatorTy &Allocator) {
    ScopedHashTableXVal *New =
        Allocator.template Allocate<ScopedHashTableXVal>();
    // Set up the value.
    new (New) ScopedHashTableXVal(key, val);
    New->NextInScope = nextInScope;
    New->NextForKey = nextForKey;
    return New;
  }

  template <typename AllocatorTy>
  void Destroy(AllocatorTy &Allocator) {
    // Free memory referenced by the item.
    this->~ScopedHashTableXVal();
    Allocator.Deallocate(this);
  }
};

template <typename K, typename V, typename KInfo = DenseMapInfo<K>,
          typename AllocatorTy = MallocAllocator>
class ScopedHashTableXScope {
  /// HT - The hashtable that we are active for.
  ScopedHashTableX<K, V, KInfo, AllocatorTy> &HT;

  /// PrevScope - This is the scope that we are shadowing in HT.
  ScopedHashTableXScope *PrevScope;

  /// LastValInScope - This is the last value that was inserted for this scope
  /// or null if none have been inserted yet.
  ScopedHashTableXVal<K, V> *LastValInScope;

 public:
  ScopedHashTableXScope(ScopedHashTableX<K, V, KInfo, AllocatorTy> &HT);
  ScopedHashTableXScope(ScopedHashTableXScope &) = delete;
  ScopedHashTableXScope &operator=(ScopedHashTableXScope &) = delete;
  ~ScopedHashTableXScope();

  ScopedHashTableXScope *getParentScope() { return PrevScope; }
  const ScopedHashTableXScope *getParentScope() const { return PrevScope; }

 private:
  friend class ScopedHashTableX<K, V, KInfo, AllocatorTy>;

  ScopedHashTableXVal<K, V> *getLastValInScope() { return LastValInScope; }

  void setLastValInScope(ScopedHashTableXVal<K, V> *Val) {
    LastValInScope = Val;
  }
};

template <typename K, typename V, typename KInfo = DenseMapInfo<K>>
class ScopedHashTableXIterator {
  ScopedHashTableXVal<K, V> *Node;

 public:
  ScopedHashTableXIterator(ScopedHashTableXVal<K, V> *node) : Node(node) {}

  V &operator*() const {
    assert(Node && "Dereference end()");
    return Node->getValue();
  }
  V *operator->() const { return &Node->getValue(); }

  bool operator==(const ScopedHashTableXIterator &RHS) const {
    return Node == RHS.Node;
  }
  bool operator!=(const ScopedHashTableXIterator &RHS) const {
    return Node != RHS.Node;
  }

  inline ScopedHashTableXIterator &operator++() {  // Preincrement
    assert(Node && "incrementing past end()");
    Node = Node->getNextForKey();
    return *this;
  }
  ScopedHashTableXIterator operator++(int) {  // Postincrement
    ScopedHashTableXIterator tmp = *this;
    ++*this;
    return tmp;
  }
};

template <typename K, typename V, typename KInfo, typename AllocatorTy>
class ScopedHashTableX {
 public:
  /// ScopeTy - This is a helpful typedef that allows clients to get easy access
  /// to the name of the scope for this hash table.
  using ScopeTy = ScopedHashTableXScope<K, V, KInfo, AllocatorTy>;
  using size_type = unsigned;

 private:
  friend class ScopedHashTableXScope<K, V, KInfo, AllocatorTy>;

  using ValTy = ScopedHashTableXVal<K, V>;

  DenseMap<K, ValTy *, KInfo> TopLevelMap;
  ScopeTy *CurScope = nullptr;

  AllocatorTy Allocator;

 public:
  ScopedHashTableX() = default;
  ScopedHashTableX(AllocatorTy A) : Allocator(A) {}
  ScopedHashTableX(const ScopedHashTableX &) = delete;
  ScopedHashTableX &operator=(const ScopedHashTableX &) = delete;

  ~ScopedHashTableX() {
    // assert(!CurScope && TopLevelMap.empty() && "Scope imbalance!");
    TopLevelMap.clear();
  }

  /// Access to the allocator.
  AllocatorTy &getAllocator() { return Allocator; }
  const AllocatorTy &getAllocator() const { return Allocator; }

  /// Return 1 if the specified key is in the table, 0 otherwise.
  size_type count(const K &Key) const { return TopLevelMap.count(Key); }

  V lookup(const K &Key) const {
    auto I = TopLevelMap.find(Key);
    if (I != TopLevelMap.end()) return I->second->getValue();

    // llvm::outs() << "-----lookup failure\n";
    return V();
  }

  void insert(const K &Key, const V &Val) {
    insertIntoScope(CurScope, Key, Val);
  }

  void clear() { TopLevelMap.clear(); }

  using iterator = ScopedHashTableXIterator<K, V, KInfo>;

  iterator end() { return iterator(0); }

  iterator begin(const K &Key) {
    typename DenseMap<K, ValTy *, KInfo>::iterator I = TopLevelMap.find(Key);
    if (I == TopLevelMap.end()) return end();
    return iterator(I->second);
  }

  ScopeTy *getCurScope() { return CurScope; }
  const ScopeTy *getCurScope() const { return CurScope; }

  /// insertIntoScope - This inserts the specified key/value at the specified
  /// (possibly not the current) scope.  While it is ok to insert into a scope
  /// that isn't the current one, it isn't ok to insert *underneath* an existing
  /// value of the specified key.
  void insertIntoScope(ScopeTy *S, const K &Key, const V &Val) {
    assert(S && "No scope active!");
    ScopedHashTableXVal<K, V> *&KeyEntry = TopLevelMap[Key];
    KeyEntry =
        ValTy::Create(S->getLastValInScope(), KeyEntry, Key, Val, Allocator);
    S->setLastValInScope(KeyEntry);
  }
};

/// ScopedHashTableXScope ctor - Install this as the current scope for the hash
/// table.
template <typename K, typename V, typename KInfo, typename Allocator>
ScopedHashTableXScope<K, V, KInfo, Allocator>::ScopedHashTableXScope(
    ScopedHashTableX<K, V, KInfo, Allocator> &ht)
    : HT(ht) {
  PrevScope = HT.CurScope;
  HT.CurScope = this;
  LastValInScope = nullptr;

  llvm::outs() << "-----enter scope\n";
}

template <typename K, typename V, typename KInfo, typename Allocator>
ScopedHashTableXScope<K, V, KInfo, Allocator>::~ScopedHashTableXScope() {
  assert(HT.CurScope == this && "Scope imbalance!");
  HT.CurScope = PrevScope;

  // Pop and delete all values corresponding to this scope.
  while (ScopedHashTableXVal<K, V> *ThisEntry = LastValInScope) {
    // Pop this value out of the TopLevelMap.
    if (!ThisEntry->getNextForKey()) {
      //   assert(HT.TopLevelMap[ThisEntry->getKey()] == ThisEntry &&
      //          "Scope imbalance!");
      HT.TopLevelMap.erase(ThisEntry->getKey());
    } else {
      ScopedHashTableXVal<K, V> *&KeyEntry =
          HT.TopLevelMap[ThisEntry->getKey()];
      // assert(KeyEntry == ThisEntry && "Scope imbalance!");
      KeyEntry = ThisEntry->getNextForKey();
    }

    // Pop this value out of the scope.
    LastValInScope = ThisEntry->getNextInScope();

    // Delete this entry.
    ThisEntry->Destroy(HT.getAllocator());
  }
  llvm::outs() << "-----leave scope\n";
}

}  // end namespace llvm

#endif  // LLVM_ADT_ScopedHashTableX_H
