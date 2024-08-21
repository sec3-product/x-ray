#ifndef RACEDETECTOR_TRIE_H
#define RACEDETECTOR_TRIE_H

#include <llvm/IR/Instruction.h>

#include <map>

extern bool USE_MAIN_CALLSTACK_HEURISTIC;
extern int SAME_FUNC_BUDGET_SIZE; // keep at most x times per func per thread 10
                                  // by default

// TODO: this file should be re-implemented to be more generic
// Right now this file is a incomplete implementation of Trie
// to record the stack-trace the race detector has traversed.
// And also prevent the race detector from traversing the same
// stack trace too many times.
// The Trie data structure can potentially be used in many other places
// we can later implentent a better version
namespace xray {
namespace trie {

// keep at most x times of the same callees
//-- this can be unsound for frequently used wrapper funcs such as STATS_LOCK
static const unsigned int CALL_STACK_BUDGET_SIZE = 2;

static std::map<const llvm::Function *, unsigned int> countSameFuncNodes;
// trie node
struct TrieNode {
  TrieNode *parent;
  std::map<const llvm::Function *, TrieNode *> children;
  const llvm::Function *key;
  uint8_t numberOfTimes;
};

static struct TrieNode *createNewNode(TrieNode *parent,
                                      const llvm::Function *key) {
  // FIXME: memory leak
  struct TrieNode *pNode = new TrieNode;
  pNode->key = key; // likely main or thread start func
  pNode->parent = parent;
  pNode->numberOfTimes = 0; // start from 0

  countSameFuncNodes[key]++; // increment by one
  return pNode;
}

// Returns new trie node
static struct TrieNode *getNode(TrieNode *parent, const llvm::Function *key) {
  if (parent == nullptr) {
    return createNewNode(nullptr, nullptr);
  } else {
    TrieNode *n = (parent->children)[key];
    if (!n) {
      n = createNewNode(parent, key);
      (parent->children)[key] = n;
    }
    n->numberOfTimes++;
    return n;
  }
}

static bool hasExceededBudget(TrieNode *n) {
  if (n && n->numberOfTimes >= CALL_STACK_BUDGET_SIZE) {
    return true; // exceeded budget
  }

  return false;
}

static bool willExceedBudget(TrieNode *parent, const llvm::Function *func) {
  if (SAME_FUNC_BUDGET_SIZE != -1 &&
      countSameFuncNodes[func] >= SAME_FUNC_BUDGET_SIZE)
    return true;
  if (parent) {
    if ((parent->children).find(func) != parent->children.end()) {
      TrieNode *n = (parent->children)[func];
      if (n->numberOfTimes >= CALL_STACK_BUDGET_SIZE)
        return true;

      // a special treatment for the main call stack
      if (USE_MAIN_CALLSTACK_HEURISTIC)
        if (parent->key->getName() == "main") {
          LOG_DEBUG("skipped repeated func from main. func={}",
                    demangle(func->getName().str()));
          // llvm::outs() << "skipping repeated func: " <<
          // demangle(func->getName()) << " from main thread\n\n";
          return true;
        }
    }
  }

  return false;
}

static void cleanTrie(TrieNode *root) {
  // TODO: free the memory allocated for this trie

  // reset countSameFuncNodes for thread
  countSameFuncNodes.clear();
}

} // namespace trie
} // namespace xray
#endif