#ifndef RACEDETECTOR_LOCKGRAPH_H
#define RACEDETECTOR_LOCKGRAPH_H

#include <list>
#include <map>

#include "Event.h"

using namespace std;

namespace aser {
namespace lock {

// trie node
class LockNode {
public:
    const ObjTy *key;                             // the lock object
    map<TID, list<const LockEvent *>> eventsMap;  // the event

    LockNode(const ObjTy *key) : key(key){};

    void addEvent(const LockEvent *e) {
        TID tid = e->getTID();
        eventsMap[tid].push_back(e);
    }
};

class LockGraph {
    int V;           // number of nodes
    list<int> *adj;  // a pointer to other linked nodes
    map<int, LockNode *> idNodeMap;
    map<const ObjTy *, int> keyIdMap;
    //TODO: to avoid different ObjTy* of the same lock value
    //map<const llvm::Value *, int> keyIdMap;

    set<list<LockNode *>> graphCycles;

public:
    LockGraph(std::set<const ObjTy *> *uniqueLocks) {
        int size = uniqueLocks->size();
        V = size;
        // FIXME: memory leak
        adj = new list<int>[V];
        for (auto key : *uniqueLocks) {
            int id = keyIdMap.size();
            keyIdMap[key] = id;
            // FIXME: memory leak
            LockNode *pNode = new LockNode(key);
            idNodeMap[id] = pNode;
        }
    };

    // given an abstract object (shared lock)
    // get it's corresponding ID if we already seen it
    // or return a new ID for it
    int getNodeId(const ObjTy *key) {
        if (keyIdMap.find(key) != keyIdMap.end()) {
            return keyIdMap[key];
        } else {
            // ideally, this branch would never be executed
            LOG_DEBUG("[WARN] this branch should never be executed!");
            int id = keyIdMap.size();
            keyIdMap[key] = id;
            // FIXME: memory leak
            LockNode *pNode = new LockNode(key);
            idNodeMap[id] = pNode;
            return id;
        }
    }

    const ObjTy *getNodeKey(int id) { return idNodeMap[id]->key; }

    void addNodeEvent(int id, const LockEvent *e) {
        if (idNodeMap.find(id) != idNodeMap.end()) {
            LockNode *pNode = idNodeMap[id];
            pNode->addEvent(e);
        } else {
            assert(false && "we should not reach this point");
        }
    }

    void addEdge(int v, int w) {
        adj[v].push_back(w);  // Add w to v’s list.
    }

    // A Recursive DFS based function used by SCC()
    // A recursive function that finds and prints strongly connected
    // components using DFS traversal
    // u --> The vertex to be visited next
    // disc[] --> Stores discovery times of visited vertices
    // low[] -- >> earliest visited vertex (the vertex with minimum
    //             discovery time) that can be reached from subtree
    //             rooted with current vertex
    // *st -- >> To store all the connected ancestors (could be part
    //           of SCC)
    // stackMember[] --> bit/index array for faster check whether
    //                  a node is in stack
    void SCCUtil(int u, int disc[], int low[], stack<int> *st, bool stackMember[]) {
        // A static variable is used for simplicity, we can avoid use
        // of static variable by passing a pointer.
        static int time = 0;

        // Initialize discovery time and low value
        disc[u] = low[u] = ++time;
        st->push(u);
        stackMember[u] = true;

        // Go through all vertices adjacent to this
        list<int>::iterator i;
        for (i = adj[u].begin(); i != adj[u].end(); ++i) {
            int v = *i;  // v is current adjacent of 'u'

            // If v is not visited yet, then recur for it
            if (disc[v] == -1) {
                SCCUtil(v, disc, low, st, stackMember);

                // Check if the subtree rooted with 'v' has a
                // connection to one of the ancestors of 'u'
                // Case 1 (per above discussion on Disc and Low value)
                low[u] = min(low[u], low[v]);
            }

            // Update low value of 'u' only of 'v' is still in stack
            // (i.e. it's a back edge, not cross edge).
            // Case 2 (per above discussion on Disc and Low value)
            else if (stackMember[v] == true)
                low[u] = min(low[u], disc[v]);
        }

        // head node found, pop the stack and print an SCC
        int w = 0;  // To store stack extracted vertices
        if (low[u] == disc[u]) {
            list<LockNode *> cycle;
            while (st->top() != u) {
                w = (int)st->top();
                // llvm::outs() << w << " ";

                // get node and insert at the beginning
                auto n = idNodeMap.at(w);
                cycle.push_front(n);

                stackMember[w] = false;
                st->pop();
            }
            w = (int)st->top();
            // llvm::outs() << w << "\n";

            // get node and insert at the beginning
            auto n = idNodeMap.at(w);
            cycle.push_front(n);
            // add the list to result
            if (cycle.size() > 1) graphCycles.insert(cycle);

            stackMember[w] = false;
            st->pop();
        }
    }

#define NIL -1
    // The function to do DFS traversal. It uses SCCUtil()
    void detectCycles() {
        // FIXME: memory leak
        int *disc = new int[V];
        int *low = new int[V];
        bool *stackMember = new bool[V];
        stack<int> *st = new stack<int>();

        // Initialize disc and low, and stackMember arrays
        for (int i = 0; i < V; i++) {
            disc[i] = NIL;
            low[i] = NIL;
            stackMember[i] = false;
        }

        // Call the recursive helper function to find strongly
        // connected components in DFS tree with vertex 'i'
        for (int i = 0; i < V; i++)
            if (disc[i] == NIL) SCCUtil(i, disc, low, st, stackMember);
    }

    bool isCyclic() {
        if (graphCycles.size() > 0)
            return true;
        else
            return false;
    }
    set<list<LockNode *>> getCycles() { return graphCycles; }
};

}  // namespace lock
}  // namespace aser
#endif