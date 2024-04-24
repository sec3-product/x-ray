// @purpose toctou based on description of security vulnerability in windows **and contains no data race**: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2003-0813
// @toctou 1

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

/* TOCTOU like this:

T1                                                        T2
node = 0x1111111                                          node = 0x1111111
if (node->work->processing) == false
                                                          if (node->work->processing) == false
node->work->processing = true;
...
delete node
                                                           node->work->processing = true;
                                                           ...
                                                           delete node


Both threads acquire the same node because of TOCTOU.
Leads to double delete.
Note the double delete is mentioned in the CVE but not required for this to be TOCTOU

*/

struct Work {
    std::atomic<bool> processing;
    int data;
    Work() : processing(false), data(0) {}
};

struct WorkNode {
    Work* work;
    WorkNode* next;
    WorkNode* prev;

    WorkNode() : next(nullptr), prev(nullptr), work(new Work()) {}
    WorkNode(WorkNode* prev) : prev(prev), next(nullptr), work(new Work()) { prev->next = this; }
    ~WorkNode() { delete work; }

    void unlink() {
        if (prev != nullptr) {
            prev->next = next;
        }

        if (next != nullptr) {
            next->prev = prev;
        }
    }
};

WorkNode* list;
pthread_mutex_t lock;
std::atomic<uint64_t> count;

void* processList(void*) {
    WorkNode* node = list;

    while (node != nullptr) {
        // find node that is not being processed
        if (node->work->processing.load()) {
            pthread_mutex_lock(&lock);
            node = node->next;
            pthread_mutex_unlock(&lock);
            continue;
        }
        // sleep added to force race
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Mark that we are processing this node
        node->work->processing.store(true);

        // Simulate processing of node
        count++;

        // Unlink node from list
        pthread_mutex_lock(&lock);
        node->unlink();
        pthread_mutex_unlock(&lock);

        // In original windows bug this led to a double delete
        // delete node;
    }

    return nullptr;
}

int main() {
    pthread_mutex_init(&lock, nullptr);

    const int size = 100;
    list = new WorkNode();
    auto tail = list;
    for (int i = 1; i < size; i++) {
        auto node = new WorkNode(tail);
        tail = node;
    }
    count.store(0);

    pthread_t t1, t2;
    pthread_create(&t1, nullptr, processList, nullptr);
    pthread_create(&t2, nullptr, processList, nullptr);

    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    std::cout << "Expected " << size << " got " << count.load() << "\n";
}