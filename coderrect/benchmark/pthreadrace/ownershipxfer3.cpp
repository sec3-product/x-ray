// @purpose producer transfers the ownership of task to the conusmer via a queue, and then access the task
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <unistd.h>
#include <stdlib.h>


/**
 * TaskNode builds a double-linked list
 */
struct TaskNode {
    TaskNode *next;
    TaskNode *prev;
    int data;
};


static pthread_mutex_t queue_mutex;
static TaskNode *head = nullptr;
static TaskNode *tail = nullptr;


static void *ProducerThreadMain(void *unused) {
    while (true) {
        sleep(5);

        TaskNode *task = new TaskNode();
        task->data = rand();

        pthread_mutex_lock(&queue_mutex);
        if (head == nullptr) {
            head = task;
            tail = task;
            task->next = nullptr;
            task->prev = nullptr;
        }
        else {
            tail->next = task;
            task->prev = tail;
            task->next = nullptr;
            tail = task;
        }

        std::cout << "enqueue a new task " << task->data << "\n";
        pthread_mutex_unlock(&queue_mutex);
        task->data = rand();
    }
}


static void *ConsumerThreadMain(void *unused) {
    while (true) {
        sleep(5);

        TaskNode *task = nullptr;
        pthread_mutex_lock(&queue_mutex);
        if (head == nullptr) {
            // no tasks
            pthread_mutex_unlock(&queue_mutex);
            continue;
        }

        task = head;
        head = nullptr;
        tail = nullptr;
        pthread_mutex_unlock(&queue_mutex);

        // now process task one by one.
        // we don't need to hold the lock because
        // we already dequeue tasks and no other consumers and producers
        // can access them anymore.
        while (task != nullptr) {
            std::cout << "process a task " << task->data << "\n";
            auto tmp = task;
            task = tmp->next;
            delete tmp;
        }
    }
}


int main(int argc, char*argv[]) {
    pthread_t thr_producer, thr_consumer;
    pthread_mutex_init(&queue_mutex, nullptr);

    pthread_create(&thr_producer, nullptr, ProducerThreadMain, nullptr);
    pthread_create(&thr_consumer, nullptr, ConsumerThreadMain, nullptr);

    pthread_join(thr_producer, nullptr);
    pthread_join(thr_consumer, nullptr);
}
