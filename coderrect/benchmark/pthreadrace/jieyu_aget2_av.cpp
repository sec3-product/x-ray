// @purpose a double-linked list show atomicity violation
// @orderViolations 1
// @seealso https://github.com/jieyu/concurrency-bugs/blob/master/apache-21285/DESCRIPTION

#include <iostream>
#include <pthread.h>


struct Node {
    Node *prev;
    Node *next;

    int x;
};


static struct Node *head = nullptr;
static pthread_mutex_t lock;


void *PurgeThread(void *arg) {
    pthread_mutex_lock(&lock);

    struct Node *tmp = head;
    while (tmp != nullptr) {
        if (tmp->x > 5) {
            if (tmp == head) {
                head = tmp->next;
            }
            else {
                tmp->prev->next = tmp->next;
            }
            delete tmp;
        }
        else {
            tmp = tmp->next;
        }
    }

    pthread_mutex_unlock(&lock);
    return nullptr;
}


int main() {
    pthread_mutex_init(&lock, nullptr);
    pthread_t th;
    pthread_create(&th, nullptr, PurgeThread, nullptr);

    struct Node *aNode = new Node();
    aNode->x = rand();

    // enqueue
    pthread_mutex_lock(&lock);
    aNode->prev = nullptr;
    aNode->next = head;
    if (head != nullptr)
        head->prev = aNode;
    head = aNode;
    pthread_mutex_unlock(&lock);

    // ... do something else ...

    // dequeue
    pthread_mutex_lock(&lock);
    if (aNode->prev == nullptr) {
        if (aNode->next != nullptr) {
            aNode->next->prev = nullptr;
        }
        head = aNode->next;
    }
    else {
        aNode->prev->next = aNode->next;
        if (aNode->next != nullptr)
            aNode->next->prev = aNode->prev;
    }
    delete aNode;
    pthread_mutex_unlock(&lock);
}
