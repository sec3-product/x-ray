// @purpose a complex case of callback, ownership transfer, and state-machine
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <pthread.h>


struct Task {
    int a;
    int b;
    int result;

    Task* next;

    void (*cb)(void *);
    void* user_data;
};


/**
 * A task queue protected by lock
 */
static struct Task *head;
static pthread_mutex_t lock;


static void *WorkerThread(void *unused) {
    while (true) {
        pthread_mutex_lock(&lock);
        Task* curr = head;
        if (curr != nullptr) {
            head = curr->next;
        }
        pthread_mutex_unlock(&lock);

        curr->result = curr->a + curr->b;

        curr->cb(curr->user_data);
    }
}


struct ControlBlock {
    bool done;
    bool submitted;
    pthread_mutex_t lock;
};


static void callback_func(void *data) {
    struct ControlBlock *ctlblk = (struct ControlBlock*)data;
    pthread_mutex_lock(&ctlblk->lock);
    ctlblk->done = true;
    pthread_mutex_unlock(&ctlblk->lock);
}


static void *StudentThread(void *unused) {
    struct ControlBlock cblk;
    struct Task task;

    task.user_data = &cblk;
    task.cb = callback_func;

    pthread_mutex_init(&cblk.lock, nullptr);
    cblk.done = false;
    cblk.submitted = false;

    while (true) {
        pthread_mutex_lock(&cblk.lock);

        if (!cblk.submitted) {
            task.a = rand();
            task.b = rand();
            task.next = nullptr;
            cblk.done = false;

            pthread_mutex_lock(&lock);
            task.next = head;
            head = &task;
            cblk.submitted = true;
            pthread_mutex_unlock(&lock);
        }
        else if (cblk.submitted && cblk.done) {
            std::cout << task.result;
            cblk.submitted = false;
        }

        pthread_mutex_unlock(&cblk.lock);
    }
}


int main(int argc, char**argv) {
    pthread_t thr_worker, thr_student;

    pthread_mutex_init(&lock, nullptr);
    pthread_create(&thr_worker, nullptr, WorkerThread, nullptr);
    pthread_create(&thr_student, nullptr, StudentThread, nullptr);

    pthread_join(thr_student, nullptr);
}