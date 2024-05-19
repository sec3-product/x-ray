// Expect no report. Nullptr assigned at line 12 and dereferenced at line 20 only if not null.

#include <pthread.h>

#include <iostream>

int *global;
pthread_mutex_t lock;

void *worker1(void *arg) {
    pthread_mutex_lock(&lock);
    global = nullptr;
    std::cout << (global);
    pthread_mutex_unlock(&lock);
}

void *worker2(void *arg) {
    pthread_mutex_lock(&lock);
    if (global != nullptr) {
        std::cout << *global << "\n";
    }
    pthread_mutex_unlock(&lock);
}

int main() {
    pthread_t th1, th2, th3, th4;

    global = new int(0);

    pthread_mutex_init(&lock, nullptr);

    pthread_create(&th1, nullptr, worker1, nullptr);
    pthread_create(&th2, nullptr, worker2, nullptr);

    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);
}