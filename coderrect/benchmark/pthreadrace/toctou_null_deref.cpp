// @purpose toctou when worker1 intlereaves worker2
// @toctou 1

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
    if (global != nullptr) {
        // TOCTOU if worker1 interleaves here
        pthread_mutex_lock(&lock);
        std::cout << *global << "\n";
        pthread_mutex_unlock(&lock);
    }
}

int main() {
    pthread_t th1, th2;

    global = new int(0);

    pthread_mutex_init(&lock, nullptr);

    pthread_create(&th1, nullptr, worker1, nullptr);
    pthread_create(&th2, nullptr, worker2, nullptr);

    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);
}