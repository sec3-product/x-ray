// Expect OV on local. nullptr assigned at line 12 and dereferenced at line 20

#include <pthread.h>

#include <iostream>

pthread_mutex_t lock;

void *worker1(void *arg) {
    pthread_mutex_lock(&lock);
    int **localptr = (int **)arg;
    *localptr = nullptr;
    std::cout << localptr;
    pthread_mutex_unlock(&lock);
}

void *worker2(void *arg) {
    pthread_mutex_lock(&lock);
    int *local = (int *)arg;
    std::cout << *local << "\n";
    pthread_mutex_unlock(&lock);
}

int main() {
    pthread_t th1, th2;

    int *local = new int();

    pthread_mutex_init(&lock, nullptr);

    pthread_create(&th1, nullptr, worker1, &local);
    pthread_create(&th2, nullptr, worker2, local);

    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);
}