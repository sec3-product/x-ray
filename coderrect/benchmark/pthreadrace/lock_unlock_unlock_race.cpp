// @purpose lock/unlock and unlock may deceit our logic
// @dataRaces 1

#include <iostream>
#include <pthread.h>


pthread_spinlock_t lock;
int x;


void *worker(void*) {
    pthread_spin_lock(&lock);
    x = rand();
    std::cout << x << "\n";
    pthread_spin_unlock(&lock);
    x = rand();
    std::cout << x << "\n";
    pthread_spin_unlock(&lock);

    return nullptr;
}


int main() {
    pthread_t th1, th2;

    pthread_spin_init(&lock, 0);
    pthread_create(&th1, nullptr, worker, nullptr);
    pthread_create(&th2, nullptr, worker, nullptr);
    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);
}