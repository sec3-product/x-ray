// @purpose support the homemade spinlock detection
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0
// @configuration homemade_spinlock.json

#include <iostream>
#include <pthread.h>
#include <atomic>


static uint32_t *buf;
std::atomic_flag locked = ATOMIC_FLAG_INIT;


static void homemade_spin_lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {}
}

static void homemade_spin_unlock() {
    locked.clear(std::memory_order_release);
}

static void *SumThread(void *unused) {
    while (true) {
        homemade_spin_lock();
        uint32_t sum = 0;
        for (int i = 0; i < 1024; i++) {
            sum += buf[i];
        }
        std::cout << sum << "\n";
        homemade_spin_unlock();
    }
}

static void *GenerateThread(void *unused) {
    while (true) {
        homemade_spin_lock();
        delete buf;
        buf = new uint32_t[1024];
        homemade_spin_unlock();
    }
}

int main(int argc, char*argv[]) {
    pthread_t thr_sum, thr_generate;

    buf = new uint32_t[1024];

    pthread_create(&thr_sum, nullptr, SumThread, nullptr);
    pthread_create(&thr_generate, nullptr, GenerateThread, nullptr);

    pthread_join(thr_sum, nullptr);
    pthread_join(thr_generate, nullptr);
}