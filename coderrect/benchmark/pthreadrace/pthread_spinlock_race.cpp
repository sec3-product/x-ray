// @purpose pthread spinlock support
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 1

#include <iostream>
#include <pthread.h>


static pthread_spinlock_t lock;
static uint32_t *buf;


static void *SumThread(void *unused) {
    while (true) {
        pthread_spin_lock(&lock);
        uint32_t sum = 0;
        for (int i = 0; i < 1024; i++) {
            sum += buf[i];
        }
        std::cout << sum << "\n";
        pthread_spin_unlock(&lock);
    }
}

static void *GenerateThread(void *unused) {
    while (true) {
        pthread_spin_lock(&lock);
        delete buf;
        buf = new uint32_t[1024];
        // pthread_spin_unlock(&lock);
    }
}

int main(int argc, char*argv[]) {
    pthread_t thr_sum, thr_generate;

    buf = new uint32_t[1024];
    pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);

    pthread_create(&thr_sum, nullptr, SumThread, nullptr);
    pthread_create(&thr_generate, nullptr, GenerateThread, nullptr);

    pthread_join(thr_sum, nullptr);
    pthread_join(thr_generate, nullptr);
}