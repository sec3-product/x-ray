// @purpose test signal handler using the old singal() api
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <signal.h>
#include <unistd.h>


static char* buf;


static void *WorkerThreadMain(void *unused) {
    while (true) {
        sleep(5);

        uint32_t sum = 0;
        for (int i = 0; i < 1024; i++) {
            sum += buf[i];
        }
        std::cout << "sum " << sum << "\n";
    }
}


static void sigusr1_handler(int unused) {
    delete []buf;
    buf = new char[1024];
    for (int i = 0; i < 1024; i++) buf[i] = rand();
}


int main(int argc, char*argv[]) {
    pthread_t thr_worker;

    buf = new char[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = rand();
    }
    pthread_create(&thr_worker, nullptr, WorkerThreadMain, nullptr);

    if (signal(SIGUSR1, sigusr1_handler) == SIG_ERR) {
        std::cout << "Failed to set signal handler\n";
        exit(1);
    }

    pthread_join(thr_worker, nullptr);
}