// @purpose signal handler support (sigaction)
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
    struct sigaction sigact;

    buf = new char[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = rand();
    }
    pthread_create(&thr_worker, nullptr, WorkerThreadMain, nullptr);

    sigact.sa_handler = sigusr1_handler;
    sigemptyset (&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigaction(SIGUSR1, &sigact, nullptr);

    pthread_join(thr_worker, nullptr);
}