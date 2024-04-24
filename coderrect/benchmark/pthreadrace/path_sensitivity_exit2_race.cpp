// @purpose one path exits from a function
// @dataRaces 1
// @tags path-sensitivity
// The exit(1) function may not always be executed due to the conditional check
// therefore we should report a race between line 20 and line 26

#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

char* buf;


static void sigusr1_handler(int unused) {
    delete []buf;
    buf = new char[1024];
    for (int i = 0; i < 1024; i++) buf[i] = rand();
}


void *worker(void* unused) {
    printf("hello world %08x\n", buf);
}


void check_db_main(bool should_exit) {
    struct sigaction sigact;

    buf = new char[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = rand();
    }

    sigact.sa_handler = sigusr1_handler;
    sigemptyset (&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigaction(SIGUSR1, &sigact, nullptr);

    if (should_exit) {
        exit(1);
    }
}


int main(int argc, char*argv[]) {
    if (strcmp(argv[1], "-check-db") == 0) {
        check_db_main(true);
    }

    pthread_t thr_worker;
    pthread_create(&thr_worker, nullptr, worker, nullptr);
    pthread_join(thr_worker, nullptr);
}
