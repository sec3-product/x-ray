// @purpose the main thread installs the signal handler
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0
// @comment abstract from redis 6.0.

#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <string.h>

static char* buf;


static void sigusr1_handler(int unused) {
    delete []buf;
    buf = new char[1024];
    for (int i = 0; i < 1024; i++) buf[i] = rand();
}


static void* worker(void* unused) {
    std::cout << "hello world\n";
    return nullptr;
}


static void main1() {
    struct sigaction sigact;

    buf = new char[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = rand();
    }

    sigact.sa_handler = sigusr1_handler;
    sigemptyset (&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigaction(SIGUSR1, &sigact, nullptr);

    int fd = open("abc.txt", O_RDONLY);
    read(fd, buf, 1024);
    close(fd);

    exit(0);
}


int main(int argc, char*argv[]) {
    if (strcmp(argv[1], "-check-rdb") == 0) {
        main1();
    }

    pthread_t thr;
    pthread_create(&thr, NULL, worker, nullptr);
    pthread_join(thr, nullptr);
}