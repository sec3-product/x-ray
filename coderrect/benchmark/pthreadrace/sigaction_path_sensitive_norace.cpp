// @purpose the main picks up a sub-main depends on the cmdline
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

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


void check_db_main() {
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

    exit(1);
}


int main(int argc, char*argv[]) {
    if (strcmp(argv[1], "-check-db") == 0) {
        check_db_main();
    }

    pthread_t thr_worker;
    pthread_create(&thr_worker, nullptr, worker, nullptr);
    pthread_join(thr_worker, nullptr);
}