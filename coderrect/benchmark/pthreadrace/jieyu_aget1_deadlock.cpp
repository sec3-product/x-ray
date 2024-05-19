// @purpose a deadlock caused by pthread_cancel and signal handler
// @dataRaces 0
// @deadlocks 1
// @orderViolations 0
// @misMatchedAPI 0
//
// https://github.com/jieyu/concurrency-bugs/blob/master/aget-bug1/DESCRIPTION


#include <iostream>
#include <pthread.h>
#include <signal.h>
//#include <zconf.h>
#include <unistd.h>


static pthread_mutex_t lock;


void *signal_waiter(void *arg)
{
    int signal;

    /* Set Cancellation Type to Asynchronous */
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    sigset_t set;

    sigemptyset(&set);
    sigaddset(&set, SIGINT);

    while(1) {
        sigwait(&set, &signal);
        pthread_mutex_lock(&lock);
        for (int i = 0; i < 10000000; i++) {
            sleep(1);
        }
        pthread_mutex_unlock(&lock);
    }
}


int main() {
    pthread_t th;

    pthread_mutex_init(&lock, nullptr);
    pthread_create(&th, nullptr, signal_waiter, nullptr);

    sleep(10);
    pthread_cancel(th);

    pthread_mutex_lock(&lock);
    for (int i = 0; i < 10000000; i++) {
        sleep(1);
    }
    pthread_mutex_unlock(&lock);
}