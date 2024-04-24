// @purpose refcount check caused double-free atomicity violation
// @orderViolations 1
// @seealso https://github.com/jieyu/concurrency-bugs/blob/master/apache-21287/DESCRIPTION

#include <iostream>
#include <pthread.h>
#include <atomic>



std::atomic_uint refcount(0);
pthread_mutex_t lock;
static int *buf;


static void *worker(void *) {
    refcount++;

    pthread_mutex_lock(&lock);
    if (rand() % 2 == 0) {
        int sum = 0;
        for (int i = 0; i < 1024; i++) {
            sum += buf[i];
        }
        std::cout << sum << "\n";
    }
    else {
        for (int i = 0; i < 1024; i++) {
            buf[i] = rand();
        }
    }
    pthread_mutex_unlock(&lock);

    refcount--;
    if (refcount == 0) {
        pthread_mutex_lock(&lock);
        delete buf;
        pthread_mutex_unlock(&lock);
    }

    return nullptr;
}


int main() {
    pthread_mutex_init(&lock, nullptr );
    buf = new int[1024];

    pthread_t t1, t2;
    pthread_create(&t1, nullptr, worker, nullptr);
    pthread_create(&t2, nullptr, worker, nullptr);

    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
}
