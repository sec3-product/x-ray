// @purpose one thread may read fields before the struct is initialized
// @dataRaces 0
// @order-violation 1


#include <iostream>
#include <pthread.h>


pthread_mutex_t lock;
struct global_opt {
    int tz;
};


global_opt *opt;


void* initThread(void *unused) {
    pthread_mutex_lock(&lock);
    opt = new global_opt();
    opt->tz = 1;
    pthread_mutex_unlock(&lock);

    return nullptr;
}


void* readThread(void *unused) {
    pthread_mutex_lock(&lock);
    int tz = opt->tz;
    std::cout << tz << "\n";
    pthread_mutex_unlock(&lock);

    return nullptr;
}


int main() {
    pthread_t th1, th2;

    pthread_mutex_init(&lock, nullptr);
    pthread_create(&th1, nullptr, initThread, nullptr);
    pthread_create(&th2, nullptr, readThread, nullptr);

    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);
}