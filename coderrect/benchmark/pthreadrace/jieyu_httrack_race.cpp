// @purpose httprack - thread may take the cpu before the initialization
// @dataRaces 1

#include <iostream>
#include <pthread.h>


struct global_opt {
    pthread_mutex_t mutex;
    int x;
};


global_opt *opt;


void *worker(void *) {
    pthread_mutex_lock(&opt->mutex);
    opt->x = rand();
    std::cout << opt->x << "\n";
    pthread_mutex_unlock(&opt->mutex);
    return nullptr;
}


int main() {
    pthread_t th;

    pthread_create(&th, nullptr, worker, nullptr);

    opt = new global_opt();
    pthread_mutex_init(&(opt->mutex), nullptr);

    pthread_join(th, nullptr);
}