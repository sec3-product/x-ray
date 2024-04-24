// @purpose demonstrates the use of pthread_once
// @dataRaces 0

#include <pthread.h>
#include <iostream>


static pthread_once_t walModuleInit = PTHREAD_ONCE_INIT;
static int *walTmrCtrl = nullptr;

static void walModuleInitFunc() {
    walTmrCtrl = new int[32];
    if (walTmrCtrl == NULL)
        walModuleInit = PTHREAD_ONCE_INIT;
    else
        std::cout << "WAL module is initialized\n";
}

void *walOpen() {
    pthread_once(&walModuleInit, walModuleInitFunc);
    if (walTmrCtrl == NULL) {
        std::cout << "unable to init wal\n";
    }

    return nullptr;
}


static void *worker(void* unused) {
    walOpen();
}


int main() {
    pthread_t t1, t2;

    pthread_create(&t1, nullptr, worker, nullptr);
    pthread_create(&t2, nullptr, worker, nullptr);

    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
}