// @purpose a callback causes an order violation in firefox
// @dataRaces 0
// @order-violation 1
//
// the buggy interleave - main thread never gets a chance to escape
// from the while-loop
//
//   main thread                          worker thread
//   -------------                        ------------
//   submitAsync()
//                                        process request
//                                        io_pending = false
//   io_pending = true
//
//   while (io_pending) {
//       wait ...
//   }
//


#include <iostream>
#include <pthread.h>
#include <unistd.h>


struct IORequest {
    int x;
};


volatile int io_pending = 0;

pthread_mutex_t lock;
IORequest *io_request = nullptr;


void* worker(void* unused) {
    while (true) {
        pthread_mutex_lock(&lock);
        if (io_request == nullptr) {
            pthread_mutex_unlock(&lock);
            sleep(1);
            continue;
        }

        std::cout << io_request->x << "\n";
        delete io_request;
        io_request = nullptr;

        io_pending = false;
        pthread_mutex_unlock(&lock);
    }
}


void submitAsync(IORequest* ioRequest) {
    pthread_mutex_lock(&lock);
    io_request = ioRequest;
    pthread_mutex_unlock(&lock);
}


int main() {
    pthread_t th;

    pthread_mutex_init(&lock, nullptr);
    pthread_create(&th, nullptr, worker, nullptr);

    IORequest *ioRequest = new IORequest();
    ioRequest->x = rand();
    submitAsync(ioRequest);
    io_pending = true;

    while (io_pending) {
        std::cout << "I'm waiting\n";
    }

    std::cout << "We'are done\n";
    return 0;
}