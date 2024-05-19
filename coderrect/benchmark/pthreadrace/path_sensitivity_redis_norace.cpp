// @purpose a program mimic behavior of redis IOThreadMain
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0
// @tags path-sensitivity


#include <iostream>
#include <pthread.h>


// this is the shared variable
int x;


static void processRequest(bool is_io_thread) {
    // read request from the client connection
    // ...

    // pesudo code of request processing.
    // only the main thread can call this piece of code
    if (!is_io_thread) {
        x = rand();
        std::cout << x << "\n";
    }

}


static void* IOThreadMain(void *unused) {
    while (1) {
        // ... other code ...
        //

        processRequest(true);
    }
}


int main() {
    pthread_t* ths = new pthread_t[16];
    for (int i = 0; i < 16; i++) {
        pthread_create(&ths[i], nullptr, IOThreadMain, nullptr);
    }

    while (1) {
        processRequest(false);
    }
}