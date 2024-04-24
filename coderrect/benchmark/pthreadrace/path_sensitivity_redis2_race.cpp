// @purpose a program mimic behavior of redis IOThreadMain
// @dataRaces 1
// @tags path-sensitivity
// There should be a race on x=2
// The interesting point about this case is there're 2 paths that can reach x=2
// either the `is_io_thread` is assigned to true or false


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
        is_io_thread = true;
    } else {
        is_io_thread = false;
    }

    if (is_io_thread) {
        x = 2;
    } else {
        x = 999;
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
