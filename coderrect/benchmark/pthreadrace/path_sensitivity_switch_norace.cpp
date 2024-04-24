// @purpose demo the path sensitivity analysis of 'switch' statement
// @dataRaces 0
// @tags path-sensitivity


#include <iostream>
#include <pthread.h>
#include <stdlib.h>


int x;


void* worker1(void *unused) {
    x = rand();
    std::cout << x << "\n";
    return nullptr;
}


void* worker2(void *unused) {
    x = rand() + 2;
    std::cout << x << "\n";
    return nullptr;
}


int main(int argc, char**argv) {
    pthread_t th;

    int k = atoi(argv[1]);
    switch (k) {
        case 1:
            pthread_create(&th, nullptr, worker1, nullptr);
            break;

        case 2:
            pthread_create(&th, nullptr, worker2, nullptr);
            break;

        default:
            std::cout << "Unknown arg\n";
    }

    pthread_join(th, nullptr);
    return 0;
}