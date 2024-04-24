// @purpose fork 2 threads using std::thread and function pointer
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0


#include <iostream>
#include <thread>
using namespace std;


constexpr size_t SIZE = 1024;


void sum(uint32_t *buf) {
    uint32_t sum = 0;
    for (int i = 0; i < SIZE; i++) {
        sum += buf[i];
    }

    cout << sum << "\n";
}


void randomize(uint32_t *buf) {
    for (int i = 0; i < SIZE; i++) {
        buf[i] = rand();
    }
}


int main(int argc, char**argv) {
    uint32_t *buf = new uint32_t[SIZE];

    thread sumth(&sum, buf);
    thread randomizeth(&randomize, buf);

    sumth.join();
    randomizeth.join();

    delete []buf;
    return 0;
}