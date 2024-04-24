// @purpose create a list of thread into a vector
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <vector>
#include <pthread.h>
using namespace std;

int accum = 0;

static void* square(void* x) {
    int64_t k = (int64_t)x;
    accum += k * k;
    return nullptr;
}

int main() {
    vector<pthread_t> ths;
    for (int i = 1; i <= 20; i++) {
        pthread_t th;
        pthread_create(&th, nullptr, square, (void *)i);
        ths.push_back(th);
    }

    for (auto& th : ths) {
        pthread_join(th, nullptr);
    }

    cout << "accum = " << accum << endl;
    return 0;
}