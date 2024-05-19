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
int count = 0;

static void* square(void* x) {
    int64_t k = (int64_t)x;
    accum += k * k;
    return nullptr;
}

static void my_create_thread(pthread_t *th, int64_t x) {
    pthread_create(th, nullptr, square, (void*)x);
}

static void increase_count() {
    count++;
}

int main() {
    vector<pthread_t> ths;
    for (int i = 1; i <= 20; i++) {
        pthread_t th;
        increase_count();
        my_create_thread(&th, i);
        ths.push_back(th);
    }

    for (auto& th : ths) {
        pthread_join(th, nullptr);
    }

    cout << "accum = " << accum << endl;
    return 0;
}
