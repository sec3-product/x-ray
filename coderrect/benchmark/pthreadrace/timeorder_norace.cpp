// @purpose demo how time order impact the race detection
// @dataRaces 0
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0


#include <iostream>
#include <pthread.h>


static pthread_mutex_t lock;
static int x;


static void* square(void* unused) {
    pthread_mutex_lock(&lock);
    x = rand();
    std::cout << x << "\n";
    pthread_mutex_unlock(&lock);
    return nullptr;
}


int main() {
    pthread_mutex_init(&lock, nullptr);

    // main thread update x
    for (int i = 0; i < 10; i++) {
        x = i;
        std::cout << x << "\n";
    }

    // then it creates a thread that updates x, too.
    // there isn't any race because there isn't 'square'
    // thread when main thread update x.
    pthread_t th;
    for (int i = 0; i < 10; i++) {
        pthread_create(&th, nullptr, square, nullptr);
    }

    pthread_join(th, nullptr);
}
