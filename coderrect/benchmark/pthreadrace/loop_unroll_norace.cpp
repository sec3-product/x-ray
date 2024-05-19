// @purpose two threads access two different elements of an array
// @dataRaces 0

#include <iostream>
#include <cstdlib>

using namespace std;

int64_t v_array[10];
long long *global_ll;

// simplified from redis/util.c, string2ll
int string2ll(long id, long long *value) {
    if (id < 5) {
        *value = 0;
    } else {
        *value = 1;
    }
    return 1;
}

int64_t f1(long tid) {
    long id = (unsigned long) tid;
    int64_t v;
    long long ll;

    int retval = string2ll(id, &ll);
    v = ll;

    cout << "value of v: " << v << "\n";
    
	return v;
}

void *worker(void *tid) {
    long id = (unsigned long) tid;
    int64_t v = f1(id);
    v_array[10] = v;

    return NULL;
}

int main () {
    int io_threads_num = 10;
    pthread_t threads[10];

    // below is important for this test case
    // it will invalidate our inferNoAliasPass
    // therefore in our older version this will results in an FP
    // due to not unrolling the loop
    long long ll = 298392;
    global_ll = &ll;
    int retval = string2ll(rand(), &ll);
    cout << "value of ll: " << *global_ll << "\n";

    for (int i = 0; i < io_threads_num; i++) {
        pthread_create(&threads[i], NULL, worker, (void *)(long)i);
    }
}
