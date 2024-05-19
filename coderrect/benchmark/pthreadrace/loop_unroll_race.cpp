// @purpose two threads access two different elements of an array
// @dataRaces 1

#include <iostream>
#include <cstdlib>

using namespace std;
int v[10];

void *f1(void *tid) {
    long id = (unsigned long) tid;
    v[id] = rand();

    cout << "value of v[id]: " << v[id] << "\n";
    
    for (int i = 0; i < 10; i++) {
        cout<< "value: " << v[i] << "\n";
    }

	return NULL;
}

int main () {
    int io_threads_num = 10;
    pthread_t threads[10];

    for (int i = 0; i < io_threads_num; i++) {
        pthread_create(&threads[i], NULL, f1, (void *)(long)i);
    }
}
