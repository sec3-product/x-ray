// @purpose access on different branch within a function
// @dataRaces 0
// @tags path-sensitivity

#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

int global1, global2;


// Two threads are spawned. One with id=0 and another with id=1
// Thread id=0 will execute the true branch in the owrker function
// Thread id=1 will execute the false branch
// Each branch accesses a different global object.
// No overlappting accesses == no race.
void *worker(void* arg) {
    int *id = (int*)arg;
    if (*id == 0) {
        global1 = *id;
    } else {
        global2 = *id;
    }

    return nullptr;
}


int main(int argc, char*argv[]) {
    pthread_t t1, t2;
    int id0 = 0;
    int id1 = 1;
    pthread_create(&t1, nullptr, worker, &id0);
    pthread_create(&t2, nullptr, worker, &id1);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
}
