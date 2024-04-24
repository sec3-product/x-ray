// @purpose toctou between checking if file is open and writing to file
// @toctou 1

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

FILE *file;
pthread_mutex_t lock;

void *worker1(void *arg) {
    pthread_mutex_lock(&lock);
    if (file != NULL) {
        // fclose from worker2 can interleave here
        fprintf(file, "msg");
    }
    pthread_mutex_unlock(&lock);
}

void *worker2(void *arg) { fclose(file); }

int main() {
    file = fopen("log.txt", "w");
    pthread_t th1, th2;

    pthread_create(&th1, NULL, worker1, NULL);
    pthread_create(&th2, NULL, worker2, NULL);

    pthread_join(th1, NULL);
    pthread_join(th2, NULL);
}