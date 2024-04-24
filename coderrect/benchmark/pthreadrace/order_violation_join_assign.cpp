// @purpose an order violation caused by assignment using pthread_join
// @dataRaces 0
// @order-violation 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

pthread_mutex_t n;

// Thread 2
void* t2f(void *args) {
    char *string = *(char**)args;
    const char source[] = "Hello world";

    pthread_mutex_lock(&n);
    memcpy(string, source, strlen(source));
    pthread_mutex_unlock(&n);
    return NULL;
}

// Thread 1
void* t1f(void *args) {
    pthread_mutex_lock(&n);
    char *string = (char *)malloc(sizeof(char)*20);
    pthread_mutex_unlock(&n);

    return string;
}

int main() {
    pthread_t t1, t2;
    char *string;

    pthread_create(&t1, NULL, t1f, (void *)NULL);
    pthread_create(&t2, NULL, t2f, (void **)&string);

    pthread_join(t1, (void**)&string);
    pthread_join(t2, NULL);

    printf("1: %s\n", string);
    free(string);
    return 0;
}