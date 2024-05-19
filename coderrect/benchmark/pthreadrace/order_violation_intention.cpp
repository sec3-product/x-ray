// @purpose how does racetectect know developer's intention
// @dataRaces 0
// @order-violation 1
//
// this is an interesting case. The original intention of the developer
// is to let thread1 process an object in 3 sequential steps. Then thread2
// will provide a new object for thread1 to repeat its processing.
//
// Since these 3 sequential steps are't put into a single transaction, thread2
// may reset the object in the middle of processing.
//
// The tricky thing is that how racedetect knows "process an object in 3 sequential
// steps" is the intention of the developer?
//
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

pthread_mutex_t m;
volatile int i = 1;

// Thread 1
void* t1f(void *args)
{

    for (int j = 0; j < 3; ++j)
    {
        pthread_mutex_lock(&m);
        switch(i) {
            case 1:
                printf("%d\n", i++);
                break;
            case 2:
                printf("%d\n", i++);
                break;
            default:
                printf("%d\n", i);
                i = 0;
        }
        pthread_mutex_unlock(&m);
    }
    return NULL;
}

// Thread 2
void* t2f(void *args)
{
    pthread_mutex_lock(&m);
    i = 1;
    pthread_mutex_unlock(&m);
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, t1f, NULL);
    pthread_create(&t2, NULL, t2f, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    return 0;
}