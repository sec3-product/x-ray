// @purpose create three threads in main to load data
// @dataRaces 1
// @deadlocks 0
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <pthread.h>
using namespace std;

int x = 0;

void *PrintHello(void *id)
{
    long tid = (long)id;
    x++;
    return nullptr;
}

pthread_t load_data_in_thread(long id)
{
    pthread_t thread;
    void *arg = (void *)id;
    int rc = pthread_create(&thread, NULL, PrintHello, arg);
    if (rc)
    {
        cout << "Error: unable to create thread, " << rc << endl;
        exit(-1);
    }
    return thread;
}

int main()
{
    pthread_t thread1, thread2, thread3;
    thread1 = load_data_in_thread(1);
    thread2 = load_data_in_thread(2);
    thread3 = load_data_in_thread(3);

    pthread_join(thread1, nullptr);
    pthread_join(thread2, nullptr);
    pthread_join(thread3, nullptr);

    cout << "x = " << x << endl;
    return 0;
}
