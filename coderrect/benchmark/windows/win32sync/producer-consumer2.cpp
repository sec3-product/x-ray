#include <windows.h>
#include <stdlib.h>
#include <stdio.h>

static int g_x = 0;
HANDLE hEvent1;
HANDLE hEvent2;

HANDLE aThread[2];
DWORD ThreadID;

//tread 1
void Producer()
{
    for (int i = 0; i < 100; ++i)
    {
        WaitForSingleObject(hEvent1, INFINITE);
        g_x = i;
        SetEvent(hEvent2);
    }
}
//thread 2
void Consumer()
{
    for (;;)
    {
        WaitForSingleObject(hEvent2, INFINITE);
        SetEvent(hEvent1);
    }
}

int createthreads() {
    hEvent1 = CreateEvent(NULL, FALSE, FALSE, NULL);
    hEvent2 = CreateEvent(NULL, FALSE, FALSE, NULL);

    // Create worker threads
    aThread[0] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)Producer, NULL, 0, &ThreadID);
    aThread[1] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)Consumer, NULL, 0, &ThreadID);
    return 0;
}
int main() {
    createthreads();
    SetEvent(hEvent1);
}