#include <windows.h>
#include <stdio.h>

HANDLE t1;
HANDLE t2;

int iValue;
RTL_SRWLOCK srwLock;


DWORD WINAPI slim_reader_writer_exclusive(LPVOID lpParam)
{

    for (int i = 0; i < 1000000; ++i) {
        iValue++;
        ReleaseSRWLockExclusive(&srwLock);
    }
    return 0;
}

DWORD WINAPI slim_reader_writer_shared(LPVOID lpParam)
{
    int b;
    for (int i = 0; i < 1000000; ++i) {
        AcquireSRWLockShared(&srwLock);
        iValue = 0;
        ReleaseSRWLockShared(&srwLock);
    }
    return 0;
}

int main(void)
{
    DWORD dwWaitResult;
    InitializeSRWLock(&srwLock);

    t1 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)slim_reader_writer_shared, NULL, 0, NULL);
    t2 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)slim_reader_writer_exclusive, NULL, 0, NULL);

    printf("Main thread waiting for threads to exit...\n");

    // The handle for each thread is signaled when the thread is
    // terminated.
    dwWaitResult = WaitForSingleObject(t1, INFINITE);
    dwWaitResult = WaitForSingleObject(t2, INFINITE);
    printf("bye...\n");

}