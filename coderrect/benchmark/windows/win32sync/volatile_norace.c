#include <windows.h>
#include <stdio.h>

HANDLE readerThread;
HANDLE writerThread;

volatile int  iValue;
volatile BOOL fValueHasBeenComputed = FALSE;

int RangedRand(int range_min, int range_max)
{
    // Generate random numbers in the half-closed interval  
    // [range_min, range_max). In other words,  
    // range_min <= random number < range_max  
    int u;
    u = (double)rand() / (RAND_MAX + 1) * (range_max - range_min)
        + range_min;
    printf("rand  %6d\n", u);
    return u;
}

int ComputeValue() {
    Sleep(128);
    return RangedRand(0, 10000);
}

void CacheComputedValue()
{
    if (!fValueHasBeenComputed)
    {
        iValue = ComputeValue();
        fValueHasBeenComputed = TRUE;
    }
}

BOOL FetchComputedValue(int* piResult)
{
    if (fValueHasBeenComputed)
    {
        *piResult = iValue;
        return TRUE;
    }

    else return FALSE;
}

DWORD WINAPI WriterThreadProc(LPVOID lpParam)
{
    for (int i = 0; i < 100; i++) {
        CacheComputedValue();
    }
    return 1;
}


DWORD WINAPI ReaderThreadProc(LPVOID lpParam)
{
    for (int i = 0; i < 100; i++) {
        Sleep(256);
        int val;
        FetchComputedValue(&val);
        printf("%d", val);
    }
    return 1;
}

int main(void)
{
    DWORD dwWaitResult;

    srand((unsigned)time(NULL));

    readerThread = CreateThread(NULL, 0, ReaderThreadProc, NULL, 0, NULL);
    writerThread = CreateThread(NULL, 0, WriterThreadProc, NULL, 0, NULL);

    printf("Main thread waiting for threads to exit...\n");

    // The handle for each thread is signaled when the thread is
    // terminated.
    dwWaitResult = WaitForSingleObject(readerThread, INFINITE);
    dwWaitResult = WaitForSingleObject(writerThread, INFINITE);

    return 0;
}

