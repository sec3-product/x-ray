#define UNICODE 1
#define _UNICODE 1

#include <windows.h>
#include <stdio.h>
#include <tchar.h>

#define _SECOND 10000000

typedef struct _MYDATA {
    TCHAR* szText;
    DWORD dwValue;
} MYDATA;

VOID CALLBACK TimerAPCProc(
    LPVOID lpArg,               // Data value
    DWORD dwTimerLowValue,      // Timer low value
    DWORD dwTimerHighValue)    // Timer high value

{
    // Formal parameters not used in this example.
    UNREFERENCED_PARAMETER(dwTimerLowValue);
    UNREFERENCED_PARAMETER(dwTimerHighValue);

    MYDATA* pMyData = (MYDATA*)lpArg;
    pMyData->dwValue++;

    _tprintf(TEXT("Message: %s\nValue: %d\n\n"), pMyData->szText,
        pMyData->dwValue);
    MessageBeep(0);

}

int RangedRand(int range_min, int range_max)
{
    // Generate random numbers in the half-closed interval  
    // [range_min, range_max). In other words,  
    // range_min <= random number < range_max  
    int u;
    u = (double)rand() / (RAND_MAX + 1) * (range_max - range_min)
        + range_min;
    printf("  %6d\n", u);
    return u;
}

DWORD WINAPI ThreadProc(LPVOID lpParam)
{
    srand((unsigned)time(NULL));

    SleepEx(RangedRand(4990, 5010),TRUE);       

    MYDATA* pMyData = lpParam;
    pMyData->dwValue++;
    
    return 1;
}

int main(void)
{
    HANDLE          hTimer;
    BOOL            bSuccess;
    __int64         qwDueTime;
    LARGE_INTEGER   liDueTime;
    MYDATA          MyData;
    DWORD dwThreadID;

    MyData.szText = TEXT("This is my data");
    MyData.dwValue = 100;
    
    HANDLE h = CreateThread(NULL, 0, ThreadProc, &MyData, 0, &dwThreadID);

    hTimer = CreateWaitableTimer(
        NULL,                   // Default security attributes
        FALSE,                  // Create auto-reset timer
        TEXT("MyTimer"));       // Name of waitable timer
    if (hTimer != NULL)
    {
        __try
        {
            // Create an integer that will be used to signal the timer 
            // 5 seconds from now.
            qwDueTime = -5 * _SECOND;

            // Copy the relative time into a LARGE_INTEGER.
            liDueTime.LowPart = (DWORD)(qwDueTime & 0xFFFFFFFF);
            liDueTime.HighPart = (LONG)(qwDueTime >> 32);

            bSuccess = SetWaitableTimer(
                hTimer,           // Handle to the timer object
                &liDueTime,       // When timer will become signaled
                2000,             // Periodic timer interval of 2 seconds
                TimerAPCProc,     // Completion routine
                &MyData,          // Argument to the completion routine
                FALSE);          // Do not restore a suspended system

            if (bSuccess)
            {
                for (; MyData.dwValue < 1000; MyData.dwValue += 100)
                {
                    SleepEx(
                        INFINITE,     // Wait forever
                        TRUE);       // Put thread in an alertable state
                    // execute this when apc called
                }

            }
            else
            {
                printf("SetWaitableTimer failed with error %d\n", GetLastError());
            }

        }
        __finally
        {
            CloseHandle(hTimer);
        }
    }
    else
    {
        printf("CreateWaitableTimer failed with error %d\n", GetLastError());
    }

    _tprintf(TEXT("end"));


    return 0;
}