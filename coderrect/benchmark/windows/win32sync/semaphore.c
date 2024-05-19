#include <windows.h>
#include <stdio.h>

#define MAX_SEM_COUNT 1
#define THREADCOUNT 12

HANDLE ghSemaphore;
int GValue = 0;

DWORD WINAPI ThreadProc(LPVOID);

int main(void)
{
    HANDLE aThread[THREADCOUNT];
    DWORD ThreadID;
    int i;

    // Create a semaphore with initial and max counts of MAX_SEM_COUNT

    ghSemaphore = CreateSemaphore(
        NULL,           // default security attributes
        MAX_SEM_COUNT,  // initial count
        MAX_SEM_COUNT,  // maximum count
        NULL);          // unnamed semaphore

    if (ghSemaphore == NULL)
    {
        printf("CreateSemaphore error: %d\n", GetLastError());
        return 1;
    }

    // Create worker threads

    for (i = 0; i < THREADCOUNT; i++)
    {
        aThread[i] = CreateThread(
            NULL,       // default security attributes
            0,          // default stack size
            (LPTHREAD_START_ROUTINE)ThreadProc,
            NULL,       // no thread function arguments
            0,          // default creation flags
            &ThreadID); // receive thread identifier

        if (aThread[i] == NULL)
        {
            printf("CreateThread error: %d\n", GetLastError());
            return 1;
        }
    }

    // Wait for all threads to terminate

    WaitForMultipleObjects(THREADCOUNT, aThread, TRUE, INFINITE);

    // Close thread and semaphore handles

    for (i = 0; i < THREADCOUNT; i++)
        CloseHandle(aThread[i]);

    CloseHandle(ghSemaphore);

    return 0;
}

DWORD WINAPI ThreadProc(LPVOID lpParam)
{

    // lpParam not used in this example
    UNREFERENCED_PARAMETER(lpParam);

    DWORD dwWaitResult;
    BOOL bContinue = TRUE;

    while (bContinue)
    {
        // Try to enter the semaphore gate.

        dwWaitResult = WaitForSingleObject(
            ghSemaphore,   // handle to semaphore
            0L);           // zero-second time-out interval

        switch (dwWaitResult)
        {
            // The semaphore object was signaled.
        case WAIT_OBJECT_0:
            printf("Thread %d: wait succeeded\n", GetCurrentThreadId());
            bContinue = FALSE;

            // Simulate thread spending time on task
            GValue++;
            Sleep(5);
            printf("GValue=%d\n", GValue);

            // Release the semaphore when task is finished

            if (!ReleaseSemaphore(
                ghSemaphore,  // handle to semaphore
                1,            // increase count by one
                NULL))       // not interested in previous count
            {
                printf("ReleaseSemaphore error: %d\n", GetLastError());
            }
            break;

            // The semaphore was nonsignaled, so a time-out occurred.
        case WAIT_TIMEOUT:
            printf("Thread %d: wait timed out\n", GetCurrentThreadId());
            break;
        }
    }
    return TRUE;
}