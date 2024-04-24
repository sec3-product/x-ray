#include <windows.h>
#include <stdio.h>

#define THREADCOUNT 4 

HANDLE ghWriteEvent;
HANDLE ghThreads;
HANDLE bugThreads;

DWORD WINAPI ThreadProc(LPVOID);
DWORD WINAPI BugThreadProc(LPVOID);


int buffer = 0;

void CreateEventsAndThreads(void)
{
    int i;
    DWORD dwThreadID;

    ghWriteEvent = CreateEvent(NULL,TRUE,FALSE,TEXT("WriteEvent"));

    ghThreads = CreateThread(NULL,0,ThreadProc,NULL,0,&dwThreadID);
    ghThreads = CreateThread(NULL, 0, BugThreadProc, NULL, 0, &dwThreadID);
}

void WriteToBuffer(VOID)
{
    buffer++;

    printf("Main thread writing to the shared buffer...\n");

    // Set ghWriteEvent to signaled

    if (!SetEvent(ghWriteEvent))
    {
        printf("SetEvent failed (%d)\n", GetLastError());
        return;
    }
}

void CloseEvents()
{
    // Close all event handles (currently, only one global handle).

    CloseHandle(ghWriteEvent);
}

int main(void)
{
    DWORD dwWaitResult;

    CreateEventsAndThreads();

    WriteToBuffer();

    printf("Main thread waiting for threads to exit...\n");

    // The handle for each thread is signaled when the thread is
    // terminated.
    dwWaitResult = WaitForSingleObject(ghThreads,INFINITE);
    dwWaitResult = WaitForSingleObject(bugThreads,INFINITE);
    CloseEvents();

    return 0;
}

DWORD WINAPI ThreadProc(LPVOID lpParam)
{
    // lpParam not used in this example.
    UNREFERENCED_PARAMETER(lpParam);

    DWORD dwWaitResult;
    printf("Thread %d waiting for write event...\n", GetCurrentThreadId());

    dwWaitResult = WaitForSingleObject(ghWriteEvent, INFINITE);
    printf("Get %d buffer\n", buffer);
    return 1;
}


DWORD WINAPI BugThreadProc(LPVOID lpParam)
{
    // lpParam not used in this example.
    UNREFERENCED_PARAMETER(lpParam);

    DWORD dwWaitResult;

    dwWaitResult = WaitForSingleObject(ghWriteEvent, INFINITE);
    printf("Get %d buffer\n", buffer);
    return 1;
}