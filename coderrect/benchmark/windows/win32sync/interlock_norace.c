#include <windows.h>
#include <stdio.h>

volatile LONG TotalCountOfOuts = 0;

void ThreadMain(void)
{
    static DWORD i;
    DWORD dwIncre;

    for (;;)
    {
        wprintf(L"Standard output print, pass # % u\n", i);
        // Increments (increases by one) the value of the specified 32-bit
        // variable as an atomic operation.
        // To operate on 64-bit values, use the  InterlockedIncrement64 function.
        dwIncre = InterlockedIncrement((LPLONG)&TotalCountOfOuts);

        // The function returns the resulting incremented value.
        wprintf(L"Increment value is % u\n", dwIncre);
        Sleep(100);

        i++;
    }

}

void CreateChildThread(void)
{
    HANDLE hThread;
    DWORD dwId;

    hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)ThreadMain, (LPVOID)NULL, 0, &dwId);
    if (hThread != NULL)
        wprintf(L"CreateThread() is OK, thread ID % u\n", dwId);
    else
        wprintf(L"CreateThread() failed, error % u\n", GetLastError());

    if (CloseHandle(hThread) != 0)
        wprintf(L"hThread's handle was closed successfully!\n");
    else
        wprintf(L"CloseHandle() failed, error % u\n", GetLastError());
}

int wmain(void)
{
    CreateChildThread();
    CreateChildThread();

    for (;;)
    {
        // 500/100 (from ThreadMain())= 5; Then 5 x 2 threads = 10.
        Sleep(500);
        wprintf(L"Current count of the printed lines by child threads = % u\n", TotalCountOfOuts);
    }

    return 0;

}