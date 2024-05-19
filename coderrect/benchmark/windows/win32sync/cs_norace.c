#include <windows.h>
#include <stdio.h>

#define THREADCOUNT 2

// Global variable
CRITICAL_SECTION CriticalSection;
int SHARED_VAR = 0;

DWORD WINAPI ThreadProc(LPVOID);

int main(void)
{
	int i;
	HANDLE aThread[THREADCOUNT];
	DWORD ThreadID;
	
	// Initialize the critical section one time only.
	if (!InitializeCriticalSectionAndSpinCount(&CriticalSection,
		0x00000400))
		return;

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

	WaitForMultipleObjects(THREADCOUNT, aThread, TRUE, INFINITE);

	// Close thread and semaphore handles

	for (i = 0; i < THREADCOUNT; i++)
		CloseHandle(aThread[i]);

	// Release resources used by the critical section object.
	DeleteCriticalSection(&CriticalSection);
}

DWORD WINAPI ThreadProc(LPVOID lpParameter)
{

	// Request ownership of the critical section.
	EnterCriticalSection(&CriticalSection);

	// Access the shared resource.
	SHARED_VAR++;

	// Release ownership of the critical section.
	LeaveCriticalSection(&CriticalSection);

	return 1;
}