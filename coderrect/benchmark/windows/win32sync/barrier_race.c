#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#define WATCHERS 4

SYNCHRONIZATION_BARRIER barrier;
HANDLE movieFinished;

// Watchers gonna watch.
DWORD CALLBACK MovieWatcherThread(void* p)
{
    int id = PtrToInt(p);

    // Build a string that we use to prefix our messages.
    char tag[WATCHERS + 1];
    for (int i = 0; i < WATCHERS; i++) {
        tag[i] = (i == id) ? (char)('1' + i) : ' ';
    }
    tag[WATCHERS] = 0;

    printf("%s Sitting down\n", tag);

    if (EnterSynchronizationBarrier(&barrier, 0)) {
        // We are the one who should start the movie.
        printf("%s Starting the movie\n", tag);

        // For demonstration purposes, the movie is only one second long.
        LARGE_INTEGER dueTime;
        dueTime.QuadPart = -1000000LL;
        SetWaitableTimer(movieFinished, &dueTime, 0,
            NULL, NULL, FALSE);
    }

    // Watch the movie until it ends.
    printf("%s Enjoying the movie\n", tag);
    WaitForSingleObject(movieFinished, INFINITE);

    // Now leave the room.
    printf("%s Leaving the room\n", tag);

    if (EnterSynchronizationBarrier(&barrier, 0)) {
        // We are the one who should lock the door.
        printf("%s Locking the door\n", tag);
    }

    printf("%s Saying good-bye and going home\n", tag);
    return 0;
}

int main(void)
{
    movieFinished = CreateWaitableTimer(NULL, TRUE, NULL);
    InitializeSynchronizationBarrier(&barrier, WATCHERS, -1);

    HANDLE threads[WATCHERS];
    for (int i = 0; i < WATCHERS; i++) {
        DWORD threadId;
        threads[i] = CreateThread(NULL, 0, MovieWatcherThread,
            IntToPtr(i), 0, &threadId);
    }

    // Wait for the demonstration to complete.
    WaitForMultipleObjects(WATCHERS, threads, TRUE, INFINITE);

    CloseHandle(movieFinished);
    DeleteSynchronizationBarrier(&barrier);
    return 0;
}