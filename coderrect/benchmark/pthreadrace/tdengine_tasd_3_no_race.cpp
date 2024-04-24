// @purpose this is a false positive from tdengine taosd
// @dataRaces 0


#include <pthread.h>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

typedef struct {
    int32_t fileNum;
    int32_t maxLines;
    int32_t lines;
    int32_t flag;
    int32_t openInProgress;
    pid_t   pid;
    pthread_mutex_t logMutex;
} SLogObj;

static SLogObj   tsLogObj = { .fileNum = 1 };


static void *taosThreadToOpenNewFile(void *param) {
    char name[128];

    tsLogObj.flag ^= 1;
    tsLogObj.lines = 0;

    int32_t fd = open(name, O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU | S_IRWXG | S_IRWXO);
    if (fd < 0) {
        return NULL;
    }

    tsLogObj.lines = 0;
    tsLogObj.openInProgress = 0;

    return NULL;
}


static int32_t taosOpenNewLogFile() {
    pthread_mutex_lock(&tsLogObj.logMutex);

    if (tsLogObj.lines > tsLogObj.maxLines && tsLogObj.openInProgress == 0) {
        tsLogObj.openInProgress = 1;

        pthread_t      thread;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

        pthread_create(&thread, &attr, taosThreadToOpenNewFile, NULL);
        pthread_attr_destroy(&attr);
    }

    pthread_mutex_unlock(&tsLogObj.logMutex);

    return 0;
}


static void* worker(void *unused) {
    taosOpenNewLogFile();
    return nullptr;
}


int main() {
    pthread_t t1, t2;

    pthread_create(&t1, nullptr, worker, nullptr);
    pthread_create(&t2, nullptr, worker, nullptr);

    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
}