// @purpose demo how tasks executed by a apr thread pool cause data races
// @dataRaces 1
// @commandLine clang -I/usr/local/apr/include/apr-2 -L/usr/local/apr/lib apr_thread_pool_race.cpp -lapr-2


/**
 * apr tutorial sample code
 * http://dev.ariel-networks.com/apr/
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <apr_general.h>
#include <apr_thread_proc.h>
#include <apr_thread_pool.h>


int g_x = 0;

static void* APR_THREAD_FUNC worker(apr_thread_t *tid, void* unused) {
    g_x ++;
    printf("%d\n", g_x);
    return NULL;
}

int main(int argc,char *argv[],char *envp[]) {
    apr_pool_t *mempool;
    apr_thread_pool_t *threadpool;

    apr_initialize();
    apr_pool_create(&mempool, NULL);

    if (apr_thread_pool_create(&threadpool, 2, 50, mempool) != APR_SUCCESS) {
        return (1);
    }
    apr_thread_pool_idle_max_set(threadpool, 10);

    apr_thread_pool_push(threadpool, worker, NULL, 0, NULL);
    apr_thread_pool_push(threadpool, worker, NULL, 0, NULL);
}
