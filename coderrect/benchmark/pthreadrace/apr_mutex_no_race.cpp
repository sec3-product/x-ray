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
#include <apr_thread_mutex.h>

/**
 * The number of running threads concurrently
 */
#define NUM_THREADS    64

/**
 * Each threads makes this number of loop to increment the count
 * As a result, the final count should become NUM_THREADS x NUM_LOOPS_PER_THREAD
 */
#define NUM_LOOPS_PER_THREAD        8192

typedef struct {
    int mutexed_count;
    apr_thread_mutex_t *mutex;

    /* for comparison */
    int unmutexed_count;
} shared_ctx_t;

/* Use macros for readability */
#define MUTEX_LOCK(ctx)		apr_thread_mutex_lock((ctx)->mutex)
#define MUTEX_UNLOCK(ctx)	apr_thread_mutex_unlock((ctx)->mutex)

static void* APR_THREAD_FUNC doit(apr_thread_t *thd, void *data);

/**
 * Thread execution and mutex sample code
 * @remark Error checks almost omitted
 */
int main(int argc, const char *argv[])
{
    apr_status_t rv;
    apr_pool_t *mp;
    apr_thread_t *thd_arr[NUM_THREADS];
    shared_ctx_t shared_ctx = { 0, NULL, 0 };
    int i;
        
    apr_initialize();
    apr_pool_create(&mp, NULL);

    apr_thread_mutex_create(&shared_ctx.mutex, APR_THREAD_MUTEX_UNNESTED, mp);

    for (i = 0; i < NUM_THREADS; i++) {
        rv = apr_thread_create(&thd_arr[i], NULL, doit, &shared_ctx, mp);
        assert(rv == APR_SUCCESS);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        rv = apr_thread_join(&rv, thd_arr[i]);
        assert(rv == APR_SUCCESS);
    }

    printf("shared_ctx::mutexed_count = %d, shared_ctx::unmutexed_count = %d\n",
           shared_ctx.mutexed_count, shared_ctx.unmutexed_count);
    assert(shared_ctx.mutexed_count == NUM_THREADS * NUM_LOOPS_PER_THREAD);

    apr_terminate();
    return 0;
}

/**
 * Thread entry point
 */
static void* APR_THREAD_FUNC doit(apr_thread_t *thd, void *data)
{
    shared_ctx_t *ctx = data;
    int i;

    for (i = 0; i < NUM_LOOPS_PER_THREAD; i++) {
        MUTEX_LOCK(ctx);
        ctx->mutexed_count++;
        MUTEX_UNLOCK(ctx);

        /* XXX Since this is not protected by mutex lock, the value is undetermined. */
        ctx->unmutexed_count++;
    }
    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}
