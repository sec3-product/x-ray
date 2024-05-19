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
#include <apr_thread_cond.h>

/**
 * Shared context between main-thread and sub-thread.
 * In this sample, main-thread wakes sub-thread up.
 */
typedef struct {
    /* condition variable should be used with a mutex variable */
    apr_thread_mutex_t *mutex;
    apr_thread_cond_t  *cond;

    /* shared context depends on application */
    int input_num;
} my_production_t;

static void* APR_THREAD_FUNC do_consume(apr_thread_t *thd, void *data);

/**
 * @return TRUE when we wake the consumer up, othewrise FALSE.
 */
static int do_produce(my_production_t *prod)
{
    int c;

    puts("type any key and return-key to wake consumer-thread up");
    c = getchar();
    if (c > 0) {
        apr_thread_mutex_lock(prod->mutex);
        prod->input_num = c;
        puts("to wake consumer-thread up");
        apr_thread_cond_signal(prod->cond);
        apr_thread_mutex_unlock(prod->mutex);
        return TRUE;
    } else {
        return FALSE;
    }
}

/**
 * Thread execution and condition variable sample code
 * @remark Error checks omitted
 */
int main(int argc, const char *argv[])
{
    apr_status_t rv;
    apr_pool_t *mp;
    apr_thread_t *thd;
    my_production_t prod = { NULL, NULL, -1 };
        
    apr_initialize();
    apr_pool_create(&mp, NULL);

    apr_thread_mutex_create(&prod.mutex, APR_THREAD_MUTEX_UNNESTED, mp);
    apr_thread_cond_create(&prod.cond, mp);
    
    apr_thread_create(&thd, NULL, do_consume, &prod, mp);

    /* if it returns FALSE, the consumer thread still sleeps */
    do_produce(&prod);

    apr_thread_join(&rv, thd);

    apr_terminate();
    return 0;
}

/**
 * Thread entry point
 * This thread sleeps until the other thread wakes up by my_production_t::cond.
 */
static void* APR_THREAD_FUNC do_consume(apr_thread_t *thd, void *data)
{
    my_production_t *prod = data;

    puts("consumer thread is sleeping(blocking)...");
    apr_thread_mutex_lock(prod->mutex);
    while (prod->input_num == -1) {
        apr_thread_cond_wait(prod->cond, prod->mutex);
    }
    apr_thread_mutex_unlock(prod->mutex);
    printf("consumer thread is waken, getting the input %d\n", prod->input_num);

    apr_thread_exit(thd, APR_SUCCESS);
    return NULL;
}
