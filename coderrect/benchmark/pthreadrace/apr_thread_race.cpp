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

/**
 * The number of running threads concurrently
 */
#define NUM_THREADS	64

/**
 * In general, the following macros are not portable.
 * However, if the integer value is small, converting it to pointer and re-converting from it could work safely.
 */
#define INT_TO_POINTER(i)	((void*)(i))
#define POINTER_TO_INT(p)	((int)(p))

int x;

static void* APR_THREAD_FUNC doit(apr_thread_t *thd, void *data);

/**
 * Thread execution sample code
 * @remark Error checks omitted
 */
int main(int argc, const char *argv[])
{
    apr_status_t rv;
    apr_pool_t *mp;
    apr_thread_t *thd_arr[NUM_THREADS];
    apr_threadattr_t *thd_attr;
    int i;
        
    apr_initialize();
    apr_pool_create(&mp, NULL);

    /* The default thread attribute: detachable */
    apr_threadattr_create(&thd_attr, mp);

    for (i = 0; i < NUM_THREADS; i++) {
        /* If the thread attribute is a default value, you can pass NULL to the second argument */
        rv = apr_thread_create(&thd_arr[i], thd_attr, doit, INT_TO_POINTER(i), mp);
        assert(rv == APR_SUCCESS);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        rv = apr_thread_join(&rv, thd_arr[i]);
        assert(rv == APR_SUCCESS);
    }

    apr_terminate();
    return 0;
}

/**
 * Thread entry point
 */
static void* APR_THREAD_FUNC doit(apr_thread_t *thd, void *data)
{
    long num = long(data);
    x += 1;
    
    printf("doit:%d, x:%d\n", num, x);
    apr_thread_exit(thd, APR_SUCCESS);

    return NULL;
}
