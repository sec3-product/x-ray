/******************************************
 * thread example with apr
 * Sylvain Marechal
 * 25/10/2005
 *****************************************/
#include "apr_general.h"
#include "apr_thread_proc.h"
#include "apr_thread_mutex.h"
#include "apr_thread_rwlock.h"


static int global_rdlock_counter = 0;

/*********************************************
 * thread_writer()
 ********************************************/
static void * APR_THREAD_FUNC thread_writer( apr_thread_t * thread, void *data )
{
  apr_thread_rwlock_t * rwlock = (apr_thread_rwlock_t *)data;
  int i;
  for( i = 0; i < 1000000; i ++ )
  {
    apr_thread_rwlock_wrlock( rwlock );
    global_rdlock_counter ++;
    apr_thread_rwlock_unlock( rwlock );
  }
  return NULL;
}

/*********************************************
 * thread_reader()
 ********************************************/
static void * APR_THREAD_FUNC thread_reader( apr_thread_t * thread, void *data )
{
  apr_thread_rwlock_t * rwlock = (apr_thread_rwlock_t *)data;
  int i;
  for( i = 0; i < 1000000; i ++ )
  {
    int a;
    apr_thread_rwlock_rdlock( rwlock );
    a = global_rdlock_counter;
    global_rdlock_counter++;
    apr_thread_rwlock_unlock( rwlock );
    a = a + 1;
  }
  return NULL;
}

#define READER_THREADS  10
#define WRITER_THREADS  2
/*********************************************
 * test_thread_rwlock()
 ********************************************/
void test_thread_rwlock( apr_pool_t *pool )
{
  apr_thread_t *athreads[WRITER_THREADS+READER_THREADS];
  apr_status_t status;
  apr_thread_rwlock_t * rwlock;
  int i;

  /* create the rd lock */
  status = apr_thread_rwlock_create( &rwlock, pool );
  if( status != APR_SUCCESS )
  {
    printf( "apr_thread_rwlock_create() failed %d\n", status );
    exit(1);
  }
  
  /* create some writer threads */
  for( i = 0; i < WRITER_THREADS; i ++ )
  {
    if( apr_thread_create(&athreads[i], NULL, 
      thread_writer, (void *)rwlock, pool) != APR_SUCCESS ) 
    {
      printf( "Could not create the thread\n");
      exit( -1);
    }
  }

  /* create some reader threads */
  for( i = WRITER_THREADS; i < READER_THREADS + WRITER_THREADS; i ++ )
  {
    if( apr_thread_create(&athreads[i], NULL, 
          thread_reader, (void *)rwlock, pool) != APR_SUCCESS ) 
    {
      printf( "Could not create the thread (i=%d)\n", i);
      exit( -1);
    }
  }

  /* wait end */
  for( i = 0; i < READER_THREADS + WRITER_THREADS; i ++ )
  {
    apr_thread_join( &status, athreads[i] );
  }

  apr_thread_rwlock_destroy( rwlock );

  /* */
  printf( "global_rdlock_counter=%d\n", global_rdlock_counter );
}



/*********************************************
 * main
 ********************************************/
int main( int argc, char * argv[] )
{
  apr_pool_t *pool;
  if (apr_initialize() != APR_SUCCESS) 
  {
      printf( "Could not initialize\n");
              exit(-1);
  }
  if (apr_pool_create(&pool, NULL) != APR_SUCCESS) 
  {
    printf( "Could not allocate pool\n");
    exit( -1);
  }
  test_thread_rwlock( pool );
  apr_pool_destroy( pool );
  return 0;
}

