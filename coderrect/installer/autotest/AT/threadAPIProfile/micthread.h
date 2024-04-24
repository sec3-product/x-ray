#include <pthread.h>

extern int micthread_create (pthread_t * threadName,
                           const pthread_attr_t * attr,
                           void *(*func) (void *),
                           void * args);

extern pthread_t micthread_self (void);

extern void micthread_exit (void * retval);

extern int micthread_mutex_init (pthread_mutex_t * mutex, const pthread_mutexattr_t * mutexattr);

extern int micthread_mutex_lock (pthread_mutex_t * mutex);

extern int micthread_mutex_unlock (pthread_mutex_t * mutex);

extern int micthread_join (pthread_t threadName, void ** threadret);

extern int micthread_rwlock_rdlock (pthread_rwlock_t * rwlock);

extern int micthread_rwlock_wrlock (pthread_rwlock_t * rwlock);

extern int micthread_rwlock_unlock (pthread_rwlock_t * rwlock);

extern int micthread_rwlock_init (pthread_rwlock_t *  rwlock, const pthread_rwlockattr_t * attr);

extern int micthread_spin_lock (pthread_spinlock_t * lock);

extern int micthread_spin_unlock (pthread_spinlock_t * lock);

extern int micthread_spin_init (pthread_spinlock_t * lock, int pshared);
