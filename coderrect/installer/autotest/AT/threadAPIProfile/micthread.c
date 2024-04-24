#include <pthread.h>
#include <stdio.h>

int micthread_create (pthread_t * threadName,
                           const pthread_attr_t * attr,
                           void *(*func) (void *),
                           void * args) {
	printf("a new thread is created.\n");
	return pthread_create(threadName, attr, func, args);
}

pthread_t micthread_self (void) {
	return pthread_self();
}

void micthread_exit (void * retval) {
	int tid = micthread_self();
	printf("thread %d exit running.\n", tid);	
	pthread_exit(retval);
}

int micthread_mutex_init (pthread_mutex_t * mutex, const pthread_mutexattr_t * mutexattr) {
        return pthread_mutex_init(mutex, mutexattr);
}

int micthread_mutex_lock (pthread_mutex_t * mutex) {
	return pthread_mutex_lock(mutex);
}

int micthread_mutex_unlock (pthread_mutex_t * mutex) {
	return pthread_mutex_unlock(mutex);
}

int micthread_join (pthread_t threadName, void ** threadret) {
	return pthread_join(threadName, threadret);
}

int micthread_rwlock_rdlock (pthread_rwlock_t * rwlock) {
	return pthread_rwlock_rdlock(rwlock);
}

int micthread_rwlock_wrlock (pthread_rwlock_t * rwlock) {
	return pthread_rwlock_wrlock(rwlock);
}

int micthread_rwlock_unlock (pthread_rwlock_t * rwlock) {
	return pthread_rwlock_unlock(rwlock);
}

int micthread_rwlock_init (pthread_rwlock_t *  rwlock, const pthread_rwlockattr_t * attr) {
	return pthread_rwlock_init(rwlock, attr);
}

int micthread_spin_lock (pthread_spinlock_t * lock) {
	return pthread_spin_lock(lock);
}

int micthread_spin_unlock (pthread_spinlock_t * lock) {
	return pthread_spin_unlock(lock);
}

int micthread_spin_init (pthread_spinlock_t * lock, int pshared) {
	return pthread_spin_init(lock, pshared);
}
