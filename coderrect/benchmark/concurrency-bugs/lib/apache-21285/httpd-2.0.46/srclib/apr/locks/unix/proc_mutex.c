/* ====================================================================
 * The Apache Software License, Version 1.1
 *
 * Copyright (c) 2000-2003 The Apache Software Foundation.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The end-user documentation included with the redistribution,
 *    if any, must include the following acknowledgment:
 *       "This product includes software developed by the
 *        Apache Software Foundation (http://www.apache.org/)."
 *    Alternately, this acknowledgment may appear in the software itself,
 *    if and wherever such third-party acknowledgments normally appear.
 *
 * 4. The names "Apache" and "Apache Software Foundation" must
 *    not be used to endorse or promote products derived from this
 *    software without prior written permission. For written
 *    permission, please contact apache@apache.org.
 *
 * 5. Products derived from this software may not be called "Apache",
 *    nor may "Apache" appear in their name, without prior written
 *    permission of the Apache Software Foundation.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation.  For more
 * information on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 */

#include "apr.h"
#include "apr_strings.h"
#include "apr_arch_proc_mutex.h"
#include "apr_arch_file_io.h" /* for apr_mkstemp() */

APR_DECLARE(apr_status_t) apr_proc_mutex_destroy(apr_proc_mutex_t *mutex)
{
    return apr_pool_cleanup_run(mutex->pool, mutex, apr_proc_mutex_cleanup);
}


#if APR_HAS_POSIXSEM_SERIALIZE

#ifndef SEM_FAILED
#define SEM_FAILED (-1)
#endif

static void proc_mutex_posix_setup(void)
{
}

static apr_status_t proc_mutex_posix_cleanup(void *mutex_)
{
    apr_proc_mutex_t *mutex=mutex_;
    apr_status_t stat = APR_SUCCESS;
    
    if (mutex->interproc->filedes != -1) {
        if (sem_close((sem_t *)mutex->interproc->filedes) < 0) {
            stat = errno;
        }
    }
    return stat;
}    

static apr_status_t proc_mutex_posix_create(apr_proc_mutex_t *new_mutex,
                                            const char *fname)
{
    sem_t *psem;
    apr_status_t stat;
    char semname[31];
    apr_time_t now;
    unsigned long sec;
    unsigned long usec;
    
    new_mutex->interproc = apr_palloc(new_mutex->pool,
                                      sizeof(*new_mutex->interproc));
    new_mutex->interproc->filedes = -1;
    /*
     * This bogusness is to follow what appears to be the
     * lowest common denominator in Posix semaphore naming:
     *   - start with '/'
     *   - be at most 14 chars
     *   - be unique and not match anything on the filesystem
     *
     * Because of this, we ignore fname, and try our
     * own naming system. We tuck the name away, since it might
     * be useful for debugging. to  make this as robust as possible,
     * we initially try something larger (and hopefully more unique)
     * and gracefully fail down to the LCD above.
     *
     * NOTE: Darwin (Mac OS X) seems to be the most restrictive
     * implementation. Versions previous to Darwin 6.2 had the 14
     * char limit, but later rev's allow up to 31 characters.
     *
     * FIXME: There is a small window of opportunity where
     * instead of getting a new semaphore descriptor, we get
     * a previously obtained one. This can happen if the requests
     * are made at the "same time" and in the small span of time between
     * the sem_open and the sem_unlink. Use of O_EXCL does not
     * help here however...
     *
     */
    now = apr_time_now();
    sec = apr_time_sec(now);
    usec = apr_time_usec(now);
    apr_snprintf(semname, sizeof(semname), "/ApR.%lxZ%lx", sec, usec);
    psem = sem_open((const char *) semname, O_CREAT, 0644, 1);
    if ((psem == (sem_t *)SEM_FAILED) && (errno == ENAMETOOLONG)) {
        /* Oh well, good try */
        semname[13] = '\0';
        psem = sem_open((const char *) semname, O_CREAT, 0644, 1);
    }

    if (psem == (sem_t *)SEM_FAILED) {
        stat = errno;
        proc_mutex_posix_cleanup(new_mutex);
        return stat;
    }
    /* Ahhh. The joys of Posix sems. Predelete it... */
    sem_unlink((const char *) semname);
    new_mutex->interproc->filedes = (int)psem;	/* Ugg */
    new_mutex->fname = apr_pstrdup(new_mutex->pool, semname);
    apr_pool_cleanup_register(new_mutex->pool, (void *)new_mutex,
                              apr_proc_mutex_cleanup, 
                              apr_pool_cleanup_null);
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_posix_acquire(apr_proc_mutex_t *mutex)
{
    int rc;

    if ((rc = sem_wait((sem_t *)mutex->interproc->filedes)) < 0) {
        return errno;
    }
    mutex->curr_locked = 1;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_posix_release(apr_proc_mutex_t *mutex)
{
    int rc;

    if ((rc = sem_post((sem_t *)mutex->interproc->filedes)) < 0) {
        return errno;
    }
    mutex->curr_locked = 0;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_posix_child_init(apr_proc_mutex_t **mutex,
                                                apr_pool_t *cont,
                                                const char *fname)
{
    return APR_SUCCESS;
}

const apr_proc_mutex_unix_lock_methods_t apr_proc_mutex_unix_posix_methods =
{
#if APR_PROCESS_LOCK_IS_GLOBAL || !APR_HAS_THREADS || defined(POSIXSEM_IS_GLOBAL)
    APR_PROCESS_LOCK_MECH_IS_GLOBAL,
#else
    0,
#endif
    proc_mutex_posix_create,
    proc_mutex_posix_acquire,
    NULL, /* no tryacquire */
    proc_mutex_posix_release,
    proc_mutex_posix_cleanup,
    proc_mutex_posix_child_init,
    "posixsem"
};

#endif /* Posix sem implementation */

#if APR_HAS_SYSVSEM_SERIALIZE

static struct sembuf proc_mutex_op_on;
static struct sembuf proc_mutex_op_off;

static void proc_mutex_sysv_setup(void)
{
    proc_mutex_op_on.sem_num = 0;
    proc_mutex_op_on.sem_op = -1;
    proc_mutex_op_on.sem_flg = SEM_UNDO;
    proc_mutex_op_off.sem_num = 0;
    proc_mutex_op_off.sem_op = 1;
    proc_mutex_op_off.sem_flg = SEM_UNDO;
}

static apr_status_t proc_mutex_sysv_cleanup(void *mutex_)
{
    apr_proc_mutex_t *mutex=mutex_;
    union semun ick;
    
    if (mutex->interproc->filedes != -1) {
        ick.val = 0;
        semctl(mutex->interproc->filedes, 0, IPC_RMID, ick);
    }
    return APR_SUCCESS;
}    

static apr_status_t proc_mutex_sysv_create(apr_proc_mutex_t *new_mutex,
                                           const char *fname)
{
    union semun ick;
    apr_status_t rv;
    
    new_mutex->interproc = apr_palloc(new_mutex->pool, sizeof(*new_mutex->interproc));
    new_mutex->interproc->filedes = semget(IPC_PRIVATE, 1, IPC_CREAT | 0600);

    if (new_mutex->interproc->filedes < 0) {
        rv = errno;
        proc_mutex_sysv_cleanup(new_mutex);
        return rv;
    }
    ick.val = 1;
    if (semctl(new_mutex->interproc->filedes, 0, SETVAL, ick) < 0) {
        rv = errno;
        proc_mutex_sysv_cleanup(new_mutex);
        return rv;
    }
    new_mutex->curr_locked = 0;
    apr_pool_cleanup_register(new_mutex->pool,
                              (void *)new_mutex, apr_proc_mutex_cleanup, 
                              apr_pool_cleanup_null);
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_sysv_acquire(apr_proc_mutex_t *mutex)
{
    int rc;

    do {
        rc = semop(mutex->interproc->filedes, &proc_mutex_op_on, 1);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
        return errno;
    }
    mutex->curr_locked = 1;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_sysv_release(apr_proc_mutex_t *mutex)
{
    int rc;

    do {
        rc = semop(mutex->interproc->filedes, &proc_mutex_op_off, 1);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
        return errno;
    }
    mutex->curr_locked = 0;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_sysv_child_init(apr_proc_mutex_t **mutex, apr_pool_t *cont, const char *fname)
{
    return APR_SUCCESS;
}

const apr_proc_mutex_unix_lock_methods_t apr_proc_mutex_unix_sysv_methods =
{
#if APR_PROCESS_LOCK_IS_GLOBAL || !APR_HAS_THREADS || defined(SYSVSEM_IS_GLOBAL)
    APR_PROCESS_LOCK_MECH_IS_GLOBAL,
#else
    0,
#endif
    proc_mutex_sysv_create,
    proc_mutex_sysv_acquire,
    NULL, /* no tryacquire */
    proc_mutex_sysv_release,
    proc_mutex_sysv_cleanup,
    proc_mutex_sysv_child_init,
    "sysvsem"
};

#endif /* SysV sem implementation */

#if APR_HAS_PROC_PTHREAD_SERIALIZE

static void proc_mutex_proc_pthread_setup(void)
{
}

static apr_status_t proc_mutex_proc_pthread_cleanup(void *mutex_)
{
    apr_proc_mutex_t *mutex=mutex_;
    apr_status_t rv;

    if (mutex->curr_locked == 1) {
        if ((rv = pthread_mutex_unlock(mutex->pthread_interproc))) {
#ifdef PTHREAD_SETS_ERRNO
            rv = errno;
#endif
            return rv;
        } 
        if (munmap((caddr_t)mutex->pthread_interproc, sizeof(pthread_mutex_t))){
            return errno;
        }
    }
    return APR_SUCCESS;
}    

static apr_status_t proc_mutex_proc_pthread_create(apr_proc_mutex_t *new_mutex,
                                                   const char *fname)
{
    apr_status_t rv;
    int fd;
    pthread_mutexattr_t mattr;

    fd = open("/dev/zero", O_RDWR);
    if (fd < 0) {
        return errno;
    }

    new_mutex->pthread_interproc = (pthread_mutex_t *)mmap(
                                       (caddr_t) 0, 
                                       sizeof(pthread_mutex_t), 
                                       PROT_READ | PROT_WRITE, MAP_SHARED,
                                       fd, 0); 
    if (new_mutex->pthread_interproc == (pthread_mutex_t *) (caddr_t) -1) {
        return errno;
    }
    close(fd);
    if ((rv = pthread_mutexattr_init(&mattr))) {
#ifdef PTHREAD_SETS_ERRNO
        rv = errno;
#endif
        proc_mutex_proc_pthread_cleanup(new_mutex);
        return rv;
    }
    if ((rv = pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED))) {
#ifdef PTHREAD_SETS_ERRNO
        rv = errno;
#endif
        proc_mutex_proc_pthread_cleanup(new_mutex);
        return rv;
    }

#ifdef HAVE_PTHREAD_MUTEXATTR_SETROBUST_NP
    if ((rv = pthread_mutexattr_setrobust_np(&mattr, 
                                               PTHREAD_MUTEX_ROBUST_NP))) {
#ifdef PTHREAD_SETS_ERRNO
        rv = errno;
#endif
        proc_mutex_proc_pthread_cleanup(new_mutex);
        return rv;
    }
    if ((rv = pthread_mutexattr_setprotocol(&mattr, PTHREAD_PRIO_INHERIT))) {
#ifdef PTHREAD_SETS_ERRNO
        rv = errno;
#endif
        proc_mutex_proc_pthread_cleanup(new_mutex);
        return rv;
    }
#endif

    if ((rv = pthread_mutex_init(new_mutex->pthread_interproc, &mattr))) {
#ifdef PTHREAD_SETS_ERRNO
        rv = errno;
#endif
        proc_mutex_proc_pthread_cleanup(new_mutex);
        return rv;
    }

    if ((rv = pthread_mutexattr_destroy(&mattr))) {
#ifdef PTHREAD_SETS_ERRNO
        rv = errno;
#endif
        proc_mutex_proc_pthread_cleanup(new_mutex);
        return rv;
    }

    new_mutex->curr_locked = 0;
    apr_pool_cleanup_register(new_mutex->pool,
                              (void *)new_mutex,
                              apr_proc_mutex_cleanup, 
                              apr_pool_cleanup_null);
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_proc_pthread_acquire(apr_proc_mutex_t *mutex)
{
    apr_status_t rv;

    if ((rv = pthread_mutex_lock(mutex->pthread_interproc))) {
#ifdef PTHREAD_SETS_ERRNO
        rv = errno;
#endif
#ifdef HAVE_PTHREAD_MUTEXATTR_SETROBUST_NP
        /* Okay, our owner died.  Let's try to make it consistent again. */
        if (rv == EOWNERDEAD) {
            pthread_mutex_consistent_np(mutex->pthread_interproc);
        }
        else
            return rv;
#else
        return rv;
#endif
    }
    mutex->curr_locked = 1;
    return APR_SUCCESS;
}

/* TODO: Add proc_mutex_proc_pthread_tryacquire(apr_proc_mutex_t *mutex) */

static apr_status_t proc_mutex_proc_pthread_release(apr_proc_mutex_t *mutex)
{
    apr_status_t rv;

    if ((rv = pthread_mutex_unlock(mutex->pthread_interproc))) {
#ifdef PTHREAD_SETS_ERRNO
        rv = errno;
#endif
        return rv;
    }
    mutex->curr_locked = 0;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_proc_pthread_child_init(apr_proc_mutex_t **mutex,
                                            apr_pool_t *cont, 
                                            const char *fname)
{
    return APR_SUCCESS;
}

const apr_proc_mutex_unix_lock_methods_t apr_proc_mutex_unix_proc_pthread_methods =
{
    APR_PROCESS_LOCK_MECH_IS_GLOBAL,
    proc_mutex_proc_pthread_create,
    proc_mutex_proc_pthread_acquire,
    NULL, /* no tryacquire */
    proc_mutex_proc_pthread_release,
    proc_mutex_proc_pthread_cleanup,
    proc_mutex_proc_pthread_child_init,
    "pthread"
};

#endif

#if APR_HAS_FCNTL_SERIALIZE

static struct flock proc_mutex_lock_it;
static struct flock proc_mutex_unlock_it;

static apr_status_t proc_mutex_fcntl_release(apr_proc_mutex_t *);

static void proc_mutex_fcntl_setup(void)
{
    proc_mutex_lock_it.l_whence = SEEK_SET;   /* from current point */
    proc_mutex_lock_it.l_start = 0;           /* -"- */
    proc_mutex_lock_it.l_len = 0;             /* until end of file */
    proc_mutex_lock_it.l_type = F_WRLCK;      /* set exclusive/write lock */
    proc_mutex_lock_it.l_pid = 0;             /* pid not actually interesting */
    proc_mutex_unlock_it.l_whence = SEEK_SET; /* from current point */
    proc_mutex_unlock_it.l_start = 0;         /* -"- */
    proc_mutex_unlock_it.l_len = 0;           /* until end of file */
    proc_mutex_unlock_it.l_type = F_UNLCK;    /* set exclusive/write lock */
    proc_mutex_unlock_it.l_pid = 0;           /* pid not actually interesting */
}

static apr_status_t proc_mutex_fcntl_cleanup(void *mutex_)
{
    apr_status_t status;
    apr_proc_mutex_t *mutex=mutex_;

    if (mutex->curr_locked == 1) {
        status = proc_mutex_fcntl_release(mutex);
        if (status != APR_SUCCESS)
            return status;
    }
    if (mutex->interproc) { /* if it was opened successfully */
        apr_file_close(mutex->interproc);
    }
    return APR_SUCCESS;
}    

static apr_status_t proc_mutex_fcntl_create(apr_proc_mutex_t *new_mutex,
                                            const char *fname)
{
    int rv;
 
    if (fname) {
        new_mutex->fname = apr_pstrdup(new_mutex->pool, fname);
        rv = apr_file_open(&new_mutex->interproc, new_mutex->fname,
                           APR_CREATE | APR_WRITE | APR_EXCL, 
                           APR_UREAD | APR_UWRITE | APR_GREAD | APR_WREAD,
                           new_mutex->pool);
    }
    else {
        new_mutex->fname = apr_pstrdup(new_mutex->pool, "/tmp/aprXXXXXX");
        rv = apr_file_mktemp(&new_mutex->interproc, new_mutex->fname, 0,
                             new_mutex->pool);
    }
 
    if (rv != APR_SUCCESS) {
        proc_mutex_fcntl_cleanup(new_mutex);
        return rv;
    }

    new_mutex->curr_locked = 0;
    /* XXX currently, apr_file_mktemp() always specifies that the file should
     *     be removed when closed; that unlink() will fail since we're 
     *     removing it here; we want to remove it here since we don't need
     *     it visible and we want it cleaned up if we exit catastrophically
     */
    unlink(new_mutex->fname);
    apr_pool_cleanup_register(new_mutex->pool,
                              (void*)new_mutex,
                              apr_proc_mutex_cleanup, 
                              apr_pool_cleanup_null);
    return APR_SUCCESS; 
}

static apr_status_t proc_mutex_fcntl_acquire(apr_proc_mutex_t *mutex)
{
    int rc;

    do {
        rc = fcntl(mutex->interproc->filedes, F_SETLKW, &proc_mutex_lock_it);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
        return errno;
    }
    mutex->curr_locked=1;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_fcntl_release(apr_proc_mutex_t *mutex)
{
    int rc;

    do {
        rc = fcntl(mutex->interproc->filedes, F_SETLKW, &proc_mutex_unlock_it);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
        return errno;
    }
    mutex->curr_locked=0;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_fcntl_child_init(apr_proc_mutex_t **mutex,
                                                apr_pool_t *pool, 
                                                const char *fname)
{
    return APR_SUCCESS;
}

const apr_proc_mutex_unix_lock_methods_t apr_proc_mutex_unix_fcntl_methods =
{
#if APR_PROCESS_LOCK_IS_GLOBAL || !APR_HAS_THREADS || defined(FCNTL_IS_GLOBAL)
    APR_PROCESS_LOCK_MECH_IS_GLOBAL,
#else
    0,
#endif
    proc_mutex_fcntl_create,
    proc_mutex_fcntl_acquire,
    NULL, /* no tryacquire */
    proc_mutex_fcntl_release,
    proc_mutex_fcntl_cleanup,
    proc_mutex_fcntl_child_init,
    "fcntl"
};

#endif /* fcntl implementation */

#if APR_HAS_FLOCK_SERIALIZE

static apr_status_t proc_mutex_flock_release(apr_proc_mutex_t *);

static void proc_mutex_flock_setup(void)
{
}

static apr_status_t proc_mutex_flock_cleanup(void *mutex_)
{
    apr_status_t status;
    apr_proc_mutex_t *mutex=mutex_;

    if (mutex->curr_locked == 1) {
        status = proc_mutex_flock_release(mutex);
        if (status != APR_SUCCESS)
            return status;
    }
    if (mutex->interproc) { /* if it was opened properly */
        apr_file_close(mutex->interproc);
    }
    unlink(mutex->fname);
    return APR_SUCCESS;
}    

static apr_status_t proc_mutex_flock_create(apr_proc_mutex_t *new_mutex,
                                            const char *fname)
{
    int rv;
 
    if (fname) {
        new_mutex->fname = apr_pstrdup(new_mutex->pool, fname);
        rv = apr_file_open(&new_mutex->interproc, new_mutex->fname,
                           APR_CREATE | APR_WRITE | APR_EXCL, 
                           APR_UREAD | APR_UWRITE,
                           new_mutex->pool);
    }
    else {
        new_mutex->fname = apr_pstrdup(new_mutex->pool, "/tmp/aprXXXXXX");
        rv = apr_file_mktemp(&new_mutex->interproc, new_mutex->fname, 0,
                             new_mutex->pool);
    }
 
    if (rv != APR_SUCCESS) {
        proc_mutex_flock_cleanup(new_mutex);
        return errno;
    }
    new_mutex->curr_locked = 0;
    apr_pool_cleanup_register(new_mutex->pool, (void *)new_mutex,
                              apr_proc_mutex_cleanup,
                              apr_pool_cleanup_null);
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_flock_acquire(apr_proc_mutex_t *mutex)
{
    int rc;

    do {
        rc = flock(mutex->interproc->filedes, LOCK_EX);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
        return errno;
    }
    mutex->curr_locked = 1;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_flock_release(apr_proc_mutex_t *mutex)
{
    int rc;

    do {
        rc = flock(mutex->interproc->filedes, LOCK_UN);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
        return errno;
    }
    mutex->curr_locked = 0;
    return APR_SUCCESS;
}

static apr_status_t proc_mutex_flock_child_init(apr_proc_mutex_t **mutex,
                                                apr_pool_t *pool, 
                                                const char *fname)
{
    apr_proc_mutex_t *new_mutex;
    int rv;

    new_mutex = (apr_proc_mutex_t *)apr_palloc(pool, sizeof(apr_proc_mutex_t));

    memcpy(new_mutex, *mutex, sizeof *new_mutex);
    new_mutex->pool = pool;
    new_mutex->fname = apr_pstrdup(pool, fname);
    rv = apr_file_open(&new_mutex->interproc, new_mutex->fname,
                       APR_WRITE, 0, new_mutex->pool);
    if (rv != APR_SUCCESS) {
        proc_mutex_flock_cleanup(new_mutex);
        return rv;
    }
    *mutex = new_mutex;
    return APR_SUCCESS;
}

const apr_proc_mutex_unix_lock_methods_t apr_proc_mutex_unix_flock_methods =
{
#if APR_PROCESS_LOCK_IS_GLOBAL || !APR_HAS_THREADS || defined(FLOCK_IS_GLOBAL)
    APR_PROCESS_LOCK_MECH_IS_GLOBAL,
#else
    0,
#endif
    proc_mutex_flock_create,
    proc_mutex_flock_acquire,
    NULL, /* no tryacquire */
    proc_mutex_flock_release,
    proc_mutex_flock_cleanup,
    proc_mutex_flock_child_init,
    "flock"
};

#endif /* flock implementation */

void apr_proc_mutex_unix_setup_lock(void)
{
#if APR_HAS_POSIXSEM_SERIALIZE
    proc_mutex_posix_setup();
#endif
#if APR_HAS_SYSVSEM_SERIALIZE
    proc_mutex_sysv_setup();
#endif
#if APR_HAS_PROC_PTHREAD_SERIALIZE
    proc_mutex_proc_pthread_setup();
#endif
#if APR_HAS_FCNTL_SERIALIZE
    proc_mutex_fcntl_setup();
#endif
#if APR_HAS_FLOCK_SERIALIZE
    proc_mutex_flock_setup();
#endif
}

static apr_status_t proc_mutex_choose_method(apr_proc_mutex_t *new_mutex, apr_lockmech_e mech)
{
    switch (mech) {
    case APR_LOCK_FCNTL:
#if APR_HAS_FCNTL_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_fcntl_methods;
#else
        return APR_ENOTIMPL;
#endif
        break;
    case APR_LOCK_FLOCK:
#if APR_HAS_FLOCK_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_flock_methods;
#else
        return APR_ENOTIMPL;
#endif
        break;
    case APR_LOCK_SYSVSEM:
#if APR_HAS_SYSVSEM_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_sysv_methods;
#else
        return APR_ENOTIMPL;
#endif
        break;
    case APR_LOCK_POSIXSEM:
#if APR_HAS_POSIXSEM_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_posix_methods;
#else
        return APR_ENOTIMPL;
#endif
        break;
    case APR_LOCK_PROC_PTHREAD:
#if APR_HAS_PROC_PTHREAD_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_proc_pthread_methods;
#else
        return APR_ENOTIMPL;
#endif
        break;
    case APR_LOCK_DEFAULT:
#if APR_USE_FLOCK_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_flock_methods;
#elif APR_USE_SYSVSEM_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_sysv_methods;
#elif APR_USE_FCNTL_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_fcntl_methods;
#elif APR_USE_PROC_PTHREAD_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_proc_pthread_methods;
#elif APR_USE_POSIXSEM_SERIALIZE
        new_mutex->inter_meth = &apr_proc_mutex_unix_posix_methods;
#else
        return APR_ENOTIMPL;
#endif
        break;
    default:
        return APR_ENOTIMPL;
    }
    return APR_SUCCESS;
}

APR_DECLARE(const char *) apr_proc_mutex_defname(void)
{
    apr_status_t rv;
    apr_proc_mutex_t mutex;

    if ((rv = proc_mutex_choose_method(&mutex, APR_LOCK_DEFAULT)) != APR_SUCCESS) {
        return "unknown";
    }
    mutex.meth = mutex.inter_meth;

    return apr_proc_mutex_name(&mutex);
}
   
static apr_status_t proc_mutex_create(apr_proc_mutex_t *new_mutex, apr_lockmech_e mech, const char *fname)
{
    apr_status_t rv;

    if ((rv = proc_mutex_choose_method(new_mutex, mech)) != APR_SUCCESS) {
        return rv;
    }

    new_mutex->meth = new_mutex->inter_meth;

    if ((rv = new_mutex->meth->create(new_mutex, fname)) != APR_SUCCESS) {
        return rv;
    }

    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_proc_mutex_create(apr_proc_mutex_t **mutex,
                                                const char *fname,
                                                apr_lockmech_e mech,
                                                apr_pool_t *pool)
{
    apr_proc_mutex_t *new_mutex;
    apr_status_t rv;

    new_mutex = (apr_proc_mutex_t *)apr_pcalloc(pool,
                                                sizeof(apr_proc_mutex_t));

    new_mutex->pool  = pool;
#if APR_HAS_SYSVSEM_SERIALIZE || APR_HAS_FCNTL_SERIALIZE || APR_HAS_FLOCK_SERIALIZE || APR_HAS_POSIXSEM_SERIALIZE
    new_mutex->interproc = NULL;
#endif

    if ((rv = proc_mutex_create(new_mutex, mech, fname)) != APR_SUCCESS)
        return rv;

    *mutex = new_mutex;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_proc_mutex_child_init(apr_proc_mutex_t **mutex,
                                                    const char *fname,
                                                    apr_pool_t *pool)
{
    return (*mutex)->meth->child_init(mutex, pool, fname);
}

APR_DECLARE(apr_status_t) apr_proc_mutex_lock(apr_proc_mutex_t *mutex)
{
    return mutex->meth->acquire(mutex);
}

APR_DECLARE(apr_status_t) apr_proc_mutex_trylock(apr_proc_mutex_t *mutex)
{
    return mutex->meth->tryacquire(mutex);
}

APR_DECLARE(apr_status_t) apr_proc_mutex_unlock(apr_proc_mutex_t *mutex)
{
    return mutex->meth->release(mutex);
}

APR_DECLARE(apr_status_t) apr_proc_mutex_cleanup(void *mutex)
{
    return ((apr_proc_mutex_t *)mutex)->meth->cleanup(mutex);
}

APR_DECLARE(const char *) apr_proc_mutex_name(apr_proc_mutex_t *mutex)
{
    return mutex->meth->name;
}

APR_POOL_IMPLEMENT_ACCESSOR(proc_mutex)

/* Implement OS-specific accessors defined in apr_portable.h */

APR_DECLARE(apr_status_t) apr_os_proc_mutex_get(apr_os_proc_mutex_t *ospmutex,
                                                apr_proc_mutex_t *pmutex)
{
#if APR_HAS_SYSVSEM_SERIALIZE || APR_HAS_FCNTL_SERIALIZE || APR_HAS_FLOCK_SERIALIZE || APR_HAS_POSIXSEM_SERIALIZE
    ospmutex->crossproc = pmutex->interproc->filedes;
#endif
#if APR_HAS_PROC_PTHREAD_SERIALIZE
    ospmutex->pthread_interproc = pmutex->pthread_interproc;
#endif
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_os_proc_mutex_put(apr_proc_mutex_t **pmutex,
                                                apr_os_proc_mutex_t *ospmutex,
                                                apr_pool_t *pool)
{
    if (pool == NULL) {
        return APR_ENOPOOL;
    }
    if ((*pmutex) == NULL) {
        (*pmutex) = (apr_proc_mutex_t *)apr_pcalloc(pool,
                                                    sizeof(apr_proc_mutex_t));
        (*pmutex)->pool = pool;
    }
#if APR_HAS_SYSVSEM_SERIALIZE || APR_HAS_FCNTL_SERIALIZE || APR_HAS_FLOCK_SERIALIZE || APR_HAS_POSIXSEM_SERIALIZE
    apr_os_file_put(&(*pmutex)->interproc, &ospmutex->crossproc, 0, pool);
#endif
#if APR_HAS_PROC_PTHREAD_SERIALIZE
    (*pmutex)->pthread_interproc = ospmutex->pthread_interproc;
#endif
    return APR_SUCCESS;
}

