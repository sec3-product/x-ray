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

#include "apr_arch_shm.h"

#include "apr_general.h"
#include "apr_errno.h"
#include "apr_user.h"
#include "apr_strings.h"

static apr_status_t shm_cleanup_owner(void *m_)
{
    apr_shm_t *m = (apr_shm_t *)m_;

    /* anonymous shared memory */
    if (m->filename == NULL) {
#if APR_USE_SHMEM_MMAP_ZERO || APR_USE_SHMEM_MMAP_ANON
        if (munmap(m->base, m->realsize) == -1) {
            return errno;
        }
        return APR_SUCCESS;
#endif
#if APR_USE_SHMEM_SHMGET_ANON
        if (shmdt(m->base) == -1) {
            return errno;
        }
        /* This segment will automatically remove itself after all
         * references have detached. */
        return APR_SUCCESS;
#endif
    }

    /* name-based shared memory */
    else {
#if APR_USE_SHMEM_MMAP_TMP
        apr_status_t rv;

        if (munmap(m->base, m->realsize) == -1) {
            return errno;
        }
        rv = apr_file_remove(m->filename, m->pool);
        if (rv != APR_SUCCESS) {
            return rv;
        }
        return APR_SUCCESS;
#endif
#if APR_USE_SHMEM_MMAP_SHM
        if (munmap(m->base, m->realsize) == -1) {
            return errno;
        }
        if (shm_unlink(m->filename) == -1) {
            return errno;
        }
        return APR_SUCCESS;
#endif
#if APR_USE_SHMEM_SHMGET
        apr_status_t rv;

        /* Indicate that the segment is to be destroyed as soon
         * as all processes have detached. This also disallows any
         * new attachments to the segment. */
        if (shmctl(m->shmid, IPC_RMID, NULL) == -1) {
            return errno;
        }
        if (shmdt(m->base) == -1) {
            return errno;
        }
        rv = apr_file_remove(m->filename, m->pool);
        if (rv != APR_SUCCESS) {
            return rv;
        }
        return APR_SUCCESS;
#endif
    }

    return APR_ENOTIMPL;
}

APR_DECLARE(apr_status_t) apr_shm_create(apr_shm_t **m,
                                         apr_size_t reqsize, 
                                         const char *filename,
                                         apr_pool_t *pool)
{
    apr_shm_t *new_m;
    apr_status_t status;
#if APR_USE_SHMEM_SHMGET || APR_USE_SHMEM_SHMGET_ANON
    struct shmid_ds shmbuf;
    apr_uid_t uid;
    apr_gid_t gid;
#endif
#if APR_USE_SHMEM_MMAP_TMP || APR_USE_SHMEM_MMAP_SHM || \
    APR_USE_SHMEM_MMAP_ZERO
    int tmpfd;
#endif
#if APR_USE_SHMEM_SHMGET
    apr_size_t nbytes;
    key_t shmkey;
#endif
#if APR_USE_SHMEM_MMAP_ZERO || APR_USE_SHMEM_SHMGET || \
    APR_USE_SHMEM_MMAP_TMP || APR_USE_SHMEM_MMAP_SHM
    apr_file_t *file;   /* file where metadata is stored */
#endif

    /* Check if they want anonymous or name-based shared memory */
    if (filename == NULL) {
#if APR_USE_SHMEM_MMAP_ZERO || APR_USE_SHMEM_MMAP_ANON
        new_m = apr_palloc(pool, sizeof(apr_shm_t));
        if (!new_m) {
            return APR_ENOMEM;
        }
        new_m->pool = pool;
        new_m->reqsize = reqsize;
        new_m->realsize = reqsize + 
            APR_ALIGN_DEFAULT(sizeof(apr_size_t)); /* room for metadata */
        new_m->filename = NULL;
    
#if APR_USE_SHMEM_MMAP_ZERO
        status = apr_file_open(&file, "/dev/zero", APR_READ | APR_WRITE, 
                               APR_OS_DEFAULT, pool);
        if (status != APR_SUCCESS) {
            return status;
        }
        status = apr_os_file_get(&tmpfd, file);
        if (status != APR_SUCCESS) {
            return status;
        }

        new_m->base = mmap(NULL, new_m->realsize, PROT_READ|PROT_WRITE,
                           MAP_SHARED, tmpfd, 0);
        if (new_m->base == MAP_FAILED) {
            return errno;
        }

        status = apr_file_close(file);
        if (status != APR_SUCCESS) {
            return status;
        }

        /* store the real size in the metadata */
        *(apr_size_t*)(new_m->base) = new_m->realsize;
        /* metadata isn't usable */
        new_m->usable = (char *)new_m->base + APR_ALIGN_DEFAULT(sizeof(apr_size_t));

        apr_pool_cleanup_register(new_m->pool, new_m, shm_cleanup_owner,
                                  apr_pool_cleanup_null);
        *m = new_m;
        return APR_SUCCESS;

#elif APR_USE_SHMEM_MMAP_ANON
        new_m->base = mmap(NULL, new_m->realsize, PROT_READ|PROT_WRITE,
                           MAP_ANON|MAP_SHARED, -1, 0);
        if (new_m->base == MAP_FAILED) {
            return errno;
        }

        /* store the real size in the metadata */
        *(apr_size_t*)(new_m->base) = new_m->realsize;
        /* metadata isn't usable */
        new_m->usable = (char *)new_m->base + APR_ALIGN_DEFAULT(sizeof(apr_size_t));

        apr_pool_cleanup_register(new_m->pool, new_m, shm_cleanup_owner,
                                  apr_pool_cleanup_null);
        *m = new_m;
        return APR_SUCCESS;

#endif /* APR_USE_SHMEM_MMAP_ZERO */
#endif /* APR_USE_SHMEM_MMAP_ZERO || APR_USE_SHMEM_MMAP_ANON */
#if APR_USE_SHMEM_SHMGET_ANON

        new_m = apr_palloc(pool, sizeof(apr_shm_t));
        if (!new_m) {
            return APR_ENOMEM;
        }
        new_m->pool = pool;
        new_m->reqsize = reqsize;
        new_m->realsize = reqsize;
        new_m->filename = NULL;

        if ((new_m->shmid = shmget(IPC_PRIVATE, new_m->realsize,
                                   SHM_R | SHM_W | IPC_CREAT)) < 0) {
            return errno;
        }

        if ((new_m->base = shmat(new_m->shmid, NULL, 0)) == (void *)-1) {
            return errno;
        }
        new_m->usable = new_m->base;

        if (shmctl(new_m->shmid, IPC_STAT, &shmbuf) == -1) {
            return errno;
        }
        apr_uid_current(&uid, &gid, pool);
        shmbuf.shm_perm.uid = uid;
        shmbuf.shm_perm.gid = gid;
        if (shmctl(new_m->shmid, IPC_SET, &shmbuf) == -1) {
            return errno;
        }

        /* Remove the segment once use count hits zero.
         * We will not attach to this segment again, since it is
         * anonymous memory, so it is ok to mark it for deletion.
         */
        if (shmctl(new_m->shmid, IPC_RMID, NULL) == -1) {
            return errno;
        }

        apr_pool_cleanup_register(new_m->pool, new_m, shm_cleanup_owner,
                                  apr_pool_cleanup_null);
        *m = new_m;
        return APR_SUCCESS;
#endif /* APR_USE_SHMEM_SHMGET_ANON */
        /* It is an error if they want anonymous memory but we don't have it. */
        return APR_ENOTIMPL; /* requested anonymous but we don't have it */
    }

    /* Name-based shared memory */
    else {
        new_m = apr_palloc(pool, sizeof(apr_shm_t));
        if (!new_m) {
            return APR_ENOMEM;
        }
        new_m->pool = pool;
        new_m->reqsize = reqsize;
        new_m->filename = apr_pstrdup(pool, filename);

#if APR_USE_SHMEM_MMAP_TMP || APR_USE_SHMEM_MMAP_SHM
        new_m->realsize = reqsize + 
            APR_ALIGN_DEFAULT(sizeof(apr_size_t)); /* room for metadata */
        /* FIXME: Ignore error for now. *
         * status = apr_file_remove(file, pool);*/
        status = APR_SUCCESS;
    
#if APR_USE_SHMEM_MMAP_TMP
        /* FIXME: Is APR_OS_DEFAULT sufficient? */
        status = apr_file_open(&file, filename, 
                               APR_READ | APR_WRITE | APR_CREATE | APR_EXCL,
                               APR_OS_DEFAULT, pool);
        if (status != APR_SUCCESS) {
            return status;
        }

        status = apr_os_file_get(&tmpfd, file);
        if (status != APR_SUCCESS) {
            apr_file_close(file); /* ignore errors, we're failing */
            apr_file_remove(new_m->filename, new_m->pool);
            return status;
        }

        status = apr_file_trunc(file, new_m->realsize);
        if (status != APR_SUCCESS) {
            apr_file_close(file); /* ignore errors, we're failing */
            apr_file_remove(new_m->filename, new_m->pool);
            return status;
        }

        new_m->base = mmap(NULL, new_m->realsize, PROT_READ | PROT_WRITE,
                           MAP_SHARED, tmpfd, 0);
        /* FIXME: check for errors */

        status = apr_file_close(file);
        if (status != APR_SUCCESS) {
            return status;
        }
#endif /* APR_USE_SHMEM_MMAP_TMP */
#if APR_USE_SHMEM_MMAP_SHM
        /* FIXME: Is APR_OS_DEFAULT sufficient? */
        tmpfd = shm_open(filename, O_RDWR | O_CREAT | O_EXCL, APR_OS_DEFAULT);
        if (tmpfd == -1) {
            return errno;
        }

        status = apr_os_file_put(&file, &tmpfd,
                                 APR_READ | APR_WRITE | APR_CREATE | APR_EXCL,
                                 pool); 
        if (status != APR_SUCCESS) {
            return status;
        }

        status = apr_file_trunc(file, new_m->realsize);
        if (status != APR_SUCCESS) {
            shm_unlink(filename); /* we're failing, remove the object */
            return status;
        }
        new_m->base = mmap(NULL, reqsize, PROT_READ | PROT_WRITE,
                           MAP_SHARED, tmpfd, 0);

        /* FIXME: check for errors */

        /* FIXME: Is it ok to close this file when using shm_open?? */
        status = apr_file_close(file);
        if (status != APR_SUCCESS) {
            return status;
        }
#endif /* APR_USE_SHMEM_MMAP_SHM */

        /* store the real size in the metadata */
        *(apr_size_t*)(new_m->base) = new_m->realsize;
        /* metadata isn't usable */
        new_m->usable = (char *)new_m->base + APR_ALIGN_DEFAULT(sizeof(apr_size_t));

        apr_pool_cleanup_register(new_m->pool, new_m, shm_cleanup_owner,
                                  apr_pool_cleanup_null);
        *m = new_m;
        return APR_SUCCESS;

#endif /* APR_USE_SHMEM_MMAP_TMP || APR_USE_SHMEM_MMAP_SHM */

#if APR_USE_SHMEM_SHMGET
        new_m->realsize = reqsize;

        /* FIXME: APR_OS_DEFAULT is too permissive, switch to 600 I think. */
        status = apr_file_open(&file, filename, 
                               APR_WRITE | APR_CREATE | APR_EXCL,
                               APR_OS_DEFAULT, pool);
        if (status != APR_SUCCESS) {
            return status;
        }

        /* ftok() (on solaris at least) requires that the file actually
         * exist before calling ftok(). */
        shmkey = ftok(filename, 1);
        if (shmkey == (key_t)-1) {
            return errno;
        }

        if ((new_m->shmid = shmget(shmkey, new_m->realsize,
                                   SHM_R | SHM_W | IPC_CREAT | IPC_EXCL)) < 0) {
            return errno;
        }

        if ((new_m->base = shmat(new_m->shmid, NULL, 0)) == (void *)-1) {
            return errno;
        }
        new_m->usable = new_m->base;

        if (shmctl(new_m->shmid, IPC_STAT, &shmbuf) == -1) {
            return errno;
        }
        apr_uid_current(&uid, &gid, pool);
        shmbuf.shm_perm.uid = uid;
        shmbuf.shm_perm.gid = gid;
        if (shmctl(new_m->shmid, IPC_SET, &shmbuf) == -1) {
            return errno;
        }

        nbytes = sizeof(reqsize);
        status = apr_file_write(file, (const void *)&reqsize,
                                &nbytes);
        if (status != APR_SUCCESS) {
            return status;
        }
        status = apr_file_close(file);
        if (status != APR_SUCCESS) {
            return status;
        }

        apr_pool_cleanup_register(new_m->pool, new_m, shm_cleanup_owner,
                                  apr_pool_cleanup_null);
        *m = new_m; 
        return APR_SUCCESS;

#endif /* APR_USE_SHMEM_SHMGET */
    }

    return APR_ENOTIMPL;
}

APR_DECLARE(apr_status_t) apr_shm_destroy(apr_shm_t *m)
{
    return apr_pool_cleanup_run(m->pool, m, shm_cleanup_owner);
}

static apr_status_t shm_cleanup_attach(void *m_)
{
    apr_shm_t *m = (apr_shm_t *)m_;

    if (m->filename == NULL) {
        /* It doesn't make sense to detach from an anonymous memory segment. */
        return APR_EINVAL;
    }
    else {
#if APR_USE_SHMEM_MMAP_TMP || APR_USE_SHMEM_MMAP_SHM
        if (munmap(m->base, m->realsize) == -1) {
            return errno;
        }
        return APR_SUCCESS;
#endif /* APR_USE_SHMEM_MMAP_TMP || APR_USE_SHMEM_MMAP_SHM */
#if APR_USE_SHMEM_SHMGET
        if (shmdt(m->base) == -1) {
            return errno;
        }
        return APR_SUCCESS;
#endif
    }

    return APR_ENOTIMPL;
}

APR_DECLARE(apr_status_t) apr_shm_attach(apr_shm_t **m,
                                         const char *filename,
                                         apr_pool_t *pool)
{
    if (filename == NULL) {
        /* It doesn't make sense to attach to a segment if you don't know
         * the filename. */
        return APR_EINVAL;
    }
    else {
#if APR_USE_SHMEM_MMAP_TMP || APR_USE_SHMEM_MMAP_SHM
        apr_shm_t *new_m;
        apr_status_t status;
        int tmpfd;
        apr_file_t *file;   /* file where metadata is stored */
        apr_size_t nbytes;

        new_m = apr_palloc(pool, sizeof(apr_shm_t));
        if (!new_m) {
            return APR_ENOMEM;
        }
        new_m->pool = pool;
        new_m->filename = apr_pstrdup(pool, filename);

        status = apr_file_open(&file, filename, 
                               APR_READ | APR_WRITE,
                               APR_OS_DEFAULT, pool);
        if (status != APR_SUCCESS) {
            return status;
        }
        status = apr_os_file_get(&tmpfd, file);
        if (status != APR_SUCCESS) {
            return status;
        }

        nbytes = sizeof(new_m->realsize);
        status = apr_file_read(file, (void *)&(new_m->realsize),
                               &nbytes);
        if (status != APR_SUCCESS) {
            return status;
        }

        status = apr_os_file_get(&tmpfd, file);
        if (status != APR_SUCCESS) {
            apr_file_close(file); /* ignore errors, we're failing */
            apr_file_remove(new_m->filename, new_m->pool);
            return status;
        }

        new_m->reqsize = new_m->realsize - sizeof(apr_size_t);

        new_m->base = mmap(NULL, new_m->realsize, PROT_READ | PROT_WRITE,
                           MAP_SHARED, tmpfd, 0);
        /* FIXME: check for errors */
        
        status = apr_file_close(file);
        if (status != APR_SUCCESS) {
            return status;
        }

        /* metadata isn't part of the usable segment */
        new_m->usable = (char *)new_m->base + APR_ALIGN_DEFAULT(sizeof(apr_size_t));

        apr_pool_cleanup_register(new_m->pool, new_m, shm_cleanup_attach,
                                  apr_pool_cleanup_null);
        *m = new_m;
        return APR_SUCCESS;

#endif /* APR_USE_SHMEM_MMAP_TMP || APR_USE_SHMEM_MMAP_SHM */
#if APR_USE_SHMEM_SHMGET
        apr_shm_t *new_m;
        apr_status_t status;
        apr_file_t *file;   /* file where metadata is stored */
        apr_size_t nbytes;
        key_t shmkey;

        new_m = apr_palloc(pool, sizeof(apr_shm_t));
        if (!new_m) {
            return APR_ENOMEM;
        }

        /* FIXME: does APR_OS_DEFAULT matter for reading? */
        status = apr_file_open(&file, filename, 
                               APR_READ, APR_OS_DEFAULT, pool);
        if (status != APR_SUCCESS) {
            return status;
        }

        nbytes = sizeof(new_m->reqsize);
        status = apr_file_read(file, (void *)&(new_m->reqsize),
                               &nbytes);
        if (status != APR_SUCCESS) {
            return status;
        }
        status = apr_file_close(file);
        if (status != APR_SUCCESS) {
            return status;
        }

        new_m->filename = apr_pstrdup(pool, filename);
        new_m->pool = pool;
        shmkey = ftok(filename, 1);
        if (shmkey == (key_t)-1) {
            return errno;
        }
        if ((new_m->shmid = shmget(shmkey, 0, SHM_R | SHM_W)) == -1) {
            return errno;
        }
        if ((new_m->base = shmat(new_m->shmid, NULL, 0)) == (void *)-1) {
            return errno;
        }
        new_m->usable = new_m->base;
        new_m->realsize = new_m->reqsize;

        apr_pool_cleanup_register(new_m->pool, new_m, shm_cleanup_attach,
                                  apr_pool_cleanup_null);
        *m = new_m;
        return APR_SUCCESS;

#endif /* APR_USE_SHMEM_SHMGET */
    }

    return APR_ENOTIMPL;
}

APR_DECLARE(apr_status_t) apr_shm_detach(apr_shm_t *m)
{
    apr_status_t rv = shm_cleanup_attach(m);
    apr_pool_cleanup_kill(m->pool, m, shm_cleanup_attach);
    return rv;
}

APR_DECLARE(void *) apr_shm_baseaddr_get(const apr_shm_t *m)
{
    return m->usable;
}

APR_DECLARE(apr_size_t) apr_shm_size_get(const apr_shm_t *m)
{
    return m->reqsize;
}

APR_POOL_IMPLEMENT_ACCESSOR(shm)

APR_DECLARE(apr_status_t) apr_os_shm_get(apr_os_shm_t *osshm,
                                         apr_shm_t *shm)
{
    return APR_ENOTIMPL;
}

APR_DECLARE(apr_status_t) apr_os_shm_put(apr_shm_t **m,
                                         apr_os_shm_t *osshm,
                                         apr_pool_t *pool)
{
    return APR_ENOTIMPL;
}    

