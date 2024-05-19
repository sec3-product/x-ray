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

#include "apr_general.h"
#include "apr_lib.h"
#include "apr_strings.h"
#include "apr_portable.h"
#include "apr_arch_thread_mutex.h"
#include "apr_arch_file_io.h"
#include <string.h>
#include <stddef.h>

static apr_status_t thread_mutex_cleanup(void *themutex)
{
    apr_thread_mutex_t *mutex = themutex;
    return apr_thread_mutex_destroy(mutex);
}



/* XXX: Need to respect APR_THREAD_MUTEX_[UN]NESTED flags argument
 *      or return APR_ENOTIMPL!!!
 */
APR_DECLARE(apr_status_t) apr_thread_mutex_create(apr_thread_mutex_t **mutex,
                                                  unsigned int flags,
                                                  apr_pool_t *pool)
{
    apr_thread_mutex_t *new_mutex;
    ULONG rc;

    new_mutex = (apr_thread_mutex_t *)apr_palloc(pool, sizeof(apr_thread_mutex_t));
    new_mutex->pool = pool;

    rc = DosCreateMutexSem(NULL, &(new_mutex->hMutex), 0, FALSE);
    *mutex = new_mutex;

    if (!rc)
        apr_pool_cleanup_register(pool, new_mutex, thread_mutex_cleanup, apr_pool_cleanup_null);

    return APR_OS2_STATUS(rc);
}



APR_DECLARE(apr_status_t) apr_thread_mutex_lock(apr_thread_mutex_t *mutex)
{
    ULONG rc = DosRequestMutexSem(mutex->hMutex, SEM_INDEFINITE_WAIT);
    return APR_OS2_STATUS(rc);
}



APR_DECLARE(apr_status_t) apr_thread_mutex_trylock(apr_thread_mutex_t *mutex)
{
    ULONG rc = DosRequestMutexSem(mutex->hMutex, SEM_IMMEDIATE_RETURN);
    return APR_OS2_STATUS(rc);
}



APR_DECLARE(apr_status_t) apr_thread_mutex_unlock(apr_thread_mutex_t *mutex)
{
    ULONG rc = DosReleaseMutexSem(mutex->hMutex);
    return APR_OS2_STATUS(rc);
}



APR_DECLARE(apr_status_t) apr_thread_mutex_destroy(apr_thread_mutex_t *mutex)
{
    ULONG rc;

    if (mutex->hMutex == 0)
        return APR_SUCCESS;

    while (DosReleaseMutexSem(mutex->hMutex) == 0);

    rc = DosCloseMutexSem(mutex->hMutex);

    if (!rc) {
        mutex->hMutex = 0;
        return APR_SUCCESS;
    }

    return APR_FROM_OS_ERROR(rc);
}

APR_POOL_IMPLEMENT_ACCESSOR(thread_mutex)

