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

#include <stdio.h>
#include <nks/fsio.h>
#include <nks/errno.h>

#include "apr_arch_file_io.h"
#include "apr_strings.h"
#include "apr_portable.h"
#include "apr_arch_inherit.h"

static apr_status_t pipeblock(apr_file_t *thepipe)
{
#ifdef USE_FLAGS
    int				err;
	unsigned long	flags;

	if (fcntl(thepipe->filedes, F_GETFL, &flags) != -1)
	{
		flags &= ~FNDELAY;
		fcntl(thepipe->filedes, F_SETFL, flags);
	}
#else
        errno = 0;
		fcntl(thepipe->filedes, F_SETFL, 0);
#endif

    if (errno)
        return errno;

    thepipe->blocking = BLK_ON;
    return APR_SUCCESS;
}

static apr_status_t pipenonblock(apr_file_t *thepipe)
{
#ifdef USE_FLAGS
	int				err;
	unsigned long	flags;

    errno = 0;
	if (fcntl(thepipe->filedes, F_GETFL, &flags) != -1)
	{
		flags |= FNDELAY;
		fcntl(thepipe->filedes, F_SETFL, flags);
	}
#else
        errno = 0;
		fcntl(thepipe->filedes, F_SETFL, FNDELAY);
#endif

    if (errno)
        return errno;

    thepipe->blocking = BLK_OFF;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_file_pipe_timeout_set(apr_file_t *thepipe, apr_interval_time_t timeout)
{
    if (thepipe->is_pipe == 1) {
        thepipe->timeout = timeout;
        if (timeout >= 0) {
            if (thepipe->blocking != BLK_OFF) { /* blocking or unknown state */
                return pipenonblock(thepipe);
            }
        }
        else {
            if (thepipe->blocking != BLK_ON) { /* non-blocking or unknown state */
                return pipeblock(thepipe);
            }
        }
        return APR_SUCCESS;
    }
    return APR_EINVAL;
}

APR_DECLARE(apr_status_t) apr_file_pipe_timeout_get(apr_file_t *thepipe, apr_interval_time_t *timeout)
{
    if (thepipe->is_pipe == 1) {
        *timeout = thepipe->timeout;
        return APR_SUCCESS;
    }
    return APR_EINVAL;
}

APR_DECLARE(apr_status_t) apr_os_pipe_put(apr_file_t **file,
                                          apr_os_file_t *thefile,
                                          apr_pool_t *pool)
{
    int *dafile = thefile;
    
    (*file) = apr_pcalloc(pool, sizeof(apr_file_t));
    (*file)->pool = pool;
    (*file)->eof_hit = 0;
    (*file)->is_pipe = 1;
    (*file)->blocking = BLK_UNKNOWN; /* app needs to make a timeout call */
    (*file)->timeout = -1;
    (*file)->ungetchar = -1; /* no char avail */
    (*file)->filedes = *dafile;
    (*file)->flags = APR_FILE_NOCLEANUP;
    (*file)->buffered = 0;
#if APR_HAS_THREADS
    (*file)->thlock = NULL;
#endif
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_file_pipe_create(apr_file_t **in, apr_file_t **out, apr_pool_t *pool)
{
	int     	filedes[2];
	int 		err;

    if (pipe(filedes) == -1) {
        return errno;
    }

    (*in) = (apr_file_t *)apr_pcalloc(pool, sizeof(apr_file_t));
    (*out) = (apr_file_t *)apr_pcalloc(pool, sizeof(apr_file_t));

    (*in)->pool     =
    (*out)->pool    = pool;
    (*in)->filedes   = filedes[0];
    (*out)->filedes  = filedes[1];
    (*in)->flags     = APR_INHERIT;
    (*out)->flags    = APR_INHERIT;
    (*in)->is_pipe      =
    (*out)->is_pipe     = 1;
    (*out)->fname    = 
    (*in)->fname     = NULL;
    (*in)->buffered  =
    (*out)->buffered = 0;
    (*in)->blocking  =
    (*out)->blocking = BLK_ON;
    (*in)->timeout   =
    (*out)->timeout  = -1;
    (*in)->ungetchar = -1;
    (*in)->thlock    =
    (*out)->thlock   = NULL;

    apr_pool_cleanup_register((*in)->pool, (void *)(*in), apr_unix_file_cleanup,
                         apr_pool_cleanup_null);
    apr_pool_cleanup_register((*out)->pool, (void *)(*out), apr_unix_file_cleanup,
                         apr_pool_cleanup_null);

    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_file_namedpipe_create(const char *filename, 
                                                    apr_fileperms_t perm, apr_pool_t *pool)
{
    return APR_ENOTIMPL;
} 

    

