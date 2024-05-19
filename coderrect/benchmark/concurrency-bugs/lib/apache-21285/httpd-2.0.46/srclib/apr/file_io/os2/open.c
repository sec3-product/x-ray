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

#include "apr_arch_file_io.h"
#include "apr_file_io.h"
#include "apr_lib.h"
#include "apr_portable.h"
#include "apr_strings.h"
#include "apr_arch_inherit.h"
#include <string.h>

apr_status_t apr_file_cleanup(void *thefile)
{
    apr_file_t *file = thefile;
    return apr_file_close(file);
}



APR_DECLARE(apr_status_t) apr_file_open(apr_file_t **new, const char *fname, apr_int32_t flag,  apr_fileperms_t perm, apr_pool_t *pool)
{
    int oflags = 0;
    int mflags = OPEN_FLAGS_FAIL_ON_ERROR|OPEN_SHARE_DENYNONE;
    int rv;
    ULONG action;
    apr_file_t *dafile = (apr_file_t *)apr_palloc(pool, sizeof(apr_file_t));

    dafile->pool = pool;
    dafile->isopen = FALSE;
    dafile->eof_hit = FALSE;
    dafile->buffer = NULL;
    dafile->flags = flag;
    dafile->blocking = BLK_ON;
    
    if ((flag & APR_READ) && (flag & APR_WRITE)) {
        mflags |= OPEN_ACCESS_READWRITE;
    } else if (flag & APR_READ) {
        mflags |= OPEN_ACCESS_READONLY;
    } else if (flag & APR_WRITE) {
        mflags |= OPEN_ACCESS_WRITEONLY;
    } else {
        dafile->filedes = -1;
        return APR_EACCES;
    }

    dafile->buffered = (flag & APR_BUFFERED) > 0;

    if (dafile->buffered) {
        dafile->buffer = apr_palloc(pool, APR_FILE_BUFSIZE);
        rv = apr_thread_mutex_create(&dafile->mutex, 0, pool);

        if (rv)
            return rv;
    }

    if (flag & APR_CREATE) {
        oflags |= OPEN_ACTION_CREATE_IF_NEW; 
        if (!(flag & APR_EXCL)) {
            if (flag & APR_APPEND)
                oflags |= OPEN_ACTION_OPEN_IF_EXISTS;
            else
                oflags |= OPEN_ACTION_REPLACE_IF_EXISTS;
        }
    }
    
    if ((flag & APR_EXCL) && !(flag & APR_CREATE))
        return APR_EACCES;

    if (flag & APR_TRUNCATE) {
        oflags |= OPEN_ACTION_REPLACE_IF_EXISTS;
    } else if ((oflags & 0xFF) == 0) {
        oflags |= OPEN_ACTION_OPEN_IF_EXISTS;
    }
    
    rv = DosOpen(fname, &(dafile->filedes), &action, 0, 0, oflags, mflags, NULL);
    
    if (rv == 0 && (flag & APR_APPEND)) {
        ULONG newptr;
        rv = DosSetFilePtr(dafile->filedes, 0, FILE_END, &newptr );
        
        if (rv)
            DosClose(dafile->filedes);
    }
    
    if (rv != 0)
        return APR_FROM_OS_ERROR(rv);
    
    dafile->isopen = TRUE;
    dafile->fname = apr_pstrdup(pool, fname);
    dafile->filePtr = 0;
    dafile->bufpos = 0;
    dafile->dataRead = 0;
    dafile->direction = 0;
    dafile->pipe = FALSE;

    if (!(flag & APR_FILE_NOCLEANUP)) { 
        apr_pool_cleanup_register(dafile->pool, dafile, apr_file_cleanup, apr_file_cleanup);
    }

    *new = dafile;
    return APR_SUCCESS;
}



APR_DECLARE(apr_status_t) apr_file_close(apr_file_t *file)
{
    ULONG rc;
    apr_status_t status;
    
    if (file && file->isopen) {
        apr_file_flush(file);
        rc = DosClose(file->filedes);
    
        if (rc == 0) {
            file->isopen = FALSE;
            status = APR_SUCCESS;

            if (file->flags & APR_DELONCLOSE) {
                status = APR_FROM_OS_ERROR(DosDelete(file->fname));
            }
        } else {
            return APR_FROM_OS_ERROR(rc);
        }
    }

    if (file->buffered)
        apr_thread_mutex_destroy(file->mutex);

    return APR_SUCCESS;
}



APR_DECLARE(apr_status_t) apr_file_remove(const char *path, apr_pool_t *pool)
{
    ULONG rc = DosDelete(path);
    return APR_FROM_OS_ERROR(rc);
}



APR_DECLARE(apr_status_t) apr_file_rename(const char *from_path, const char *to_path,
                                   apr_pool_t *p)
{
    ULONG rc = DosMove(from_path, to_path);

    if (rc == ERROR_ACCESS_DENIED) {
        rc = DosDelete(to_path);

        if (rc == 0 || rc == ERROR_FILE_NOT_FOUND) {
            rc = DosMove(from_path, to_path);
        }
    }

    return APR_FROM_OS_ERROR(rc);
}



APR_DECLARE(apr_status_t) apr_os_file_get(apr_os_file_t *thefile, apr_file_t *file)
{
    *thefile = file->filedes;
    return APR_SUCCESS;
}



APR_DECLARE(apr_status_t) apr_os_file_put(apr_file_t **file, apr_os_file_t *thefile, apr_int32_t flags, apr_pool_t *pool)
{
    apr_os_file_t *dafile = thefile;

    (*file) = apr_palloc(pool, sizeof(apr_file_t));
    (*file)->pool = pool;
    (*file)->filedes = *dafile;
    (*file)->isopen = TRUE;
    (*file)->eof_hit = FALSE;
    (*file)->flags = flags;
    (*file)->pipe = FALSE;
    (*file)->buffered = (flags & APR_BUFFERED) > 0;

    if ((*file)->buffered) {
        apr_status_t rv;

        (*file)->buffer = apr_palloc(pool, APR_FILE_BUFSIZE);
        rv = apr_thread_mutex_create(&(*file)->mutex, 0, pool);

        if (rv)
            return rv;
    }
    return APR_SUCCESS;
}    


APR_DECLARE(apr_status_t) apr_file_eof(apr_file_t *fptr)
{
    if (!fptr->isopen || fptr->eof_hit == 1) {
        return APR_EOF;
    }
    return APR_SUCCESS;
}   


APR_DECLARE(apr_status_t) apr_file_open_stderr(apr_file_t **thefile, apr_pool_t *pool)
{
    apr_os_file_t fd = 2;

    return apr_os_file_put(thefile, &fd, 0, pool);
}



APR_DECLARE(apr_status_t) apr_file_open_stdout(apr_file_t **thefile, apr_pool_t *pool)
{
    apr_os_file_t fd = 1;

    return apr_os_file_put(thefile, &fd, 0, pool);
}


APR_DECLARE(apr_status_t) apr_file_open_stdin(apr_file_t **thefile, apr_pool_t *pool)
{
    apr_os_file_t fd = 0;

    return apr_os_file_put(thefile, &fd, 0, pool);
}

APR_POOL_IMPLEMENT_ACCESSOR(file);

APR_IMPLEMENT_INHERIT_SET(file, flags, pool, apr_file_cleanup)

APR_IMPLEMENT_INHERIT_UNSET(file, flags, pool, apr_file_cleanup)

