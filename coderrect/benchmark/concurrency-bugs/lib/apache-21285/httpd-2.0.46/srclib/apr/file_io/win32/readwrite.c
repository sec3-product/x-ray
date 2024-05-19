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

#include "win32/apr_arch_file_io.h"
#include "apr_file_io.h"
#include "apr_general.h"
#include "apr_strings.h"
#include "apr_lib.h"
#include "apr_errno.h"
#include <malloc.h>
#include "apr_arch_atime.h"
#include "apr_arch_misc.h"

/*
 * read_with_timeout() 
 * Uses async i/o to emulate unix non-blocking i/o with timeouts.
 */
static apr_status_t read_with_timeout(apr_file_t *file, void *buf, apr_size_t len, apr_size_t *nbytes)
{
    apr_status_t rv;
    *nbytes = 0;

    /* Handle the zero timeout non-blocking case */
    if (file->timeout == 0) {
        /* Peek at the pipe. If there is no data available, return APR_EAGAIN.
         * If data is available, go ahead and read it.
         */
        if (file->pipe) {
            DWORD bytes;
            if (!PeekNamedPipe(file->filehand, NULL, 0, NULL, &bytes, NULL)) {
                rv = apr_get_os_error();
                if (rv == APR_FROM_OS_ERROR(ERROR_BROKEN_PIPE)) {
                    rv = APR_EOF;
                }
                return rv;
            }
            else {
                if (bytes == 0) {
                    return APR_EAGAIN;
                }
                if (len > bytes) {
                    len = bytes;
                }
            }
        }
        else {
            /* ToDo: Handle zero timeout non-blocking file i/o 
             * This is not needed until an APR application needs to
             * timeout file i/o (which means setting file i/o non-blocking)
             */
        }
    }

    if (file->pOverlapped && !file->pipe) {
        file->pOverlapped->Offset     = (DWORD)file->filePtr;
        file->pOverlapped->OffsetHigh = (DWORD)(file->filePtr >> 32);
    }

    rv = ReadFile(file->filehand, buf, len, nbytes, file->pOverlapped);

    if (!rv) {
        rv = apr_get_os_error();
        if (rv == APR_FROM_OS_ERROR(ERROR_IO_PENDING)) {
            /* Wait for the pending i/o */
            if (file->timeout > 0) {
                /* timeout in milliseconds... */
                rv = WaitForSingleObject(file->pOverlapped->hEvent, 
                                         (DWORD)(file->timeout/1000)); 
            }
            else if (file->timeout == -1) {
                rv = WaitForSingleObject(file->pOverlapped->hEvent, INFINITE);
            }
            switch (rv) {
            case WAIT_OBJECT_0:
                GetOverlappedResult(file->filehand, file->pOverlapped, 
                                    nbytes, TRUE);
                rv = APR_SUCCESS;
                break;
            case WAIT_TIMEOUT:
                rv = APR_TIMEUP;
                break;
            case WAIT_FAILED:
                rv = apr_get_os_error();
                break;
            default:
                break;
            }
            if (rv != APR_SUCCESS) {
                if (apr_os_level >= APR_WIN_98)
                    CancelIo(file->filehand);
            }
        }
        else if (rv == APR_FROM_OS_ERROR(ERROR_BROKEN_PIPE)) {
            /* Assume ERROR_BROKEN_PIPE signals an EOF reading from a pipe */
            rv = APR_EOF;
        }
    } else {
        /* OK and 0 bytes read ==> end of file */
        if (*nbytes == 0)
            rv = APR_EOF;
        else
            rv = APR_SUCCESS;
    }
    if (rv == APR_SUCCESS && file->pOverlapped && !file->pipe) {
        file->filePtr += *nbytes;
    }
    return rv;
}

APR_DECLARE(apr_status_t) apr_file_read(apr_file_t *thefile, void *buf, apr_size_t *len)
{
    apr_status_t rv;
    DWORD bytes_read = 0;

    if (*len <= 0) {
        *len = 0;
        return APR_SUCCESS;
    }

    /* If the file is open for xthread support, allocate and
     * initialize the overlapped and io completion event (hEvent). 
     * Threads should NOT share an apr_file_t or its hEvent.
     */
    if ((thefile->flags & APR_XTHREAD) && !thefile->pOverlapped ) {
        thefile->pOverlapped = (OVERLAPPED*) apr_pcalloc(thefile->pool, 
                                                         sizeof(OVERLAPPED));
        thefile->pOverlapped->hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        if (!thefile->pOverlapped->hEvent) {
            rv = apr_get_os_error();
            return rv;
        }
    }

    /* Handle the ungetchar if there is one */
    if (thefile->ungetchar != -1) {
        bytes_read = 1;
        *(char *)buf = (char)thefile->ungetchar;
        buf = (char *)buf + 1;
        (*len)--;
        thefile->ungetchar = -1;
        if (*len == 0) {
            *len = bytes_read;
            return APR_SUCCESS;
        }
    }
    if (thefile->buffered) {
        char *pos = (char *)buf;
        apr_size_t blocksize;
        apr_size_t size = *len;

        apr_thread_mutex_lock(thefile->mutex);

        if (thefile->direction == 1) {
            apr_file_flush(thefile);
            thefile->bufpos = 0;
            thefile->direction = 0;
            thefile->dataRead = 0;
        }

        rv = 0;
        while (rv == 0 && size > 0) {
            if (thefile->bufpos >= thefile->dataRead) {
                apr_size_t read;
                rv = read_with_timeout(thefile, thefile->buffer, 
                                       APR_FILE_BUFSIZE, &read);
                if (read == 0) {
                    if (rv == APR_EOF)
                        thefile->eof_hit = TRUE;
                    break;
                }
                else {
                    thefile->dataRead = read;
                    thefile->filePtr += thefile->dataRead;
                    thefile->bufpos = 0;
                }
            }

            blocksize = size > thefile->dataRead - thefile->bufpos ? thefile->dataRead - thefile->bufpos : size;
            memcpy(pos, thefile->buffer + thefile->bufpos, blocksize);
            thefile->bufpos += blocksize;
            pos += blocksize;
            size -= blocksize;
        }

        *len = pos - (char *)buf;
        if (*len) {
            rv = APR_SUCCESS;
        }
        apr_thread_mutex_unlock(thefile->mutex);
    } else {  
        /* Unbuffered i/o */
        apr_size_t nbytes;
        rv = read_with_timeout(thefile, buf, *len, &nbytes);
        *len = nbytes;
    }

    return rv;
}

APR_DECLARE(apr_status_t) apr_file_write(apr_file_t *thefile, const void *buf, apr_size_t *nbytes)
{
    apr_status_t rv;
    DWORD bwrote;

    /* If the file is open for xthread support, allocate and
     * initialize the overlapped and io completion event (hEvent). 
     * Threads should NOT share an apr_file_t or its hEvent.
     */
    if ((thefile->flags & APR_XTHREAD) && !thefile->pOverlapped ) {
        thefile->pOverlapped = (OVERLAPPED*) apr_pcalloc(thefile->pool, 
                                                         sizeof(OVERLAPPED));
        thefile->pOverlapped->hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        if (!thefile->pOverlapped->hEvent) {
            rv = apr_get_os_error();
            return rv;
        }
    }

    if (thefile->buffered) {
        char *pos = (char *)buf;
        apr_size_t blocksize;
        apr_size_t size = *nbytes;

        apr_thread_mutex_lock(thefile->mutex);

        if (thefile->direction == 0) {
            // Position file pointer for writing at the offset we are logically reading from
            apr_off_t offset = thefile->filePtr - thefile->dataRead + thefile->bufpos;
            DWORD offlo = (DWORD)offset;
            DWORD offhi = (DWORD)(offset >> 32);
            if (offset != thefile->filePtr)
                SetFilePointer(thefile->filehand, offlo, &offhi, FILE_BEGIN);
            thefile->bufpos = thefile->dataRead = 0;
            thefile->direction = 1;
        }

        rv = 0;
        while (rv == 0 && size > 0) {
            if (thefile->bufpos == APR_FILE_BUFSIZE)   // write buffer is full
                rv = apr_file_flush(thefile);

            blocksize = size > APR_FILE_BUFSIZE - thefile->bufpos ? APR_FILE_BUFSIZE - thefile->bufpos : size;
            memcpy(thefile->buffer + thefile->bufpos, pos, blocksize);
            thefile->bufpos += blocksize;
            pos += blocksize;
            size -= blocksize;
        }

        apr_thread_mutex_unlock(thefile->mutex);
        return rv;
    } else {
        if (!thefile->pipe) {
            apr_off_t offset = 0;
            apr_status_t rc;
            if (thefile->append) {
                /* apr_file_lock will mutex the file across processes.
                 * The call to apr_thread_mutex_lock is added to avoid
                 * a race condition between LockFile and WriteFile 
                 * that occasionally leads to deadlocked threads.
                 */
                apr_thread_mutex_lock(thefile->mutex);
                rc = apr_file_lock(thefile, APR_FLOCK_EXCLUSIVE);
                if (rc != APR_SUCCESS) {
                    apr_thread_mutex_unlock(thefile->mutex);
                    return rc;
                }
                rc = apr_file_seek(thefile, APR_END, &offset);
                if (rc != APR_SUCCESS) {
                    apr_thread_mutex_unlock(thefile->mutex);
                    return rc;
                }
            }
            if (thefile->pOverlapped) {
                thefile->pOverlapped->Offset     = (DWORD)thefile->filePtr;
                thefile->pOverlapped->OffsetHigh = (DWORD)(thefile->filePtr >> 32);
            }
            rv = WriteFile(thefile->filehand, buf, *nbytes, &bwrote,
                           thefile->pOverlapped);
            if (thefile->append) {
                apr_file_unlock(thefile);
                apr_thread_mutex_unlock(thefile->mutex);
            }
        }
        else {
            rv = WriteFile(thefile->filehand, buf, *nbytes, &bwrote,
                           thefile->pOverlapped);
        }
        if (rv) {
            *nbytes = bwrote;
            rv = APR_SUCCESS;
        }
        else {
            (*nbytes) = 0;
            rv = apr_get_os_error();
            if (rv == APR_FROM_OS_ERROR(ERROR_IO_PENDING)) {
                /* Wait for the pending i/o (put a timeout here?) */
                rv = WaitForSingleObject(thefile->pOverlapped->hEvent, INFINITE);
                switch (rv) {
                    case WAIT_OBJECT_0:
                        GetOverlappedResult(thefile->filehand, thefile->pOverlapped, nbytes, TRUE);
                        rv = APR_SUCCESS;
                        break;
                    case WAIT_TIMEOUT:
                        rv = APR_TIMEUP;
                        break;
                    case WAIT_FAILED:
                        rv = apr_get_os_error();
                        break;
                    default:
                        break;
                }
                if (rv != APR_SUCCESS) {
                    if (apr_os_level >= APR_WIN_98)
                        CancelIo(thefile->filehand);
                }
            }
        }
        if (rv == APR_SUCCESS && thefile->pOverlapped && !thefile->pipe) {
            thefile->filePtr += *nbytes;
        }
    }
    return rv;
}
/* ToDo: Write for it anyway and test the oslevel!
 * Too bad WriteFileGather() is not supported on 95&98 (or NT prior to SP2)
 */
APR_DECLARE(apr_status_t) apr_file_writev(apr_file_t *thefile,
                                     const struct iovec *vec,
                                     apr_size_t nvec, 
                                     apr_size_t *nbytes)
{
    apr_status_t rv = APR_SUCCESS;
    apr_size_t i;
    DWORD bwrote = 0;
    char *buf;

    *nbytes = 0;
    for (i = 0; i < nvec; i++) {
        buf = vec[i].iov_base;
        bwrote = vec[i].iov_len;
        rv = apr_file_write(thefile, buf, &bwrote);
        *nbytes += bwrote;
        if (rv != APR_SUCCESS) {
            break;
        }
    }
    return rv;
}

APR_DECLARE(apr_status_t) apr_file_putc(char ch, apr_file_t *thefile)
{
    DWORD len = 1;

    return apr_file_write(thefile, &ch, &len);
}

APR_DECLARE(apr_status_t) apr_file_ungetc(char ch, apr_file_t *thefile)
{
    thefile->ungetchar = (unsigned char) ch;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_file_getc(char *ch, apr_file_t *thefile)
{
    apr_status_t rc;
    int bread;

    bread = 1;
    rc = apr_file_read(thefile, ch, &bread);

    if (rc) {
        return rc;
    }
    
    if (bread == 0) {
        thefile->eof_hit = TRUE;
        return APR_EOF;
    }
    return APR_SUCCESS; 
}

APR_DECLARE(apr_status_t) apr_file_puts(const char *str, apr_file_t *thefile)
{
    DWORD len = strlen(str);

    return apr_file_write(thefile, str, &len);
}

APR_DECLARE(apr_status_t) apr_file_gets(char *str, int len, apr_file_t *thefile)
{
    apr_size_t readlen;
    apr_status_t rv = APR_SUCCESS;
    int i;    

    for (i = 0; i < len-1; i++) {
        readlen = 1;
        rv = apr_file_read(thefile, str+i, &readlen);

        if (readlen != 1) {
            rv = APR_EOF;
            break;
        }
        
        if (str[i] == '\n') {
            i++; /* don't clobber this char below */
            break;
        }
    }
    str[i] = 0;
    if (i > 0) {
        /* we stored chars; don't report EOF or any other errors;
         * the app will find out about that on the next call
         */
        return APR_SUCCESS;
    }
    return rv;
}

APR_DECLARE(apr_status_t) apr_file_flush(apr_file_t *thefile)
{
    if (thefile->buffered) {
        DWORD written = 0;
        apr_status_t rc = 0;

        if (thefile->direction == 1 && thefile->bufpos) {
            if (!WriteFile(thefile->filehand, thefile->buffer, thefile->bufpos, &written, NULL))
                rc = apr_get_os_error();
            thefile->filePtr += written;

            if (rc == 0)
                thefile->bufpos = 0;
        }

        return rc;
    } else {
        FlushFileBuffers(thefile->filehand);
        return APR_SUCCESS;
    }
}

static int printf_flush(apr_vformatter_buff_t *vbuff)
{
    /* I would love to print this stuff out to the file, but I will
     * get that working later.  :)  For now, just return.
     */
    return -1;
}

APR_DECLARE_NONSTD(int) apr_file_printf(apr_file_t *fptr, 
                                        const char *format, ...)
{
    int cc;
    va_list ap;
    char *buf;
    int len;

    buf = malloc(HUGE_STRING_LEN);
    if (buf == NULL) {
        return 0;
    }
    va_start(ap, format);
    len = apr_vsnprintf(buf, HUGE_STRING_LEN, format, ap);
    cc = apr_file_puts(buf, fptr);
    va_end(ap);
    free(buf);
    return (cc == APR_SUCCESS) ? len : -1;
}


