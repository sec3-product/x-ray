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

#include "apr_arch_networkio.h"
#include "apr_support.h"

#if APR_HAS_SENDFILE
/* This file is needed to allow us access to the apr_file_t internals. */
#include "apr_arch_file_io.h"
#endif /* APR_HAS_SENDFILE */

/* sys/sysctl.h is only needed on FreeBSD for include_hdrs_in_length() */
#if defined(__FreeBSD__) && defined(HAVE_SYS_SYSCTL_H)
#include <sys/sysctl.h>
#endif

apr_status_t apr_socket_send(apr_socket_t *sock, const char *buf, 
                             apr_size_t *len)
{
    apr_ssize_t rv;
    
    if (sock->netmask & APR_INCOMPLETE_WRITE) {
        sock->netmask &= ~APR_INCOMPLETE_WRITE;
        goto do_select;
    }

    do {
        rv = write(sock->socketdes, buf, (*len));
    } while (rv == -1 && errno == EINTR);

    if (rv == -1 && (errno == EAGAIN || errno == EWOULDBLOCK) 
      && sock->timeout != 0) {
        apr_status_t arv;
do_select:
        arv = apr_wait_for_io_or_timeout(NULL, sock, 0);
        if (arv != APR_SUCCESS) {
            *len = 0;
            return arv;
        }
        else {
            do {
                rv = write(sock->socketdes, buf, (*len));
            } while (rv == -1 && errno == EINTR);
        }
    }
    if (rv == -1) {
        *len = 0;
        return errno;
    }
    if (sock->timeout && rv < *len) {
    sock->netmask |= APR_INCOMPLETE_WRITE;
    }
    (*len) = rv;
    return APR_SUCCESS;
}

apr_status_t apr_socket_recv(apr_socket_t *sock, char *buf, apr_size_t *len)
{
    apr_ssize_t rv;
    apr_status_t arv;

    if (sock->netmask & APR_INCOMPLETE_READ) {
    sock->netmask &= ~APR_INCOMPLETE_READ;
        goto do_select;
    }

    do {
        rv = read(sock->socketdes, buf, (*len));
    } while (rv == -1 && errno == EINTR);

    if (rv == -1 && (errno == EAGAIN || errno == EWOULDBLOCK) && 
      sock->timeout != 0) {
do_select:
    arv = apr_wait_for_io_or_timeout(NULL, sock, 1);
        if (arv != APR_SUCCESS) {
            *len = 0;
            return arv;
        }
        else {
            do {
                rv = read(sock->socketdes, buf, (*len));
            } while (rv == -1 && errno == EINTR);
        }
    }
    if (rv == -1) {
        (*len) = 0;
        return errno;
    }
    if (sock->timeout && rv < *len) {
    sock->netmask |= APR_INCOMPLETE_READ;
    }
    (*len) = rv;
    if (rv == 0) {
        return APR_EOF;
    }
    return APR_SUCCESS;
}

apr_status_t apr_socket_sendto(apr_socket_t *sock, apr_sockaddr_t *where,
                               apr_int32_t flags, const char *buf,
                               apr_size_t *len)
{
    apr_ssize_t rv;

    do {
        rv = sendto(sock->socketdes, buf, (*len), flags, 
                    (const struct sockaddr*)&where->sa, 
                    where->salen);
    } while (rv == -1 && errno == EINTR);

    if (rv == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)
        && sock->timeout != 0) {
        apr_status_t arv = apr_wait_for_io_or_timeout(NULL, sock, 0);
        if (arv != APR_SUCCESS) {
            *len = 0;
            return arv;
        } else {
            do {
                rv = sendto(sock->socketdes, buf, (*len), flags,
                            (const struct sockaddr*)&where->sa,
                            where->salen);
            } while (rv == -1 && errno == EINTR);
        }
    }
    if (rv == -1) {
        *len = 0;
        return errno;
    }
    *len = rv;
    return APR_SUCCESS;
}

apr_status_t apr_socket_recvfrom(apr_sockaddr_t *from, apr_socket_t *sock,
                                 apr_int32_t flags, char *buf, 
                                 apr_size_t *len)
{
    apr_ssize_t rv;

    do {
        rv = recvfrom(sock->socketdes, buf, (*len), flags, 
                      (struct sockaddr*)&from->sa, &from->salen);
    } while (rv == -1 && errno == EINTR);

    if (rv == -1 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
        sock->timeout != 0) {
        apr_status_t arv = apr_wait_for_io_or_timeout(NULL, sock, 1);
        if (arv != APR_SUCCESS) {
            *len = 0;
            return arv;
        } else {
            do {
                rv = recvfrom(sock->socketdes, buf, (*len), flags,
                              (struct sockaddr*)&from->sa, &from->salen);
                } while (rv == -1 && errno == EINTR);
        }
    }
    if (rv == -1) {
        (*len) = 0;
        return errno;
    }

    (*len) = rv;
    if (rv == 0 && sock->type == SOCK_STREAM)
        return APR_EOF;

    return APR_SUCCESS;
}

#ifdef HAVE_WRITEV
apr_status_t apr_socket_sendv(apr_socket_t * sock, const struct iovec *vec,
                              apr_int32_t nvec, apr_size_t *len)
{
    apr_ssize_t rv;
    apr_size_t requested_len = 0;
    apr_int32_t i;

    for (i = 0; i < nvec; i++) {
        requested_len += vec[i].iov_len;
    }

    if (sock->netmask & APR_INCOMPLETE_WRITE) {
        sock->netmask &= ~APR_INCOMPLETE_WRITE;
        goto do_select;
    }

    do {
        rv = writev(sock->socketdes, vec, nvec);
    } while (rv == -1 && errno == EINTR);

    if (rv == -1 && (errno == EAGAIN || errno == EWOULDBLOCK) && 
      sock->timeout != 0) {
        apr_status_t arv;
do_select:
        arv = apr_wait_for_io_or_timeout(NULL, sock, 0);
        if (arv != APR_SUCCESS) {
            *len = 0;
            return arv;
        }
        else {
            do {
                rv = writev(sock->socketdes, vec, nvec);
            } while (rv == -1 && errno == EINTR);
        }
    }
    if (rv == -1) {
        *len = 0;
        return errno;
    }
    if (sock->timeout && rv < requested_len) {
    sock->netmask |= APR_INCOMPLETE_WRITE;
    }
    (*len) = rv;
    return APR_SUCCESS;
}
#endif

#if APR_HAS_SENDFILE

/* TODO: Verify that all platforms handle the fd the same way,
 * i.e. that they don't move the file pointer.
 */
/* TODO: what should flags be?  int_32? */

/* Define a structure to pass in when we have a NULL header value */
static apr_hdtr_t no_hdtr;

#if defined(__linux__) && defined(HAVE_WRITEV)

apr_status_t apr_socket_sendfile(apr_socket_t *sock, apr_file_t *file,
                                 apr_hdtr_t *hdtr, apr_off_t *offset,
                                 apr_size_t *len, apr_int32_t flags)
{
    off_t off = *offset;
    int rv, nbytes = 0, total_hdrbytes, i;
    apr_status_t arv;

    if (!hdtr) {
        hdtr = &no_hdtr;
    }

    /* Ignore flags for now. */
    flags = 0;

    if (hdtr->numheaders > 0) {
        apr_size_t hdrbytes;

        /* cork before writing headers */
        rv = apr_socket_opt_set(sock, APR_TCP_NOPUSH, 1);
        if (rv != APR_SUCCESS) {
            return rv;
        }

        /* Now write the headers */
        arv = apr_socket_sendv(sock, hdtr->headers, hdtr->numheaders,
                               &hdrbytes);
        if (arv != APR_SUCCESS) {
        *len = 0;
            return errno;
        }
        nbytes += hdrbytes;

        /* If this was a partial write and we aren't doing timeouts, 
         * return now with the partial byte count; this is a non-blocking 
         * socket.
         */
        total_hdrbytes = 0;
        for (i = 0; i < hdtr->numheaders; i++) {
            total_hdrbytes += hdtr->headers[i].iov_len;
        }
        if (hdrbytes < total_hdrbytes) {
            *len = hdrbytes;
            return apr_socket_opt_set(sock, APR_TCP_NOPUSH, 0);
        }
    }

    if (sock->netmask & APR_INCOMPLETE_WRITE) {
        sock->netmask &= ~APR_INCOMPLETE_WRITE;
        goto do_select;
    }

    do {
        rv = sendfile(sock->socketdes,    /* socket */
                  file->filedes,    /* open file descriptor of the file to be sent */
                  &off,    /* where in the file to start */
                  *len    /* number of bytes to send */
            );
    } while (rv == -1 && errno == EINTR);

    if (rv == -1 && 
        (errno == EAGAIN || errno == EWOULDBLOCK) && 
        sock->timeout > 0) {
do_select:
    arv = apr_wait_for_io_or_timeout(NULL, sock, 0);
    if (arv != APR_SUCCESS) {
        *len = 0;
        return arv;
    }
        else {
            do {
            rv = sendfile(sock->socketdes,    /* socket */
                      file->filedes,    /* open file descriptor of the file to be sent */
                      &off,    /* where in the file to start */
                      *len);    /* number of bytes to send */
            } while (rv == -1 && errno == EINTR);
        }
    }

    if (rv == -1) {
    *len = nbytes;
        rv = errno;
        apr_socket_opt_set(sock, APR_TCP_NOPUSH, 0);
        return rv;
    }

    nbytes += rv;

    if (rv < *len) {
        *len = nbytes;
        arv = apr_socket_opt_set(sock, APR_TCP_NOPUSH, 0);
        if (rv > 0) {
                
            /* If this was a partial write, return now with the 
             * partial byte count;  this is a non-blocking socket.
             */

            if (sock->timeout) {
                sock->netmask |= APR_INCOMPLETE_WRITE;
            }
            return arv;
        }
        else {
            /* If the file got smaller mid-request, eventually the offset
             * becomes equal to the new file size and the kernel returns 0.  
             * Make this an error so the caller knows to log something and
             * exit.
             */
            return APR_EOF;
        }
    }

    /* Now write the footers */
    if (hdtr->numtrailers > 0) {
        apr_size_t trbytes;
        arv = apr_socket_sendv(sock, hdtr->trailers, hdtr->numtrailers, 
                               &trbytes);
        nbytes += trbytes;
        if (arv != APR_SUCCESS) {
        *len = nbytes;
            rv = errno;
            apr_socket_opt_set(sock, APR_TCP_NOPUSH, 0);
            return rv;
        }
    }

    apr_socket_opt_set(sock, APR_TCP_NOPUSH, 0);
    
    (*len) = nbytes;
    return rv < 0 ? errno : APR_SUCCESS;
}

#elif defined(__FreeBSD__)

static int include_hdrs_in_length(void)
{
#ifdef HAVE_SYS_SYSCTL_H
/* this assumes: 
 *   if the header exists, so does the sysctlbyname() syscall, and 
 *   if the header doesn't exist, the kernel is really old
 */

/* see 
 * http://www.freebsd.org/cgi/cvsweb.cgi/src/sys/sys/param.h#rev1.61.2.29 
 * for kernel version number
 * 
 * http://www.freebsd.org/cgi/cvsweb.cgi/src/sys/kern/uipc_syscalls.c#rev1.65.2.10
 * for the sendfile patch
 */
#define KERNEL_WITH_SENDFILE_LENGTH_FIX 460001

    typedef enum { UNKNOWN = 0, 
                   NEW, 
                   OLD 
                 } api_e;

    static api_e api; 
    int kernel_version;  
    apr_size_t kernel_version_size;

    if (api != UNKNOWN) {
        return (api == OLD);
    }
    kernel_version = 0;    /* silence compiler warning */
    kernel_version_size = sizeof(kernel_version);
    if (sysctlbyname("kern.osreldate", &kernel_version, 
                     &kernel_version_size, NULL, NULL) == 0 &&
            kernel_version < KERNEL_WITH_SENDFILE_LENGTH_FIX) {
        api = OLD;
        return 1;
    }
    /* size of kern.osreldate's output might change in the future 
     * causing the sysctlbyname to fail,
     * but if it's the future, we should use the newer API 
     */
    api = NEW;
    return 0;
#else
    /* the build system's kernel is older than 3.4.  Use the old API */
    return 1;
#endif
}

/* Release 3.1 or greater */
apr_status_t apr_socket_sendfile(apr_socket_t * sock, apr_file_t * file,
                                 apr_hdtr_t * hdtr, apr_off_t * offset,
                                 apr_size_t * len, apr_int32_t flags)
{
    off_t nbytes = 0;
    int rv, i;
    struct sf_hdtr headerstruct;
    apr_size_t bytes_to_send = *len;

    /* Ignore flags for now. */
    flags = 0;

    if (!hdtr) {
        hdtr = &no_hdtr;
    }

    else if (hdtr->numheaders && include_hdrs_in_length()) {

        /* On early versions of FreeBSD sendfile, the number of bytes to send 
         * must include the length of the headers.  Don't look at the man page 
         * for this :(  Instead, look at the the logic in 
         * src/sys/kern/uipc_syscalls::sendfile().
         *
         * This was fixed in the middle of 4.6-STABLE
         */
        for (i = 0; i < hdtr->numheaders; i++) {
            bytes_to_send += hdtr->headers[i].iov_len;
        }
    }

    headerstruct.headers = hdtr->headers;
    headerstruct.hdr_cnt = hdtr->numheaders;
    headerstruct.trailers = hdtr->trailers;
    headerstruct.trl_cnt = hdtr->numtrailers;

    /* FreeBSD can send the headers/footers as part of the system call */
    do {
        if (sock->netmask & APR_INCOMPLETE_WRITE) {
            apr_status_t arv;
            sock->netmask &= ~APR_INCOMPLETE_WRITE;
            arv = apr_wait_for_io_or_timeout(NULL, sock, 0);
            if (arv != APR_SUCCESS) {
                *len = 0;
                return arv;
            }
        }
        if (bytes_to_send) {
            /* We won't dare call sendfile() if we don't have
             * header or file bytes to send because bytes_to_send == 0
             * means send the whole file.
             */
            rv = sendfile(file->filedes, /* file to be sent */
                          sock->socketdes, /* socket */
                          *offset,       /* where in the file to start */
                          bytes_to_send, /* number of bytes to send */
                          &headerstruct, /* Headers/footers */
                          &nbytes,       /* number of bytes written */
                          flags);        /* undefined, set to 0 */

            if (rv == -1) {
                if (errno == EAGAIN) {
                    if (sock->timeout) {
                        sock->netmask |= APR_INCOMPLETE_WRITE;
                    }
                    /* FreeBSD's sendfile can return -1/EAGAIN even if it
                     * sent bytes.  Sanitize the result so we get normal EAGAIN
                     * semantics w.r.t. bytes sent.
                     */
                    if (nbytes) {
                        /* normal exit for a big file & non-blocking io */
                        (*len) = nbytes;
                        return APR_SUCCESS;
                    }
                }
            }
            else {       /* rv == 0 (or the kernel is broken) */
                if (nbytes == 0) {
                    /* Most likely the file got smaller after the stat.
                     * Return an error so the caller can do the Right Thing.
                     */
                    (*len) = nbytes;
                    return APR_EOF;
                }
            }
        }    
        else {
            /* just trailer bytes... use writev()
             */
            rv = writev(sock->socketdes,
                        hdtr->trailers,
                        hdtr->numtrailers);
            if (rv > 0) {
                nbytes = rv;
                rv = 0;
            }
            else {
                nbytes = 0;
            }
        }
        if (rv == -1 &&
            errno == EAGAIN && 
            sock->timeout > 0) {
            apr_status_t arv = apr_wait_for_io_or_timeout(NULL, sock, 0);
            if (arv != APR_SUCCESS) {
                *len = 0;
                return arv;
            }
        }
    } while (rv == -1 && (errno == EINTR || errno == EAGAIN));

    (*len) = nbytes;
    if (rv == -1) {
        return errno;
    }
    return APR_SUCCESS;
}

#elif defined(__hpux) || defined(__hpux__)

/* HP cc in ANSI mode defines __hpux; gcc defines __hpux__ */

/* HP-UX Version 10.30 or greater
 * (no worries, because we only get here if autoconfiguration found sendfile)
 */

/* ssize_t sendfile(int s, int fd, off_t offset, size_t nbytes,
 *                  const struct iovec *hdtrl, int flags);
 *
 * nbytes is the number of bytes to send just from the file; as with FreeBSD, 
 * if nbytes == 0, the rest of the file (from offset) is sent
 */

apr_status_t apr_socket_sendfile(apr_socket_t *sock, apr_file_t *file,
                                 apr_hdtr_t *hdtr, apr_off_t *offset,
                                 apr_size_t *len, apr_int32_t flags)
{
    int i;
    apr_ssize_t rc;
    apr_size_t nbytes = *len, headerlen, trailerlen;
    struct iovec hdtrarray[2];
    char *headerbuf, *trailerbuf;

    if (!hdtr) {
        hdtr = &no_hdtr;
    }

    /* Ignore flags for now. */
    flags = 0;

    /* HP-UX can only send one header iovec and one footer iovec; try to
     * only allocate storage to combine input iovecs when we really have to
     */

    switch(hdtr->numheaders) {
    case 0:
        hdtrarray[0].iov_base = NULL;
        hdtrarray[0].iov_len = 0;
        break;
    case 1:
        hdtrarray[0] = hdtr->headers[0];
        break;
    default:
        headerlen = 0;
        for (i = 0; i < hdtr->numheaders; i++) {
            headerlen += hdtr->headers[i].iov_len;
        }  

        /* XXX:  BUHHH? wow, what a memory leak! */
        headerbuf = hdtrarray[0].iov_base = apr_palloc(sock->cntxt, headerlen);
        hdtrarray[0].iov_len = headerlen;

        for (i = 0; i < hdtr->numheaders; i++) {
            memcpy(headerbuf, hdtr->headers[i].iov_base,
                   hdtr->headers[i].iov_len);
            headerbuf += hdtr->headers[i].iov_len;
        }
    }

    switch(hdtr->numtrailers) {
    case 0:
        hdtrarray[1].iov_base = NULL;
        hdtrarray[1].iov_len = 0;
        break;
    case 1:
        hdtrarray[1] = hdtr->trailers[0];
        break;
    default:
        trailerlen = 0;
        for (i = 0; i < hdtr->numtrailers; i++) {
            trailerlen += hdtr->trailers[i].iov_len;
        }

        /* XXX:  BUHHH? wow, what a memory leak! */
        trailerbuf = hdtrarray[1].iov_base = apr_palloc(sock->cntxt, trailerlen);
        hdtrarray[1].iov_len = trailerlen;

        for (i = 0; i < hdtr->numtrailers; i++) {
            memcpy(trailerbuf, hdtr->trailers[i].iov_base,
                   hdtr->trailers[i].iov_len);
            trailerbuf += hdtr->trailers[i].iov_len;
        }
    }

    do {
        if (nbytes) {       /* any bytes to send from the file? */
            rc = sendfile(sock->socketdes,      /* socket  */
                          file->filedes,        /* file descriptor to send */
                          *offset,              /* where in the file to start */
                          nbytes,               /* number of bytes to send from file */
                          hdtrarray,            /* Headers/footers */
                          flags);               /* undefined, set to 0 */
        }
        else {              /* we can't call sendfile() with no bytes to send from the file */
            rc = writev(sock->socketdes, hdtrarray, 2);
        }
    } while (rc == -1 && errno == EINTR);

    if (rc == -1 && 
        (errno == EAGAIN || errno == EWOULDBLOCK) && 
        sock->timeout > 0) {
        apr_status_t arv = apr_wait_for_io_or_timeout(NULL, sock, 0);

        if (arv != APR_SUCCESS) {
            *len = 0;
            return arv;
        }
        else {
            do {
                if (nbytes) {
                    rc = sendfile(sock->socketdes,    /* socket  */
                                  file->filedes,      /* file descriptor to send */
                                  *offset,            /* where in the file to start */
                                  nbytes,             /* number of bytes to send from file */
                                  hdtrarray,          /* Headers/footers */
                                  flags);             /* undefined, set to 0 */
                }
                else {      /* we can't call sendfile() with no bytes to send from the file */
                    rc = writev(sock->socketdes, hdtrarray, 2);
                }
            } while (rc == -1 && errno == EINTR);
        }
    }

    if (rc == -1) {
    *len = 0;
        return errno;
    }

    /* Set len to the number of bytes written */
    *len = rc;
    return APR_SUCCESS;
}
#elif defined(_AIX) || defined(__MVS__)
/* AIX and OS/390 have the same send_file() interface.
 *
 * subtle differences:
 *   AIX doesn't update the file ptr but OS/390 does
 *
 * availability (correctly determined by autoconf):
 *
 * AIX -  version 4.3.2 with APAR IX85388, or version 4.3.3 and above
 * OS/390 - V2R7 and above
 */
apr_status_t apr_socket_sendfile(apr_socket_t * sock, apr_file_t * file,
                                 apr_hdtr_t * hdtr, apr_off_t * offset,
                                 apr_size_t * len, apr_int32_t flags)
{
    int i, ptr, rv = 0;
    void * hbuf=NULL, * tbuf=NULL;
    apr_status_t arv;
    struct sf_parms parms;

    if (!hdtr) {
        hdtr = &no_hdtr;
    }

    /* Ignore flags for now. */
    flags = 0;

    /* AIX can also send the headers/footers as part of the system call */
    parms.header_length = 0;
    if (hdtr && hdtr->numheaders) {
        if (hdtr->numheaders == 1) {
            parms.header_data = hdtr->headers[0].iov_base;
            parms.header_length = hdtr->headers[0].iov_len;
        }
        else {
            for (i = 0; i < hdtr->numheaders; i++) {
                parms.header_length += hdtr->headers[i].iov_len;
            }
#if 0
            /* Keepalives make apr_palloc a bad idea */
            hbuf = malloc(parms.header_length);
#else
            /* but headers are small, so maybe we can hold on to the
             * memory for the life of the socket...
             */
            hbuf = apr_palloc(sock->cntxt, parms.header_length);
#endif
            ptr = 0;
            for (i = 0; i < hdtr->numheaders; i++) {
                memcpy((char *)hbuf + ptr, hdtr->headers[i].iov_base,
                       hdtr->headers[i].iov_len);
                ptr += hdtr->headers[i].iov_len;
            }
            parms.header_data = hbuf;
        }
    }
    else parms.header_data = NULL;
    parms.trailer_length = 0;
    if (hdtr && hdtr->numtrailers) {
        if (hdtr->numtrailers == 1) {
            parms.trailer_data = hdtr->trailers[0].iov_base;
            parms.trailer_length = hdtr->trailers[0].iov_len;
        }
        else {
            for (i = 0; i < hdtr->numtrailers; i++) {
                parms.trailer_length += hdtr->trailers[i].iov_len;
            }
#if 0
            /* Keepalives make apr_palloc a bad idea */
            tbuf = malloc(parms.trailer_length);
#else
            tbuf = apr_palloc(sock->cntxt, parms.trailer_length);
#endif
            ptr = 0;
            for (i = 0; i < hdtr->numtrailers; i++) {
                memcpy((char *)tbuf + ptr, hdtr->trailers[i].iov_base,
                       hdtr->trailers[i].iov_len);
                ptr += hdtr->trailers[i].iov_len;
            }
            parms.trailer_data = tbuf;
        }
    }
    else parms.trailer_data = NULL;

    /* Whew! Headers and trailers set up. Now for the file data */

    parms.file_descriptor = file->filedes;
    parms.file_offset = *offset;
    parms.file_bytes = *len;

    /* O.K. All set up now. Let's go to town */

    if (sock->netmask & APR_INCOMPLETE_WRITE) {
        sock->netmask &= ~APR_INCOMPLETE_WRITE;
        goto do_select;
    }

    do {
        rv = send_file(&(sock->socketdes), /* socket */
                       &(parms),           /* all data */
                       flags);             /* flags */
    } while (rv == -1 && errno == EINTR);

    if (rv == -1 &&
        (errno == EAGAIN || errno == EWOULDBLOCK) &&
        sock->timeout > 0) {
do_select:
        arv = apr_wait_for_io_or_timeout(NULL, sock, 0);
        if (arv != APR_SUCCESS) {
            *len = 0;
            return arv;
        }
        else {
            do {
                rv = send_file(&(sock->socketdes), /* socket */
                               &(parms),           /* all data */
                               flags);             /* flags */
            } while (rv == -1 && errno == EINTR);
        }
    }

    (*len) = parms.bytes_sent;

#if 0
    /* Clean up after ourselves */
    if(hbuf) free(hbuf);
    if(tbuf) free(tbuf);
#endif

    if (rv == -1) {
        return errno;
    }

    if (sock->timeout &&
        (parms.bytes_sent < (parms.file_bytes + parms.header_length + parms.trailer_length))) {
        sock->netmask |= APR_INCOMPLETE_WRITE;
    }

    return APR_SUCCESS;
}
#elif defined(__osf__) && defined (__alpha)
/* Tru64's sendfile implementation doesn't work, and we need to make sure that
 * we don't use it until it is fixed.  If it is used as it is now, it will
 * hang the machine and the only way to fix it is a reboot.
 */
#elif defined(HAVE_SENDFILEV)
/* Solaris 8's sendfilev() interface 
 *
 * SFV_FD_SELF refers to our memory space.
 *
 * Required Sparc patches (or newer):
 * 111297-01, 108528-09, 109472-06, 109234-03, 108995-02, 111295-01, 109025-03,
 * 108991-13
 * Required x86 patches (or newer):
 * 111298-01, 108529-09, 109473-06, 109235-04, 108996-02, 111296-01, 109026-04,
 * 108992-13
 */
apr_status_t apr_socket_sendfile(apr_socket_t *sock, apr_file_t *file,
                                 apr_hdtr_t *hdtr, apr_off_t *offset,
                                 apr_size_t *len, apr_int32_t flags)
{
    apr_status_t rv, arv;
    apr_size_t nbytes;
    sendfilevec_t *sfv;
    int vecs, curvec, i, repeat;
    apr_size_t requested_len = 0;

    if (!hdtr) {
        hdtr = &no_hdtr;
    }

    /* Ignore flags for now. */
    flags = 0;

    /* Calculate how much space we need. */
    vecs = hdtr->numheaders + hdtr->numtrailers + 1;
    sfv = apr_palloc(sock->cntxt, sizeof(sendfilevec_t) * vecs);

    curvec = 0;

    /* Add the headers */
    for (i = 0; i < hdtr->numheaders; i++, curvec++) {
        sfv[curvec].sfv_fd = SFV_FD_SELF;
        sfv[curvec].sfv_flag = 0;
        sfv[curvec].sfv_off = (off_t)hdtr->headers[i].iov_base;
        sfv[curvec].sfv_len = hdtr->headers[i].iov_len;
        requested_len += sfv[curvec].sfv_len;
    }

    /* If the len is 0, we skip the file. */
    if (*len)
    {
        sfv[curvec].sfv_fd = file->filedes;
        sfv[curvec].sfv_flag = 0;
        sfv[curvec].sfv_off = *offset;
        sfv[curvec].sfv_len = *len; 
        requested_len += sfv[curvec].sfv_len;

        curvec++;
    }
    else
        vecs--;

    /* Add the footers */
    for (i = 0; i < hdtr->numtrailers; i++, curvec++) {
        sfv[curvec].sfv_fd = SFV_FD_SELF;
        sfv[curvec].sfv_flag = 0;
        sfv[curvec].sfv_off = (off_t)hdtr->trailers[i].iov_base;
        sfv[curvec].sfv_len = hdtr->trailers[i].iov_len;
        requested_len += sfv[curvec].sfv_len;
    }

    /* If the last write couldn't send all the requested data,
     * wait for the socket to become writable before proceeding
     */
    if (sock->netmask & APR_INCOMPLETE_WRITE) {
        sock->netmask &= ~APR_INCOMPLETE_WRITE;
        arv = apr_wait_for_io_or_timeout(NULL, sock, 0);
        if (arv != APR_SUCCESS) {
            *len = 0;
            return arv;
        }
    }
 
    /* Actually do the sendfilev
     *
     * Solaris may return -1/EAGAIN even if it sent bytes on a non-block sock.
     *
     * If no bytes were originally sent (nbytes == 0) and we are on a TIMEOUT 
     * socket (which as far as the OS is concerned is a non-blocking socket), 
     * we want to retry after waiting for the other side to read the data (as 
     * determined by poll).  Once it is clear to send, we want to retry
     * sending the sendfilevec_t once more.
     */
    arv = 0;
    do {
        /* Clear out the repeat */
        repeat = 0;

        /* socket, vecs, number of vecs, bytes written */
        rv = sendfilev(sock->socketdes, sfv, vecs, &nbytes);

        if (rv == -1 && errno == EAGAIN)
        {
            if (nbytes)
                rv = 0; 
            else if (!arv &&
                     apr_is_option_set(sock->netmask, APR_SO_TIMEOUT) == 1)
            {
                apr_status_t t = apr_wait_for_io_or_timeout(NULL, sock, 0);

                if (t != APR_SUCCESS)
                {
                    *len = 0;
                    return t;
                }

                arv = 1; 
                repeat = 1;
            }
        }
    } while ((rv == -1 && errno == EINTR) || repeat);

    if (rv == -1)
    {
        *len = 0;
        return errno;
    }

    /* Update how much we sent */
    *len = nbytes;
    if (sock->timeout && (*len < requested_len)) {
    sock->netmask |= APR_INCOMPLETE_WRITE;
    }
    return APR_SUCCESS;
}
#else
#error APR has detected sendfile on your system, but nobody has written a
#error version of it for APR yet.  To get past this, either write apr_sendfile
#error or change APR_HAS_SENDFILE in apr.h to 0. 
#endif /* __linux__, __FreeBSD__, __HPUX__, _AIX, __MVS__, Tru64/OSF1 */

/* deprecated */
apr_status_t apr_sendfile(apr_socket_t *sock, apr_file_t *file,
                          apr_hdtr_t *hdtr, apr_off_t *offset, apr_size_t *len,
                          apr_int32_t flags)
{
    return apr_socket_sendfile(sock, file, hdtr, offset, len, flags);
}

#endif /* APR_HAS_SENDFILE */

/* deprecated */
apr_status_t apr_send(apr_socket_t *sock, const char *buf, apr_size_t *len)
{
    return apr_socket_send(sock, buf, len);
}

/* deprecated */
#ifdef HAVE_WRITEV
apr_status_t apr_sendv(apr_socket_t * sock, const struct iovec *vec,
                       apr_int32_t nvec, apr_size_t *len)
{
    return apr_socket_sendv(sock, vec, nvec, len);
}
#endif

/* deprecated */
apr_status_t apr_sendto(apr_socket_t *sock, apr_sockaddr_t *where,
                        apr_int32_t flags, const char *buf, apr_size_t *len)
{
    return apr_socket_sendto(sock, where, flags, buf, len);
}

/* deprecated */
apr_status_t apr_recvfrom(apr_sockaddr_t *from, apr_socket_t *sock,
                          apr_int32_t flags, char *buf, 
                          apr_size_t *len)
{
    return apr_socket_recvfrom(from, sock, flags, buf, len);
}

/* deprecated */
apr_status_t apr_recv(apr_socket_t *sock, char *buf, apr_size_t *len)
{
    return apr_socket_recv(sock, buf, len);
}
