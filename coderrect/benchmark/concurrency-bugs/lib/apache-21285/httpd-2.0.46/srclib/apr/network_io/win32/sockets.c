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
#include "apr_network_io.h"
#include "apr_general.h"
#include "apr_lib.h"
#include "apr_portable.h"
#include <string.h>
#include "apr_arch_inherit.h"
#include "apr_arch_misc.h"

static char generic_inaddr_any[16] = {0}; /* big enough for IPv4 or IPv6 */

static apr_status_t socket_cleanup(void *sock)
{
    apr_socket_t *thesocket = sock;

    if (thesocket->socketdes != INVALID_SOCKET) {
        if (closesocket(thesocket->socketdes) == SOCKET_ERROR) {
            return apr_get_netos_error();
        }
        thesocket->socketdes = INVALID_SOCKET;
    }
    return APR_SUCCESS;
}

static void set_socket_vars(apr_socket_t *sock, int family, int type, int protocol)
{
    sock->type = type;
    sock->protocol = protocol;
    apr_sockaddr_vars_set(sock->local_addr, family, 0);
    apr_sockaddr_vars_set(sock->remote_addr, family, 0);
}                                                                                                  
static void alloc_socket(apr_socket_t **new, apr_pool_t *p)
{
    *new = (apr_socket_t *)apr_pcalloc(p, sizeof(apr_socket_t));
    (*new)->cntxt = p;
    (*new)->local_addr = (apr_sockaddr_t *)apr_pcalloc((*new)->cntxt,
                                                       sizeof(apr_sockaddr_t));
    (*new)->local_addr->pool = p;
    (*new)->remote_addr = (apr_sockaddr_t *)apr_pcalloc((*new)->cntxt,
                                                        sizeof(apr_sockaddr_t));
    (*new)->remote_addr->pool = p;
}

APR_DECLARE(apr_status_t) apr_socket_protocol_get(apr_socket_t *sock,
                                                  int *protocol)
{
    *protocol = sock->protocol;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_socket_create_ex(apr_socket_t **new, int family,
                                               int type, int protocol, 
                                               apr_pool_t *cont)
{
    int downgrade = (family == AF_UNSPEC);

    if (family == AF_UNSPEC) {
#if APR_HAVE_IPV6
        family = AF_INET6;
#else
        family = AF_INET;
#endif
    }

    alloc_socket(new, cont);

    /* For right now, we are not using socket groups.  We may later.
     * No flags to use when creating a socket, so use 0 for that parameter as well.
     */
    (*new)->socketdes = socket(family, type, protocol);
#if APR_HAVE_IPV6
    if ((*new)->socketdes == INVALID_SOCKET && downgrade) {
        family = AF_INET;
        (*new)->socketdes = socket(family, type, protocol);
    }
#endif

    if ((*new)->socketdes == INVALID_SOCKET) {
        return apr_get_netos_error();
    }

#ifdef WIN32
    /* Socket handles are never truly inheritable, there are too many
     * bugs associated.  WSADuplicateSocket will copy them, but for our
     * purposes, always transform the socket() created as a non-inherited
     * handle
     */
#if APR_HAS_UNICODE_FS
    IF_WIN_OS_IS_UNICODE {
        /* A different approach.  Many users report errors such as 
         * (32538)An operation was attempted on something that is not 
         * a socket.  : Parent: WSADuplicateSocket failed...
         *
         * This appears that the duplicated handle is no longer recognized
         * as a socket handle.  SetHandleInformation should overcome that
         * problem by not altering the handle identifier.  But this won't
         * work on 9x - it's unsupported.
         */
        SetHandleInformation((HANDLE) (*new)->socketdes, 
                             HANDLE_FLAG_INHERIT, 0);
    }
#endif
#if APR_HAS_ANSI_FS
    ELSE_WIN_OS_IS_ANSI {
        HANDLE hProcess = GetCurrentProcess();
        HANDLE dup;
        if (DuplicateHandle(hProcess, (HANDLE) (*new)->socketdes, hProcess, 
                            &dup, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            closesocket((*new)->socketdes);
            (*new)->socketdes = (SOCKET) dup;
        }
    }
#endif

#endif /* def WIN32 */

    set_socket_vars(*new, family, type, protocol);

    (*new)->timeout = -1;
    (*new)->disconnected = 0;

    apr_pool_cleanup_register((*new)->cntxt, (void *)(*new), 
                        socket_cleanup, apr_pool_cleanup_null);

    return APR_SUCCESS;
} 

APR_DECLARE(apr_status_t) apr_socket_create(apr_socket_t **new, int family,
                                            int type, apr_pool_t *cont)
{
    return apr_socket_create_ex(new, family, type, 0, cont);
}

APR_DECLARE(apr_status_t) apr_socket_shutdown(apr_socket_t *thesocket,
                                              apr_shutdown_how_e how)
{
    int winhow = 0;

#ifdef SD_RECEIVE
    switch (how) {
        case APR_SHUTDOWN_READ: {
            winhow = SD_RECEIVE;
            break;
        }
        case APR_SHUTDOWN_WRITE: {
            winhow = SD_SEND;
            break;
        }
        case APR_SHUTDOWN_READWRITE: {
            winhow = SD_BOTH;
            break;
        }
        default:
            return APR_BADARG;
    }
#endif
    if (shutdown(thesocket->socketdes, winhow) == 0) {
        return APR_SUCCESS;
    }
    else {
        return apr_get_netos_error();
    }
}

APR_DECLARE(apr_status_t) apr_socket_close(apr_socket_t *thesocket)
{
    apr_pool_cleanup_kill(thesocket->cntxt, thesocket, socket_cleanup);
    return socket_cleanup(thesocket);
}

APR_DECLARE(apr_status_t) apr_socket_bind(apr_socket_t *sock,
                                          apr_sockaddr_t *sa)
{
    if (bind(sock->socketdes, 
             (struct sockaddr *)&sa->sa, 
             sa->salen) == -1) {
        return apr_get_netos_error();
    }
    else {
        sock->local_addr = sa;
        if (sock->local_addr->sa.sin.sin_port == 0) {
            sock->local_port_unknown = 1; /* ephemeral port */
        }
        return APR_SUCCESS;
    }
}

APR_DECLARE(apr_status_t) apr_socket_listen(apr_socket_t *sock,
                                            apr_int32_t backlog)
{
    if (listen(sock->socketdes, backlog) == SOCKET_ERROR)
        return apr_get_netos_error();
    else
        return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_socket_accept(apr_socket_t **new, 
                                            apr_socket_t *sock, apr_pool_t *p)
{
    SOCKET s;
    struct sockaddr sa;
    int salen = sizeof(sock->remote_addr->sa);

    /* Don't allocate the memory until after we call accept. This allows
       us to work with nonblocking sockets. */
    s = accept(sock->socketdes, (struct sockaddr *)&sa, &salen);
    if (s == INVALID_SOCKET) {
        return apr_get_netos_error();
    }

    alloc_socket(new, p);
    set_socket_vars(*new, sock->local_addr->sa.sin.sin_family, SOCK_STREAM, 
                    sock->protocol);

    (*new)->timeout = -1;   
    (*new)->disconnected = 0;

    (*new)->socketdes = s;
    /* XXX next line looks bogus w.r.t. AF_INET6 support */
    (*new)->remote_addr->salen = sizeof((*new)->remote_addr->sa);
    memcpy (&(*new)->remote_addr->sa, &sa, salen);
    *(*new)->local_addr = *sock->local_addr;

    /* The above assignment just overwrote the pool entry. Setting the local_addr 
       pool for the accepted socket back to what it should be.  Otherwise all 
       allocations for this socket will come from a server pool that is not
       freed until the process goes down.*/
    (*new)->local_addr->pool = p;

    /* fix up any pointers which are no longer valid */
    if (sock->local_addr->sa.sin.sin_family == AF_INET) {
        (*new)->local_addr->ipaddr_ptr = &(*new)->local_addr->sa.sin.sin_addr;
    }
#if APR_HAVE_IPV6
    else if (sock->local_addr->sa.sin.sin_family == AF_INET6) {
        (*new)->local_addr->ipaddr_ptr = &(*new)->local_addr->sa.sin6.sin6_addr;
    }
#endif
    (*new)->remote_addr->port = ntohs((*new)->remote_addr->sa.sin.sin_port);
    if (sock->local_port_unknown) {
        /* not likely for a listening socket, but theoretically possible :) */
        (*new)->local_port_unknown = 1;
    }

#if APR_TCP_NODELAY_INHERITED
    if (apr_is_option_set(sock->netmask, APR_TCP_NODELAY) == 1) {
        apr_set_option(&(*new)->netmask, APR_TCP_NODELAY, 1);
    }
#endif /* TCP_NODELAY_INHERITED */
#if APR_O_NONBLOCK_INHERITED
    if (apr_is_option_set(sock->netmask, APR_SO_NONBLOCK) == 1) {
        apr_set_option(&(*new)->netmask, APR_SO_NONBLOCK, 1);
    }
#endif /* APR_O_NONBLOCK_INHERITED */

    if (sock->local_interface_unknown ||
        !memcmp(sock->local_addr->ipaddr_ptr,
                generic_inaddr_any,
                sock->local_addr->ipaddr_len)) {
        /* If the interface address inside the listening socket's local_addr wasn't
         * up-to-date, we don't know local interface of the connected socket either.
         *
         * If the listening socket was not bound to a specific interface, we
         * don't know the local_addr of the connected socket.
         */
        (*new)->local_interface_unknown = 1;
    }

    apr_pool_cleanup_register((*new)->cntxt, (void *)(*new), 
                        socket_cleanup, apr_pool_cleanup_null);
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_socket_connect(apr_socket_t *sock, 
                                             apr_sockaddr_t *sa)
{
    apr_status_t rv;

    if ((sock->socketdes == INVALID_SOCKET) || (!sock->local_addr)) {
        return APR_ENOTSOCK;
    }

    if (connect(sock->socketdes, (const struct sockaddr *)&sa->sa.sin,
                sa->salen) == SOCKET_ERROR) {
        int rc;
        struct timeval tv, *tvptr;
        fd_set wfdset, efdset;

        rv = apr_get_netos_error();
        if (rv != APR_FROM_OS_ERROR(WSAEWOULDBLOCK)) {
            return rv;
        }

        if (sock->timeout == 0) {
            /* Tell the app that the connect is in progress...
             * Gotta play some games here.  connect on Unix will return 
             * EINPROGRESS under the same circumstances that Windows 
             * returns WSAEWOULDBLOCK. Do some adhoc canonicalization...
             */
            return APR_FROM_OS_ERROR(WSAEINPROGRESS);
        }

        /* wait for the connect to complete or timeout */
        FD_ZERO(&wfdset);
        FD_SET(sock->socketdes, &wfdset);
        FD_ZERO(&efdset);
        FD_SET(sock->socketdes, &efdset);

        if (sock->timeout < 0) {
            tvptr = NULL;
        }
        else {
            /* casts for winsock/timeval definition */
            tv.tv_sec =  (long)apr_time_sec(sock->timeout);
            tv.tv_usec = (int)apr_time_usec(sock->timeout);
            tvptr = &tv;
        }
        rc = select(FD_SETSIZE+1, NULL, &wfdset, &efdset, tvptr);
        if (rc == SOCKET_ERROR) {
            return apr_get_netos_error();
        }
        else if (!rc) {
            return APR_FROM_OS_ERROR(WSAETIMEDOUT);
        }
        /* Evaluate the efdset */
        if (FD_ISSET(sock->socketdes, &efdset)) {
            /* The connect failed. */
            int rclen = sizeof(rc);
            if (getsockopt(sock->socketdes, SOL_SOCKET, SO_ERROR, (char*) &rc, &rclen)) {
                return apr_get_netos_error();
            }
            return APR_FROM_OS_ERROR(rc);
        }
    }
    /* connect was OK .. amazing */
    sock->remote_addr = sa;
    if (sock->local_addr->sa.sin.sin_port == 0) {
        sock->local_port_unknown = 1;
    }
    if (!memcmp(sock->local_addr->ipaddr_ptr,
                generic_inaddr_any,
                sock->local_addr->ipaddr_len)) {
        /* not bound to specific local interface; connect() had to assign
         * one for the socket
         */
        sock->local_interface_unknown = 1;
    }
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_socket_data_get(void **data, const char *key,
                                             apr_socket_t *socket)
{
    return apr_pool_userdata_get(data, key, socket->cntxt);
}

APR_DECLARE(apr_status_t) apr_socket_data_set(apr_socket_t *socket, void *data,
                                             const char *key,
                                             apr_status_t (*cleanup)(void *))
{
    return apr_pool_userdata_set(data, key, cleanup, socket->cntxt);
}

APR_DECLARE(apr_status_t) apr_os_sock_get(apr_os_sock_t *thesock,
                                          apr_socket_t *sock)
{
    *thesock = sock->socketdes;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_os_sock_make(apr_socket_t **apr_sock,
                                           apr_os_sock_info_t *os_sock_info,
                                           apr_pool_t *cont)
{
    alloc_socket(apr_sock, cont);
#ifdef APR_ENABLE_FOR_1_0 /* no protocol field yet */
    set_socket_vars(*apr_sock, os_sock_info->family, os_sock_info->type, os_sock_info->protocol);
#else
    set_socket_vars(*apr_sock, os_sock_info->family, os_sock_info->type, 0);
#endif
    (*apr_sock)->timeout = -1;
    (*apr_sock)->disconnected = 0;
    (*apr_sock)->socketdes = *os_sock_info->os_sock;
    if (os_sock_info->local) {
        memcpy(&(*apr_sock)->local_addr->sa.sin, 
               os_sock_info->local, 
               (*apr_sock)->local_addr->salen);
        (*apr_sock)->local_addr->pool = cont;
        /* XXX IPv6 - this assumes sin_port and sin6_port at same offset */
        (*apr_sock)->local_addr->port = ntohs((*apr_sock)->local_addr->sa.sin.sin_port);
    }
    else {
        (*apr_sock)->local_port_unknown = (*apr_sock)->local_interface_unknown = 1;
    }
    if (os_sock_info->remote) {
        memcpy(&(*apr_sock)->remote_addr->sa.sin, 
               os_sock_info->remote,
               (*apr_sock)->remote_addr->salen);
        (*apr_sock)->remote_addr->pool = cont;
        /* XXX IPv6 - this assumes sin_port and sin6_port at same offset */
        (*apr_sock)->remote_addr->port = ntohs((*apr_sock)->remote_addr->sa.sin.sin_port);
    }
    else {
        (*apr_sock)->remote_addr_unknown = 1;
    }
        
    apr_pool_cleanup_register((*apr_sock)->cntxt, (void *)(*apr_sock), 
                        socket_cleanup, apr_pool_cleanup_null);

    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_os_sock_put(apr_socket_t **sock,
                                          apr_os_sock_t *thesock,
                                          apr_pool_t *cont)
{
    if ((*sock) == NULL) {
        alloc_socket(sock, cont);
        /* XXX figure out the actual socket type here */
        /* *or* just decide that apr_os_sock_put() has to be told the family and type */
        set_socket_vars(*sock, AF_INET, SOCK_STREAM, 0);
        (*sock)->timeout = -1;
        (*sock)->disconnected = 0;
    }
    (*sock)->local_port_unknown = (*sock)->local_interface_unknown = 1;
    (*sock)->remote_addr_unknown = 1;
    (*sock)->socketdes = *thesock;
    return APR_SUCCESS;
}


/* Sockets cannot be inherited through the standard sockets
 * inheritence.  WSADuplicateSocket must be used.
 * This is not trivial to implement.
 */

APR_DECLARE(apr_status_t) apr_socket_inherit_set(apr_socket_t *socket)    
{    
    return APR_ENOTIMPL;
}    
/* Deprecated */    
APR_DECLARE(void) apr_socket_set_inherit(apr_socket_t *socket)    
{    
    apr_socket_inherit_set(socket);    
}

APR_DECLARE(apr_status_t) apr_socket_inherit_unset(apr_socket_t *socket)    
{    
    return APR_ENOTIMPL;
}    
/* Deprecated */    
APR_DECLARE(void) apr_socket_unset_inherit(apr_socket_t *socket)    
{    
    apr_socket_inherit_unset(socket);    
}
/* Deprecated */
APR_DECLARE(apr_status_t) apr_shutdown(apr_socket_t *thesocket,
                                       apr_shutdown_how_e how)
{
    return apr_socket_shutdown(thesocket, how);
}

/* Deprecated */
APR_DECLARE(apr_status_t) apr_bind(apr_socket_t *sock, apr_sockaddr_t *sa)
{
    return apr_socket_bind(sock, sa);
}

/* Deprecated */
APR_DECLARE(apr_status_t) apr_listen(apr_socket_t *sock, apr_int32_t backlog)
{
    return apr_socket_listen(sock, backlog);
}

/* Deprecated */
APR_DECLARE(apr_status_t) apr_accept(apr_socket_t **new, apr_socket_t *sock,
                                     apr_pool_t *p)
{
    return apr_socket_accept(new, sock, p);
}

/* Deprecated */
APR_DECLARE(apr_status_t) apr_connect(apr_socket_t *sock, apr_sockaddr_t *sa)
{
    return apr_socket_connect(sock, sa);
}
