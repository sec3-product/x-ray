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
 *
 * Portions of this software are based upon public domain software
 * originally written at the National Center for Supercomputing Applications,
 * University of Illinois, Urbana-Champaign.
 */

#ifndef MOD_PROXY_H
#define MOD_PROXY_H 

/*
 * Main include file for the Apache proxy
 */

/*

   Also note numerous FIXMEs and CHECKMEs which should be eliminated.

   This code is once again experimental!

   Things to do:

   1. Make it completely work (for FTP too)

   2. HTTP/1.1

   Chuck Murcko <chuck@topsail.org> 02-06-01

 */

#define CORE_PRIVATE

#include "apr_hooks.h"
#include "apr.h"
#include "apr_lib.h"
#include "apr_strings.h"
#include "apr_buckets.h"
#include "apr_md5.h"
#include "apr_network_io.h"
#include "apr_pools.h"
#include "apr_strings.h"
#include "apr_uri.h"
#include "apr_date.h"
#include "apr_fnmatch.h"
#define APR_WANT_STRFUNC
#include "apr_want.h"

#include "httpd.h"
#include "http_config.h"
#include "ap_config.h"
#include "http_core.h"
#include "http_protocol.h"
#include "http_request.h"
#include "http_vhost.h"
#include "http_main.h"
#include "http_log.h"
#include "http_connection.h"
#include "util_filter.h"
#include "util_ebcdic.h"

#if APR_HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#if APR_HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif

/* for proxy_canonenc() */
enum enctype {
    enc_path, enc_search, enc_user, enc_fpath, enc_parm
};

#if APR_CHARSET_EBCDIC
#define CRLF   "\r\n"
#else /*APR_CHARSET_EBCDIC*/
#define CRLF   "\015\012"
#endif /*APR_CHARSET_EBCDIC*/

/* default Max-Forwards header setting */
#define DEFAULT_MAX_FORWARDS	10

/* static information about a remote proxy */
struct proxy_remote {
    const char *scheme;		/* the schemes handled by this proxy, or '*' */
    const char *protocol;	/* the scheme used to talk to this proxy */
    const char *hostname;	/* the hostname of this proxy */
    apr_port_t  port;		/* the port for this proxy */
    regex_t *regexp;		/* compiled regex (if any) for the remote */
    int use_regex;		/* simple boolean. True if we have a regex pattern */
};

struct proxy_alias {
    const char *real;
    const char *fake;
};

struct dirconn_entry {
    char *name;
    struct in_addr addr, mask;
    struct apr_sockaddr_t *hostaddr;
    int (*matcher) (struct dirconn_entry * This, request_rec *r);
};

struct noproxy_entry {
    const char *name;
    struct apr_sockaddr_t *addr;
};

typedef struct {
    apr_array_header_t *proxies;
    apr_array_header_t *sec_proxy;
    apr_array_header_t *aliases;
    apr_array_header_t *raliases;
    apr_array_header_t *noproxies;
    apr_array_header_t *dirconn;
    apr_array_header_t *allowed_connect_ports;
    const char *domain;		/* domain name to use in absence of a domain name in the request */
    int req;			/* true if proxy requests are enabled */
    char req_set;
    enum {
      via_off,
      via_on,
      via_block,
      via_full
    } viaopt;                   /* how to deal with proxy Via: headers */
    char viaopt_set;
    apr_size_t recv_buffer_size;
    char recv_buffer_size_set;
    apr_size_t io_buffer_size;
    char io_buffer_size_set;
    long maxfwd;
    char maxfwd_set;
    /** 
     * the following setting masks the error page
     * returned from the 'proxied server' and just 
     * forwards the status code upwards.
     * This allows the main server (us) to generate
     * the error page, (so it will look like a error
     * returned from the rest of the system 
     */
    int error_override;
    int error_override_set;
    int preserve_host;
    int preserve_host_set;
    apr_interval_time_t timeout;
    apr_interval_time_t timeout_set;
    enum {
      bad_error,
      bad_ignore,
      bad_body
    } badopt;                   /* how to deal with bad headers */
    char badopt_set;

} proxy_server_conf;

typedef struct {
    const char *p;            /* The path */
    int         p_is_fnmatch; /* Is this path an fnmatch candidate? */
    regex_t    *r;            /* Is this a regex? */
} proxy_dir_conf;

typedef struct {
    conn_rec *connection;
    char *hostname;
    apr_port_t port;
    int is_ssl;
} proxy_conn_rec;

typedef struct {
        float cache_completion; /* completion percentage */
        int content_length; /* length of the content */
} proxy_completion;


/* hooks */

/* Create a set of PROXY_DECLARE(type), PROXY_DECLARE_NONSTD(type) and 
 * PROXY_DECLARE_DATA with appropriate export and import tags for the platform
 */
#if !defined(WIN32)
#define PROXY_DECLARE(type)            type
#define PROXY_DECLARE_NONSTD(type)     type
#define PROXY_DECLARE_DATA
#elif defined(PROXY_DECLARE_STATIC)
#define PROXY_DECLARE(type)            type __stdcall
#define PROXY_DECLARE_NONSTD(type)     type
#define PROXY_DECLARE_DATA
#elif defined(PROXY_DECLARE_EXPORT)
#define PROXY_DECLARE(type)            __declspec(dllexport) type __stdcall
#define PROXY_DECLARE_NONSTD(type)     __declspec(dllexport) type
#define PROXY_DECLARE_DATA             __declspec(dllexport)
#else
#define PROXY_DECLARE(type)            __declspec(dllimport) type __stdcall
#define PROXY_DECLARE_NONSTD(type)     __declspec(dllimport) type
#define PROXY_DECLARE_DATA             __declspec(dllimport)
#endif

/**
 * Hook an optional proxy hook.  Unlike static hooks, this uses a macro
 * instead of a function.
 */
#define PROXY_OPTIONAL_HOOK(name,fn,pre,succ,order) \
        APR_OPTIONAL_HOOK(proxy,name,fn,pre,succ,order)

APR_DECLARE_EXTERNAL_HOOK(proxy, PROXY, int, scheme_handler, (request_rec *r, 
                          proxy_server_conf *conf, char *url, 
                          const char *proxyhost, apr_port_t proxyport))
APR_DECLARE_EXTERNAL_HOOK(proxy, PROXY, int, canon_handler, (request_rec *r, 
                          char *url))

APR_DECLARE_EXTERNAL_HOOK(proxy, PROXY, int, create_req, (request_rec *r, request_rec *pr))
APR_DECLARE_EXTERNAL_HOOK(proxy, PROXY, int, fixups, (request_rec *r)) 

/* proxy_util.c */

PROXY_DECLARE(request_rec *)ap_proxy_make_fake_req(conn_rec *c, request_rec *r);
PROXY_DECLARE(int) ap_proxy_hex2c(const char *x);
PROXY_DECLARE(void) ap_proxy_c2hex(int ch, char *x);
PROXY_DECLARE(char *)ap_proxy_canonenc(apr_pool_t *p, const char *x, int len, enum enctype t,
			int isenc);
PROXY_DECLARE(char *)ap_proxy_canon_netloc(apr_pool_t *p, char **const urlp, char **userp,
			 char **passwordp, char **hostp, apr_port_t *port);
PROXY_DECLARE(const char *)ap_proxy_date_canon(apr_pool_t *p, const char *x);
PROXY_DECLARE(apr_table_t *)ap_proxy_read_headers(request_rec *r, request_rec *rp, char *buffer, int size, conn_rec *c);
PROXY_DECLARE(int) ap_proxy_liststr(const char *list, const char *val);
PROXY_DECLARE(char *)ap_proxy_removestr(apr_pool_t *pool, const char *list, const char *val);
PROXY_DECLARE(int) ap_proxy_hex2sec(const char *x);
PROXY_DECLARE(void) ap_proxy_sec2hex(int t, char *y);
PROXY_DECLARE(int) ap_proxyerror(request_rec *r, int statuscode, const char *message);
PROXY_DECLARE(int) ap_proxy_is_ipaddr(struct dirconn_entry *This, apr_pool_t *p);
PROXY_DECLARE(int) ap_proxy_is_domainname(struct dirconn_entry *This, apr_pool_t *p);
PROXY_DECLARE(int) ap_proxy_is_hostname(struct dirconn_entry *This, apr_pool_t *p);
PROXY_DECLARE(int) ap_proxy_is_word(struct dirconn_entry *This, apr_pool_t *p);
PROXY_DECLARE(int) ap_proxy_checkproxyblock(request_rec *r, proxy_server_conf *conf, apr_sockaddr_t *uri_addr);
PROXY_DECLARE(int) ap_proxy_pre_http_request(conn_rec *c, request_rec *r);
PROXY_DECLARE(apr_status_t) ap_proxy_string_read(conn_rec *c, apr_bucket_brigade *bb, char *buff, size_t bufflen, int *eos);
PROXY_DECLARE(void) ap_proxy_table_unmerge(apr_pool_t *p, apr_table_t *t, char *key);
PROXY_DECLARE(int) ap_proxy_connect_to_backend(apr_socket_t **, const char *, apr_sockaddr_t *, const char *, proxy_server_conf *, server_rec *, apr_pool_t *);
PROXY_DECLARE(int) ap_proxy_ssl_enable(conn_rec *c);
PROXY_DECLARE(int) ap_proxy_ssl_disable(conn_rec *c);

/* For proxy_util */
extern module AP_MODULE_DECLARE_DATA proxy_module;

#endif /*MOD_PROXY_H*/
