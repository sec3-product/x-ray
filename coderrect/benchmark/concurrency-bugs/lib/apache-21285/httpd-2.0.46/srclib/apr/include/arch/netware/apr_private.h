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

/*
 * Note: 
 * This is the windows specific autoconf-like config file
 * which unix would create at build time.
 */

#ifdef NETWARE

#ifndef APR_PRIVATE_H
#define APR_PRIVATE_H

/* Include the public APR symbols, include our idea of the 'right'
 * subset of the Windows.h header.  This saves us repetition.
 */
#include "apr.h"

#include <sys/types.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <library.h>

/* Use this section to define all of the HAVE_FOO_H
 * that are required to build properly.
 */
#define HAVE_DLFCN_H    1
#define HAVE_LIMITS_H   1
#define HAVE_SIGNAL_H   1
#define HAVE_STDDEF_H   1
#define HAVE_STDLIB_H   1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_MMAN_H 1
#define HAVE_FCNTL_H    1
#define HAVE_ICONV_H    1

#define HAVE_STRICMP    1
#define HAVE_STRNICMP   1
#define HAVE_STRDUP     1
#define HAVE_STRSTR     1
#define HAVE_MEMCHR     1
#define HAVE_CALLOC     1

/*#define DSO_USE_DLFCN */

#ifdef NW_BUILD_IPV6
#define HAVE_GETADDRINFO 1
#define HAVE_GETNAMEINFO 1
#endif

/* 1 is used for SIGABRT on netware */
/* 2 is used for SIGFPE on netware */
/* 3 is used for SIGILL on netware */
/* 4 is used for SIGINT on netware */
/* 5 is used for SIGSEGV on netware */
/* 6 is used for SIGTERM on netware */
/* 7 is used for SIGPOLL on netware */

#define SIGKILL         11
#define SA_NOCLDSTOP    12
#define SIGALRM         13
#define SIGCHLD         14 
#define SIGCONT         15
#define SIGHUP          16
#define SIGPIPE         17
#define SIGQUIT         18
#define SIGSTOP         19
#define SIGTSTP         20
#define SIGTTIN         21
#define SIGTTOU         22
#define SIGUSR1         23
#define SIGUSR2         24
    
#define SIGTRAP         25
#define SIGIOT          26
#define SIGBUS          27
#define SIGSTKFLT       28
#define SIGURG          29
#define SIGXCPU         30
#define SIGXFSZ         31
#define SIGVTALRM       32
#define SIGPROF         33
#define SIGWINCH        34
#define SIGIO           35

#if 0
#define __attribute__(__x) 

/* APR COMPATABILITY FUNCTIONS
 * This section should be used to define functions and
 * macros which are need to make Windows features look
 * like POSIX features.
 */
typedef void (Sigfunc)(int);
#endif

#define strcasecmp(s1, s2)       stricmp(s1, s2)
#define Sleep(t)                 delay(t)
#define lstat(a,b)               stat(a,b)
#define _getch()                 getcharacter()

#define SIZEOF_SHORT           2
#define SIZEOF_INT             4
#define SIZEOF_LONGLONG        8
#define SIZEOF_CHAR            1
#define SIZEOF_SSIZE_T         SIZEOF_INT

void netware_pool_proc_cleanup ();

/* NLM registration routines for managing which NLMs
    are using the library. */
int register_NLM(void *NLMHandle);
int unregister_NLM(void *NLMHandle);

/* Application global data management */
extern int  gLibId;
extern void *gLibHandle;

typedef struct app_data {
    int     initialized;
    void*   gPool;
    void*   gs_aHooksToSort;
    void*   gs_phOptionalHooks;
    void*   gs_phOptionalFunctions;
} APP_DATA;

int setGlobalPool(void *data);
void* getGlobalPool();
int setStatCache(void *data);
void* getStatCache();

/* Redefine malloc to use the library malloc call so 
    that all of the memory resources will be owned
    and can be shared by the library. */
#undef malloc
#define malloc(x) library_malloc(gLibHandle,x)

/*
 * Include common private declarations.
 */
#include "../apr_private_common.h"

#endif  /*APR_PRIVATE_H*/
#endif  /*NETWARE*/
