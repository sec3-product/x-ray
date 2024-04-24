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

#ifndef APR_LIB_H
#define APR_LIB_H

/**
 * @file apr_lib.h
 * This is collection of oddballs that didn't fit anywhere else,
 * and might move to more appropriate headers with the release
 * of APR 1.0.
 * @brief APR general purpose library routines
 */

#include "apr.h"
#include "apr_errno.h"

#if APR_HAVE_CTYPE_H
#include <ctype.h>
#endif
#if APR_HAVE_STDARG_H
#include <stdarg.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @defgroup apr_lib General Purpose Library Routines
 * @ingroup APR 
 * This is collection of oddballs that didn't fit anywhere else,
 * and might move to more appropriate headers with the release
 * of APR 1.0.
 * @{
 */

/** A constant representing a 'large' string. */
#define HUGE_STRING_LEN 8192

/*
 * Define the structures used by the APR general-purpose library.
 */

/** @see apr_vformatter_buff_t */
typedef struct apr_vformatter_buff_t apr_vformatter_buff_t;

/**
 * Structure used by the variable-formatter routines.
 */
struct apr_vformatter_buff_t {
    /** The current position */
    char *curpos;
    /** The end position of the format string */
    char *endpos;
};

/**
 * return the final element of the pathname
 * @param pathname The path to get the final element of
 * @return the final element of the path
 * @remark
 * <PRE>
 * For example:
 *                 "/foo/bar/gum"    -> "gum"
 *                 "/foo/bar/gum/"   -> ""
 *                 "gum"             -> "gum"
 *                 "bs\\path\\stuff" -> "stuff"
 * </PRE>
 */
APR_DECLARE(const char *) apr_filepath_name_get(const char *pathname);

/** @deprecated @see apr_filepath_name_get */
APR_DECLARE(const char *) apr_filename_of_pathname(const char *pathname);

/**
 * apr_killpg
 * Small utility macros to make things easier to read.  Not usually a
 * goal, to be sure..
 */

#ifdef WIN32
#define apr_killpg(x, y)
#else /* WIN32 */
#ifdef NO_KILLPG
#define apr_killpg(x, y)        (kill (-(x), (y)))
#else /* NO_KILLPG */
#define apr_killpg(x, y)        (killpg ((x), (y)))
#endif /* NO_KILLPG */
#endif /* WIN32 */

/**
 * apr_vformatter() is a generic printf-style formatting routine
 * with some extensions.
 * @param flush_func The function to call when the buffer is full
 * @param c The buffer to write to
 * @param fmt The format string
 * @param ap The arguments to use to fill out the format string.
 *
 * @remark
 * <PRE>
 * The extensions are:
 *
 * %%pA	takes a struct in_addr *, and prints it as a.b.c.d
 * %%pI	takes an apr_sockaddr_t * and prints it as a.b.c.d:port or
 *      [ipv6-address]:port
 * %%pT takes an apr_os_thread_t * and prints it in decimal
 *      ('0' is printed if !APR_HAS_THREADS)
 * %%pp takes a void * and outputs it in hex
 *
 * The %%p hacks are to force gcc's printf warning code to skip
 * over a pointer argument without complaining.  This does
 * mean that the ANSI-style %%p (output a void * in hex format) won't
 * work as expected at all, but that seems to be a fair trade-off
 * for the increased robustness of having printf-warnings work.
 *
 * Additionally, apr_vformatter allows for arbitrary output methods
 * using the apr_vformatter_buff and flush_func.
 *
 * The apr_vformatter_buff has two elements curpos and endpos.
 * curpos is where apr_vformatter will write the next byte of output.
 * It proceeds writing output to curpos, and updating curpos, until
 * either the end of output is reached, or curpos == endpos (i.e. the
 * buffer is full).
 *
 * If the end of output is reached, apr_vformatter returns the
 * number of bytes written.
 *
 * When the buffer is full, the flush_func is called.  The flush_func
 * can return -1 to indicate that no further output should be attempted,
 * and apr_vformatter will return immediately with -1.  Otherwise
 * the flush_func should flush the buffer in whatever manner is
 * appropriate, re apr_pool_t nitialize curpos and endpos, and return 0.
 *
 * Note that flush_func is only invoked as a result of attempting to
 * write another byte at curpos when curpos >= endpos.  So for
 * example, it's possible when the output exactly matches the buffer
 * space available that curpos == endpos will be true when
 * apr_vformatter returns.
 *
 * apr_vformatter does not call out to any other code, it is entirely
 * self-contained.  This allows the callers to do things which are
 * otherwise "unsafe".  For example, apr_psprintf uses the "scratch"
 * space at the unallocated end of a block, and doesn't actually
 * complete the allocation until apr_vformatter returns.  apr_psprintf
 * would be completely broken if apr_vformatter were to call anything
 * that used a apr_pool_t.  Similarly http_bprintf() uses the "scratch"
 * space at the end of its output buffer, and doesn't actually note
 * that the space is in use until it either has to flush the buffer
 * or until apr_vformatter returns.
 * </PRE>
 */
APR_DECLARE(int) apr_vformatter(int (*flush_func)(apr_vformatter_buff_t *b),
			        apr_vformatter_buff_t *c, const char *fmt,
			        va_list ap);

/**
 * Display a prompt and read in the password from stdin.
 * @param prompt The prompt to display
 * @param pwbuf Buffer to store the password
 * @param bufsize The length of the password buffer.
 */
APR_DECLARE(apr_status_t) apr_password_get(const char *prompt, char *pwbuf, 
                                           apr_size_t *bufsize);

/** @} */

/**
 * @defgroup apr_ctype ctype functions
 * These macros allow correct support of 8-bit characters on systems which
 * support 8-bit characters.  Pretty dumb how the cast is required, but
 * that's legacy libc for ya.  These new macros do not support EOF like
 * the standard macros do.  Tough.
 * @{
 */
/** @see isalnum */
#define apr_isalnum(c) (isalnum(((unsigned char)(c))))
/** @see isalpha */
#define apr_isalpha(c) (isalpha(((unsigned char)(c))))
/** @see iscntrl */
#define apr_iscntrl(c) (iscntrl(((unsigned char)(c))))
/** @see isdigit */
#define apr_isdigit(c) (isdigit(((unsigned char)(c))))
/** @see isgraph */
#define apr_isgraph(c) (isgraph(((unsigned char)(c))))
/** @see islower*/
#define apr_islower(c) (islower(((unsigned char)(c))))
/** @see isascii */
#ifdef isascii
#define apr_isascii(c) (isascii(((unsigned char)(c))))
#else
#define apr_isascii(c) (((c) & ~0x7f)==0)
#endif
/** @see isprint */
#define apr_isprint(c) (isprint(((unsigned char)(c))))
/** @see ispunct */
#define apr_ispunct(c) (ispunct(((unsigned char)(c))))
/** @see isspace */
#define apr_isspace(c) (isspace(((unsigned char)(c))))
/** @see isupper */
#define apr_isupper(c) (isupper(((unsigned char)(c))))
/** @see isxdigit */
#define apr_isxdigit(c) (isxdigit(((unsigned char)(c))))
/** @see tolower */
#define apr_tolower(c) (tolower(((unsigned char)(c))))
/** @see toupper */
#define apr_toupper(c) (toupper(((unsigned char)(c))))

/** @} */

#ifdef __cplusplus
}
#endif

#endif	/* ! APR_LIB_H */
