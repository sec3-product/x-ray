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

#ifndef APR_DATE_H
#define APR_DATE_H

/**
 * @file apr_date.h
 * @brief APR-UTIL date routines
 */

/**
 * @defgroup APR_Util_Date Date routines
 * @ingroup APR_Util
 * @{
 */

/*
 * apr_date.h: prototypes for date parsing utility routines
 */

#include "apu.h"
#include "apr_time.h"

#ifdef __cplusplus
extern "C" {
#endif

/** A bad date. */
#define APR_DATE_BAD ((apr_time_t)0)

/**
 * Compare a string to a mask
 * @param data The string to compare
 * @param mask Mask characters (arbitrary maximum is 256 characters):
 * <PRE>
 *   '\@' - uppercase letter
 *   '\$' - lowercase letter
 *   '\&' - hex digit
 *   '#' - digit
 *   '~' - digit or space
 *   '*' - swallow remaining characters
 * </PRE>
 * @remark The mask tests for an exact match for any other character
 * @return 1 if the string matches, 0 otherwise
 */
APU_DECLARE(int) apr_date_checkmask(const char *data, const char *mask);

/**
 * Parses an HTTP date in one of three standard forms:
 * <PRE>
 *     Sun, 06 Nov 1994 08:49:37 GMT  ; RFC 822, updated by RFC 1123
 *     Sunday, 06-Nov-94 08:49:37 GMT ; RFC 850, obsoleted by RFC 1036
 *     Sun Nov  6 08:49:37 1994       ; ANSI C's asctime() format
 * </PRE>
 * @param date The date in one of the three formats above
 * @return the apr_time_t number of microseconds since 1 Jan 1970 GMT, or
 *         0 if this would be out of range or if the date is invalid.
 */
APU_DECLARE(apr_time_t) apr_date_parse_http(const char *date);

/**
 * Parses a string resembling an RFC 822 date.  This is meant to be
 * leinent in its parsing of dates.  Hence, this will parse a wider 
 * range of dates than apr_date_parse_http.
 *
 * The prominent mailer (or poster, if mailer is unknown) that has
 * been seen in the wild is included for the unknown formats.
 * <PRE>
 *     Sun, 06 Nov 1994 08:49:37 GMT  ; RFC 822, updated by RFC 1123
 *     Sunday, 06-Nov-94 08:49:37 GMT ; RFC 850, obsoleted by RFC 1036
 *     Sun Nov  6 08:49:37 1994       ; ANSI C's asctime() format
 *     Sun, 6 Nov 1994 08:49:37 GMT   ; RFC 822, updated by RFC 1123
 *     Sun, 06 Nov 94 08:49:37 GMT    ; RFC 822
 *     Sun, 6 Nov 94 08:49:37 GMT     ; RFC 822
 *     Sun, 06 Nov 94 08:49 GMT       ; Unknown [drtr\@ast.cam.ac.uk] 
 *     Sun, 6 Nov 94 08:49 GMT        ; Unknown [drtr\@ast.cam.ac.uk]
 *     Sun, 06 Nov 94 8:49:37 GMT     ; Unknown [Elm 70.85]
 *     Sun, 6 Nov 94 8:49:37 GMT      ; Unknown [Elm 70.85] 
 * </PRE>
 *
 * @param date The date in one of the formats above
 * @return the apr_time_t number of microseconds since 1 Jan 1970 GMT, or
 *         0 if this would be out of range or if the date is invalid.
 */
APU_DECLARE(apr_time_t) apr_date_parse_rfc(const char *date);

/** @} */
#ifdef __cplusplus
}
#endif

#endif	/* !APR_DATE_H */
