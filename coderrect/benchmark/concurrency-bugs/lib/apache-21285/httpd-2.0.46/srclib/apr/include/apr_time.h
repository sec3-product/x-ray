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

#ifndef APR_TIME_H
#define APR_TIME_H

/**
 * @file apr_time.h
 * @brief APR Time Library
 */

#include "apr.h"
#include "apr_pools.h"
#include "apr_errno.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @defgroup apr_time Time Routines
 * @ingroup APR 
 * @{
 */

/** month names */
APR_DECLARE_DATA extern const char apr_month_snames[12][4];
/** day names */
APR_DECLARE_DATA extern const char apr_day_snames[7][4];


/** number of microseconds since 00:00:00 january 1, 1970 UTC */
typedef apr_int64_t apr_time_t;


/** mechanism to properly type apr_time_t literals */
#define APR_TIME_C(val) APR_INT64_C(val)

/** mechanism to properly print apr_time_t values */
#define APR_TIME_T_FMT APR_INT64_T_FMT

/** intervals for I/O timeouts, in microseconds */
typedef apr_int64_t apr_interval_time_t;
/** short interval for I/O timeouts, in microseconds */
typedef apr_int32_t apr_short_interval_time_t;

/** number of microseconds per second */
#define APR_USEC_PER_SEC APR_TIME_C(1000000)

/** @return apr_time_t as a second */
#define apr_time_sec(time) ((time) / APR_USEC_PER_SEC)

/** @return apr_time_t as a usec */
#define apr_time_usec(time) ((time) % APR_USEC_PER_SEC)

/** @return apr_time_t as a msec */
#define apr_time_msec(time) (((time) / 1000) % 1000)

/** @return apr_time_t as a msec */
#define apr_time_as_msec(time) ((time) / 1000)

/** @return a second as an apr_time_t */
#define apr_time_from_sec(sec) ((apr_time_t)(sec) * APR_USEC_PER_SEC)

/** @return a second and usec combination as an apr_time_t */
#define apr_time_make(sec, usec) ((apr_time_t)(sec) * APR_USEC_PER_SEC \
                                + (apr_time_t)(usec))

/**
 * @return the current time
 */
APR_DECLARE(apr_time_t) apr_time_now(void);

/** @see apr_time_exp_t */
typedef struct apr_time_exp_t apr_time_exp_t;

/**
 * a structure similar to ANSI struct tm with the following differences:
 *  - tm_usec isn't an ANSI field
 *  - tm_gmtoff isn't an ANSI field (it's a bsdism)
 */
struct apr_time_exp_t {
    /** microseconds past tm_sec */
    apr_int32_t tm_usec;
    /** (0-61) seconds past tm_min */
    apr_int32_t tm_sec;
    /** (0-59) minutes past tm_hour */
    apr_int32_t tm_min;
    /** (0-23) hours past midnight */
    apr_int32_t tm_hour;
    /** (1-31) day of the month */
    apr_int32_t tm_mday;
    /** (0-11) month of the year */
    apr_int32_t tm_mon;
    /** year since 1900 */
    apr_int32_t tm_year;
    /** (0-6) days since sunday */
    apr_int32_t tm_wday;
    /** (0-365) days since jan 1 */
    apr_int32_t tm_yday;
    /** daylight saving time */
    apr_int32_t tm_isdst;
    /** seconds east of UTC */
    apr_int32_t tm_gmtoff;
};

/**
 * convert an ansi time_t to an apr_time_t
 * @param result the resulting apr_time_t
 * @param input the time_t to convert
 */
APR_DECLARE(apr_status_t) apr_time_ansi_put(apr_time_t *result, 
                                                    time_t input);

/**
 * convert a time to its human readable components using an offset
 * from GMT
 * @param result the exploded time
 * @param input the time to explode
 * @param offs the number of seconds offset to apply
 */
APR_DECLARE(apr_status_t) apr_time_exp_tz(apr_time_exp_t *result,
                                          apr_time_t input,
                                          apr_int32_t offs);

/** @deprecated @see apr_time_exp_tz */
APR_DECLARE(apr_status_t) apr_explode_time(apr_time_exp_t *result,
                                           apr_time_t input,
                                           apr_int32_t offs);

/**
 * convert a time to its human readable components in GMT timezone
 * @param result the exploded time
 * @param input the time to explode
 */
APR_DECLARE(apr_status_t) apr_time_exp_gmt(apr_time_exp_t *result, 
                                           apr_time_t input);

/**
 * convert a time to its human readable components in local timezone
 * @param result the exploded time
 * @param input the time to explode
 */
APR_DECLARE(apr_status_t) apr_time_exp_lt(apr_time_exp_t *result, 
                                          apr_time_t input);

/** @deprecated @see apr_time_exp_lt */
APR_DECLARE(apr_status_t) apr_explode_localtime(apr_time_exp_t *result, 
                                                apr_time_t input);

/**
 * Convert time value from human readable format to a numeric apr_time_t 
 * e.g. elapsed usec since epoch
 * @param result the resulting imploded time
 * @param input the input exploded time
 */
APR_DECLARE(apr_status_t) apr_time_exp_get(apr_time_t *result, 
                                           apr_time_exp_t *input);

/**
 * Convert time value from human readable format to a numeric apr_time_t that
 * always represents GMT
 * @param result the resulting imploded time
 * @param input the input exploded time
 */
APR_DECLARE(apr_status_t) apr_time_exp_gmt_get(apr_time_t *result, 
                                               apr_time_exp_t *input);

/** @deprecated @see apr_time_exp_gmt_get */
APR_DECLARE(apr_status_t) apr_implode_gmt(apr_time_t *result, 
                                          apr_time_exp_t *input);

/**
 * Sleep for the specified number of micro-seconds.
 * @param t desired amount of time to sleep.
 * @warning May sleep for longer than the specified time. 
 */
APR_DECLARE(void) apr_sleep(apr_interval_time_t t);

/** length of a RFC822 Date */
#define APR_RFC822_DATE_LEN (30)
/**
 * apr_rfc822_date formats dates in the RFC822
 * format in an efficient manner.  It is a fixed length
 * format which requires the indicated amount of storage,
 * including the trailing null byte.
 * @param date_str String to write to.
 * @param t the time to convert 
 */
APR_DECLARE(apr_status_t) apr_rfc822_date(char *date_str, apr_time_t t);

/** length of a CTIME date */
#define APR_CTIME_LEN (25)
/**
 * apr_ctime formats dates in the ctime() format
 * in an efficient manner.  it is a fixed length format
 * and requires the indicated amount of storage including
 * the trailing null byte.
 * Unlike ANSI/ISO C ctime(), apr_ctime() does not include
 * a \n at the end of the string.
 * @param date_str String to write to.
 * @param t the time to convert 
 */
APR_DECLARE(apr_status_t) apr_ctime(char *date_str, apr_time_t t);

/**
 * formats the exploded time according to the format specified
 * @param s string to write to
 * @param retsize The length of the returned string
 * @param max The maximum length of the string
 * @param format The format for the time string
 * @param tm The time to convert
 */
APR_DECLARE(apr_status_t) apr_strftime(char *s, apr_size_t *retsize, 
                                       apr_size_t max, const char *format, 
                                       apr_time_exp_t *tm);

/**
 * Improve the clock resolution for the lifetime of the given pool.
 * Generally this is only desireable on benchmarking and other very
 * time-sensitive applications, and has no impact on most platforms.
 * @param p The pool to associate the finer clock resolution 
 */
APR_DECLARE(void) apr_time_clock_hires(apr_pool_t *p);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  /* ! APR_TIME_H */
