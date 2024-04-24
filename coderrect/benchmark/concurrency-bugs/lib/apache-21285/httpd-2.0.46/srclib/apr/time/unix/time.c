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

#include "apr_portable.h"
#include "apr_time.h"
#include "apr_lib.h"
#include "apr_private.h"
#include "apr_strings.h"

/* private APR headers */
#include "apr_arch_internal_time.h"

/* System Headers required for time library */
#if APR_HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#if APR_HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_TIME_H
#include <time.h>
#endif
/* End System Headers */

#if !defined(HAVE_GMTOFF) && !defined(HAVE___OFFSET)
static apr_int32_t server_gmt_offset;
#endif /* if !defined(HAVE_GMTOFF) && !defined(HAVE___OFFSET) */

static apr_int32_t get_offset(struct tm *tm)
{
#ifdef HAVE_GMTOFF
    return tm->tm_gmtoff;
#elif defined(HAVE___OFFSET)
    return tm->__tm_gmtoff;
#else
#ifdef NETWARE
    /* Need to adjust the global variable each time otherwise
        the web server would have to be restarted when daylight
        savings changes.
    */
    if (daylightOnOff) {
        return server_gmt_offset + daylightOffset;
    }
#else
    if(tm->tm_isdst)
        return server_gmt_offset + 3600;
#endif
    return server_gmt_offset;
#endif
}

APR_DECLARE(apr_status_t) apr_time_ansi_put(apr_time_t *result,
                                            time_t input)
{
    *result = (apr_time_t)input * APR_USEC_PER_SEC;
    return APR_SUCCESS;
}

/* NB NB NB NB This returns GMT!!!!!!!!!! */
APR_DECLARE(apr_time_t) apr_time_now(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * APR_USEC_PER_SEC + tv.tv_usec;
}

static void explode_time(apr_time_exp_t *xt, apr_time_t t,
                         apr_int32_t offset, int use_localtime)
{
    struct tm tm;
    time_t tt = (t / APR_USEC_PER_SEC) + offset;
    xt->tm_usec = t % APR_USEC_PER_SEC;

#if APR_HAS_THREADS && defined (_POSIX_THREAD_SAFE_FUNCTIONS)
    if (use_localtime)
        localtime_r(&tt, &tm);
    else
        gmtime_r(&tt, &tm);
#else
    if (use_localtime)
        tm = *localtime(&tt);
    else
        tm = *gmtime(&tt);
#endif

    xt->tm_sec  = tm.tm_sec;
    xt->tm_min  = tm.tm_min;
    xt->tm_hour = tm.tm_hour;
    xt->tm_mday = tm.tm_mday;
    xt->tm_mon  = tm.tm_mon;
    xt->tm_year = tm.tm_year;
    xt->tm_wday = tm.tm_wday;
    xt->tm_yday = tm.tm_yday;
    xt->tm_isdst = tm.tm_isdst;
    xt->tm_gmtoff = get_offset(&tm);
}

APR_DECLARE(apr_status_t) apr_time_exp_tz(apr_time_exp_t *result,
                                          apr_time_t input, apr_int32_t offs)
{
    explode_time(result, input, offs, 0);
    result->tm_gmtoff = offs;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_time_exp_gmt(apr_time_exp_t *result,
                                           apr_time_t input)
{
    return apr_time_exp_tz(result, input, 0);
}

APR_DECLARE(apr_status_t) apr_time_exp_lt(apr_time_exp_t *result,
                                                apr_time_t input)
{
#if defined(__EMX__)
    /* EMX gcc (OS/2) has a timezone global we can use */
    return apr_time_exp_tz(result, input, -timezone);
#else
    explode_time(result, input, 0, 1);
    return APR_SUCCESS;
#endif /* __EMX__ */
}

APR_DECLARE(apr_status_t) apr_time_exp_get(apr_time_t *t, apr_time_exp_t *xt)
{
    int year;
    time_t days;
    static const int dayoffset[12] =
    {306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275};

    year = xt->tm_year;
    if (year < 70 || ((sizeof(time_t) <= 4) && (year >= 138))) {
        return APR_EBADDATE;
    }

    /* shift new year to 1st March in order to make leap year calc easy */

    if (xt->tm_mon < 2)
        year--;

    /* Find number of days since 1st March 1900 (in the Gregorian calendar). */

    days = year * 365 + year / 4 - year / 100 + (year / 100 + 3) / 4;
    days += dayoffset[xt->tm_mon] + xt->tm_mday - 1;
    days -= 25508;              /* 1 jan 1970 is 25508 days since 1 mar 1900 */
    days = ((days * 24 + xt->tm_hour) * 60 + xt->tm_min) * 60 + xt->tm_sec;

    if (days < 0) {
        return APR_EBADDATE;
    }
    *t = days * APR_USEC_PER_SEC + xt->tm_usec;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_time_exp_gmt_get(apr_time_t *t, 
                                               apr_time_exp_t *xt)
{
    apr_status_t status = apr_time_exp_get(t, xt);
    if (status == APR_SUCCESS)
        *t -= (apr_time_t) xt->tm_gmtoff * APR_USEC_PER_SEC;
    return status;
}

APR_DECLARE(apr_status_t) apr_os_imp_time_get(apr_os_imp_time_t **ostime,
                                              apr_time_t *aprtime)
{
    (*ostime)->tv_usec = *aprtime % APR_USEC_PER_SEC;
    (*ostime)->tv_sec = *aprtime / APR_USEC_PER_SEC;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_os_exp_time_get(apr_os_exp_time_t **ostime,
                                              apr_time_exp_t *aprtime)
{
    (*ostime)->tm_sec  = aprtime->tm_sec;
    (*ostime)->tm_min  = aprtime->tm_min;
    (*ostime)->tm_hour = aprtime->tm_hour;
    (*ostime)->tm_mday = aprtime->tm_mday;
    (*ostime)->tm_mon  = aprtime->tm_mon;
    (*ostime)->tm_year = aprtime->tm_year;
    (*ostime)->tm_wday = aprtime->tm_wday;
    (*ostime)->tm_yday = aprtime->tm_yday;
    (*ostime)->tm_isdst = aprtime->tm_isdst;

#if HAVE_GMTOFF
    (*ostime)->tm_gmtoff = aprtime->tm_gmtoff;
#elif defined(HAVE__OFFSET)
    (*ostime)->__tm_gmtoff = aprtime->tm_gmtoff;
#endif

    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_os_imp_time_put(apr_time_t *aprtime,
                                              apr_os_imp_time_t **ostime,
                                              apr_pool_t *cont)
{
    *aprtime = (*ostime)->tv_sec * APR_USEC_PER_SEC + (*ostime)->tv_usec;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_os_exp_time_put(apr_time_exp_t *aprtime,
                                              apr_os_exp_time_t **ostime,
                                              apr_pool_t *cont)
{
    aprtime->tm_sec = (*ostime)->tm_sec;
    aprtime->tm_min = (*ostime)->tm_min;
    aprtime->tm_hour = (*ostime)->tm_hour;
    aprtime->tm_mday = (*ostime)->tm_mday;
    aprtime->tm_mon = (*ostime)->tm_mon;
    aprtime->tm_year = (*ostime)->tm_year;
    aprtime->tm_wday = (*ostime)->tm_wday;
    aprtime->tm_yday = (*ostime)->tm_yday;
    aprtime->tm_isdst = (*ostime)->tm_isdst;

#if HAVE_GMTOFF
    aprtime->tm_gmtoff = (*ostime)->tm_gmtoff;
#elif defined(HAVE__OFFSET)
    aprtime->tm_gmtoff = (*ostime)->__tm_gmtoff;
#endif

    return APR_SUCCESS;
}

APR_DECLARE(void) apr_sleep(apr_interval_time_t t)
{
#ifdef OS2
    DosSleep(t/1000);
#elif defined(BEOS)
    snooze(t);
#elif defined(NETWARE)
    delay(t/1000);
#else
    struct timeval tv;
    tv.tv_usec = t % APR_USEC_PER_SEC;
    tv.tv_sec = t / APR_USEC_PER_SEC;
    select(0, NULL, NULL, NULL, &tv);
#endif
}

#ifdef OS2
APR_DECLARE(apr_status_t) apr_os2_time_to_apr_time(apr_time_t *result,
                                                   FDATE os2date,
                                                   FTIME os2time)
{
  struct tm tmpdate;

  memset(&tmpdate, 0, sizeof(tmpdate));
  tmpdate.tm_hour  = os2time.hours;
  tmpdate.tm_min   = os2time.minutes;
  tmpdate.tm_sec   = os2time.twosecs * 2;

  tmpdate.tm_mday  = os2date.day;
  tmpdate.tm_mon   = os2date.month - 1;
  tmpdate.tm_year  = os2date.year + 80;
  tmpdate.tm_isdst = -1;

  *result = mktime(&tmpdate) * APR_USEC_PER_SEC;
  return APR_SUCCESS;
}
#endif

#ifdef NETWARE
APR_DECLARE(void) apr_netware_setup_time(void)
{
    tzset();
    server_gmt_offset = -TZONE;
}
#else
APR_DECLARE(void) apr_unix_setup_time(void)
{
#if !defined(HAVE_GMTOFF) && !defined(HAVE___OFFSET)
    /* Precompute the offset from GMT on systems where it's not
       in struct tm.

       Note: This offset is normalized to be independent of daylight
       savings time; if the calculation happens to be done in a
       time/place where a daylight savings adjustment is in effect,
       the returned offset has the same value that it would have
       in the same location if daylight savings were not in effect.
       The reason for this is that the returned offset can be
       applied to a past or future timestamp in explode_time(),
       so the DST adjustment obtained from the current time won't
       necessarily be applicable.

       mktime() is the inverse of localtime(); so, presumably,
       passing in a struct tm made by gmtime() let's us calculate
       the true GMT offset. However, there's a catch: if daylight
       savings is in effect, gmtime()will set the tm_isdst field
       and confuse mktime() into returning a time that's offset
       by one hour. In that case, we must adjust the calculated GMT
       offset.

     */

    struct timeval now;
    time_t t1, t2;
    struct tm t;

    gettimeofday(&now, NULL);
    t1 = now.tv_sec;
    t2 = 0;

#if APR_HAS_THREADS && defined(_POSIX_THREAD_SAFE_FUNCTIONS)
    gmtime_r(&t1, &t);
#else
    t = *gmtime(&t1);
#endif
    t.tm_isdst = 0; /* we know this GMT time isn't daylight-savings */
    t2 = mktime(&t);
    server_gmt_offset = (apr_int32_t) difftime(t1, t2);
#endif
}

#endif

/* A noop on all known Unix implementations */
APR_DECLARE(void) apr_time_clock_hires(apr_pool_t *p)
{
    return;
}

/* Deprecated */
APR_DECLARE(apr_status_t) apr_explode_time(apr_time_exp_t *result,
                                          apr_time_t input,
                                          apr_int32_t offs)
{
    return apr_time_exp_tz(result, input, offs);
}

/* Deprecated */
APR_DECLARE(apr_status_t) apr_explode_localtime(apr_time_exp_t *result, 
                                                apr_time_t input)
{
    return apr_time_exp_lt(result, input);
}

/* Deprecated */
APR_DECLARE(apr_status_t) apr_implode_gmt(apr_time_t *t, apr_time_exp_t *xt)
{
    return apr_time_exp_gmt_get(t, xt);
}

