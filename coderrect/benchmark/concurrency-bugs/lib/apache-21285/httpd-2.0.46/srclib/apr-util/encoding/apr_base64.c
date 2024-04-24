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

/* base64 encoder/decoder. Originally part of main/util.c
 * but moved here so that support/ab and apr_sha1.c could
 * use it. This meant removing the apr_palloc()s and adding
 * ugly 'len' functions, which is quite a nasty cost.
 */

#include "apr_base64.h"
#if APR_CHARSET_EBCDIC
#include "apr_xlate.h"
#endif				/* APR_CHARSET_EBCDIC */

/* aaaack but it's fast and const should make it shared text page. */
static const unsigned char pr2six[256] =
{
#if !APR_CHARSET_EBCDIC
    /* ASCII table */
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
    64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
    64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
#else /*APR_CHARSET_EBCDIC*/
    /* EBCDIC table */
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 64, 64, 64, 64, 64, 64,
    64, 35, 36, 37, 38, 39, 40, 41, 42, 43, 64, 64, 64, 64, 64, 64,
    64, 64, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64,  0,  1,  2,  3,  4,  5,  6,  7,  8, 64, 64, 64, 64, 64, 64,
    64,  9, 10, 11, 12, 13, 14, 15, 16, 17, 64, 64, 64, 64, 64, 64,
    64, 64, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64, 64,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64
#endif /*APR_CHARSET_EBCDIC*/
};

#if APR_CHARSET_EBCDIC
static apr_xlate_t *xlate_to_ebcdic;
static unsigned char os_toascii[256];

APU_DECLARE(apr_status_t) apr_base64init_ebcdic(apr_xlate_t *to_ascii,
                                             apr_xlate_t *to_ebcdic)
{
    int i;
    apr_size_t inbytes_left, outbytes_left;
    apr_status_t rv;
    int onoff;
    
    /* Only single-byte conversion is supported.
     */
    rv = apr_xlate_get_sb(to_ascii, &onoff);
    if (rv) {
        return rv;
    }
    if (!onoff) { /* If conversion is not single-byte-only */
        return APR_EINVAL;
    }
    rv = apr_xlate_get_sb(to_ebcdic, &onoff);
    if (rv) {
        return rv;
    }
    if (!onoff) { /* If conversion is not single-byte-only */
        return APR_EINVAL;
    }
    xlate_to_ebcdic = to_ebcdic;
    for (i = 0; i < sizeof(os_toascii); i++) {
        os_toascii[i] = i;
    }
    inbytes_left = outbytes_left = sizeof(os_toascii);
    apr_xlate_conv_buffer(to_ascii, os_toascii, &inbytes_left,
                          os_toascii, &outbytes_left);

    return APR_SUCCESS;
}
#endif /*APR_CHARSET_EBCDIC*/

APU_DECLARE(int) apr_base64_decode_len(const char *bufcoded)
{
    int nbytesdecoded;
    register const unsigned char *bufin;
    register int nprbytes;

    bufin = (const unsigned char *) bufcoded;
    while (pr2six[*(bufin++)] <= 63);

    nprbytes = (bufin - (const unsigned char *) bufcoded) - 1;
    nbytesdecoded = ((nprbytes + 3) / 4) * 3;

    return nbytesdecoded + 1;
}

APU_DECLARE(int) apr_base64_decode(char *bufplain, const char *bufcoded)
{
#if APR_CHARSET_EBCDIC
    apr_size_t inbytes_left, outbytes_left;
#endif				/* APR_CHARSET_EBCDIC */
    int len;
    
    len = apr_base64_decode_binary((unsigned char *) bufplain, bufcoded);
#if APR_CHARSET_EBCDIC
    inbytes_left = outbytes_left = len;
    apr_xlate_conv_buffer(xlate_to_ebcdic, bufplain, &inbytes_left,
                          bufplain, &outbytes_left);
#endif				/* APR_CHARSET_EBCDIC */
    bufplain[len] = '\0';
    return len;
}

/* This is the same as apr_base64_decode() except on EBCDIC machines, where
 * the conversion of the output to ebcdic is left out.
 */
APU_DECLARE(int) apr_base64_decode_binary(unsigned char *bufplain,
				   const char *bufcoded)
{
    int nbytesdecoded;
    register const unsigned char *bufin;
    register unsigned char *bufout;
    register int nprbytes;

    bufin = (const unsigned char *) bufcoded;
    while (pr2six[*(bufin++)] <= 63);
    nprbytes = (bufin - (const unsigned char *) bufcoded) - 1;
    nbytesdecoded = ((nprbytes + 3) / 4) * 3;

    bufout = (unsigned char *) bufplain;
    bufin = (const unsigned char *) bufcoded;

    while (nprbytes > 4) {
	*(bufout++) =
	    (unsigned char) (pr2six[*bufin] << 2 | pr2six[bufin[1]] >> 4);
	*(bufout++) =
	    (unsigned char) (pr2six[bufin[1]] << 4 | pr2six[bufin[2]] >> 2);
	*(bufout++) =
	    (unsigned char) (pr2six[bufin[2]] << 6 | pr2six[bufin[3]]);
	bufin += 4;
	nprbytes -= 4;
    }

    /* Note: (nprbytes == 1) would be an error, so just ingore that case */
    if (nprbytes > 1) {
	*(bufout++) =
	    (unsigned char) (pr2six[*bufin] << 2 | pr2six[bufin[1]] >> 4);
    }
    if (nprbytes > 2) {
	*(bufout++) =
	    (unsigned char) (pr2six[bufin[1]] << 4 | pr2six[bufin[2]] >> 2);
    }
    if (nprbytes > 3) {
	*(bufout++) =
	    (unsigned char) (pr2six[bufin[2]] << 6 | pr2six[bufin[3]]);
    }

    nbytesdecoded -= (4 - nprbytes) & 3;
    return nbytesdecoded;
}

static const char basis_64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

APU_DECLARE(int) apr_base64_encode_len(int len)
{
    return ((len + 2) / 3 * 4) + 1;
}

APU_DECLARE(int) apr_base64_encode(char *encoded, const char *string, int len)
{
#if !APR_CHARSET_EBCDIC
    return apr_base64_encode_binary(encoded, (const unsigned char *) string, len);
#else /* APR_CHARSET_EBCDIC */
    int i;
    char *p;

    p = encoded;
    for (i = 0; i < len - 2; i += 3) {
	*p++ = basis_64[(os_toascii[string[i]] >> 2) & 0x3F];
	*p++ = basis_64[((os_toascii[string[i]] & 0x3) << 4) |
	                ((int) (os_toascii[string[i + 1]] & 0xF0) >> 4)];
	*p++ = basis_64[((os_toascii[string[i + 1]] & 0xF) << 2) |
	                ((int) (os_toascii[string[i + 2]] & 0xC0) >> 6)];
	*p++ = basis_64[os_toascii[string[i + 2]] & 0x3F];
    }
    if (i < len) {
	*p++ = basis_64[(os_toascii[string[i]] >> 2) & 0x3F];
	if (i == (len - 1)) {
	    *p++ = basis_64[((os_toascii[string[i]] & 0x3) << 4)];
	    *p++ = '=';
	}
	else {
	    *p++ = basis_64[((os_toascii[string[i]] & 0x3) << 4) |
	                    ((int) (os_toascii[string[i + 1]] & 0xF0) >> 4)];
	    *p++ = basis_64[((os_toascii[string[i + 1]] & 0xF) << 2)];
	}
	*p++ = '=';
    }

    *p++ = '\0';
    return p - encoded;
#endif				/* APR_CHARSET_EBCDIC */
}

/* This is the same as apr_base64_encode() except on EBCDIC machines, where
 * the conversion of the input to ascii is left out.
 */
APU_DECLARE(int) apr_base64_encode_binary(char *encoded,
                                      const unsigned char *string, int len)
{
    int i;
    char *p;

    p = encoded;
    for (i = 0; i < len - 2; i += 3) {
	*p++ = basis_64[(string[i] >> 2) & 0x3F];
	*p++ = basis_64[((string[i] & 0x3) << 4) |
	                ((int) (string[i + 1] & 0xF0) >> 4)];
	*p++ = basis_64[((string[i + 1] & 0xF) << 2) |
	                ((int) (string[i + 2] & 0xC0) >> 6)];
	*p++ = basis_64[string[i + 2] & 0x3F];
    }
    if (i < len) {
	*p++ = basis_64[(string[i] >> 2) & 0x3F];
	if (i == (len - 1)) {
	    *p++ = basis_64[((string[i] & 0x3) << 4)];
	    *p++ = '=';
	}
	else {
	    *p++ = basis_64[((string[i] & 0x3) << 4) |
	                    ((int) (string[i + 1] & 0xF0) >> 4)];
	    *p++ = basis_64[((string[i + 1] & 0xF) << 2)];
	}
	*p++ = '=';
    }

    *p++ = '\0';
    return p - encoded;
}
