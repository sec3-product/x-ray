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

#include "apr.h"
#include "apr_private.h"
#include "apr_errno.h"
#include "apr_arch_utf8.h"

/* Implement the design principal specified by RFC 2718 2.2.5 
 * Guidelines for new URL Schemes - within the APR.
 *
 * Since many architectures support unicode, and UCS2 is the most
 * efficient storage used by those archictures, these functions
 * exist to validate a UCS string.  It is up to the operating system
 * to determine the validitity of the string in the context of it's
 * native language support.  File systems that support filename 
 * characters of 0x80-0xff but have no support of Unicode will find 
 * this function useful only for validating the character sequences 
 * and rejecting poorly encoded strings, if RFC 2718 2.2.5 naming is
 * desired.
 *
 * from RFC 2279 UTF-8, a transformation format of ISO 10646
 *
 *     UCS-4 range (hex.)    UTF-8 octet sequence (binary)
 * 1:2 0000 0000-0000 007F   0xxxxxxx
 * 2:2 0000 0080-0000 07FF   110XXXXx 10xxxxxx
 * 3:2 0000 0800-0000 FFFF   1110XXXX 10Xxxxxx 10xxxxxx
 * 4:4 0001 0000-001F FFFF   11110zXX 10XXxxxx 10xxxxxx 10xxxxxx
 * inv 0020 0000-03FF FFFF   111110XX 10XXXxxx 10xxxxxx 10xxxxxx 10xxxxxx
 * inv 0400 0000-7FFF FFFF   1111110X 10XXXXxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
 *
 * One of the X values must be one for the encoding length to be legit.
 * Neither the z bit, nor the final two forms, are used for ucs-2
 *
 *   "Pairs of UCS-2 values between D800 and DFFF (surrogate pairs in 
 *   Unicode parlance), being actually UCS-4 characters transformed 
 *   through UTF-16, need special treatment: the UTF-16 transformation 
 *   must be undone, yielding a UCS-4 character that is then transformed 
 *   as above."
 *
 * from RFC2781 UTF-16: the compressed ISO 10646 encoding bitmask
 *
 *  U' = U - 0x10000
 *  U' = 000000000000yyyyyyyyyyxxxxxxxxxx
 *                  W1 = 110110yyyyyyyyyy
 *                  W2 = 110111xxxxxxxxxx
 *
 * apr_conv_utf8_to_ucs2 out bytes:sizeof(in) * 1 <= Req <= sizeof(in) * 2
 *
 * apr_conv_ucs2_to_utf8 out words:sizeof(in) / 2 <= Req <= sizeof(in) * 3 / 2
 */

APR_DECLARE(apr_status_t) apr_conv_utf8_to_ucs2(const char *in, 
                                                apr_size_t *inbytes,
                                                apr_wchar_t *out, 
                                                apr_size_t *outwords)
{
    apr_int64_t newch, mask;
    apr_size_t expect, eating;
    int ch;
    
    while (*inbytes && *outwords) 
    {
        ch = (unsigned char)(*in++);
        if (!(ch & 0200)) {
            /* US-ASCII-7 plain text
             */
            --*inbytes;
            --*outwords;
            *(out++) = ch;
        }
        else
        {
            if ((ch & 0300) != 0300) { 
                /* Multibyte Continuation is out of place
                 */
                return APR_EINVAL;
            }
            else
            {
                /* Multibyte Sequence Lead Character
                 *
                 * Compute the expected bytes while adjusting
                 * or lead byte and leading zeros mask.
                 */
                mask = 0340;
                expect = 1;
                while ((ch & mask) == mask) {
                    mask |= mask >> 1;
                    if (++expect > 3) /* (truly 5 for ucs-4) */
                        return APR_EINVAL;
                }
                newch = ch & ~mask;
                eating = expect + 1;
                if (*inbytes <= expect)
                    return APR_INCOMPLETE;
                /* Reject values of excessive leading 0 bits
                 * utf-8 _demands_ the shortest possible byte length
                 */
                if (expect == 1) {
                    if (!(newch & 0036))
                        return APR_EINVAL;
                }
                else {
                    /* Reject values of excessive leading 0 bits
                     */
                    if (!newch && !((unsigned char)*in & 0077 & (mask << 1)))
                        return APR_EINVAL;
                    if (expect == 2) {
                        /* Reject values D800-DFFF when not utf16 encoded
                         * (may not be an appropriate restriction for ucs-4)
                         */
                        if (newch == 0015 && ((unsigned char)*in & 0040))
                            return APR_EINVAL;
                    }
                    else if (expect == 3) {
                        /* Short circuit values > 110000
                         */
                        if (newch > 4)
                            return APR_EINVAL;
                        if (newch == 4 && ((unsigned char)*in & 0060))
                            return APR_EINVAL;
                    }
                }
                /* Where the boolean (expect > 2) is true, we will need
                 * an extra word for the output.
                 */
                if (*outwords < (apr_size_t)(expect > 2) + 1) 
                    break; /* buffer full */
                while (expect--)
                {
                    /* Multibyte Continuation must be legal */
                    if (((ch = (unsigned char)*(in++)) & 0300) != 0200)
                        return APR_EINVAL;
                    newch <<= 6;
                    newch |= (ch & 0077);
                }
                *inbytes -= eating;
                /* newch is now a true ucs-4 character
                 *
                 * now we need to fold to ucs-2
                 */
                if (newch < 0x10000) 
                {
                    --*outwords;
                    *(out++) = (apr_wchar_t) newch;
                }
                else 
                {
                    *outwords -= 2;
                    newch -= 0x10000;
                    *(out++) = (apr_wchar_t) (0xD800 | (newch >> 10));
                    *(out++) = (apr_wchar_t) (0xDC00 | (newch & 0x03FF));                    
                }
            }
        }
    }
    /* Buffer full 'errors' aren't errors, the client must inspect both
     * the inbytes and outwords values
     */
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_conv_ucs2_to_utf8(const apr_wchar_t *in, 
                                                apr_size_t *inwords,
                                                char *out, 
                                                apr_size_t *outbytes)
{
    apr_int64_t newch, require;
    apr_size_t need;
    char *invout;
    int ch;
    
    while (*inwords && *outbytes) 
    {
        ch = (unsigned short)(*in++);
        if (ch < 0x80)
        {
            --*inwords;
            --*outbytes;
            *(out++) = (unsigned char) ch;
        }
        else 
        {
            if ((ch & 0xFC00) == 0xDC00) {
                /* Invalid Leading ucs-2 Multiword Continuation Character
                 */
                return APR_EINVAL;
            }
            if ((ch & 0xFC00) == 0xD800) {
                /* Leading ucs-2 Multiword Character
                 */
                if (*inwords < 2) {
                    /* Missing ucs-2 Multiword Continuation Character
                     */
                    return APR_INCOMPLETE;
                }
                if (((unsigned short)(*in) & 0xFC00) != 0xDC00) {
                    /* Invalid ucs-2 Multiword Continuation Character
                     */
                    return APR_EINVAL;
                }
                newch = (ch & 0x03FF) << 10 | ((unsigned short)(*in++) & 0x03FF);
                newch += 0x10000;
            }
            else {
                /* ucs-2 Single Word Character
                 */
                newch = ch;
            }
            /* Determine the absolute minimum utf-8 bytes required
             */
            require = newch >> 11;
            need = 1;
            while (require)
                require >>= 5, ++need;
            if (need >= *outbytes)
                break; /* Insufficient buffer */
            *inwords -= (need > 2) + 1;
            *outbytes -= need + 1;
            /* Compute the utf-8 characters in last to first order,
             * calculating the lead character length bits along the way.
             */
            ch = 0200;
            out += need + 1;
            invout = out;
            while (need--) {
                ch |= ch >> 1;
                *(--invout) = (unsigned char)(0200 | (newch & 0077));
                newch >>= 6;
            }
            /* Compute the lead utf-8 character and move the dest offset
             */
            *(--invout) = (unsigned char)(ch | newch);
        }
    }
    /* Buffer full 'errors' aren't errors, the client must inspect both
     * the inwords and outbytes values
     */
    return APR_SUCCESS;    
}
