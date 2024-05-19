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

#ifndef APR_XLATE_H
#define APR_XLATE_H

#include "apu.h"
#include "apr_pools.h"
#include "apr_errno.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @file apr_xlate.h
 * @brief APR I18N translation library
 */

/**
 * @defgroup APR_XLATE I18N translation library
 * @ingroup APR
 * @{
 */
/** Opaque translation buffer */
typedef struct apr_xlate_t            apr_xlate_t;

/**
 * Set up for converting text from one charset to another.
 * @param convset The handle to be filled in by this function
 * @param topage The name of the target charset
 * @param frompage The name of the source charset
 * @param pool The pool to use
 * @remark
 *  Specify APR_DEFAULT_CHARSET for one of the charset
 *  names to indicate the charset of the source code at
 *  compile time.  This is useful if there are literal
 *  strings in the source code which must be translated
 *  according to the charset of the source code.
 *  APR_DEFAULT_CHARSET is not useful if the source code
 *  of the caller was not encoded in the same charset as
 *  APR at compile time.
 *
 * @remark
 *  Specify APR_LOCALE_CHARSET for one of the charset
 *  names to indicate the charset of the current locale.
 *
 * @remark
 *  Return APR_EINVAL if unable to procure a convset, or APR_ENOTIMPL
 *  if charset transcoding is not available in this instance of
 *  apr-util at all (i.e., APR_HAS_XLATE is undefined).
 */
APU_DECLARE(apr_status_t) apr_xlate_open(apr_xlate_t **convset, 
                                         const char *topage, 
                                         const char *frompage, 
                                         apr_pool_t *pool);

/** 
 * This is to indicate the charset of the sourcecode at compile time
 * names to indicate the charset of the source code at
 * compile time.  This is useful if there are literal
 * strings in the source code which must be translated
 * according to the charset of the source code.
 */
#define APR_DEFAULT_CHARSET (const char *)0
/**
 * To indicate charset names of the current locale 
 */
#define APR_LOCALE_CHARSET (const char *)1

/**
 * Find out whether or not the specified conversion is single-byte-only.
 * @param convset The handle allocated by apr_xlate_open, specifying the 
 *                parameters of conversion
 * @param onoff Output: whether or not the conversion is single-byte-only
 * @remark
 *  Return APR_ENOTIMPL if charset transcoding is not available
 *  in this instance of apr-util (i.e., APR_HAS_XLATE is undefined).
 */
APU_DECLARE(apr_status_t) apr_xlate_sb_get(apr_xlate_t *convset, int *onoff);

/** @deprecated @see apr_xlate_sb_get */
APU_DECLARE(apr_status_t) apr_xlate_get_sb(apr_xlate_t *convset, int *onoff);

/**
 * Convert a buffer of text from one codepage to another.
 * @param convset The handle allocated by apr_xlate_open, specifying 
 *                the parameters of conversion
 * @param inbuf The address of the source buffer
 * @param inbytes_left Input: the amount of input data to be translated
 *                     Output: the amount of input data not yet translated    
 * @param outbuf The address of the destination buffer
 * @param outbytes_left Input: the size of the output buffer
 *                      Output: the amount of the output buffer not yet used
 * @remark
 *  Return APR_ENOTIMPL if charset transcoding is not available
 *  in this instance of apr-util (i.e., APR_HAS_XLATE is undefined).
 */
APU_DECLARE(apr_status_t) apr_xlate_conv_buffer(apr_xlate_t *convset, 
                                                const char *inbuf, 
                                                apr_size_t *inbytes_left, 
                                                char *outbuf,
                                                apr_size_t *outbytes_left);

/* @see apr_file_io.h the comment in apr_file_io.h about this hack */
#ifdef APR_NOT_DONE_YET
/**
 * The purpose of apr_xlate_conv_char is to translate one character
 * at a time.  This needs to be written carefully so that it works
 * with double-byte character sets. 
 * @param convset The handle allocated by apr_xlate_open, specifying the
 *                parameters of conversion
 * @param inchar The character to convert
 * @param outchar The converted character
 */
APU_DECLARE(apr_status_t) apr_xlate_conv_char(apr_xlate_t *convset, 
                                              char inchar, char outchar);
#endif

/**
 * Convert a single-byte character from one charset to another.
 * @param convset The handle allocated by apr_xlate_open, specifying the 
 *                parameters of conversion
 * @param inchar The single-byte character to convert.
 * @warning This only works when converting between single-byte character sets.
 *          -1 will be returned if the conversion can't be performed.
 */
APU_DECLARE(apr_int32_t) apr_xlate_conv_byte(apr_xlate_t *convset, 
                                             unsigned char inchar);

/**
 * Close a codepage translation handle.
 * @param convset The codepage translation handle to close
 * @remark
 *  Return APR_ENOTIMPL if charset transcoding is not available
 *  in this instance of apr-util (i.e., APR_HAS_XLATE is undefined).
 */
APU_DECLARE(apr_status_t) apr_xlate_close(apr_xlate_t *convset);

/** @} */
#ifdef __cplusplus
}
#endif

#endif  /* ! APR_XLATE_H */
