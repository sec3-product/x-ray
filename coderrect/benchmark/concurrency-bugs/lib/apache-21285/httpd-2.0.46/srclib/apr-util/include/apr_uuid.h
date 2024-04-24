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

/**
 * @file apr_uuid.h
 * @brief APR UUID library
 */
#ifndef APR_UUID_H
#define APR_UUID_H

#include "apu.h"
#include "apr_errno.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @defgroup APR_UUID UUID Handling
 * @ingroup APR
 * @{
 */

/**
 * we represent a UUID as a block of 16 bytes.
 */

typedef struct {
    unsigned char data[16]; /**< the actual UUID */
} apr_uuid_t;

/** UUIDs are formatted as: 00112233-4455-6677-8899-AABBCCDDEEFF */
#define APR_UUID_FORMATTED_LENGTH 36


/**
 * Generate and return a (new) UUID
 * @param uuid The resulting UUID
 */ 
APU_DECLARE(void) apr_uuid_get(apr_uuid_t *uuid);

/**
 * Format a UUID into a string, following the standard format
 * @param buffer The buffer to place the formatted UUID string into. It must
 *               be at least APR_UUID_FORMATTED_LENGTH + 1 bytes long to hold
 *               the formatted UUID and a null terminator
 * @param uuid The UUID to format
 */ 
APU_DECLARE(void) apr_uuid_format(char *buffer, const apr_uuid_t *uuid);

/**
 * Parse a standard-format string into a UUID
 * @param uuid The resulting UUID
 * @param uuid_str The formatted UUID
 */ 
APU_DECLARE(apr_status_t) apr_uuid_parse(apr_uuid_t *uuid, const char *uuid_str);

/** @} */
#ifdef __cplusplus
}
#endif

#endif /* APR_UUID_H */
