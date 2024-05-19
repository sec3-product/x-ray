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

#ifndef INHERIT_H
#define INHERIT_H

#include "apr_inherit.h"

#define APR_INHERIT (1 << 24)    /* Must not conflict with other bits */

#define APR_IMPLEMENT_INHERIT_SET(name, flag, pool, cleanup)        \
APR_DECLARE(apr_status_t) apr_##name##_inherit_set(apr_##name##_t *the##name) \
{                                                                   \
    IF_WIN_OS_IS_UNICODE                                            \
    {                                                               \
        if (!SetHandleInformation(the##name->filehand,              \
                                  HANDLE_FLAG_INHERIT,              \
                                  HANDLE_FLAG_INHERIT))             \
            return apr_get_os_error();                              \
    }                                                               \
    ELSE_WIN_OS_IS_ANSI                                             \
    {                                                               \
        HANDLE temp, hproc = GetCurrentProcess();                   \
        if (!DuplicateHandle(hproc, the##name->filehand,            \
                             hproc, &temp, 0, TRUE,                 \
                             DUPLICATE_SAME_ACCESS))                \
            return apr_get_os_error();                              \
        CloseHandle(the##name->filehand);                           \
        the##name->filehand = temp;                                 \
    }                                                               \
    return APR_SUCCESS;                                             \
}                                                                   \
/* Deprecated */                                                    \
APR_DECLARE(void) apr_##name##_set_inherit(apr_##name##_t *the##name) \
{                                                                   \
    apr_##name##_inherit_set(the##name);                            \
}

#define APR_IMPLEMENT_INHERIT_UNSET(name, flag, pool, cleanup)      \
APR_DECLARE(apr_status_t) apr_##name##_inherit_unset(apr_##name##_t *the##name)\
{                                                                   \
    IF_WIN_OS_IS_UNICODE                                            \
    {                                                               \
        if (!SetHandleInformation(the##name->filehand,              \
                                  HANDLE_FLAG_INHERIT, 0))          \
            return apr_get_os_error();                              \
    }                                                               \
    ELSE_WIN_OS_IS_ANSI                                             \
    {                                                               \
        HANDLE temp, hproc = GetCurrentProcess();                   \
        if (!DuplicateHandle(hproc, the##name->filehand,            \
                             hproc, &temp, 0, FALSE,                \
                             DUPLICATE_SAME_ACCESS))                \
            return apr_get_os_error();                              \
        CloseHandle(the##name->filehand);                           \
        the##name->filehand = temp;                                 \
    }                                                               \
    return APR_SUCCESS;                                             \
}                                                                   \
/* Deprecated */                                                    \
APR_DECLARE(void) apr_##name##_unset_inherit(apr_##name##_t *the##name) \
{                                                                   \
    apr_##name##_inherit_unset(the##name);                          \
}

#endif	/* ! INHERIT_H */
