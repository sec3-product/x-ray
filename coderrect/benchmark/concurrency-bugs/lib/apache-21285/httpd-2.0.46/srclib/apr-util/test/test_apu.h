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

/* Some simple functions to make the test apps easier to write and
 * a bit more consistent...
 * this is a >copy< of apr_test.h
 */

/* Things to bear in mind when using these...
 *
 * If you include '\t' within the string passed in it won't be included
 * in the spacing, so use spaces instead :)
 * 
 */ 

#ifndef APU_TEST_INCLUDES
#define APU_TEST_INCLUDES

#include "apr_strings.h"
#include "apr_time.h"

#define TEST_EQ(str, func, value, good, bad) \
    printf("%-60s", str); \
    { \
    apr_status_t rv; \
    if ((rv = func) == value){ \
        char errmsg[200]; \
        printf("%s\n", bad); \
        fprintf(stderr, "Error was %d : %s\n", rv, \
                apr_strerror(rv, (char*)&errmsg, 200)); \
        exit(-1); \
    } \
    printf("%s\n", good); \
    }

#define TEST_NEQ(str, func, value, good, bad) \
    printf("%-60s", str); \
    { \
    apr_status_t rv; \
    if ((rv = func) != value){ \
        char errmsg[200]; \
        printf("%s\n", bad); \
        fprintf(stderr, "Error was %d : %s\n", rv, \
                apr_strerror(rv, (char*)&errmsg, 200)); \
        exit(-1); \
    } \
    printf("%s\n", good); \
    }

#define TEST_STATUS(str, func, testmacro, good, bad) \
    printf("%-60s", str); \
    { \
        apr_status_t rv = func; \
        if (!testmacro(rv)) { \
            char errmsg[200]; \
            printf("%s\n", bad); \
            fprintf(stderr, "Error was %d : %s\n", rv, \
                    apr_strerror(rv, (char*)&errmsg, 200)); \
            exit(-1); \
        } \
        printf("%s\n", good); \
    }

#define STD_TEST_NEQ(str, func) \
	TEST_NEQ(str, func, APR_SUCCESS, "OK", "Failed");

#define PRINT_ERROR(rv) \
    { \
        char errmsg[200]; \
        fprintf(stderr, "Error was %d : %s\n", rv, \
                apr_strerror(rv, (char*)&errmsg, 200)); \
        exit(-1); \
    }

#define MSG_AND_EXIT(msg) \
    printf("%s\n", msg); \
    exit (-1);

#define TIME_FUNCTION(time, function) \
    { \
        apr_time_t tt = apr_time_now(); \
        function; \
        time = apr_time_now() - tt; \
    }
    
    
#endif /* APU_TEST_INCLUDES */
