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

#include "apr_file_io.h"
#include "apr_file_info.h"
#include "apr_network_io.h"
#include "apr_errno.h"
#include "apr_general.h"
#include "apr_poll.h"
#include "apr_lib.h"
#include "test_apr.h"

#define DIRNAME "data"
#define FILENAME DIRNAME "/file_datafile.txt"
#define TESTSTR  "This is the file data file."

#define TESTREAD_BLKSIZE 1024
#define APR_BUFFERSIZE   4096 /* This should match APR's buffer size. */



static void test_open_noreadwrite(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *thefile = NULL;

    rv = apr_file_open(&thefile, FILENAME,
                       APR_CREATE | APR_EXCL, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertTrue(tc, rv != APR_SUCCESS);
    CuAssertIntEquals(tc, 1, APR_STATUS_IS_EACCES(rv));
    CuAssertPtrEquals(tc, NULL, thefile); 
}

static void test_open_excl(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *thefile = NULL;

    rv = apr_file_open(&thefile, FILENAME,
                       APR_CREATE | APR_EXCL | APR_WRITE, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertTrue(tc, rv != APR_SUCCESS);
    CuAssertIntEquals(tc, 1, APR_STATUS_IS_EEXIST(rv));
    CuAssertPtrEquals(tc, NULL, thefile); 
}

static void test_open_read(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *filetest = NULL;

    rv = apr_file_open(&filetest, FILENAME, 
                       APR_READ, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, rv, APR_SUCCESS);
    CuAssertPtrNotNull(tc, filetest);
    apr_file_close(filetest);
}

static void test_read(CuTest *tc)
{
    apr_status_t rv;
    apr_size_t nbytes = 256;
    char *str = apr_pcalloc(p, nbytes + 1);
    apr_file_t *filetest = NULL;
    
    rv = apr_file_open(&filetest, FILENAME, 
                       APR_READ, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);

    apr_assert_success(tc, "Opening test file " FILENAME, rv);
    rv = apr_file_read(filetest, str, &nbytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, strlen(TESTSTR), nbytes);
    CuAssertStrEquals(tc, TESTSTR, str);

    apr_file_close(filetest);
}

static void test_filename(CuTest *tc)
{
    const char *str;
    apr_status_t rv;
    apr_file_t *filetest = NULL;
    
    rv = apr_file_open(&filetest, FILENAME, 
                       APR_READ, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    apr_assert_success(tc, "Opening test file " FILENAME, rv);

    rv = apr_file_name_get(&str, filetest);
    CuAssertIntEquals(tc, rv, APR_SUCCESS);
    CuAssertStrEquals(tc, FILENAME, str);

    apr_file_close(filetest);
}
    
static void test_fileclose(CuTest *tc)
{
    char str;
    apr_status_t rv;
    apr_size_t one = 1;
    apr_file_t *filetest = NULL;
    
    rv = apr_file_open(&filetest, FILENAME, 
                       APR_READ, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    apr_assert_success(tc, "Opening test file " FILENAME, rv);

    rv = apr_file_close(filetest);
    CuAssertIntEquals(tc, rv, APR_SUCCESS);
    /* We just closed the file, so this should fail */
    rv = apr_file_read(filetest, &str, &one);
    CuAssertIntEquals(tc, 1, APR_STATUS_IS_EBADF(rv));
}

static void test_file_remove(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *filetest = NULL;

    rv = apr_file_remove(FILENAME, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_open(&filetest, FILENAME, APR_READ, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, 1, APR_STATUS_IS_ENOENT(rv));
}

static void test_open_write(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *filetest = NULL;

    filetest = NULL;
    rv = apr_file_open(&filetest, FILENAME, 
                       APR_WRITE, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, 1, APR_STATUS_IS_ENOENT(rv));
    CuAssertPtrEquals(tc, NULL, filetest);
}

static void test_open_writecreate(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *filetest = NULL;

    filetest = NULL;
    rv = apr_file_open(&filetest, FILENAME, 
                       APR_WRITE | APR_CREATE, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    apr_file_close(filetest);
}

static void test_write(CuTest *tc)
{
    apr_status_t rv;
    apr_size_t bytes = strlen(TESTSTR);
    apr_file_t *filetest = NULL;

    rv = apr_file_open(&filetest, FILENAME, 
                       APR_WRITE | APR_CREATE, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_write(filetest, TESTSTR, &bytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    apr_file_close(filetest);
}

static void test_open_readwrite(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *filetest = NULL;

    filetest = NULL;
    rv = apr_file_open(&filetest, FILENAME, 
                       APR_READ | APR_WRITE, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, rv, APR_SUCCESS);
    CuAssertPtrNotNull(tc, filetest);

    apr_file_close(filetest);
}

static void test_seek(CuTest *tc)
{
    apr_status_t rv;
    apr_off_t offset = 5;
    apr_size_t nbytes = 256;
    char *str = apr_pcalloc(p, nbytes + 1);
    apr_file_t *filetest = NULL;

    rv = apr_file_open(&filetest, FILENAME, 
                       APR_READ, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    apr_assert_success(tc, "Open test file " FILENAME, rv);

    rv = apr_file_read(filetest, str, &nbytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, strlen(TESTSTR), nbytes);
    CuAssertStrEquals(tc, TESTSTR, str);

    memset(str, 0, nbytes + 1);

    rv = apr_file_seek(filetest, SEEK_SET, &offset);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    
    rv = apr_file_read(filetest, str, &nbytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, strlen(TESTSTR) - 5, nbytes);
    CuAssertStrEquals(tc, TESTSTR + 5, str);

    apr_file_close(filetest);
}                

static void test_userdata_set(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *filetest = NULL;

    rv = apr_file_open(&filetest, FILENAME, 
                       APR_WRITE, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_data_set(filetest, "This is a test",
                           "test", apr_pool_cleanup_null);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    apr_file_close(filetest);
}

static void test_userdata_get(CuTest *tc)
{
    apr_status_t rv;
    char *teststr;
    apr_file_t *filetest = NULL;

    rv = apr_file_open(&filetest, FILENAME, 
                       APR_WRITE, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_data_set(filetest, "This is a test",
                           "test", apr_pool_cleanup_null);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_data_get((void **)&teststr, "test", filetest);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertStrEquals(tc, "This is a test", teststr);

    apr_file_close(filetest);
}

static void test_userdata_getnokey(CuTest *tc)
{
    apr_status_t rv;
    char *teststr;
    apr_file_t *filetest = NULL;

    rv = apr_file_open(&filetest, FILENAME, 
                       APR_WRITE, 
                       APR_UREAD | APR_UWRITE | APR_GREAD, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_data_get((void **)&teststr, "nokey", filetest);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertPtrEquals(tc, NULL, teststr);
    apr_file_close(filetest);
}

static void test_getc(CuTest *tc)
{
    apr_file_t *f = NULL;
    apr_status_t rv;
    char ch;

    rv = apr_file_open(&f, FILENAME, APR_READ, 0, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    apr_file_getc(&ch, f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, (int)TESTSTR[0], (int)ch);
    apr_file_close(f);
}

static void test_ungetc(CuTest *tc)
{
    apr_file_t *f = NULL;
    apr_status_t rv;
    char ch;

    rv = apr_file_open(&f, FILENAME, APR_READ, 0, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    apr_file_getc(&ch, f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, (int)TESTSTR[0], (int)ch);

    apr_file_ungetc('X', f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    apr_file_getc(&ch, f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, 'X', (int)ch);

    apr_file_close(f);
}

static void test_gets(CuTest *tc)
{
    apr_file_t *f = NULL;
    apr_status_t rv;
    char *str = apr_palloc(p, 256);

    rv = apr_file_open(&f, FILENAME, APR_READ, 0, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_gets(str, 256, f);
    /* Only one line in the test file, so APR will encounter EOF on the first
     * call to gets, but we should get APR_SUCCESS on this call and
     * APR_EOF on the next.
     */
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertStrEquals(tc, TESTSTR, str);
    rv = apr_file_gets(str, 256, f);
    CuAssertIntEquals(tc, APR_EOF, rv);
    CuAssertStrEquals(tc, "", str);
    apr_file_close(f);
}

static void test_bigread(CuTest *tc)
{
    apr_file_t *f = NULL;
    apr_status_t rv;
    char buf[APR_BUFFERSIZE * 2];
    apr_size_t nbytes;

    /* Create a test file with known content.
     */
    rv = apr_file_open(&f, "data/created_file", 
                       APR_CREATE | APR_WRITE | APR_TRUNCATE, 
                       APR_UREAD | APR_UWRITE, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    nbytes = APR_BUFFERSIZE;
    memset(buf, 0xFE, nbytes);

    rv = apr_file_write(f, buf, &nbytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, APR_BUFFERSIZE, nbytes);

    rv = apr_file_close(f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    f = NULL;
    rv = apr_file_open(&f, "data/created_file", APR_READ, 0, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    nbytes = sizeof buf;
    rv = apr_file_read(f, buf, &nbytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, APR_BUFFERSIZE, nbytes);

    rv = apr_file_close(f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_remove("data/created_file", p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
}

/* This is a horrible name for this function.  We are testing APR, not how
 * Apache uses APR.  And, this function tests _way_ too much stuff.
 */
static void test_mod_neg(CuTest *tc)
{
    apr_status_t rv;
    apr_file_t *f;
    const char *s;
    int i;
    apr_size_t nbytes;
    char buf[8192];
    apr_off_t cur;
    const char *fname = "data/modneg.dat";

    rv = apr_file_open(&f, fname, 
                       APR_CREATE | APR_WRITE, APR_UREAD | APR_UWRITE, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    s = "body56789\n";
    nbytes = strlen(s);
    rv = apr_file_write(f, s, &nbytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, strlen(s), nbytes);
    
    for (i = 0; i < 7980; i++) {
        s = "0";
        nbytes = strlen(s);
        rv = apr_file_write(f, s, &nbytes);
        CuAssertIntEquals(tc, APR_SUCCESS, rv);
        CuAssertIntEquals(tc, strlen(s), nbytes);
    }
    
    s = "end456789\n";
    nbytes = strlen(s);
    rv = apr_file_write(f, s, &nbytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, strlen(s), nbytes);

    for (i = 0; i < 10000; i++) {
        s = "1";
        nbytes = strlen(s);
        rv = apr_file_write(f, s, &nbytes);
        CuAssertIntEquals(tc, APR_SUCCESS, rv);
        CuAssertIntEquals(tc, strlen(s), nbytes);
    }
    
    rv = apr_file_close(f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_open(&f, fname, APR_READ, 0, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_gets(buf, 11, f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertStrEquals(tc, "body56789\n", buf);

    cur = 0;
    rv = apr_file_seek(f, APR_CUR, &cur);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, 10, cur);

    nbytes = sizeof(buf);
    rv = apr_file_read(f, buf, &nbytes);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, nbytes, sizeof(buf));

    cur = -((apr_off_t)nbytes - 7980);
    rv = apr_file_seek(f, APR_CUR, &cur);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertIntEquals(tc, 7990, cur);

    rv = apr_file_gets(buf, 11, f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
    CuAssertStrEquals(tc, "end456789\n", buf);

    rv = apr_file_close(f);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);

    rv = apr_file_remove(fname, p);
    CuAssertIntEquals(tc, APR_SUCCESS, rv);
}

CuSuite *testfile(void)
{
    CuSuite *suite = CuSuiteNew("File I/O");

    SUITE_ADD_TEST(suite, test_open_noreadwrite);
    SUITE_ADD_TEST(suite, test_open_excl);
    SUITE_ADD_TEST(suite, test_open_read);
    SUITE_ADD_TEST(suite, test_open_readwrite);
    SUITE_ADD_TEST(suite, test_read); 
    SUITE_ADD_TEST(suite, test_seek);
    SUITE_ADD_TEST(suite, test_filename);
    SUITE_ADD_TEST(suite, test_fileclose);
    SUITE_ADD_TEST(suite, test_file_remove);
    SUITE_ADD_TEST(suite, test_open_write);
    SUITE_ADD_TEST(suite, test_open_writecreate);
    SUITE_ADD_TEST(suite, test_write);
    SUITE_ADD_TEST(suite, test_userdata_set);
    SUITE_ADD_TEST(suite, test_userdata_get);
    SUITE_ADD_TEST(suite, test_userdata_getnokey);
    SUITE_ADD_TEST(suite, test_getc);
    SUITE_ADD_TEST(suite, test_ungetc);
    SUITE_ADD_TEST(suite, test_gets);
    SUITE_ADD_TEST(suite, test_bigread);
    SUITE_ADD_TEST(suite, test_mod_neg);

    return suite;
}

