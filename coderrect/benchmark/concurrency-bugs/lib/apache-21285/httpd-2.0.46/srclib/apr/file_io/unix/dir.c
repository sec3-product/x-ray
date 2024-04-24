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

#include "apr_arch_file_io.h"
#include "apr_strings.h"
#include "apr_portable.h"
#if APR_HAVE_SYS_SYSLIMITS_H
#include <sys/syslimits.h>
#endif
#if APR_HAVE_LIMITS_H
#include <limits.h>
#endif

static apr_status_t dir_cleanup(void *thedir)
{
    apr_dir_t *dir = thedir;
    if (closedir(dir->dirstruct) == 0) {
        return APR_SUCCESS;
    }
    else {
        return errno;
    }
} 

#define PATH_SEPARATOR '/'

/* Remove trailing separators that don't affect the meaning of PATH. */
static const char *path_canonicalize (const char *path, apr_pool_t *pool)
{
    /* At some point this could eliminate redundant components.  For
     * now, it just makes sure there is no trailing slash. */
    apr_size_t len = strlen (path);
    apr_size_t orig_len = len;
    
    while ((len > 0) && (path[len - 1] == PATH_SEPARATOR))
        len--;
    
    if (len != orig_len)
        return apr_pstrndup (pool, path, len);
    else
        return path;
}

/* Remove one component off the end of PATH. */
static char *path_remove_last_component (const char *path, apr_pool_t *pool)
{
    const char *newpath = path_canonicalize (path, pool);
    int i;
    
    for (i = (strlen(newpath) - 1); i >= 0; i--) {
        if (path[i] == PATH_SEPARATOR)
            break;
    }

    return apr_pstrndup (pool, path, (i < 0) ? 0 : i);
}

apr_status_t apr_dir_open(apr_dir_t **new, const char *dirname, 
                          apr_pool_t *pool)
{
    /* On some platforms (e.g., Linux+GNU libc), d_name[] in struct 
     * dirent is declared with enough storage for the name.  On other
     * platforms (e.g., Solaris 8 for Intel), d_name is declared as a
     * one-byte array.  Note: gcc evaluates this at compile time.
     */
    apr_size_t dirent_size = 
        (sizeof((*new)->entry->d_name) > 1 ? 
         sizeof(struct dirent) : sizeof (struct dirent) + 255);

    (*new) = (apr_dir_t *)apr_palloc(pool, sizeof(apr_dir_t));

    (*new)->pool = pool;
    (*new)->dirname = apr_pstrdup(pool, dirname);
    (*new)->dirstruct = opendir(dirname);
    (*new)->entry = apr_pcalloc(pool, dirent_size);

    if ((*new)->dirstruct == NULL) {
        return errno;
    }    
    else {
        apr_pool_cleanup_register((*new)->pool, (void *)(*new), dir_cleanup,
	                          apr_pool_cleanup_null);
        return APR_SUCCESS;
    }
}

apr_status_t apr_dir_close(apr_dir_t *thedir)
{
    return apr_pool_cleanup_run(thedir->pool, thedir, dir_cleanup);
}

#ifdef DIRENT_TYPE
static apr_filetype_e filetype_from_dirent_type(int type)
{
    switch (type) {
    case DT_REG:
        return APR_REG;
    case DT_DIR:
        return APR_DIR;
    case DT_LNK:
        return APR_LNK;
    case DT_CHR:
        return APR_CHR;
    case DT_BLK:
        return APR_BLK;
#if defined(DT_FIFO)
    case DT_FIFO:
        return APR_PIPE;
#endif
#if !defined(BEOS) && defined(DT_SOCK)
    case DT_SOCK:
        return APR_SOCK;
#endif
    default:
        return APR_UNKFILE;
    }
}
#endif

apr_status_t apr_dir_read(apr_finfo_t *finfo, apr_int32_t wanted,
                          apr_dir_t *thedir)
{
    apr_status_t ret = 0;
#ifdef DIRENT_TYPE
    apr_filetype_e type;
#endif
#if APR_HAS_THREADS && defined(_POSIX_THREAD_SAFE_FUNCTIONS) \
                    && !defined(READDIR_IS_THREAD_SAFE)
    struct dirent *retent;

    ret = readdir_r(thedir->dirstruct, thedir->entry, &retent);

    /* Avoid the Linux problem where at end-of-directory thedir->entry
     * is set to NULL, but ret = APR_SUCCESS.
     */
    if(!ret && thedir->entry != retent)
        ret = APR_ENOENT;

    /* Solaris is a bit strange, if there are no more entries in the
     * directory, it returns EINVAL.  Since this is against POSIX, we
     * hack around the problem here.  EINVAL is possible from other
     * readdir implementations, but only if the result buffer is too small.
     * since we control the size of that buffer, we should never have
     * that problem.
     */
    if (ret == EINVAL) {
        ret = ENOENT;
    }
#else
    /* We're about to call a non-thread-safe readdir() that may
       possibly set `errno', and the logic below actually cares about
       errno after the call.  Therefore we need to clear errno first. */
    errno = 0;
    thedir->entry = readdir(thedir->dirstruct);
    if (thedir->entry == NULL) {
        /* If NULL was returned, this can NEVER be a success. Can it?! */
        if (errno == APR_SUCCESS) {
            ret = APR_ENOENT;
        }
        else
            ret = errno;
    }
#endif

    /* No valid bit flag to test here - do we want one? */
    finfo->fname = NULL;

    if (ret) {
        finfo->valid = 0;
        return ret;
    }

#ifdef DIRENT_TYPE
    type = filetype_from_dirent_type(thedir->entry->DIRENT_TYPE);
    if (type != APR_UNKFILE) {
        wanted &= ~APR_FINFO_TYPE;
    }
#endif
#ifdef DIRENT_INODE
    if (thedir->entry->DIRENT_INODE && thedir->entry->DIRENT_INODE != -1) {
        wanted &= ~APR_FINFO_INODE;
    }
#endif

    wanted &= ~APR_FINFO_NAME;

    if (wanted)
    {
        char fspec[APR_PATH_MAX];
        int off;
        apr_cpystrn(fspec, thedir->dirname, sizeof(fspec));
        off = strlen(fspec);
        if ((fspec[off - 1] != '/') && (off + 1 < sizeof(fspec)))
            fspec[off++] = '/';
        apr_cpystrn(fspec + off, thedir->entry->d_name, sizeof(fspec) - off);
        ret = apr_lstat(finfo, fspec, wanted, thedir->pool);
        /* We passed a stack name that will disappear */
        finfo->fname = NULL;
    }

    if (wanted && (ret == APR_SUCCESS || ret == APR_INCOMPLETE)) {
        wanted &= ~finfo->valid;
    }
    else {
        /* We don't bail because we fail to stat, when we are only -required-
         * to readdir... but the result will be APR_INCOMPLETE
         */
        finfo->pool = thedir->pool;
        finfo->valid = 0;
#ifdef DIRENT_TYPE
        if (type != APR_UNKFILE) {
            finfo->filetype = type;
            finfo->valid |= APR_FINFO_TYPE;
        }
#endif
#ifdef DIRENT_INODE
        if (thedir->entry->DIRENT_INODE && thedir->entry->DIRENT_INODE != -1) {
            finfo->inode = thedir->entry->DIRENT_INODE;
            finfo->valid |= APR_FINFO_INODE;
        }
#endif
    }

    finfo->name = apr_pstrdup(thedir->pool, thedir->entry->d_name);
    finfo->valid |= APR_FINFO_NAME;

    if (wanted)
        return APR_INCOMPLETE;

    return APR_SUCCESS;
}

apr_status_t apr_dir_rewind(apr_dir_t *thedir)
{
    rewinddir(thedir->dirstruct);
    return APR_SUCCESS;
}

apr_status_t apr_dir_make(const char *path, apr_fileperms_t perm, 
                          apr_pool_t *pool)
{
    mode_t mode = apr_unix_perms2mode(perm);

    if (mkdir(path, mode) == 0) {
        return APR_SUCCESS;
    }
    else {
        return errno;
    }
}

apr_status_t apr_dir_make_recursive(const char *path, apr_fileperms_t perm,
                                           apr_pool_t *pool) 
{
    apr_status_t apr_err = 0;
    
    apr_err = apr_dir_make (path, perm, pool); /* Try to make PATH right out */
    
    if (apr_err == EEXIST) /* It's OK if PATH exists */
        return APR_SUCCESS;
    
    if (apr_err == ENOENT) { /* Missing an intermediate dir */
        char *dir;
        
        dir = path_remove_last_component(path, pool);
        apr_err = apr_dir_make_recursive(dir, perm, pool);
        
        if (!apr_err) 
            apr_err = apr_dir_make (path, perm, pool);
    }

    return apr_err;
}

apr_status_t apr_dir_remove(const char *path, apr_pool_t *pool)
{
    if (rmdir(path) == 0) {
        return APR_SUCCESS;
    }
    else {
        return errno;
    }
}

apr_status_t apr_os_dir_get(apr_os_dir_t **thedir, apr_dir_t *dir)
{
    if (dir == NULL) {
        return APR_ENODIR;
    }
    *thedir = dir->dirstruct;
    return APR_SUCCESS;
}

apr_status_t apr_os_dir_put(apr_dir_t **dir, apr_os_dir_t *thedir,
                          apr_pool_t *pool)
{
    if ((*dir) == NULL) {
        (*dir) = (apr_dir_t *)apr_pcalloc(pool, sizeof(apr_dir_t));
        (*dir)->pool = pool;
    }
    (*dir)->dirstruct = thedir;
    return APR_SUCCESS;
}

  
