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

#include "apr_arch_threadproc.h"
#include "apr_strings.h"

struct send_pipe {
	int in;
	int out;
	int err;
};

APR_DECLARE(apr_status_t) apr_procattr_create(apr_procattr_t **new, apr_pool_t *pool)
{
    (*new) = (apr_procattr_t *)apr_palloc(pool, 
              sizeof(apr_procattr_t));

    if ((*new) == NULL) {
        return APR_ENOMEM;
    }
    (*new)->pool = pool;
    (*new)->parent_in = NULL;
    (*new)->child_in = NULL;
    (*new)->parent_out = NULL;
    (*new)->child_out = NULL;
    (*new)->parent_err = NULL;
    (*new)->child_err = NULL;
    (*new)->currdir = NULL; 
    (*new)->cmdtype = APR_PROGRAM;
    (*new)->detached = 0;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_procattr_io_set(apr_procattr_t *attr, apr_int32_t in, 
                                              apr_int32_t out, apr_int32_t err)
{
    apr_status_t status;
    if (in != 0) {
        if ((status = apr_file_pipe_create(&attr->child_in, &attr->parent_in, 
                                   attr->pool)) != APR_SUCCESS) {
            return status;
        }
        switch (in) {
        case APR_FULL_BLOCK:
            apr_file_pipe_timeout_set(attr->child_in, -1);
            apr_file_pipe_timeout_set(attr->parent_in, -1);
            break;
        case APR_PARENT_BLOCK:
            apr_file_pipe_timeout_set(attr->child_in, -1);
            break;
        case APR_CHILD_BLOCK:
            apr_file_pipe_timeout_set(attr->parent_in, -1);
            break;
        default:
            break;
        }
    } 
    if (out) {
        if ((status = apr_file_pipe_create(&attr->parent_out, &attr->child_out, 
                                   attr->pool)) != APR_SUCCESS) {
            return status;
        }
        switch (out) {
        case APR_FULL_BLOCK:
            apr_file_pipe_timeout_set(attr->child_out, -1);
            apr_file_pipe_timeout_set(attr->parent_out, -1);       
            break;
        case APR_PARENT_BLOCK:
            apr_file_pipe_timeout_set(attr->child_out, -1);
            break;
        case APR_CHILD_BLOCK:
            apr_file_pipe_timeout_set(attr->parent_out, -1);
            break;
        default:
            break;
        }
    } 
    if (err) {
        if ((status = apr_file_pipe_create(&attr->parent_err, &attr->child_err, 
                                   attr->pool)) != APR_SUCCESS) {
            return status;
        }
        switch (err) {
        case APR_FULL_BLOCK:
            apr_file_pipe_timeout_set(attr->child_err, -1);
            apr_file_pipe_timeout_set(attr->parent_err, -1);
            break;
        case APR_PARENT_BLOCK:
            apr_file_pipe_timeout_set(attr->child_err, -1);
            break;
        case APR_CHILD_BLOCK:
            apr_file_pipe_timeout_set(attr->parent_err, -1);
            break;
        default:
            break;
        }
    } 
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_procattr_dir_set(apr_procattr_t *attr, 
                                               const char *dir) 
{
    char * cwd;
    if (strncmp("/",dir,1) != 0 ) {
        cwd = (char*)malloc(sizeof(char) * PATH_MAX);
        getcwd(cwd, PATH_MAX);
        strncat(cwd,"/\0",2);
        strcat(cwd,dir);
        attr->currdir = (char *)apr_pstrdup(attr->pool, cwd);
        free(cwd);
    } else {
        attr->currdir = (char *)apr_pstrdup(attr->pool, dir);
    }
    if (attr->currdir) {
        return APR_SUCCESS;
    }
    return APR_ENOMEM;
}

APR_DECLARE(apr_status_t) apr_procattr_cmdtype_set(apr_procattr_t *attr,
                                                   apr_cmdtype_e cmd) 
{
    attr->cmdtype = cmd;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_procattr_detach_set(apr_procattr_t *attr, apr_int32_t detach) 
{
    attr->detached = detach;
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_proc_fork(apr_proc_t *proc, apr_pool_t *pool)
{
    int pid;
    
    if ((pid = fork()) < 0) {
        return errno;
    }
    else if (pid == 0) {
        proc->pid = pid;
        proc->in = NULL; 
        proc->out = NULL; 
        proc->err = NULL; 
        return APR_INCHILD;
    }
    proc->pid = pid;
    proc->in = NULL; 
    proc->out = NULL; 
    proc->err = NULL; 
    return APR_INPARENT;
}

APR_DECLARE(apr_status_t) apr_procattr_child_errfn_set(apr_procattr_t *attr,
                                                       apr_child_errfn_t *errfn)
{
    /* won't ever be called on this platform, so don't save the function pointer */
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_procattr_error_check_set(apr_procattr_t *attr,
                                                       apr_int32_t chk)
{
    /* won't ever be used on this platform, so don't save the flag */
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_proc_create(apr_proc_t *new, const char *progname, 
                                          const char * const *args,
                                          const char * const *env, 
                                          apr_procattr_t *attr, apr_pool_t *pool)
{
    int i=0,nargs=0;
    char **newargs = NULL;
    thread_id newproc, sender;
    struct send_pipe *sp;        
    char * dir = NULL;
	    
    sp = (struct send_pipe *)apr_palloc(pool, sizeof(struct send_pipe));

    new->in = attr->parent_in;
    new->err = attr->parent_err;
    new->out = attr->parent_out;
	sp->in  = attr->child_in  ? attr->child_in->filedes  : -1;
	sp->out = attr->child_out ? attr->child_out->filedes : -1;
	sp->err = attr->child_err ? attr->child_err->filedes : -1;

    i = 0;
    while (args && args[i]) {
        i++;
    }

	newargs = (char**)malloc(sizeof(char *) * (i + 4));
	newargs[0] = strdup("/boot/home/config/bin/apr_proc_stub");
    if (attr->currdir == NULL) {
        /* we require the directory , so use a temp. variable */
        dir = malloc(sizeof(char) * PATH_MAX);
        getcwd(dir, PATH_MAX);
        newargs[1] = strdup(dir);
        free(dir);
    } else {
        newargs[1] = strdup(attr->currdir);
    }
    newargs[2] = strdup(progname);
    i=0;nargs = 3;

    while (args && args[i]) {
        newargs[nargs] = strdup(args[i]);
        i++;nargs++;
    }
    newargs[nargs] = NULL;

    /* ### we should be looking at attr->cmdtype in here... */

    newproc = load_image(nargs, (const char**)newargs, (const char**)env);

    /* load_image copies the data so now we can free it... */
    while (--nargs >= 0)
        free (newargs[nargs]);
    free(newargs);
        
    if ( newproc < B_NO_ERROR) {
        return errno;
    }

    resume_thread(newproc);

    if (attr->child_in) {
        apr_file_close(attr->child_in);
    }
    if (attr->child_out) {
        apr_file_close(attr->child_out);
    }
    if (attr->child_err) {
        apr_file_close(attr->child_err);
    }

    send_data(newproc, 0, (void*)sp, sizeof(struct send_pipe));
    new->pid = newproc;

    /* before we go charging on we need the new process to get to a 
     * certain point.  When it gets there it'll let us know and we
     * can carry on. */
    receive_data(&sender, (void*)NULL,0);
    
    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_proc_wait_all_procs(apr_proc_t *proc,
                                                  int *exitcode,
                                                  apr_exit_why_e *exitwhy,
                                                  apr_wait_how_e waithow, 
                                                  apr_pool_t *p)
{
    proc->pid = -1;
    return apr_proc_wait(proc, exitcode, exitwhy, waithow);
} 

APR_DECLARE(apr_status_t) apr_proc_wait(apr_proc_t *proc,
                                        int *exitcode, 
                                        apr_exit_why_e *exitwhy,
                                        apr_wait_how_e waithow)
{
    pid_t pstatus;
    int waitpid_options = WUNTRACED;
    int exit_int;
    int ignore;
    apr_exit_why_e ignorewhy;

    if (exitcode == NULL) {
        exitcode = &ignore;
    }
    if (exitwhy == NULL) {
        exitwhy = &ignorewhy;
    }

    if (waithow != APR_WAIT) {
        waitpid_options |= WNOHANG;
    }
    
    if ((pstatus = waitpid(proc->pid, &exit_int, waitpid_options)) > 0) {
        proc->pid = pstatus;
        if (WIFEXITED(exit_int)) {
            *exitwhy = APR_PROC_EXIT;
            *exitcode = WEXITSTATUS(exit_int);
        }
        else if (WIFSIGNALED(exit_int)) {
            *exitwhy = APR_PROC_SIGNAL;
            *exitcode = WTERMSIG(exit_int);
        }
        else {
            /* unexpected condition */
            return APR_EGENERAL;
        }
        return APR_CHILD_DONE;
    }
    else if (pstatus == 0) {
        return APR_CHILD_NOTDONE;
    }
    return errno;
} 

APR_DECLARE(apr_status_t) apr_procattr_child_in_set(apr_procattr_t *attr, apr_file_t *child_in,
                                   apr_file_t *parent_in)
{
    if (attr->child_in == NULL && attr->parent_in == NULL)
        apr_file_pipe_create(&attr->child_in, &attr->parent_in, attr->pool);

    if (child_in != NULL)
        apr_file_dup(&attr->child_in, child_in, attr->pool);

    if (parent_in != NULL)
        apr_file_dup(&attr->parent_in, parent_in, attr->pool);

    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_procattr_child_out_set(apr_procattr_t *attr, apr_file_t *child_out,
                                                     apr_file_t *parent_out)
{
    if (attr->child_out == NULL && attr->parent_out == NULL)
        apr_file_pipe_create(&attr->child_out, &attr->parent_out, attr->pool);

    if (child_out != NULL)
        apr_file_dup(&attr->child_out, child_out, attr->pool);

    if (parent_out != NULL)
        apr_file_dup(&attr->parent_out, parent_out, attr->pool);

    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_procattr_child_err_set(apr_procattr_t *attr, apr_file_t *child_err,
                                                     apr_file_t *parent_err)
{
    if (attr->child_err == NULL && attr->parent_err == NULL)
        apr_file_pipe_create(&attr->child_err, &attr->parent_err, attr->pool);

    if (child_err != NULL)
        apr_file_dup(&attr->child_err, child_err, attr->pool);

    if (parent_err != NULL)
        apr_file_dup(&attr->parent_err, parent_err, attr->pool);

    return APR_SUCCESS;
}

APR_DECLARE(apr_status_t) apr_procattr_limit_set(apr_procattr_t *attr, apr_int32_t what, 
                                                  void *limit)
{
    return APR_ENOTIMPL;
}
