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

/*
 * mod_actions.c: executes scripts based on MIME type or HTTP method
 *
 * by Alexei Kosut; based on mod_cgi.c, mod_mime.c and mod_includes.c,
 * adapted by rst from original NCSA code by Rob McCool
 *
 * Usage instructions:
 *
 * Action mime/type /cgi-bin/script
 * 
 * will activate /cgi-bin/script when a file of content type mime/type is 
 * requested. It sends the URL and file path of the requested document using 
 * the standard CGI PATH_INFO and PATH_TRANSLATED environment variables.
 *
 * Script PUT /cgi-bin/script
 *
 * will activate /cgi-bin/script when a request is received with the
 * HTTP method "PUT".  The available method names are defined in httpd.h.
 * If the method is GET, the script will only be activated if the requested
 * URI includes query information (stuff after a ?-mark).
 */

#include "apr_strings.h"
#include "ap_config.h"
#include "httpd.h"
#include "http_config.h"
#include "http_request.h"
#include "http_core.h"
#include "http_protocol.h"
#include "http_main.h"
#include "http_log.h"
#include "util_script.h"

typedef struct {
    apr_table_t *action_types;       /* Added with Action... */
    const char *scripted[METHODS];   /* Added with Script... */
    int configured;  /* True if Action or Script has been
                      * called at least once
                      */
} action_dir_config;

module AP_MODULE_DECLARE_DATA actions_module;

static void *create_action_dir_config(apr_pool_t *p, char *dummy)
{
    action_dir_config *new =
    (action_dir_config *) apr_pcalloc(p, sizeof(action_dir_config));

    new->action_types = apr_table_make(p, 4);

    return new;
}

static void *merge_action_dir_configs(apr_pool_t *p, void *basev, void *addv)
{
    action_dir_config *base = (action_dir_config *) basev;
    action_dir_config *add = (action_dir_config *) addv;
    action_dir_config *new = (action_dir_config *) apr_palloc(p,
                                  sizeof(action_dir_config));
    int i;

    new->action_types = apr_table_overlay(p, add->action_types,
				       base->action_types);

    for (i = 0; i < METHODS; ++i) {
        new->scripted[i] = add->scripted[i] ? add->scripted[i]
                                            : base->scripted[i];
    }

    new->configured = (base->configured || add->configured);
    return new;
}

static const char *add_action(cmd_parms *cmd, void *m_v, 
                              const char *type, const char *script)
{
    action_dir_config *m = (action_dir_config *)m_v;
    apr_table_setn(m->action_types, type, script);
    m->configured = 1;
    return NULL;
}

static const char *set_script(cmd_parms *cmd, void *m_v,
                              const char *method, const char *script)
{
    action_dir_config *m = (action_dir_config *)m_v;    
    int methnum;

    methnum = ap_method_number_of(method);
    if (methnum == M_TRACE)
        return "TRACE not allowed for Script";
    else if (methnum == M_INVALID)
        return "Unknown method type for Script";
    else
        m->scripted[methnum] = script;

    m->configured = 1;
    return NULL;
}

static const command_rec action_cmds[] =
{
    AP_INIT_TAKE2("Action", add_action, NULL, OR_FILEINFO,
                  "a media type followed by a script name"),
    AP_INIT_TAKE2("Script", set_script, NULL, ACCESS_CONF | RSRC_CONF,
                  "a method followed by a script name"),
    {NULL}
};

static int action_handler(request_rec *r)
{
    action_dir_config *conf = (action_dir_config *)
        ap_get_module_config(r->per_dir_config, &actions_module);
    const char *t, *action;
    const char *script;
    int i;

    if (!conf->configured) {
        return DECLINED;
    }

    /* Note that this handler handles _all_ types, so handler is unchecked */

    /* Set allowed stuff */
    for (i = 0; i < METHODS; ++i) {
        if (conf->scripted[i])
            r->allowed |= (AP_METHOD_BIT << i);
    }

    /* First, check for the method-handling scripts */
    if (r->method_number == M_GET) {
        if (r->args)
            script = conf->scripted[M_GET];
        else
            script = NULL;
    }
    else {
        script = conf->scripted[r->method_number];
    }

    /* Check for looping, which can happen if the CGI script isn't */
    if (script && r->prev && r->prev->prev)
	return DECLINED;

    /* Second, check for actions (which override the method scripts) */
    action = r->handler ? r->handler :
	ap_field_noparam(r->pool, r->content_type);
    if ((t = apr_table_get(conf->action_types,
		       action ? action : ap_default_type(r)))) {
	script = t;
	if (r->finfo.filetype == 0) {
	    ap_log_rerror(APLOG_MARK, APLOG_ERR, 0, r,
			"File does not exist: %s", r->filename);
	    return HTTP_NOT_FOUND;
	}
    }

    if (script == NULL)
	return DECLINED;

    ap_internal_redirect_handler(apr_pstrcat(r->pool, script,
                                             ap_escape_uri(r->pool, r->uri),
                                             r->args ? "?" : NULL,
                                             r->args, NULL), r);
    return OK;
}

static void register_hooks(apr_pool_t *p)
{
    ap_hook_handler(action_handler,NULL,NULL,APR_HOOK_LAST);
}

module AP_MODULE_DECLARE_DATA actions_module =
{
    STANDARD20_MODULE_STUFF,
    create_action_dir_config,	/* dir config creater */
    merge_action_dir_configs,	/* dir merger --- default is to override */
    NULL,			/* server config */
    NULL,			/* merge server config */
    action_cmds,		/* command apr_table_t */
    register_hooks		/* register hooks */
};
