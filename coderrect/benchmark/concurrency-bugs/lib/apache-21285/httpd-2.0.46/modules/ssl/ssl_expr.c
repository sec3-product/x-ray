/*                      _             _
**  _ __ ___   ___   __| |    ___ ___| |  mod_ssl
** | '_ ` _ \ / _ \ / _` |   / __/ __| |  Apache Interface to OpenSSL
** | | | | | | (_) | (_| |   \__ \__ \ |  www.modssl.org
** |_| |_| |_|\___/ \__,_|___|___/___/_|  ftp.modssl.org
**                      |_____|
**  ssl_expr.c
**  Expression Handling
*/

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
 */
                             /* ``It is hard to fly with
                                  the eagles when you work
                                  with the turkeys.''
                                          -- Unknown  */
#include "mod_ssl.h"

/*  _________________________________________________________________
**
**  Expression Handling
**  _________________________________________________________________
*/

ssl_expr_info_type ssl_expr_info;
char              *ssl_expr_error;

ssl_expr *ssl_expr_comp(apr_pool_t *p, char *expr)
{
    ssl_expr_info.pool       = p;
    ssl_expr_info.inputbuf   = expr;
    ssl_expr_info.inputlen   = strlen(expr);
    ssl_expr_info.inputptr   = ssl_expr_info.inputbuf;
    ssl_expr_info.expr       = FALSE;

    ssl_expr_error = NULL;
    if (ssl_expr_yyparse())
        return NULL;
    return ssl_expr_info.expr;
}

char *ssl_expr_get_error(void)
{
    if (ssl_expr_error == NULL)
        return "";
    return ssl_expr_error;
}

ssl_expr *ssl_expr_make(ssl_expr_node_op op, void *a1, void *a2)
{
    ssl_expr *node;

    node = (ssl_expr *)apr_palloc(ssl_expr_info.pool, sizeof(ssl_expr));
    node->node_op   = op;
    node->node_arg1 = (char *)a1;
    node->node_arg2 = (char *)a2;
    return node;
}

int ssl_expr_exec(request_rec *r, ssl_expr *expr)
{
    BOOL rc;

    rc = ssl_expr_eval(r, expr);
    if (ssl_expr_error != NULL)
        return (-1);
    else
        return (rc ? 1 : 0);
}
