dnl -----------------------------------------------------------------
dnl apr_network.m4: APR's autoconf macros for testing network support
dnl

dnl
dnl check for working getaddrinfo()
dnl
dnl Note that if the system doesn't have gai_strerror(), we
dnl can't use getaddrinfo() because we can't get strings
dnl describing the error codes.
dnl
AC_DEFUN(APR_CHECK_WORKING_GETADDRINFO,[
  AC_CACHE_CHECK(for working getaddrinfo, ac_cv_working_getaddrinfo,[
  AC_TRY_RUN( [
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif

void main(void) {
    struct addrinfo hints, *ai;
    int error;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    error = getaddrinfo("127.0.0.1", NULL, &hints, &ai);
    if (error) {
        exit(1);
    }
    if (ai->ai_addr->sa_family != AF_INET) {
        exit(1);
    }
    exit(0);
}
],[
  ac_cv_working_getaddrinfo="yes"
],[
  ac_cv_working_getaddrinfo="no"
],[
  ac_cv_working_getaddrinfo="yes"
])])
if test "$ac_cv_working_getaddrinfo" = "yes"; then
  if test "$ac_cv_func_gai_strerror" != "yes"; then
    ac_cv_working_getaddrinfo="no"
  else
    AC_DEFINE(HAVE_GETADDRINFO, 1, [Define if getaddrinfo exists and works well enough for APR])
  fi
fi
])

dnl
dnl check for working getnameinfo()
dnl
AC_DEFUN(APR_CHECK_WORKING_GETNAMEINFO,[
  AC_CACHE_CHECK(for working getnameinfo, ac_cv_working_getnameinfo,[
  AC_TRY_RUN( [
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif

void main(void) {
    struct sockaddr_in sa;
    char hbuf[256];
    int error;

    sa.sin_family = AF_INET;
    sa.sin_port = 0;
    sa.sin_addr.s_addr = inet_addr("127.0.0.1");
#ifdef SIN6_LEN
    sa.sin_len = sizeof(sa);
#endif

    error = getnameinfo((const struct sockaddr *)&sa, sizeof(sa),
                        hbuf, 256, NULL, 0,
                        NI_NUMERICHOST);
    if (error) {
        exit(1);
    } else {
        exit(0);
    }
}
],[
  ac_cv_working_getnameinfo="yes"
],[
  ac_cv_working_getnameinfo="no"
],[
  ac_cv_working_getnameinfo="yes"
])])
if test "$ac_cv_working_getnameinfo" = "yes"; then
  AC_DEFINE(HAVE_GETNAMEINFO, 1, [Define if getnameinfo exists])
fi
])

dnl
dnl check for negative error codes for getaddrinfo()
dnl
AC_DEFUN(APR_CHECK_NEGATIVE_EAI,[
  AC_CACHE_CHECK(for negative error codes for getaddrinfo, ac_cv_negative_eai,[
  AC_TRY_RUN( [
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

void main(void) {
    if (EAI_ADDRFAMILY < 0) {
        exit(0);
    }
    exit(1);
}
],[
  ac_cv_negative_eai="yes"
],[
  ac_cv_negative_eai="no"
],[
  ac_cv_negative_eai="no"
])])
if test "$ac_cv_negative_eai" = "yes"; then
  AC_DEFINE(NEGATIVE_EAI, 1, [Define if EAI_ error codes from getaddrinfo are negative])
fi
])

dnl
dnl check for presence of  retrans/retry variables in the res_state structure
dnl
AC_DEFUN(APR_CHECK_RESOLV_RETRANS,[
  AC_CACHE_CHECK(for presence of retrans/retry fields in res_state/resolv.h , ac_cv_retransretry,[
  AC_TRY_RUN( [
#include <sys/types.h>
#if defined(__sun__)
#include <inet/ip.h>
#endif
#include <resolv.h>
/* _res is a global defined in resolv.h */
int main(void) {
    _res.retrans = 2;
    _res.retry = 1;
    exit(0);
    return 0;
}
],[
  ac_cv_retransretry="yes"
],[
  ac_cv_retransretry="no"
],[
  ac_cv_retransretry="no"
])])
if test "$ac_cv_retransretry" = "yes"; then
  AC_DEFINE(RESOLV_RETRANSRETRY, 1, [Define if resolv.h's res_state has the fields retrans/rety])
fi
])

dnl
dnl Checks the definition of gethostbyname_r and gethostbyaddr_r
dnl which are different for glibc, solaris and assorted other operating
dnl systems
dnl
dnl Note that this test is executed too early to see if we have all of
dnl the headers.
AC_DEFUN(APR_CHECK_GETHOSTBYNAME_R_STYLE,[

dnl Try and compile a glibc2 gethostbyname_r piece of code, and set the
dnl style of the routines to glibc2 on success
AC_CACHE_CHECK([style of gethostbyname_r routine], ac_cv_gethostbyname_r_style,
APR_TRY_COMPILE_NO_WARNING([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
],[
int tmp = gethostbyname_r((const char *) 0, (struct hostent *) 0, 
                          (char *) 0, 0, (struct hostent **) 0, &tmp);
], ac_cv_gethostbyname_r_style=glibc2, ac_cv_gethostbyname_r_style=none))

if test "$ac_cv_gethostbyname_r_style" = "glibc2"; then
    AC_DEFINE(GETHOSTBYNAME_R_GLIBC2, 1, [Define if gethostbyname_r has the glibc style])
fi

AC_CACHE_CHECK([3rd argument to the gethostbyname_r routines], ac_cv_gethostbyname_r_arg,
APR_TRY_COMPILE_NO_WARNING([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
],[
int tmp = gethostbyname_r((const char *) 0, (struct hostent *) 0, 
                          (struct hostent_data *) 0);],
ac_cv_gethostbyname_r_arg=hostent_data, ac_cv_gethostbyname_r_arg=char))

if test "$ac_cv_gethostbyname_r_arg" = "hostent_data"; then
    AC_DEFINE(GETHOSTBYNAME_R_HOSTENT_DATA, 1, [Define if gethostbyname_r has the hostent_data for the third argument])
fi
])

dnl
dnl see if TCP_NODELAY setting is inherited from listening sockets
dnl
AC_DEFUN(APR_CHECK_TCP_NODELAY_INHERITED,[
  AC_CACHE_CHECK(if TCP_NODELAY setting is inherited from listening sockets, ac_cv_tcp_nodelay_inherited,[
  AC_TRY_RUN( [
#include <stdio.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_TCP_H
#include <netinet/tcp.h>
#endif
#ifndef HAVE_SOCKLEN_T
typedef int socklen_t;
#endif
int main(void) {
    int listen_s, connected_s, client_s;
    int listen_port, rc;
    struct sockaddr_in sa;
    socklen_t sa_len;
    socklen_t option_len;
    int option;

    listen_s = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_s < 0) {
        perror("socket");
        exit(1);
    }
    option = 1;
    rc = setsockopt(listen_s, IPPROTO_TCP, TCP_NODELAY, &option, sizeof option);
    if (rc < 0) {
        perror("setsockopt TCP_NODELAY");
        exit(1);
    }
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
#ifdef BEOS
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#endif
    /* leave port 0 to get ephemeral */
    rc = bind(listen_s, (struct sockaddr *)&sa, sizeof sa);
    if (rc < 0) {
        perror("bind for ephemeral port");
        exit(1);
    }
    /* find ephemeral port */
    sa_len = sizeof(sa);
    rc = getsockname(listen_s, (struct sockaddr *)&sa, &sa_len);
    if (rc < 0) {
        perror("getsockname");
        exit(1);
    }
    listen_port = sa.sin_port;
    rc = listen(listen_s, 5);
    if (rc < 0) {
        perror("listen");
        exit(1);
    }
    client_s = socket(AF_INET, SOCK_STREAM, 0);
    if (client_s < 0) {
        perror("socket");
        exit(1);
    }
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_port   = listen_port;
#ifdef BEOS
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#endif
    /* leave sin_addr all zeros to use loopback */
    rc = connect(client_s, (struct sockaddr *)&sa, sizeof sa);
    if (rc < 0) {
        perror("connect");
        exit(1);
    }
    sa_len = sizeof sa;
    connected_s = accept(listen_s, (struct sockaddr *)&sa, &sa_len);
    if (connected_s < 0) {
        perror("accept");
        exit(1);
    }
    option_len = sizeof option;
    rc = getsockopt(connected_s, IPPROTO_TCP, TCP_NODELAY, &option, &option_len);
    if (rc < 0) {
        perror("getsockopt");
        exit(1);
    }
    if (!option) {
        fprintf(stderr, "TCP_NODELAY is not set in the child.\n");
        exit(1);
    }
    return 0;
}
],[
    ac_cv_tcp_nodelay_inherited="yes"
],[
    ac_cv_tcp_nodelay_inherited="no"
],[
    ac_cv_tcp_nodelay_inherited="yes"
])])
if test "$ac_cv_tcp_nodelay_inherited" = "yes"; then
    tcp_nodelay_inherited=1
else
    tcp_nodelay_inherited=0
fi
])

dnl
dnl see if O_NONBLOCK setting is inherited from listening sockets
dnl
AC_DEFUN(APR_CHECK_O_NONBLOCK_INHERITED,[
  AC_CACHE_CHECK(if O_NONBLOCK setting is inherited from listening sockets, ac_cv_o_nonblock_inherited,[
  AC_TRY_RUN( [
#include <stdio.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_TCP_H
#include <netinet/tcp.h>
#endif
#ifndef HAVE_SOCKLEN_T
typedef int socklen_t;
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
int main(void) {
    int listen_s, connected_s, client_s;
    int listen_port, rc;
    struct sockaddr_in sa;
    socklen_t sa_len;

    listen_s = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_s < 0) {
        perror("socket");
        exit(1);
    }
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
#ifdef BEOS
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#endif
    /* leave port 0 to get ephemeral */
    rc = bind(listen_s, (struct sockaddr *)&sa, sizeof sa);
    if (rc < 0) {
        perror("bind for ephemeral port");
        exit(1);
    }
    /* find ephemeral port */
    sa_len = sizeof(sa);
    rc = getsockname(listen_s, (struct sockaddr *)&sa, &sa_len);
    if (rc < 0) {
        perror("getsockname");
        exit(1);
    }
    listen_port = sa.sin_port;
    rc = listen(listen_s, 5);
    if (rc < 0) {
        perror("listen");
        exit(1);
    }
    rc = fcntl(listen_s, F_SETFL, O_NONBLOCK);
    if (rc < 0) {
        perror("fcntl(F_SETFL)");
        exit(1);
    }
    client_s = socket(AF_INET, SOCK_STREAM, 0);
    if (client_s < 0) {
        perror("socket");
        exit(1);
    }
    memset(&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_port   = listen_port;
#ifdef BEOS
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#endif
    /* leave sin_addr all zeros to use loopback */
    rc = connect(client_s, (struct sockaddr *)&sa, sizeof sa);
    if (rc < 0) {
        perror("connect");
        exit(1);
    }
    sa_len = sizeof sa;
    connected_s = accept(listen_s, (struct sockaddr *)&sa, &sa_len);
    if (connected_s < 0) {
        perror("accept");
        exit(1);
    }
    rc = fcntl(connected_s, F_GETFL, 0);
    if (rc < 0) {
        perror("fcntl(F_GETFL)");
        exit(1);
    }
    if (!(rc & O_NONBLOCK)) {
        fprintf(stderr, "O_NONBLOCK is not set in the child.\n");
        exit(1);
    }
    return 0;
}
],[
    ac_cv_o_nonblock_inherited="yes"
],[
    ac_cv_o_nonblock_inherited="no"
],[
    ac_cv_o_nonblock_inherited="yes"
])])
if test "$ac_cv_o_nonblock_inherited" = "yes"; then
    o_nonblock_inherited=1
else
    o_nonblock_inherited=0
fi
])

dnl 
dnl check for socklen_t, fall back to unsigned int
dnl
AC_DEFUN(APR_CHECK_SOCKLEN_T,[
AC_CACHE_CHECK(for socklen_t, ac_cv_socklen_t,[
AC_TRY_COMPILE([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
],[
socklen_t foo = (socklen_t) 0;
],[
    ac_cv_socklen_t=yes
],[
    ac_cv_socklen_t=no
])
])

if test "$ac_cv_socklen_t" = "yes"; then
  AC_DEFINE(HAVE_SOCKLEN_T, 1, [Whether you have socklen_t])
fi
])


AC_DEFUN(APR_CHECK_INET_ADDR,[
AC_CACHE_CHECK(for inet_addr, ac_cv_func_inet_addr,[
AC_TRY_COMPILE([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
],[
inet_addr("127.0.0.1");
],[
    ac_cv_func_inet_addr=yes
],[
    ac_cv_func_inet_addr=no
])
])

if test "$ac_cv_func_inet_addr" = "yes"; then
  have_inet_addr=1
else
  have_inet_addr=0
fi
])


AC_DEFUN(APR_CHECK_INET_NETWORK,[
AC_CACHE_CHECK(for inet_network, ac_cv_func_inet_network,[
AC_TRY_COMPILE([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
],[
inet_network("127.0.0.1");
],[
    ac_cv_func_inet_network=yes
],[
    ac_cv_func_inet_network=no
])
])

if test "$ac_cv_func_inet_network" = "yes"; then
  have_inet_network=1
else
  have_inet_network=0
fi
])


AC_DEFUN(APR_CHECK_SOCKADDR_IN6,[
AC_CACHE_CHECK(for sockaddr_in6, ac_cv_define_sockaddr_in6,[
AC_TRY_COMPILE([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
],[
struct sockaddr_in6 sa;
],[
    ac_cv_define_sockaddr_in6=yes
],[
    ac_cv_define_sockaddr_in6=no
])
])

if test "$ac_cv_define_sockaddr_in6" = "yes"; then
  have_sockaddr_in6=1
else
  have_sockaddr_in6=0
fi
])


dnl
dnl Check to see if this platform includes sa_len in it's
dnl struct sockaddr.  If it does it changes the length of sa_family
dnl which could cause us problems
dnl
AC_DEFUN(APR_CHECK_SOCKADDR_SA_LEN,[
AC_CACHE_CHECK(for sockaddr sa_len, ac_cv_define_sockaddr_sa_len,[
AC_TRY_COMPILE([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
],[
struct sockaddr_in sai;
int i = sai.sin_len;
],[
  ac_cv_define_sockaddr_sa_len=yes
],[
  ac_cv_define_sockaddr_sa_len=no
])
])

if test "$ac_cv_define_sockaddr_sa_len" = "yes"; then
  AC_DEFINE(HAVE_SOCKADDR_SA_LEN, 1 ,[Define if we have length field in sockaddr_in])
fi
])


dnl
dnl APR_INADDR_NONE
dnl
dnl checks for missing INADDR_NONE macro
dnl
AC_DEFUN(APR_INADDR_NONE,[
  AC_CACHE_CHECK(whether system defines INADDR_NONE, ac_cv_inaddr_none,[
  AC_TRY_COMPILE([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
],[
unsigned long foo = INADDR_NONE;
],[
    ac_cv_inaddr_none=yes
],[
    ac_cv_inaddr_none=no
])])
  if test "$ac_cv_inaddr_none" = "no"; then
    apr_inaddr_none="((unsigned int) 0xffffffff)"
  else
    apr_inaddr_none="INADDR_NONE"
  fi
])


dnl
dnl APR_H_ERRNO_COMPILE_CHECK
dnl
AC_DEFUN(APR_H_ERRNO_COMPILE_CHECK,[
  if test x$1 != x; then
    CPPFLAGS="-D$1 $CPPFLAGS"
  fi
  AC_TRY_COMPILE([
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
],[
int h_e = h_errno;
],[
  if test x$1 != x; then
    ac_cv_h_errno_cppflags="$1"
  else
    ac_cv_h_errno_cppflags=yes
  fi
],[
  ac_cv_h_errno_cppflags=no
])])


dnl
dnl APR_CHECK_SCTP
dnl
dnl check for presence of SCTP protocol support
dnl
AC_DEFUN(APR_CHECK_SCTP,[
  AC_CACHE_CHECK(if SCTP protocol is supported, ac_cv_sctp,[
  AC_TRY_RUN( [
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
int main(void) {
    int s = socket(AF_INET, SOCK_STREAM, IPPROTO_SCTP);
    if (s < 0) {
        exit(1);
    }
    exit(0);
}
],[
    ac_cv_sctp="yes"
],[
    ac_cv_sctp="no"
],[
    ac_cv_sctp="yes"
])])
if test "$ac_cv_sctp" = "yes"; then
    have_sctp=1
else
    have_sctp=0
fi
])

dnl
dnl APR_CHECK_H_ERRNO_FLAG
dnl
dnl checks which flags are necessary for <netdb.h> to define h_errno
dnl
AC_DEFUN(APR_CHECK_H_ERRNO_FLAG,[
  AC_MSG_CHECKING([for h_errno in netdb.h])
  AC_CACHE_VAL(ac_cv_h_errno_cppflags,[
    APR_H_ERRNO_COMPILE_CHECK
    if test "$ac_cv_h_errno_cppflags" = "no"; then
      ac_save="$CPPFLAGS"
      for flag in _XOPEN_SOURCE_EXTENDED; do
        APR_H_ERRNO_COMPILE_CHECK($flag)
        if test "$ac_cv_h_errno_cppflags" != "no"; then
          break
        fi
      done
      CPPFLAGS="$ac_save"
    fi
  ])
  if test "$ac_cv_h_errno_cppflags" != "no"; then
    if test "$ac_cv_h_errno_cppflags" != "yes"; then
      CPPFLAGS="-D$ac_cv_h_errno_cppflags $CPPFLAGS"
      AC_MSG_RESULT([yes, with -D$ac_cv_h_errno_cppflags])
    else
      AC_MSG_RESULT([$ac_cv_h_errno_cppflags])
    fi
  else
    AC_MSG_RESULT([$ac_cv_h_errno_cppflags])
  fi
])


AC_DEFUN(APR_EBCDIC,[
  AC_CACHE_CHECK([whether system uses EBCDIC],ac_cv_ebcdic,[
  AC_TRY_RUN( [
int main(void) { 
  return (unsigned char)'A' != (unsigned char)0xC1; 
} 
],[
  ac_cv_ebcdic="yes"
],[
  ac_cv_ebcdic="no"
],[
  ac_cv_ebcdic="no"
])])
  if test "$ac_cv_ebcdic" = "yes"; then
    apr_charset_ebcdic=1
  else
    apr_charset_ebcdic=0
  fi
  AC_SUBST(apr_charset_ebcdic)
])

