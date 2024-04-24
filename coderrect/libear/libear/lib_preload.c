/*  Copyright (C) 2012-2019 by László Nagy
    This file is part of Bear.

    Bear is a tool to generate compilation database for clang tooling.

    Bear is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Bear is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * This file implements a shared library. This library can be pre-loaded by
 * the dynamic linker of the Operating System (OS). It implements a few function
 * related to process creation. By pre-load this library the executed process
 * uses these functions instead of those from the standard library.
 *
 * The idea here is to inject a logic before call the real methods. The logic is
 * to dump the call into a file. To call the real method this library is doing
 * the job of the dynamic linker.
 *
 * The only input for the log writing is about the destination directory.
 * This is passed as environment variable.
 */

#include "config.h"

#include <stddef.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <locale.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <sys/wait.h>

#include "lib/intercept.h"

#if defined HAVE_POSIX_SPAWN || defined HAVE_POSIX_SPAWNP
#include <spawn.h>
#include <execinfo.h>

#endif

#if defined HAVE_NSGETENVIRON
#include <crt_externs.h>
static char **environ;
#else
extern char **environ;
#endif

// TODO: we might need anthor environment variable for the CODRRECT PATH
#define ENV_OUTPUT "SKIP_FOLLOWING"
#ifdef APPLE
#define ENV_FLAT "DYLD_FORCE_FLAT_NAMESPACE"
#define ENV_PRELOAD "DYLD_INSERT_LIBRARIES"
#define ENV_SIZE 3
#else
#define ENV_PRELOAD "LD_PRELOAD"
#define ENV_SIZE 2
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT "libear: (" __FILE__ ":" TOSTRING(__LINE__) ") "

#define PERROR(msg)       \
    do                    \
    { /*perror(AT msg);*/ \
    } while (0)

#define ERROR_AND_EXIT(msg) \
    do                      \
    {                       \
        PERROR(msg);        \
        exit(EXIT_FAILURE); \
    } while (0)

#define DLSYM(TYPE_, VAR_, SYMBOL_)                   \
    union                                             \
    {                                                 \
        void *from;                                   \
        TYPE_ to;                                     \
    } cast;                                           \
    if (0 == (cast.from = dlsym(RTLD_NEXT, SYMBOL_))) \
    {                                                 \
        PERROR("dlsym");                              \
        exit(EXIT_FAILURE);                           \
    }                                                 \
    TYPE_ const VAR_ = cast.to;

typedef char const *bear_env_t[ENV_SIZE];

static int capture_env_t(bear_env_t *env);
static void release_env_t(bear_env_t *env);
static char const **string_array_partial_update(char *const envp[], bear_env_t *env);
static char const **string_array_single_update(char const *envs[], char const *key, char const *value);
static char const **string_array_from_varargs(char const *arg, va_list *args);
static char const **string_array_copy(char const **in);
static size_t string_array_length(char const *const *in);
static void string_array_release(char const **);

int shouldInterceptFollow = 0;

static bear_env_t env_names =
    {ENV_OUTPUT, ENV_PRELOAD
#ifdef ENV_FLAT
     ,
     ENV_FLAT
#endif
};

static bear_env_t initial_env =
    {0, 0
#ifdef ENV_FLAT
     ,
     0
#endif
};

static int initialized = 0;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

#ifdef HAVE_EXECVE
static int call_execve(const char *path, char *const argv[],
                       char *const envp[]);
#endif
#ifdef HAVE_EXECVP
static int call_execvp(const char *file, char *const argv[]);
#endif
#ifdef HAVE_EXECVPE
static int call_execvpe(const char *file, char *const argv[],
                        char *const envp[]);
#endif
#ifdef HAVE_EXECVP2
static int call_execvP(const char *file, const char *search_path,
                       char *const argv[]);
#endif
#ifdef HAVE_EXECT
static int call_exect(const char *path, char *const argv[],
                      char *const envp[]);
#endif
#ifdef HAVE_POSIX_SPAWN
static int call_posix_spawn(pid_t *restrict pid, const char *restrict path,
                            const posix_spawn_file_actions_t *file_actions,
                            const posix_spawnattr_t *restrict attrp,
                            char *const argv[restrict],
                            char *const envp[restrict]);
#endif
#ifdef HAVE_POSIX_SPAWNP
static int call_posix_spawnp(pid_t *restrict pid, const char *restrict file,
                             const posix_spawn_file_actions_t *file_actions,
                             const posix_spawnattr_t *restrict attrp,
                             char *const argv[restrict],
                             char *const envp[restrict]);
#endif

static void on_load(void) __attribute__((constructor));
static void on_unload(void) __attribute__((destructor));

static int mt_safe_on_load(void);
static void mt_safe_on_unload(void);

/* Initialization method to Captures the relevant environment variables.
 */
static void on_load(void)
{
    pthread_mutex_lock(&mutex);
    if (getenv(ENV_OUTPUT) == NULL)
    {
        //    fprintf(stderr, "%d, %d, initial as no\n", getpid(), getppid());
        initial_env[0] = strdup("no");
    }
    else
    {
        //    fprintf(stderr, "%d, %d, initial %s=%s\n", getpid(), getppid(), ENV_OUTPUT, getenv(ENV_OUTPUT));
        initial_env[0] = strdup(getenv(ENV_OUTPUT));
    }
    if (0 == initialized)
        initialized = mt_safe_on_load();
    pthread_mutex_unlock(&mutex);
}

static void on_unload(void)
{
    pthread_mutex_lock(&mutex);
    if (0 != initialized)
        mt_safe_on_unload();
    initialized = 0;
    pthread_mutex_unlock(&mutex);
}

static int mt_safe_on_load(void)
{
#ifdef HAVE_NSGETENVIRON
    environ = *_NSGetEnviron();
    if (0 == environ)
        return 0;
#endif
    // Capture current relevant environment variables
    return capture_env_t(&initial_env);
}

static void mt_safe_on_unload(void)
{
    release_env_t(&initial_env);
}

void print_trace(void)
{
    char **strings;
    size_t i, size;
    enum Constexpr
    {
        MAX_SIZE = 1024
    };
    void *array[MAX_SIZE];
    size = backtrace(array, MAX_SIZE);
    strings = backtrace_symbols(array, size);
    for (i = 0; i < size; i++)
        fprintf(stderr, "%s\n", strings[i]);
    free(strings);
}

/* Initialization method to Captures the relevant environment variables.
 */
#define INTERCEPT_AND_RUN(intercepted_call, original_call)          \
    {                                                               \
        if (getenv(ENV_OUTPUT) != NULL)                             \
        {                                                           \
            if (strcmp(getenv(ENV_OUTPUT), "yes") == 0)             \
            {                                                       \
                return original_call;                               \
            }                                                       \
        }                                                           \
        result r = intercept_call(file, (char const *const *)argv); \
        char *new_file = r.file;                                    \
        char **new_argv = r.argv;                                   \
        int result;                                                 \
        if (strcmp(new_file, file) != 0)                            \
        {                                                           \
            pid_t id = fork();                                      \
            if (id == 0)                                            \
            {                                                       \
                initial_env[0] = strdup("yes");                     \
                result = intercepted_call;                          \
            }                                                       \
            else                                                    \
            {                                                       \
                pid_t id2 = -1;                                     \
                do                                                  \
                {                                                   \
                    id2 = waitpid(id, NULL, 0);                     \
                } while (id2 < 0);                                  \
                if (shouldInterceptFollow)                          \
                {                                                   \
                    initial_env[0] = strdup("yes");                 \
                }                                                   \
                result = original_call;                             \
                return result;                                      \
            }                                                       \
        }                                                           \
        else                                                        \
        {                                                           \
            if (getenv(ENV_OUTPUT) != NULL)                         \
            {                                                       \
                if (strcmp(getenv(ENV_OUTPUT), "no") == 0)          \
                {                                                   \
                    initial_env[0] = strdup("no");                  \
                }                                                   \
            }                                                       \
            result = original_call;                                 \
        }                                                           \
        return result;                                              \
    }

/* These are the methods we are try to hijack.
 */

#ifdef HAVE_EXECVE
int execve(const char *file, char *const argv[], char *const envp[])
{
    INTERCEPT_AND_RUN(
        call_execve(new_argv[0], new_argv, envp),
        call_execve(file, argv, envp));
}
#endif

#ifdef HAVE_EXECV
#ifndef HAVE_EXECVE
#error can not implement execv without execve
#endif
int execv(const char *file, char *const argv[])
{
    INTERCEPT_AND_RUN(
        call_execve(new_file, new_argv, environ),
        call_execve(file, argv, environ));
}
#endif

#ifdef HAVE_EXECVPE
int execvpe(const char *file, char *const argv[], char *const envp[])
{
    INTERCEPT_AND_RUN(
        call_execvpe(new_file, new_argv, envp),
        call_execvpe(file, argv, envp));
}
#endif

#ifdef HAVE_EXECVP
int execvp(const char *file, char *const argv[])
{
    INTERCEPT_AND_RUN(
        call_execvp(new_file, new_argv),
        call_execvp(file, argv));
}
#endif

#ifdef HAVE_EXECVP2
int execvP(const char *file, const char *search_path, char *const argv[])
{
    char **new_argv = intercept_call((char const *const *)argv);
    int result = call_execvP(file, search_path, new_argv);

    // TODO: release the memory
    return result;
}
#endif

#ifdef HAVE_EXECT
int exect(const char *path, char *const argv[], char *const envp[])
{
    char **new_argv = intercept_call((char const *const *)argv);
    int result = call_exect(path, new_argv, envp);

    // TODO: release the memory
    return result;
}
#endif

#ifdef HAVE_EXECL
#ifndef HAVE_EXECVE
#error can not implement execl without execve
#endif
int execl(const char *file, const char *arg, ...)
{
    va_list args;
    va_start(args, arg);
    char const **argv = string_array_from_varargs(arg, &args);
    va_end(args);

    INTERCEPT_AND_RUN(
        call_execve(new_file, new_argv, environ),
        call_execve(file, argv, environ));
}
#endif

#ifdef HAVE_EXECLP
#ifndef HAVE_EXECVP
#error can not implement execlp without execvp
#endif
int execlp(const char *file, const char *arg, ...)
{
    va_list args;
    va_start(args, arg);
    char const **argv = string_array_from_varargs(arg, &args);
    va_end(args);

    INTERCEPT_AND_RUN(
        call_execvp(new_file, new_argv),
        call_execvp(file, argv));
}
#endif

#ifdef HAVE_EXECLE
#ifndef HAVE_EXECVE
#error can not implement execle without execve
#endif
// int execle(const char *path, const char *arg, ..., char * const envp[]);
int execle(const char *file, const char *arg, ...)
{
    va_list args;
    va_start(args, arg);
    char const **argv = string_array_from_varargs(arg, &args);
    char const **envp = va_arg(args, char const **);
    va_end(args);

    INTERCEPT_AND_RUN(
        call_execve(new_file, new_argv, (char *const *)envp),
        call_execve(file, argv, (char *const *)envp));
}
#endif

#ifdef HAVE_POSIX_SPAWN
int posix_spawn(pid_t *restrict pid, const char *restrict file,
                const posix_spawn_file_actions_t *file_actions,
                const posix_spawnattr_t *restrict attrp,
                char *const argv[restrict], char *const envp[restrict])
{

    INTERCEPT_AND_RUN(
        call_execve(new_file, new_argv, envp),
        call_posix_spawn(pid, file, file_actions, attrp, argv, envp));
}
#endif

#ifdef HAVE_POSIX_SPAWNP
int posix_spawnp(pid_t *restrict pid, const char *restrict file,
                 const posix_spawn_file_actions_t *file_actions,
                 const posix_spawnattr_t *restrict attrp,
                 char *const argv[restrict], char *const envp[restrict])
{
    INTERCEPT_AND_RUN(
        call_execve(new_file, new_argv, envp),
        call_posix_spawnp(pid, file, file_actions, attrp, argv, envp));
}
#endif

/* These are the methods which forward the call to the standard implementation.
 */

#ifdef HAVE_EXECVE
static int call_execve(const char *path, char *const argv[],
                       char *const envp[])
{
    typedef int (*func)(const char *, char *const *, char *const *);

    DLSYM(func, fp, "execve");

    /* char const **it = envp; */
    /* for (; (it) && (*it); ++it) { */
    /*   puts(*it); */
    /* } */

    char const **const menvp = string_array_partial_update(envp, &initial_env);
    int const result = (*fp)(path, argv, (char *const *)menvp);
    string_array_release(menvp);
    return result;
}
#endif

#ifdef HAVE_EXECVPE
static int call_execvpe(const char *file, char *const argv[],
                        char *const envp[])
{
    typedef int (*func)(const char *, char *const *, char *const *);

    DLSYM(func, fp, "execvpe");

    char const **const menvp = string_array_partial_update(envp, &initial_env);
    int const result = (*fp)(file, argv, (char *const *)menvp);
    string_array_release(menvp);
    return result;
}
#endif

#ifdef HAVE_EXECVP
static int call_execvp(const char *file, char *const argv[])
{
    //return call_execvpe(file, argv, environ);
    typedef int (*func)(const char *file, char *const argv[]);

    DLSYM(func, fp, "execvp");

    char **const original = environ;
    char const **const modified = string_array_partial_update(original, &initial_env);
    environ = (char **)modified;
    char const **it = modified;

    /* for (; (it) && (*it); ++it) { */
    /*   puts(*it); */
    /* } */

    int const result = (*fp)(file, argv);
    environ = original;
    string_array_release(modified);

    return result;
}
#endif

#ifdef HAVE_EXECVP2
static int call_execvP(const char *file, const char *search_path,
                       char *const argv[])
{
    typedef int (*func)(const char *, const char *, char *const *);

    DLSYM(func, fp, "execvP");

    char **const original = environ;
    char const **const modified = string_array_partial_update(original, &initial_env);
    environ = (char **)modified;
    int const result = (*fp)(file, search_path, argv);
    environ = original;
    string_array_release(modified);

    return result;
}
#endif

#ifdef HAVE_EXECT
static int call_exect(const char *path, char *const argv[],
                      char *const envp[])
{
    typedef int (*func)(const char *, char *const *, char *const *);

    DLSYM(func, fp, "exect");

    char const **const menvp = string_array_partial_update(envp, &initial_env);
    int const result = (*fp)(path, argv, (char *const *)menvp);
    string_array_release(menvp);
    return result;
}
#endif

#ifdef HAVE_POSIX_SPAWN
static int call_posix_spawn(pid_t *restrict pid, const char *restrict path,
                            const posix_spawn_file_actions_t *file_actions,
                            const posix_spawnattr_t *restrict attrp,
                            char *const argv[restrict],
                            char *const envp[restrict])
{
    typedef int (*func)(pid_t * restrict, const char *restrict,
                        const posix_spawn_file_actions_t *,
                        const posix_spawnattr_t *restrict,
                        char *const *restrict, char *const *restrict);

    DLSYM(func, fp, "posix_spawn");

    char const **const menvp = string_array_partial_update(envp, &initial_env);
    int const result =
        (*fp)(pid, path, file_actions, attrp, argv, (char *const *restrict)menvp);
    string_array_release(menvp);
    return result;
}
#endif

#ifdef HAVE_POSIX_SPAWNP
static int call_posix_spawnp(pid_t *restrict pid, const char *restrict file,
                             const posix_spawn_file_actions_t *file_actions,
                             const posix_spawnattr_t *restrict attrp,
                             char *const argv[restrict],
                             char *const envp[restrict])
{
    typedef int (*func)(pid_t * restrict, const char *restrict,
                        const posix_spawn_file_actions_t *,
                        const posix_spawnattr_t *restrict,
                        char *const *restrict, char *const *restrict);

    DLSYM(func, fp, "posix_spawnp");

    char const **const menvp = string_array_partial_update(envp, &initial_env);
    int const result =
        (*fp)(pid, file, file_actions, attrp, argv, (char *const *restrict)menvp);
    string_array_release(menvp);
    return result;
}
#endif

/* update environment assure that chilren processes will copy the desired
 * behaviour */

static int capture_env_t(bear_env_t *env)
{

    //    for (size_t it = 0; it < ENV_SIZE; ++it) {
    //        char const * const env_value = getenv(env_names[it]);
    //        if (0 == env_value) {
    //            PERROR("getenv");
    //            return 0;
    //        }
    //
    //        char const * const env_copy = strdup(env_value);
    //        if (0 == env_copy) {
    //            PERROR("strdup");
    //            return 0;
    //        }
    //
    //        (*env)[it] = env_copy;
    //    }
    return 1;
}

static void release_env_t(bear_env_t *env)
{
    for (size_t it = 0; it < ENV_SIZE; ++it)
    {
        free((void *)(*env)[it]);
        (*env)[it] = 0;
    }
}

static char const **string_array_partial_update(char *const envp[], bear_env_t *env)
{
    char const **result = string_array_copy((char const **)envp);
    for (size_t it = 0; it < ENV_SIZE && (*env)[it]; ++it)
    {
        result = string_array_single_update(result, env_names[it], (*env)[it]);
    }
    return result;
}

static char const **string_array_single_update(char const *envs[], char const *key, char const *const value)
{
    // find the key if it's there
    size_t const key_length = strlen(key);
    char const **it = envs;
    for (; (it) && (*it); ++it)
    {
        if ((0 == strncmp(*it, key, key_length)) &&
            (strlen(*it) > key_length) && ('=' == (*it)[key_length]))
            break;
    }
    // allocate a environment entry
    size_t const value_length = strlen(value);
    size_t const env_length = key_length + value_length + 2;
    char *env = malloc(env_length);
    // fprintf(stderr, "%d, %d, updating %s=%s\n",getpid(), getppid(), key, value);

    if (0 == env)
        ERROR_AND_EXIT("malloc");
    if (-1 == snprintf(env, env_length, "%s=%s", key, value))
        ERROR_AND_EXIT("snprintf");
    // replace or append the environment entry
    if (it && *it)
    {
        free((void *)*it);
        *it = env;
        return envs;
    }
    else
    {
        size_t const size = string_array_length(envs);
        char const **result = realloc(envs, (size + 2) * sizeof(char const *));
        if (0 == result)
            ERROR_AND_EXIT("realloc");
        result[size] = env;
        result[size + 1] = 0;
        return result;
    }
}

/* util methods to deal with string arrays. environment and process arguments
 * are both represented as string arrays. */

static char const **string_array_from_varargs(char const *const arg, va_list *args)
{
    char const **result = 0;
    size_t size = 0;
    for (char const *it = arg; it; it = va_arg(*args, char const *))
    {
        result = realloc(result, (size + 1) * sizeof(char const *));
        if (0 == result)
            ERROR_AND_EXIT("realloc");
        char const *copy = strdup(it);
        if (0 == copy)
            ERROR_AND_EXIT("strdup");
        result[size++] = copy;
    }
    result = realloc(result, (size + 1) * sizeof(char const *));
    if (0 == result)
        ERROR_AND_EXIT("realloc");
    result[size++] = 0;

    return result;
}

static char const **string_array_copy(char const **const in)
{
    size_t const size = string_array_length(in);

    char const **const result = malloc((size + 1) * sizeof(char const *));
    if (0 == result)
        ERROR_AND_EXIT("malloc");

    char const **out_it = result;
    for (char const *const *in_it = in; (in_it) && (*in_it);
         ++in_it, ++out_it)
    {
        *out_it = strdup(*in_it);
        if (0 == *out_it)
            ERROR_AND_EXIT("strdup");
    }
    *out_it = 0;
    return result;
}

static size_t string_array_length(char const *const *const in)
{
    size_t result = 0;
    for (char const *const *it = in; (it) && (*it); ++it)
        ++result;
    return result;
}

static void string_array_release(char const **in)
{
    for (char const *const *it = in; (it) && (*it); ++it)
    {
        free((void *)*it);
    }
    free((void *)in);
}
