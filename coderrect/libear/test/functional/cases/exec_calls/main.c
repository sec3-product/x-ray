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

#include "config.h"

#include <sys/wait.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <paths.h>

#if defined HAVE_POSIX_SPAWN || defined HAVE_POSIX_SPAWNP
#include <spawn.h>
#endif

// ..:: environment access fixer - begin ::..
#ifdef HAVE_NSGETENVIRON
#include <crt_externs.h>
#else
extern char **environ;
#endif

char **get_environ() {
#ifdef HAVE_NSGETENVIRON
    return *_NSGetEnviron();
#else
    return environ;
#endif
}
// ..:: environment access fixer - end ::..

// ..:: test fixtures - begin ::..
static char const *cwd = NULL;
static FILE *fd = NULL;
static int need_comma = 0;

void expected_out_open(const char *expected) {
    cwd = getcwd(NULL, 0);
    fd = fopen(expected, "w");
    if (!fd) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }
    fprintf(fd, "[\n");
    need_comma = 0;
}

void expected_out_close() {
    fprintf(fd, "]\n");
    fclose(fd);
    fd = NULL;

    free((void *)cwd);
    cwd = NULL;
}

void expected_out(const char * compiler, const char *file) {
    if (need_comma)
        fprintf(fd, ",\n");
    else
        need_comma = 1;

    fprintf(fd, "{\n");
    fprintf(fd, "  \"directory\": \"%s\",\n", cwd);
    fprintf(fd, "  \"command\": \"%s -c %s\",\n", compiler, file);
    fprintf(fd, "  \"file\": \"%s\"\n", file);
    fprintf(fd, "}\n");
}

void create_source(char *file) {
    FILE *fd = fopen(file, "w");
    if (!fd) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }
    fprintf(fd, "typedef int score;\n");
    fclose(fd);
}

typedef void (*exec_fun)();

void wait_for(pid_t child) {
    int status;
    if (-1 == waitpid(child, &status, 0)) {
        perror("wait");
        exit(EXIT_FAILURE);
    }
    if (WIFEXITED(status) ? WEXITSTATUS(status) : EXIT_FAILURE) {
        fprintf(stderr, "children process has non zero exit code\n");
        exit(EXIT_FAILURE);
    }
}

#define FORK(FUNC)                                                             \
    {                                                                          \
        pid_t child = fork();                                                  \
        if (-1 == child) {                                                     \
            perror("fork");                                                    \
            exit(EXIT_FAILURE);                                                \
        } else if (0 == child) {                                               \
            FUNC fprintf(stderr, "children process failed to exec\n");         \
            exit(EXIT_FAILURE);                                                \
        } else {                                                               \
            wait_for(child);                                                   \
        }                                                                      \
    }
// ..:: test fixtures - end ::..

#ifdef HAVE_EXECV
void call_execv() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "execv.c";
    char *const compiler = "/usr/bin/cc";
    char *const argv[] = {compiler, "-c", file, 0};

    expected_out(compiler, file);
    create_source(file);

    FORK(execv(compiler, argv);)
}
#endif

#ifdef HAVE_EXECVE
void call_execve() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "execve.c";
    char *const compiler = "/usr/bin/cc";
    char *const argv[] = {compiler, "-c", file, 0};
    char *const envp[] = {"THIS=THAT", 0};

    expected_out(compiler, file);
    create_source(file);

    FORK(execve(compiler, argv, envp);)
}
#endif

#ifdef HAVE_EXECVP
void call_execvp() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "execvp.c";
    char *const compiler = "cc";
    char *const argv[] = {compiler, "-c", file, 0};

    expected_out(compiler, file);
    create_source(file);

    FORK(execvp(compiler, argv);)
}
#endif

#ifdef HAVE_EXECVP2
void call_execvP() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "execv_p.c";
    char *const compiler = "cc";
    char *const argv[] = {compiler, "-c", file, 0};

    expected_out(compiler, file);
    create_source(file);

    FORK(execvP(compiler, _PATH_DEFPATH, argv);)
}
#endif

#ifdef HAVE_EXECVPE
void call_execvpe() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "execvpe.c";
    char *const compiler = "/usr/bin/cc";
    char *const argv[] = {compiler, "-c", file, 0};
    char *const envp[] = {"THIS=THAT", 0};

    expected_out(compiler, file);
    create_source(file);

    FORK(execvpe(compiler, argv, envp);)
}
#endif

#ifdef HAVE_EXECT
void call_exect() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "exect.c";
    char *const compiler = "/usr/bin/cc";
    char *const argv[] = {compiler, "-c", file, 0};
    char *const envp[] = {"THIS=THAT", 0};

    expected_out(compiler, file);
    create_source(file);

    FORK(exect(compiler, argv, envp);)
}
#endif

#ifdef HAVE_EXECL
void call_execl() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "execl.c";
    char *const compiler = "/usr/bin/cc";

    expected_out("cc", file);
    create_source(file);

    FORK(execl(compiler, "cc", "-c", file, (char *)0);)
}
#endif

#ifdef HAVE_EXECLP
void call_execlp() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "execlp.c";
    char *const compiler = "cc";

    expected_out(compiler, file);
    create_source(file);

    FORK(execlp(compiler, compiler, "-c", file, (char *)0);)
}
#endif

#ifdef HAVE_EXECLE
void call_execle() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "execle.c";
    char *const compiler = "/usr/bin/cc";
    char *const envp[] = {"THIS=THAT", 0};

    expected_out(compiler, file);
    create_source(file);

    FORK(execle(compiler, compiler, "-c", file, (char *)0, envp);)
}
#endif

#ifdef HAVE_POSIX_SPAWN
void call_posix_spawn() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "posix_spawn.c";
    char *const compiler = "cc";
    char *const argv[] = {compiler, "-c", file, 0};

    expected_out(compiler, file);
    create_source(file);

    pid_t child;
    if (0 != posix_spawn(&child, "/usr/bin/cc", 0, 0, argv, get_environ())) {
        perror("posix_spawn");
        exit(EXIT_FAILURE);
    }
    wait_for(child);
}
#endif

#ifdef HAVE_POSIX_SPAWNP
void call_posix_spawnp() {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    char *const file = "posix_spawnp.c";
    char *const compiler = "cc";
    char *const argv[] = {compiler, "-c", file, 0};

    expected_out(compiler, file);
    create_source(file);

    pid_t child;
    if (0 != posix_spawnp(&child, "cc", 0, 0, argv, get_environ())) {
        perror("posix_spawnp");
        exit(EXIT_FAILURE);
    }
    wait_for(child);
}
#endif

int main(int argc, char *const argv[]) {

    char *workdir = NULL;
    char *output = NULL;
    int c = 0;

    opterr = 0;
    while ((c = getopt (argc, argv, "C:o:")) != -1) {
        switch (c) {
            case 'C':
                workdir = optarg;
                break;
            case 'o':
                output = optarg;
                break;
            case '?':
                if (optopt == 'C' || optopt == 'o')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                return 1;
            default:
                abort();
        }
    }
    for (int index = optind; index < argc; ++index)
        printf ("Non-option argument %s\n", argv[index]);

    if (workdir != NULL) {
        chdir(workdir);
    }
    expected_out_open(output);
#ifdef HAVE_EXECV
    call_execv();
#endif
#ifdef HAVE_EXECVE
    call_execve();
#endif
#ifdef HAVE_EXECVP
    call_execvp();
#endif
#ifdef HAVE_EXECVP2
    call_execvP();
#endif
#ifdef HAVE_EXECVPE
    call_execvpe();
#endif
#ifdef HAVE_EXECT
    call_exect();
#endif
#ifdef HAVE_EXECL
    call_execl();
#endif
#ifdef HAVE_EXECLP
    call_execlp();
#endif
#ifdef HAVE_EXECLE
    call_execle();
#endif
#ifdef HAVE_POSIX_SPAWN
    call_posix_spawn();
#endif
#ifdef HAVE_POSIX_SPAWNP
    call_posix_spawnp();
#endif
    expected_out_close();
    return 0;
}
