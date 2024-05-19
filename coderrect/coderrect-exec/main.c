#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/wait.h>


/**
 * coderrect-exec <command> <options>
 *
 * For example
 *    coderrect-exec find . -name "*.c" -print
 */
int main(int argc, char* argv[]) {
    char **myArgs = (char**) malloc(sizeof(char*) * argc);
    int i = 0;
    pid_t pid;

    for (i = 0; i < argc-1; i++) {
        myArgs[i] = argv[i+1];
    }
    myArgs[argc-1] = NULL;

    pid = fork();
    if (pid < 0) {
        fprintf(stderr, "%s\n", strerror(errno));
    }
    else if (pid == 0) {
        execvp(myArgs[0], myArgs);

        // if execv() succeeds, it won't return. That is,
        // we encounter an error if this piece of code is
        // executed.
        fprintf(stderr, "%s\n", strerror(errno));
        return 1;
    }

    int status;
    if ( waitpid(pid, &status, 0) == -1 ) {
        perror("waitpid() failed");
        exit(EXIT_FAILURE);
    }

    int es;
    if ( WIFEXITED(status) ) {
        es = WEXITSTATUS(status);
    }
    return es;
}