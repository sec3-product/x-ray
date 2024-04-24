//
// Created by peiming on 2/6/20.
//
#ifndef BEAR_INTERCEPT_H
#define BEAR_INTERCEPT_H

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

// bridge between C and C++
EXTERNC struct intercepted_result {
  char * file;
  char **argv;
};

typedef struct intercepted_result result;

EXTERNC int shouldInterceptFollow;

EXTERNC result intercept_call(const char *file, char const *const argv[]);

#undef EXTERNC

#endif //BEAR_INTERCEPT_H
