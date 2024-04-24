# Introduction

Each .cpp file corresponds to a test case. 

The developer needs to write the purpose and the expected results in the comment. Below 
is an example.

```
// @purpose A simple single-thread hello-world program without any race
// @races 0
```

# Run the benchmark

You can run the whole benchmark with the command below

```
$ benchmark
```

Or you can run a specific case with the command below

```
$ benchmark helloworld
```

There must be a file called helloworld.c (or helloworld.cpp).
