#!/bin/sh

cd bin

# load .so lib
export LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH
./main