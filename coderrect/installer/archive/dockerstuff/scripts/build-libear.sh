#!/bin/bash

cd /src
rm -rf cmake-build
mkdir cmake-build 
cd cmake-build 
cmake .. 
if [ $? -ne 0 ]; then
    exit 1
fi
make
if [ $? -ne 0 ]; then
    exit 1
fi

cp libear/libear.so /build/package/bin/.

