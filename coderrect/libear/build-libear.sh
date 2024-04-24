#!/bin/bash

cd /build
rm -fr cmake-build
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
