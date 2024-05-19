#!/bin/bash

cd /src
rm -rf bin/
mkdir bin
/usr/local/clang_9.0.0/bin/clang -o bin/coderrect-exec main.c
if [ $? -ne 0 ]; then
    exit 1
fi

cp bin/coderrect-exec /build/package/bin/.

