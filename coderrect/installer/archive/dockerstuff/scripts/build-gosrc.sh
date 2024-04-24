#!/bin/bash

cd /src
make all
if [ $? -ne 0 ]; then
    exit 1
fi

cp bin/* /build/package/bin/.

