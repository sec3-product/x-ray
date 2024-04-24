#!/bin/bash

# This check may not be needed anymore
# if [ ! -d build/llvm11/clang-11.0.0 ]
# then
#     echo "You need to build clang 11 first"
#     exit 1
# fi

imageExist=$(docker images | grep 'coderrect/installer' | grep 1.1 | wc -l)
if [ $imageExist -gt 0 ]
then
	echo "You need to delete coderrect/installer:1.1 docker image first"
	exit 2
fi

docker build -f dockerstuff/Dockerfile.installer.1.1 -t coderrect/installer:1.1 .


