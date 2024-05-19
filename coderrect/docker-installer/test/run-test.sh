#!/bin/bash

workingDir=$(pwd)
imageName=coderrect/coderrect:linux-develop

echo "**********************************"
echo 'Testing hello'
echo "**********************************"
cd hello
../../docker-coderrect -i $imageName gcc hello.c

echo "**********************************"
echo 'Testing hellomake'
echo "**********************************"
cd ${workingDir}/hello-make
make clean && ../../docker-coderrect -i $imageName make

echo "**********************************"
echo 'Testing memcached'
echo "**********************************"
cd ${workingDir}
unzip -o memcached-master.zip
cd memcached-master
./autogen.sh
./configure
# docker run --rm --user=$(id -u):$(id -g) -v $(pwd):/src -it coderrect/coderrect:linux-x64-dev 
../../docker-coderrect -i $imageName -e memcached make -j $(nproc)
cd .. && rm -rf memcached-master