#!/bin/bash

# this script is running within docker
#
# build/
#    LLVMRace/       # source code
#    llvm10/
#      clang-10.0.0/ # full package of clang 10.0.1
#    package/        # package
#

# prepare directories
cd /build/LLVMRace
rm -fr build
mkdir build
cd build

cmake -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=/build/llvm10/clang-10.0.0/lib/cmake/llvm \
      -DENABLE_TEST=Off ..  
make -j $(nproc) 
#make test > unittest.log 2>&1
#if [ $? != 0 ];then
#	echo "Unit test failed, pls check."
#	exit 2
#else
#	echo "Unit test succeed."
#fi

cp bin/racedetect /build/package/bin/
cp -r bin/stub /build/package/bin/

