#!/bin/bash

# build folder layout
#  /build
#    classic-flang-llvm-project/    # clang/flang-driver/llvm
#    flang/                         # flang 
#    coderrect/                     # go executables/libear/coderrect-exec
#    LLVMRace/                      # racedetect

# The dedicated path to install clang/flang/flang-driver/llvm
INSTALL_PREFIX=/build/custom-clang

CLANG_OPTIONS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DLLVM_STATIC_LINK_CXX_STDLIB:BOOL=On \
    -DLLVM_ENABLE_LIBCXX=On \
    -DLLVM_ENABLE_CLASSIC_FLANG=On \
    -DLLVM_ENABLE_PROJECTS='clang;openmp' \
    -DLLVM_ENABLE_TERMINFO=Off \
    -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=On \
    -DUSE_Z3_SOLVER=0 \
    -DLLVM_ENABLE_Z3_SOLVER=Off \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DLLVM_TARGETS_TO_BUILD=X86"

LIBPGMATH_OPTIONS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_CXX_COMPILER=$INSTALL_PREFIX/bin/clang++ \
    -DCMAKE_C_COMPILER=$INSTALL_PREFIX/bin/clang \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"

FLANG_OPTIONS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DLLVM_CONFIG=$INSTALL_PREFIX/bin/llvm-config \
    -DCMAKE_CXX_COMPILER=$INSTALL_PREFIX/bin/clang++ \
    -DCMAKE_C_COMPILER=$INSTALL_PREFIX/bin/clang \
    -DCMAKE_Fortran_COMPILER=$INSTALL_PREFIX/bin/flang \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_TERMINFO=Off \
    -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=On \
    -DUSE_Z3_SOLVER=0 \
    -DLLVM_ENABLE_Z3_SOLVER=Off \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DFLANG_OPENMP_GPU_NVIDIA=ON"

echo "Start building classic-flang-llvm-project..."
cd /build/classic-flang-llvm-project
mkdir -p build && cd build
cmake $CLANG_OPTIONS ../llvm
make -j $(nproc)
make install
if [ $? -ne 0 ]; then
    exit 1
fi
echo "Finish building classic-flang-llvm-project..."

echo "Start building libpgmath for flang..."
cd /build/flang/runtime/libpgmath
mkdir -p build && cd build
cmake $LIBPGMATH_OPTIONS ..
make -j $(nproc)
make install
if [ $? -ne 0 ]; then
    exit 1
fi
echo "Finish building libpgmath..."

echo "Start building flang..."
cd /build/flang
mkdir -p build && cd build
cmake $FLANG_OPTIONS ..
make -j $(nproc)
make install
if [ $? -ne 0 ]; then
    exit 1
fi
echo "Finish building flang..."

echo "Start building LLVMRace..."
cd /build/LLVMRace
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=$INSTALL_PREFIX/lib/cmake/llvm \
      -DENABLE_TEST=Off ..
make -j $(nproc)
if [ $? -ne 0 ]; then
    exit 1
fi
echo "Finish building LLVMRace..."

echo "Start building libear..."
cd /build/coderrect/libear/
mkdir -p build && cd build
# it is important to specify gcc/g++ version 4.8 (which comes with ubuntu 14.04)
# because this lowers the required glibcxx version
cmake -DCMAKE_C_COMPILER=gcc-4.8 -DCMAKE_CXX_COMPILER=g++-4.8 .. && make
if [ $? -ne 0 ]; then
    exit 1
fi
echo "Finish building libear..."

echo "Start building coderrect-exec..."
cd /build/coderrect/coderrect-exec/
clang -o coderrect-exec main.c
if [ $? -ne 0 ]; then
    exit 1
fi
echo "Finish building coderrect-exec..."

echo "Start building go executables..."
mkdir -p /build/go
cd /build/coderrect/gosrc/
make all
if [ $? -ne 0 ]; then
    exit 1
fi
echo "Finish building go executables..."
