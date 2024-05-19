#!/bin/bash

# /build is mapped to host's installer/build/llvm10
#
# folder layout
#  /build
#    clang-10.0.0/                  # the compiled package
#    custom_clang/                  # all needed files in our final tarball custom-clang-10.tar.gz
#    classic-flang-llvm-project/    # develop branch of coderrect-inc/classic-flang-llvm-project repo
#    flang/                         # develop branch of coderrect-inc/flang repo
#

INSTALL_PREFIX=/build/clang-10.0.0
TAR_DIR=/build/custom_clang

rm -fr $INSTALL_PREFIX

CMAKE_OPTIONS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
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

# Build flang's llvm and clang
cd /build/classic-flang-llvm-project
rm -fr build && mkdir -p build && cd build
cmake $CMAKE_OPTIONS -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -DLLVM_STATIC_LINK_CXX_STDLIB:BOOL=On \
      -DLLVM_ENABLE_LIBCXX=On \
      -DLLVM_ENABLE_CLASSIC_FLANG=ON \
      -DLLVM_ENABLE_PROJECTS="clang;openmp" \
      ../llvm
make -j $(nproc)
make install

# Build LLVMRace - we use a separate script to build LLVMRace
# cd /build/LLVMRace
# rm -fr build && mkdir -p build && cd build
# cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_DIR=$INSTALL_PREFIX/lib/cmake/llvm/ ..
# make -j $(nproc)

# Build Flang
cd /build/flang/runtime/libpgmath
rm -fr build && mkdir -p build && cd build
cmake $CMAKE_OPTIONS ..
make -j $(nproc)
make install 

cd /build/flang
rm -fr build && mkdir -p build && cd build
cmake $CMAKE_OPTIONS ..
make -j $(nproc)
make install 


echo "Copy binaries ..."
rm -fr $TAR_DIR 
mkdir -p $TAR_DIR
cd $TAR_DIR
mkdir bin
cp $INSTALL_PREFIX/bin/clang-10 bin/.
cp $INSTALL_PREFIX/bin/llvm-link bin/.
(cd bin && ln -s clang-10 clang && ln -s clang-10 clang++)


echo "Copy lib directory ..."
cd $TAR_DIR
mkdir lib
cp -r $INSTALL_PREFIX/lib/clang lib/.
cp -r $INSTALL_PREFIX/lib/cmake lib/.

echo "Copy the include directory ..."
cd $TAR_DIR
cp -r $INSTALL_PREFIX/include . 

echo "Copy bin directory ..."
cd $TAR_DIR
cp $INSTALL_PREFIX/bin/flang1 bin/.
cp $INSTALL_PREFIX/bin/flang2 bin/.
cd bin
ln -s clang flang


# now we are under '/build'
cd /build
tar -czvf custom_clang_10.tar.gz custom_clang
