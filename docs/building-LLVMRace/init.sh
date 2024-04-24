##
## OS: Ubuntu 20.04, ARM64
##

## Speedup apt-get with a faster mirror (optional)
# sed -i 's/htt[p|ps]:\/\/ports.ubuntu.com\/ubuntu-ports\//http:\/\/mirror.kumi.systems\/ubuntu-ports/g' /etc/apt/sources.list

## install dependence
apt-get update
apt install git
apt install vim
apt install make
apt install g++
apt install wget
apt install python3
apt install libssl-dev
apt install zlib1g-dev
apt install libxml2-dev

## install CMake
wget https://github.com/Kitware/CMake/releases/download/v3.26.0-rc1/cmake-3.26.0-rc1.tar.gz
tar xzvf cmake*.gz
cd cmake-3.26.0-rc1
./bootstrap
make
make install

## Build LLVM Locally
git clone https://github.com/llvm/llvm-project.git
mkdir -p llvm-project/build
cd llvm-project/build
git checkout release/12.x
cmake -DLLVM_ENABLE_PROJECTS="clang;openmp;compiler-rt;lld;mlir" ../llvm/ -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Release
time make -j 32
