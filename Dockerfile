FROM ubuntu:20.04

# ARG CODERRECT_LOCAL_PATH=coderrect-linux-develop.tar.gz
# ARG CODERRECT_LOCAL_FILENAME=coderrect-linux-develop

# # install coderrect
# ADD ${CODERRECT_LOCAL_PATH} /opt
# ENV PATH="/opt/${CODERRECT_LOCAL_FILENAME}/bin:${PATH}" 

# # prepare source
# RUN mkdir /src && cd /src
# WORKDIR /src

# install necessary packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:longsleep/golang-backports && \
    apt-get update && \
    apt-get install -y \
    golang-go \      
    wget \       
    # autotools-dev \
    # automake \
    # ca-certificates \
    # mpich \
    git \
    vim    

#################################################################
# TEMP: copy ssh key, for downloading private repos
RUN apt-get update && apt-get install -y \
    openssh-client

RUN mkdir -p /root/.ssh
COPY id_rsa /root/.ssh/id_rsa
COPY id_rsa.pub /root/.ssh/id_rsa.pub

# RUN chmod 600 /root/.ssh/id_rsa
# RUN chmod 644 /root/.ssh/id_rsa.pub
#################################################################

# install c++ depends
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    autoconf \
    cmake \
    # g++ \                    # GNU C++ compiler
    clang

# install specific
RUN apt-get update && apt-get install --no-install-recommends -y \
    libevent-dev \
    zlib1g-dev \             
    libssl-dev \   
    libxml2-dev \
    libbz2-dev \
    libomp-dev  \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable for Go
ENV PATH=$PATH:/usr/local/go/bin

ENV DEBIAN_FRONTEND=noninteractive

# # for LLVM 12.0.1 image
# RUN apt-get update && apt-get install -y \
#     gnupg \
#     software-properties-common \
#     build-essential \
#     python3 \
#     python3-pip \
#     ninja-build \
#     zlib1g-dev \
#     libtinfo-dev \
#     libncurses5-dev

# # pre-compiled LLVM 12.0.1 (with MLIR)
# RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
#     tar -xf clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
#     mv clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu- /usr/local/llvm-12 && \
#     ln -s /usr/local/llvm-12/bin/llvm-config /usr/bin/llvm-config-12 && \
#     ln -s /usr/local/llvm-12/bin/clang /usr/bin/clang-12 && \
#     ln -s /usr/local/llvm-12/bin/clang++ /usr/bin/clang++-12 && \
#     ln -s /usr/local/llvm-12/bin/mlir* /usr/bin/

# # Verify the installations
# RUN go version && \
#     cmake --version && \
#     git --version

# # Verify the installations: LLVM
# RUN clang-12 --version
# RUN llvm-config-12 --version

# quick install CMake 3.26
ARG CMAKE_VERSION=3.26.0

RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

ARG MAKE_THREADS=32

# build llvm12 from src
RUN git clone --branch release/12.x --single-branch https://github.com/llvm/llvm-project.git \
# RUN git clone https://github.com/llvm/llvm-project.git \
    && mkdir -p llvm-project/build \
    && cd llvm-project/build \
#    && git checkout release/12.x \
    && cmake -DLLVM_ENABLE_PROJECTS="clang;openmp;compiler-rt;lld;mlir" ../llvm/ -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Release \
    && make -j${MAKE_THREADS}

# build x-ray detector
RUN git clone git@github.com:sec3-product/x-ray-toolchain.git \
    && mkdir -p x-ray-toolchain/code-detector/build \
    && cd x-ray-toolchain/code-detector/build \
    && cmake .. \
    && make -j${MAKE_THREADS}

# Set the working directory
WORKDIR /workspace

# Default command to start an interactive shell
CMD ["/bin/bash"]

