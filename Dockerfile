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
    # autotools-dev \
    # automake \
    # ca-certificates \
    # mpich \
    git \
    vim    

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

# for LLVM 12.0.1 image
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    ninja-build \
    zlib1g-dev \
    libtinfo-dev \
    libncurses5-dev

# pre-compiled LLVM 12.0.1 (with MLIR)
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    tar -xf clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    mv clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu- /usr/local/llvm-12 && \
    ln -s /usr/local/llvm-12/bin/llvm-config /usr/bin/llvm-config-12 && \
    ln -s /usr/local/llvm-12/bin/clang /usr/bin/clang-12 && \
    ln -s /usr/local/llvm-12/bin/clang++ /usr/bin/clang++-12 && \
    ln -s /usr/local/llvm-12/bin/mlir* /usr/bin/

# Verify the installations
RUN go version && \
    cmake --version && \
    git --version

# Verify the installations: LLVM
RUN clang-12 --version
RUN llvm-config-12 --version


# Set the working directory
WORKDIR /workspace

# Default command to start an interactive shell
CMD ["/bin/bash"]

