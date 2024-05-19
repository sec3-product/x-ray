FROM ubuntu:16.04 as base

MAINTAINER Jie Song


ENV DEBIAN_FRONTEND=noninteractive 

# Install dependencies
RUN apt-get -qq update; \
    apt-get install -qqy --no-install-recommends \
        autoconf automake cmake git make build-essential ca-certificates \
        awscli \
        libz-dev libtinfo-dev wget xz-utils; \
    rm -fr /var/lib/apt/lists/*

# install g++-7(COD-676)
#RUN apt-get update; \
#    apt-get install -y software-properties-common; \
#    add-apt-repository -y ppa:ubuntu-toolchain-r/test; \
#    apt update; \
#    apt install g++-7 -y

# install clang
# 
ENV LLVM_RELEASE 9
ENV LLVM_VERSION 9.0.0 

RUN wget http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz; \
    tar xf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz; \
    rm clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz; \
    mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04 /usr/local/clang_9.0.0

# install cmake
#
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5-Linux-x86_64.tar.gz; \
    tar xzvf cmake-3.16.5-Linux-x86_64.tar.gz; \
    rm cmake-3.16.5-Linux-x86_64.tar.gz; \
    mv cmake-3.16.5-Linux-x86_64 /usr/local/

# install go
#
RUN wget --no-check-certificate https://dl.google.com/go/go1.13.4.linux-amd64.tar.gz; \
    tar zxvf go1.13.4.linux-amd64.tar.gz; \
    rm go1.13.4.linux-amd64.tar.gz; \
    mv go /usr/local/.

# install maven
#
RUN apt update; \
    apt -y install maven

# install jdk
# 
RUN cd /usr/local; \
    wget https://cr-qa-automation.s3-us-west-2.amazonaws.com/jdk-8u202-linux-x64.tar.gz; \
    tar -xzvf jdk-8u202-linux-x64.tar.gz; \
    rm -f jdk-8u202-linux-x64.tar.gz;

ENV JAVA_HOME=/usr/local/jdk1.8.0_202

ENV PATH=/usr/local/jdk1.8.0_202/bin:/usr/local/go/bin:/usr/local/clang_9.0.0/bin:/usr/local/cmake-3.16.5-Linux-x86_64/bin:$PATH
ENV GOROOT=/usr/local/go
ENV GOPATH=/build/go 

RUN git config --global http.sslVerify false

