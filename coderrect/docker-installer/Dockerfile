FROM ubuntu:18.04

ARG CODERRECT_LOCAL_PATH=coderrect-linux-develop.tar.gz
ARG CODERRECT_LOCAL_FILENAME=coderrect-linux-develop

# install coderrect
ADD ${CODERRECT_LOCAL_PATH} /opt
ENV PATH="/opt/${CODERRECT_LOCAL_FILENAME}/bin:${PATH}" 

# prepare source
RUN mkdir /src && cd /src
WORKDIR /src

# install general depends
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    git \
    autotools-dev \
    automake \
    ca-certificates \
    mpich

# install c++ depends
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    autoconf \
    cmake \
    clang

# install specific
RUN apt-get update && apt-get install --no-install-recommends -y \
    libevent-dev \ 
    libssl-dev \
    libbz2-dev \
    libomp-dev  \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
