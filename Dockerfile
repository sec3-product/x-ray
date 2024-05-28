FROM ubuntu:20.04

# ARG CODERRECT_LOCAL_PATH=coderrect-linux-develop.tar.gz
# ARG CODERRECT_LOCAL_FILENAME=coderrect-linux-develop

# # install coderrect
# ADD ${CODERRECT_LOCAL_PATH} /opt
# ENV PATH="/opt/${CODERRECT_LOCAL_FILENAME}/bin:${PATH}" 

# # prepare source
# RUN mkdir /src && cd /src
# WORKDIR /src

# install  necessary packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:longsleep/golang-backports && \
    apt-get update && \
    apt-get install -y \
    build-essential \        
    golang-go \              
    # make \                   # Utility for directing compilation
    # g++ \                    # GNU C++ compiler
    # libssl-dev \             # Development files for OpenSSL
    zlib1g-dev \             
    libxml2-dev
    
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

# Set environment variable for Go
ENV PATH=$PATH:/usr/local/go/bin

# Verify the installations
RUN go version && \
    make --version && \
    cmake --version && \
    g++ --version && \
    git --version && \
    wget --version

# Set the working directory
WORKDIR /workspace

# Default command to start an interactive shell
CMD ["/bin/bash"]