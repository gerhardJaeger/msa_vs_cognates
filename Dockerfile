# Use an official Ubuntu base image
FROM ubuntu:latest

# Set a non-interactive frontend for apt to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install the required packages
RUN apt-get update && apt-get install -y \
    g++ \
    gfortran \
    git \
    libblas-dev \
    cmake \
    wget \
    unzip \
    flex \
    bison \
    libgmp3-dev \
    python3 \
    python3-pip \
    mrbayes-mpi \
    autoconf \
    libtool \
    pkg-config \
    r-base \
    r-cran-ape \
    r-cran-tidyverse \
    && rm -rf /var/lib/apt/lists/*

# Install the required python packages

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas lingpy ete3 tqdm lingrex
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Julia by downloading the specific version from the official website
RUN mkdir /opt/julia-1.9.4 && \
    cd /opt/julia-1.9.4 && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz && \
    tar -xzf julia-1.9.4-linux-x86_64.tar.gz --strip-components=1 && \
    ln -s /opt/julia-1.9.4/bin/julia /usr/local/bin/julia

# Create a directory for your project and copy the Project.toml there
WORKDIR /root/.julia/environments/my_project
COPY code/Project.toml .


# Install the Julia packages specified in the Project.toml
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'


# Download the qdist source code
WORKDIR /tmp
RUN wget -q https://users-birc.au.dk/cstorm/software/qdist/qdist-src-2.0.zip

# Unpack the source code and build qdist
RUN unzip qdist-src-2.0.zip 
RUN sed -i '/^#include/ i #include <cstddef>' Tree.hpp
RUN cmake .
RUN make
RUN make test
RUN make install


# Clone, build, and install raxml-ng
WORKDIR /tmp
RUN git clone --recursive https://github.com/amkozlov/raxml-ng && \
    cd raxml-ng && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install



# RUN mv bin/raxml-ng /usr/local/bin/

# Set the default command for the container
CMD ["/usr/bin/bash"]
