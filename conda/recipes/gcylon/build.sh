#!/bin/bash

# export CYLON_HOME=/path/to/cylon_home
# export GCYLON_HOME=/path/to/gcylon_home
# export CUDACXX=/path/to/nvcc
# export CC=/path/to/gcc
# export CXX=/path/to/c++

./build.sh --conda_cpp --build_path ${pwd}/cbuild
#./build.sh --conda_cpp --install_path ${BUILD_PREFIX}