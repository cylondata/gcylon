#!/bin/bash

# export CYLON_HOME=/path/to/cylon_home
# export CUDACXX=/path/to/nvcc
# export CC=/path/to/gcc
# export CXX=/path/to/c++

./build.sh --conda_cpp --install_path ${BUILD_PREFIX}
# ./build.sh --conda_cpp --install_path ${BUILD_PREFIX}/cbuild --build_path ${BUILD_PREFIX}/cbuild
# ./build.sh --conda_cpp