# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

##########################################################################
#	CMake Build Rules for the Cylon                               #
##########################################################################
# Basic Usage:                                                           #
#   cmake .								 #
#   make  								 #
#   make test                                                            #
# For more information about CMake, see http://www.cmake.org             #
##########################################################################
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# help cmake find cuda
# CUDACXX must be set as an environment variable pointing to nvcc
# such as: /usr/local/cuda/bin/nvcc
set(CMAKE_CUDA_COMPILER $ENV{CUDACXX})

project(GCYLON LANGUAGES CXX CUDA)

set(GCYLON_VERSION 0.2.0)

message("CONDA_PREFIX: " $ENV{CONDA_PREFIX})
if ((NOT DEFINED ENV{CONDA_PREFIX}) AND (NOT DEFINED ENV{CONDA_BUILD}) )
    message(SEND_ERROR "Neither CONDA_PREFIX nor CONDA_BUILD is set. Activate conda environment or use conda-build")
endif ()

message("CYLON_HOME: " $ENV{CYLON_HOME})
if (NOT DEFINED ENV{CYLON_HOME})
    message(SEND_ERROR "CYLON_HOME not set as an environment variable.." )
endif ()

set(WITHOUT_CUDA OFF)

## defaults to release build
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (WITHOUT_CUDA)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif ()

set("GCYLON_SIMD_LEVEL" "SSE4_2") # options: SSE4_2| AVX512 | AVX2

set(CMAKE_CUDA_ARCHITECTURES 60 61 75)

# C++ standard
# with cudf v21.08, at least c++ 17 required
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++17")

if (WITHOUT_CUDA)
    add_compile_options(-Wall -Wextra -W)
endif ()

if (NOT HAVE_EXCLUSIVE_SCAN)
    add_definitions(-DNEED_EXCLUSIVE_SCAN)
endif ()


# if conda build is used
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CMAKE_INSTALL_INCLUDEDIR "$ENV{BUILD_PREFIX}/include")
    set(CUDF_INCLUDE_DIR "$ENV{BUILD_PREFIX}/include/libcudf/libcudacxx/" "$ENV{PREFIX}/include/libcudf/libcudacxx/")
    link_directories($ENV{BUILD_PREFIX}/lib/)
    link_directories($ENV{BUILD_PREFIX}/lib64/)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} $ENV{BUILD_PREFIX}/lib/cmake)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} $ENV{BUILD_PREFIX}/lib64/cmake)
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")

# if build.sh is used without conda build
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CUDF_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx/")
    link_directories($ENV{CONDA_PREFIX}/lib/)
    link_directories($ENV{CONDA_PREFIX}/lib64/)
    set(CMAKE_MODULE_PATH "$ENV{CONDA_PREFIX}/lib/cmake;${CMAKE_MODULE_PATH}")
    set(CMAKE_MODULE_PATH "$ENV{CONDA_PREFIX}/lib64/cmake;${CMAKE_MODULE_PATH}")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# Put the libaries and binaries that get built into directories at the top of the build.
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(TEST_DATA_DIR ${GCYLON_SOURCE_DIR}/data)

# glog
find_package (gflags REQUIRED)
find_package (glog REQUIRED)
option(WITH_GFLAGS "Use gflags" ON)
SET(WITH_GFLAGS OFF)

# find CUDA
find_package(CUDAToolkit REQUIRED)
message("CUDAToolkit_INCLUDE_DIRS: " ${CUDAToolkit_INCLUDE_DIRS})

# CYLON home directory
SET(CYLON_DIR $ENV{CYLON_HOME})

# include_directories()
set(GLOG_LIBRARIES glog)

include_directories(
        ${CYLON_DIR}/cpp/src/cylon/
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CONDA_INCLUDE_DIRS}
        ${CUDF_INCLUDE_DIR}
)

add_subdirectory(src/cylon)
add_subdirectory(src/examples)


