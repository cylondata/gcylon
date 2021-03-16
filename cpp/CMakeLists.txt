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
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(GCYLON LANGUAGES CXX CUDA)

set(GCYLON_VERSION 0.2.0)

set(WITHOUT_CUDA OFF)

## defaults to release build
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# cmake modules directories
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/Modules/" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${GCYLON_SOURCE_DIR}/CMake)

if (WITHOUT_CUDA)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif ()

set("GCYLON_SIMD_LEVEL" "SSE4_2") # options: SSE4_2| AVX512 | AVX2

set(CMAKE_CUDA_ARCHITECTURES 60 61 75)

# Add common flags
# include(SetupCxxFlags)
# message("CXX_COMMON_FLAGS: ${CXX_COMMON_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMMON_FLAGS}")

# string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)
# if (UPPERCASE_CMAKE_BUILD_TYPE MATCHES "DEBUG")
#     message("Running on debug mode...")
#    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
#
#    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer ")
# elseif (UPPERCASE_CMAKE_BUILD_TYPE MATCHES "RELEASE")
#    message("Running on Release mode...")
#    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
#else ()
#    message(FATAL_ERROR "Unknown CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}! Exiting...")
#endif ()

# C++ standard
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++17")

if (WITHOUT_CUDA)
    add_compile_options(-Wall -Wextra -W)
endif ()

if (NOT HAVE_EXCLUSIVE_SCAN)
    add_definitions(-DNEED_EXCLUSIVE_SCAN)
endif ()

# Put the libaries and binaries that get built into directories at the
# top of the build.
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

# root include directory
SET(CONDA_DIR /home/auyar/anaconda3/envs/cudf_dev/)
SET(CYLON_DIR /home/auyar/projects/cylon/)

include_directories(
        ${CONDA_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CONDA_DIR}/include/libcudf/libcudacxx/
        ${CYLON_DIR}/cpp/src/cylon/
        ${glog_DIR}/include
)


# include(Build)

add_subdirectory(src/cylon)
add_subdirectory(src/examples)


