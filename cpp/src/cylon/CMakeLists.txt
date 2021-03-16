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

# find_package(Threads REQUIRED)

# set_source_files_properties(util/murmur3.cpp util/murmur3.hpp PROPERTIES COMPILE_FLAGS -Wno-implicit-fallthrough)

add_library(gcylon SHARED
        cudf/cudf_a2a.cuh
        cudf/cudf_a2a.cu
        cudf/cudf_a2a.hpp
        cudf/cudf_a2a.cpp
        cudf/util.hpp
        cudf/gtable.hpp
        cudf/gtable.cpp
        )

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

set_target_properties(gcylon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET gcylon PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gcylon ${GLOG_LIBRARIES})
target_link_libraries(gcylon ${CYLON_DIR}/build/lib/libcylon.so)
target_link_libraries(gcylon CUDA::cudart)
target_link_libraries(gcylon
        ${CONDA_DIR}/lib/libcudf_ast.so
        ${CONDA_DIR}/lib/libcudf_interop.so
        ${CONDA_DIR}/lib/libcudf_partitioning.so
        ${CONDA_DIR}/lib/libcudf_rolling.so
        ${CONDA_DIR}/lib/libcudf_base.so
        ${CONDA_DIR}/lib/libcudf_io.so
        ${CONDA_DIR}/lib/libcudf_reductions.so
        ${CONDA_DIR}/lib/libcudf.so
        ${CONDA_DIR}/lib/libcudf_comms.so
        ${CONDA_DIR}/lib/libcudf_join.so
        ${CONDA_DIR}/lib/libcudf_replace.so
        ${CONDA_DIR}/lib/libcudf_hash.so
        ${CONDA_DIR}/lib/libcudf_merge.so
        ${CONDA_DIR}/lib/libcudf_reshape.so
        ${CONDA_DIR}/lib/libcudf_transpose.so
        )

if (WITHOUT_CUDA)
    target_compile_options(gcylon PRIVATE -Werror -Wall -Wextra -Wno-unused-parameter)
endif ()

# gcylon_install_all_headers("gcylon")

add_subdirectory(cudf)

install(TARGETS gcylon DESTINATION lib)