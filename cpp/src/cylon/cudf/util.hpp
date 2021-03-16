//
// Created by auyar on 3.02.2021.
//
#ifndef CYLON_CUDF_UTIL_H
#define CYLON_CUDF_UTIL_H

#include <cudf/types.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/strings/strings_column_view.hpp>

inline void printIntColumnPartA(const uint8_t * buff, int columnIndex, int start, int end) {
    std::cout << "column[" << columnIndex << "][" << start << "-" << end << "]: ";
    int8_t *hostArray= new int8_t [(end - start)];
    cudaMemcpy(hostArray, buff, (end-start), cudaMemcpyDeviceToHost);
    int32_t * hdata = (int32_t *) hostArray;
    int size = (end-start)/4;

    for (int i = start; i < size; ++i) {
        std::cout << hdata[i] << ", ";
    }
    std::cout << std::endl;
}

inline void printStringColumnPartA(const uint8_t * buff, int columnIndex, int start, int end) {
    std::cout << "column[" << columnIndex << "][" << start << "-" << end << "]: ";
    char *hostArray= new char[end - start + 1];
    cudaMemcpy(hostArray, buff, end-start, cudaMemcpyDeviceToHost);
    hostArray[end-start] = '\0';
    std::cout << hostArray << std::endl;
}

inline void printStringColumnA(cudf::column_view const& cv, int columnIndex) {
    cudf::strings_column_view scv(cv);
    int endIndex = cudf::detail::get_value<int32_t>(scv.offsets(), scv.offsets().size() - 1, rmm::cuda_stream_default);
    printStringColumnPartA(scv.chars().data<uint8_t>(), columnIndex,  0, endIndex);
    printIntColumnPartA(scv.offsets().data<uint8_t>(), columnIndex, 0, scv.offsets().size()*4);
}

#endif //CYLON_CUDF_UTIL_H
