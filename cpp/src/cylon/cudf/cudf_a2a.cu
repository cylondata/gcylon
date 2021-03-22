//
// Created by auyar on 3.02.2021.
//
#include "cudf_a2a.cuh"

namespace gcylon {

    __global__ void rebaseOffsets(int32_t * arr, int size, int32_t base) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        arr[i] -= base;
    }
}

int ceil(const int& numerator, const int& denominator) {
   return (numerator + denominator - 1) / denominator;
}

//todo: need to take care of the case when the size is more than
//      max_thread_count_per_block * max_number_of_blocks
void callRebaseOffsets(int32_t * arr, int size, int32_t base){
    int threads_per_block = 256;
    int number_of_blocks = ceil(size, threads_per_block);
    rebaseOffsets<<<number_of_blocks, threads_per_block>>>(arr, size, base);
    cudaDeviceSynchronize();
}

}// end of namespace gcylon
