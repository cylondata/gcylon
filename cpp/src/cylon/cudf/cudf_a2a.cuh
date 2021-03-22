//
// Created by auyar on 3.02.2021.
//
#ifndef CYLON_CUDF_A2AD_H
#define CYLON_CUDF_A2AD_H

#include <cuda_runtime.h>

namespace gcylon {

__global__ void rebaseOffsets(int32_t * arr, int size, int32_t base);

void callRebaseOffsets(int32_t * arr, int size, int32_t base);

}// end of namespace gcylon

#endif //CYLON_CUDF_A2AD_H
