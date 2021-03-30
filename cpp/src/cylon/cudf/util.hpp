//
// Created by auyar on 3.02.2021.
//
#ifndef CYLON_CUDF_UTIL_H
#define CYLON_CUDF_UTIL_H

/**
 * get one scalar value from device to host
 * @param buff
 * @return
 */
template <typename T>
inline T getScalar(const uint8_t * buff) {
    uint8_t *hostArray= new uint8_t[sizeof(T)];
    cudaMemcpy(hostArray, buff, sizeof(T), cudaMemcpyDeviceToHost);
    T * hdata = (T *) hostArray;
    return hdata[0];
}

#endif //CYLON_CUDF_UTIL_H
