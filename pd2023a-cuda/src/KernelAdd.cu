#include "KernelAdd.cuh"

__global__ void KernelAdd(int numElements, float* x, float* y, float* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numElements) {
        result[tid] = x[tid] + y[tid];
    }
}

