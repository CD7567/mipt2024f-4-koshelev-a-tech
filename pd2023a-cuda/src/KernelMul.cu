#include <KernelMul.cuh>

__global__ void KernelMul(int numElements, float* x, float* y, float* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x;

    for (int i = tid; i < numElements; i += stride) {
        result[i] = x[i] * y [i];
    }
}

