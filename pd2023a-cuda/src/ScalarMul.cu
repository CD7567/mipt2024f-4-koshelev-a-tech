#include <ScalarMul.cuh>

__global__
void ScalarMulBlock(int numElements, float* vector1, float* vector2, float *result) {
    extern __shared__ float shared[];
    shared[threadIdx.x] = 0;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numElements) {
        shared[threadIdx.x] = vector1[tid] * vector2[tid];

        __syncthreads();

        for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            }
            __syncthreads();
        }
    
        if (threadIdx.x == 0) {
            result[blockIdx.x] = shared[0];
        }
    } 
}

