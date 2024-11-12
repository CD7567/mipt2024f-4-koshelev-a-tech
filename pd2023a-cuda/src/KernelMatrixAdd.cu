#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (tid_x < width && tid_y < height) {
        int lin_index = tid_x + tid_y * pitch; 
        result[lin_index] =  A[lin_index] + B[lin_index];
    }
}

