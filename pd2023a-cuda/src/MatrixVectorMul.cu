#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < height) {
        float tmp = 0;

        for (int i = 0; i < width; ++i) {
            tmp += matrix[i + tid * width] * vector[i];
        }

        result[tid] = tmp;
    }
}

