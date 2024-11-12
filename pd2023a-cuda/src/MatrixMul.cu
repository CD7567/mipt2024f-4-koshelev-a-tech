#include <MatrixMul.cuh>

__global__
void MatrixMul(int heightA, int widthA, int widthB, float* matrixA, float* matrixB, float* matrixResult) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    
    extern __shared__ float shared_buf[];

    if (tid_x < widthB && tid_y < heightA) {
        float* A = shared_buf;
        float* B = shared_buf + blockDim.x * blockDim.y;
        float tmp = 0;

        for (int i = 0; i < widthA / blockDim.x; ++i) {
            A[threadIdx.x + threadIdx.y * blockDim.x] = matrixA[threadIdx.x + tid_y * widthA + i * blockDim.x];
            B[threadIdx.x + threadIdx.y * blockDim.x] = matrixB[tid_x + (i * blockDim.x + threadIdx.y) * widthB];

            __syncthreads();

            for (int j = 0; j < blockDim.x; ++j) {
                tmp += A[j + blockDim.x * threadIdx.y] * B[threadIdx.x + blockDim.x * j];
            }

            __syncthreads();
        }

        matrixResult[tid_x + tid_y * widthB] = tmp;
    }
}

