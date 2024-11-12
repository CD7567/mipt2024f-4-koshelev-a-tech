#include <ScalarMul.cuh>
#include <ScalarMulRunner.cuh>
#include <iostream>

__global__
void SumBlock(int numElements, float* vector, float* result) {
    extern __shared__ float shared[];
    shared[threadIdx.x] = 0;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numElements) {
         shared[threadIdx.x] = vector[tid];

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

float ScalarMulTwoReductions(int numElements, float* vector1, float* vector2, int blockSize) {
    float* dev_vec1;
    float* dev_vec2;
    float* dev_result;
    float result;

    cudaMalloc(&dev_vec1, numElements * sizeof(float));
    cudaMalloc(&dev_vec2, numElements * sizeof(float));
    cudaMalloc(&dev_result, numElements * sizeof(float));

    cudaMemcpy(dev_vec1, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vec2, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_size((numElements + blockSize - 1) / blockSize);
    int reduced_block_size = pow(2, 1 + static_cast<int>(log(grid_size.x) / log(2)));

    ScalarMulBlock<<<grid_size, blockSize, blockSize * sizeof(float)>>>(numElements, dev_vec1, dev_vec2, dev_result);
    
    if (grid_size.x > 1)
        SumBlock<<<1, reduced_block_size, reduced_block_size * sizeof(float)>>>(grid_size.x, dev_result, dev_result);

    cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_vec1);
    cudaFree(dev_vec2);
    cudaFree(dev_result);

    return result;
}

float ScalarMulSumPlusReduction(int numElements, float* vector1, float* vector2, int blockSize) {
    float* dev_vec1;
    float* dev_vec2;
    float* dev_result;

    dim3 grid_size((numElements + blockSize - 1) / blockSize);
    int reduced_block_size = pow(2, 1 + static_cast<int>(log(grid_size.x) / log(2)));

    float* result = new float[grid_size.x];
    
    cudaMalloc(&dev_vec1, numElements * sizeof(float));
    cudaMalloc(&dev_vec2, numElements * sizeof(float));
    cudaMalloc(&dev_result, grid_size.x * sizeof(float));

    cudaMemcpy(dev_vec1, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vec2, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);
 
    ScalarMulBlock<<<grid_size, blockSize, blockSize * sizeof(float)>>>(numElements, dev_vec1, dev_vec2, dev_result);

    cudaMemcpy(result, dev_result, grid_size.x * sizeof(float), cudaMemcpyDeviceToHost);

    float tmp = 0;
    for (int i = 0; i < grid_size.x; ++i) {
        tmp += result[i];
    }

    cudaFree(dev_vec1);
    cudaFree(dev_vec2);
    cudaFree(dev_result);

    free(result);

    return tmp;
}

