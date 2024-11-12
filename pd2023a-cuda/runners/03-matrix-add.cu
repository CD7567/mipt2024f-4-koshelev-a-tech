#include "KernelMatrixAdd.cuh"
#include "CommonKernels.cuh"
#include <cstdio>
#include <random>

int main(int argc, char** argv) {
    int WIDTH = std::atoi(argv[1]);
    int HEIGHT = std::atoi(argv[2]);
    int LENGTH = WIDTH * HEIGHT;
    int BLOCK_SIZE = std::atoi(argv[3]);

    float *A, *B, *result;
    float *dev_A, *dev_B, *dev_result;

    size_t pitch;
    HANDLE_ERROR(cudaMallocPitch(&dev_A, &pitch, WIDTH, HEIGHT));
    HANDLE_ERROR(cudaMallocPitch(&dev_B, &pitch, WIDTH, HEIGHT));
    HANDLE_ERROR(cudaMallocPitch(&dev_result, &pitch, WIDTH, HEIGHT));

    HANDLE_ERROR(cudaMemcpy2D(dev_A, pitch, A, WIDTH * sizeof(float), WIDTH, HEIGHT, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(dev_B, pitch, B, WIDTH * sizeof(float), WIDTH, HEIGHT, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(dev_result, pitch, result, WIDTH * sizeof(float), WIDTH, HEIGHT, cudaMemcpyHostToDevice));

    dim3 grid_size((WIDTH +  BLOCK_SIZE - 1) / WIDTH, (HEIGHT +  BLOCK_SIZE - 1) / HEIGHT);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

    cudaEventRecord(begin);
    KernelMatrixAdd<<<grid_size, block_size>>>(HEIGHT, WIDTH, pitch, dev_A, dev_B, dev_result);
    cudaEventRecord(end);

	cudaEventSynchronize(end);

	FILE* csv_output = fopen("data/data.csv", "a");
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    fprintf(csv_output, "KernelMatrixAdd, %d, %d, %f", LENGTH, BLOCK_SIZE * BLOCK_SIZE, elapsed_time);
    fclose(csv_output);

    cudaMemcpy2D(result, WIDTH * sizeof(float), dev_result, pitch, WIDTH, HEIGHT, cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_result);

    free(A);
    free(B);
    free(result);
}

