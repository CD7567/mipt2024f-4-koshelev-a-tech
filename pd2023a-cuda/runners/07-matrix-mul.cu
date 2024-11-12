#include <MatrixMul.cuh>
#include "CommonKernels.cuh"
#include <cstdio>

int main(int argc, char** argv) {
    int WIDTH = std::atoi(argv[1]);
    int HEIGHT = std::atoi(argv[2]);
    int LENGTH = WIDTH * HEIGHT;
    int BLOCK_SIZE = std::atoi(argv[3]);

    float *A, *B, *result;
    float *dev_A, *dev_B, *dev_result;

    A = (float*) malloc(LENGTH * sizeof(float));
    B = (float*) malloc(LENGTH * sizeof(float));
    result = (float*) malloc(LENGTH * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**) &dev_A, LENGTH * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**) &dev_B, LENGTH * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**) &dev_result, LENGTH * sizeof(float)));

    for (int i = 0; i < LENGTH; ++i) {
        A[i] = i;
    }

    for (int i = 0; i < WIDTH; ++i) {
        B[i] = i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_A, A, LENGTH * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_B, B, LENGTH * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid_size((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

    cudaEventRecord(begin);

    MatrixMul<<<grid_size, block_size>>>(WIDTH, HEIGHT, WIDTH, dev_A, dev_B, dev_result);

    HANDLE_ERROR(cudaMemcpy(result, dev_result, LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventRecord(end);

	cudaEventSynchronize(end);

    FILE* csv_output = fopen("data/data.csv", "a");
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    fprintf(csv_output, "MatrixMul, %d, %d, %f", LENGTH, BLOCK_SIZE * BLOCK_SIZE, elapsed_time);
    fclose(csv_output);
    
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_result);

    free(A);
    free(B);
    free(result);
}

