#include "KernelAdd.cuh"
#include "CommonKernels.cuh"
#include <cstdio>

int main(int argc, char** argv) {
    int SIZE = std::atoi(argv[1]);
    int BLOCK_SIZE = std::atoi(argv[2]);

    float *x, *y, *result;
    float *dev_x, *dev_y, *dev_result;

    x = (float*) malloc(SIZE * sizeof(float));
    y = (float*) malloc(SIZE * sizeof(float));
    result = (float*) malloc(SIZE * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**) &dev_x, SIZE * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**) &dev_y, SIZE * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**) &dev_result, SIZE * sizeof(float)));

    for (int i = 0; i < SIZE; ++i) {
        x[i] = i;
        y[i] = i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_x, x, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_y, y, SIZE * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid_size((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 block_size(BLOCK_SIZE);

	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	cudaEventRecord(begin);
    KernelAdd<<<grid_size, block_size>>>(SIZE, dev_x, dev_y, dev_result);
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    FILE* csv_output = fopen("data/data.csv", "a");
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, begin, end);
    fprintf(csv_output, "KernelAdd, %d, %d, %f", SIZE, BLOCK_SIZE, elapsed_time);
    fclose(csv_output);

    HANDLE_ERROR(cudaMemcpy(result, dev_result, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_result);

    free(x);
    free(y);
    free(result);
}
