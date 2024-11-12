#include <ScalarMulRunner.cuh>
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

	  cudaEvent_t begin_tr, end_tr, begin_sp, end_sp;
	  cudaEventCreate(&begin_tr);
	  cudaEventCreate(&end_tr);
	  cudaEventCreate(&begin_sp);
	  cudaEventCreate(&end_sp);

	  cudaEventRecord(begin_tr);
    ScalarMulTwoReductions(SIZE, dev_x, dev_y, BLOCK_SIZE);
    cudaEventRecord(end_tr);

    cudaEventSynchronize(end_tr);

    cudaEventRecord(begin_sp);
    ScalarMulSumPlusReduction(SIZE, dev_x, dev_y, BLOCK_SIZE);
    cudaEventRecord(end_sp);

    cudaEventSynchronize(end_sp);

    FILE* csv_output = fopen("data/data.csv", "a");
    float elapsed_time_tr, elapsed_time_sp;
    cudaEventElapsedTime(&elapsed_time_tr, begin_tr, end_tr);
    cudaEventElapsedTime(&elapsed_time_sp, begin_sp, end_sp);
    fprintf(csv_output, "ScalarMulTR, %d, %d, %f", SIZE, BLOCK_SIZE, elapsed_time_tr);
    fprintf(csv_output, "ScalarMulSP, %d, %d, %f", SIZE, BLOCK_SIZE, elapsed_time_sp);
    fclose(csv_output);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_result);

    free(x);
    free(y);
    free(result);
}
