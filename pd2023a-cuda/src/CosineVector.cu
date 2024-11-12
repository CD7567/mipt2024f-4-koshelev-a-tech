#include <CosineVector.cuh>
#include <ScalarMulRunner.cuh>

float CosineVector(int numElements, float* vector1, float* vector2, int blockSize) {
    float scalar_prod = ScalarMulTwoReductions(numElements, vector1, vector2, blockSize);
    float vector1_len = ScalarMulTwoReductions(numElements, vector1, vector1, blockSize);
    float vector2_len = ScalarMulTwoReductions(numElements, vector2, vector2, blockSize);

    return scalar_prod / sqrt(vector1_len * vector2_len);
}

