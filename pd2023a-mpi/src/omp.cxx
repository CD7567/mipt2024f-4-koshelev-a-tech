#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <omp.h>

inline double fast_power(double real, size_t power) {
    double buff = power ? fast_power(real, power / 2) : 1;
    return buff * buff * (power % 2 ? real : 1);
}

int main(int argc, char** argv) {    
    size_t threads_num = std::stoul(argv[1]);
    size_t N = std::stoul(argv[2]);
    long double delta = 1.0 / N;
    long double seq_result = 0.0;
    long double par_result = 0.0;
    double seq_time;
    double par_time;

    seq_time = omp_get_wtime();

    for (size_t index = 1; index <= N; ++index) {;
        seq_result += (1 / (1 + fast_power(delta * (index - 1), 2)) + 1 / (1 + fast_power(delta * index, 2))) * delta;
    }

    seq_result *= 2;

    seq_time = omp_get_wtime() - seq_time;
    std::printf("Sequential calculation result: %.11Lf\n", seq_result);
    std::printf("Sequential time: %lf\n", seq_time);

    size_t left_border = 0;
    size_t right_border = 0;
    size_t div = N / threads_num;
    size_t rem = N % threads_num;

    size_t* borders = (size_t*) malloc((threads_num + 1) * sizeof(size_t));

    for (size_t i = 0; i <= threads_num; ++i, left_border = right_border) {
        right_border = left_border + div;

        if (rem) {
            right_border += 1;
            --rem;
        }

        borders[i] = left_border;
    }

    par_time = omp_get_wtime();

    #pragma omp parallel num_threads(threads_num)
    {
        int thread_num = omp_get_thread_num();
        long double partial_integral = 0.0;

        for (size_t index = borders[thread_num] + 1; index <= borders[thread_num + 1]; ++index) {
            partial_integral += (1 / (1 + fast_power(delta * (index - 1), 2)) + 1 / (1 + fast_power(delta * index, 2))) * delta;
        }

        std::printf("Process %d resulted: %Lf\n", thread_num, partial_integral);
        par_result += partial_integral;
    }

    par_time = omp_get_wtime() - par_time;

    std::free(borders);

    par_result *= 2;
    std::printf("Parallel calculation result: %.11Lf\n", par_result);
    std::printf("Parallel time: %lf\n", par_time);
    std::printf("Difference between calculated results: %.11Lf\n", std::abs(seq_result - par_result));

    FILE* csv_output = fopen(argv[3], "a");
    fprintf(csv_output, "%lu, %lu, %.11Lf, %.11lf, %.11Lf, %.11lf\n", threads_num, N, seq_result, seq_time, par_result, par_time);
    fclose(csv_output);
}
