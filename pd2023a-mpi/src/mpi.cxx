#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <mpi.h>

inline double fast_power(double real, size_t power) {
    double buff = power ? fast_power(real, power / 2) : 1;
    return buff * buff * (power % 2 ? real : 1);
}

int main(int argc, char** argv) {
    MPI_Init(nullptr, nullptr);

    size_t N = std::stoul(argv[1]);
    long double delta = 1.0 / N;
    long double seq_result = 0.0;
    long double par_result = 0.0;
    double seq_time;
    double par_time;

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (!world_rank) {
        seq_time = MPI_Wtime();

        for (size_t index = 1; index <= N; ++index) {;
            seq_result += (1 / (1 + fast_power(delta * (index - 1), 2)) + 1 / (1 + fast_power(delta * index, 2))) * delta;
        }

        seq_result *= 2;

        seq_time = MPI_Wtime() - seq_time;
        std::printf("Sequential calculation result: %.11Lf\n", seq_result);
        std::printf("Sequential time: %lf\n", seq_time);

        size_t left_border = 0;
        size_t right_border = 0;
        size_t div = N / world_size;
        size_t rem = N % world_size;

        par_time = MPI_Wtime();
        for (size_t i = 0; i < world_size; ++i, left_border = right_border) {
            right_border = left_border + div;

            if (rem) {
                right_border += 1;
                --rem;
            }


            size_t job[2] = {left_border, right_border};
            MPI_Bsend(&job, 2, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
        }
    }

    size_t job[2];
    double task_time[2];
    MPI_Status job_recv_status;
    MPI_Recv(&job, 2, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &job_recv_status);

    long double partial_integral = 0.0;

    for (size_t index = job[0] + 1; index <= job[1]; ++index) {;
        partial_integral += (1 / (1 + fast_power(delta * (index - 1), 2)) + 1 / (1 + fast_power(delta * index, 2))) * delta;
    }

    MPI_Request req;
    MPI_Isend(&partial_integral, 1, MPI_LONG_DOUBLE, 0, 1, MPI_COMM_WORLD, &req);

    if (!world_rank) {
        for (size_t i = 0; i < world_size; ++i) {
            long double partial_integral;
            MPI_Status result_recv_status;
            MPI_Recv(&partial_integral, 1, MPI_LONG_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &result_recv_status);

            std::printf("Process %d returned %Lf\n", result_recv_status.MPI_SOURCE, partial_integral);

            par_result += partial_integral;
        }

        par_result *= 2;
        par_time = MPI_Wtime() - par_time;
        std::printf("Parallel calculation result: %.11Lf\n", par_result);
        std::printf("Parallel time: %lf\n", par_time);
        std::printf("Difference between calculated results: %.11Lf\n", std::abs(seq_result - par_result));

        FILE* csv_output = fopen(argv[2], "a");
        fprintf(csv_output, "%d, %lu, %.11Lf, %.11lf, %.11Lf, %.11lf\n", world_size, N, seq_result, seq_time, par_result, par_time);
        fclose(csv_output);
    }

    MPI_Finalize();
}
