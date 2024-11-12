cmake_minimum_required(VERSION 3.14.0)
project(MPI_hw_1 VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")

add_executable(mpi src/mpi.cxx)
add_executable(omp src/omp.cxx)
add_executable(mpi_omp src/mpi_omp.cxx)

target_link_libraries(mpi PRIVATE MPI::MPI_CXX)
target_link_libraries(mpi_omp PRIVATE MPI::MPI_CXX)
