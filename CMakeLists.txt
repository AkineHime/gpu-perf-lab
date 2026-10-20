cmake_minimum_required(VERSION 3.24)
project(gpu_perf_lab LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
add_executable(gemm_bench src/gemm_bench.cu)
target_compile_options(gemm_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas)
