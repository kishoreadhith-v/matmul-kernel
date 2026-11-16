cmake_minimum_required(VERSION 3.24)
project(matmul_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)      # GTX 1650 SUPER = sm_75
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(matmul
  src/main.cu
  src/harness.cu
  src/kernels/01_naive.cu
)

target_include_directories(matmul PRIVATE include src)
target_link_libraries(matmul PRIVATE CUDA::cudart CUDA::cublas)

target_compile_options(matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
