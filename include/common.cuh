#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,          \
                   __LINE__, cudaGetErrorString(err__));                   \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

#define CUBLAS_CHECK(call)                                                 \
  do {                                                                     \
    cublasStatus_t status__ = (call);                                      \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                               \
      std::fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, \
                   __LINE__, static_cast<int>(status__));                  \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }
