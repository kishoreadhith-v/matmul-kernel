#pragma once

#include <cuda_runtime.h>

struct GemmProblem {
  int M, N, K;
  float alpha, beta;
  const float* A;   // M x K, row-major
  const float* B;   // K x N, row-major
  float* C;         // M x N, row-major
};

enum class Metric { Gflops, Gbps, None };

using KernelLaunch = void (*)(const GemmProblem&, cudaStream_t);

struct KernelVariant {
  const char* name;    // "naive"
  const char* config;  // "block=32x32"
  KernelLaunch launch;
  Metric metric;
};
