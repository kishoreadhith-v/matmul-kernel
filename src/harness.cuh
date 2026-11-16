#pragma once

#include "kernel.cuh"

#include <ostream>

struct BenchOptions {
  int reps = 50;
  int warmup = 5;
  int seed = 0;
  bool verify = true;
};

struct BenchResult {
  double min_ms = 0.0;
  double median_ms = 0.0;
  double mean_ms = 0.0;
  double std_ms = 0.0;
  double value = 0.0;
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
  bool passed = true;
};

BenchResult bench_kernel(const KernelVariant& kernel, int M, int N, int K,
                         const BenchOptions& opts, std::ostream* csv_out);
