#include "harness.cuh"
#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {
    constexpr double kAtol = 1e-2;
    constexpr double kRtol = 1e-3;
}  // namespace

BenchResult bench_kernel(const KernelVariant& kernel, int M, int N, int K,
                         const BenchOptions& opts, std::ostream* csv_out) {
  BenchResult result;

  const size_t a_elems = static_cast<size_t>(M) * K;
  const size_t b_elems = static_cast<size_t>(K) * N;
  const size_t c_elems = static_cast<size_t>(M) * N;
  const size_t a_bytes = a_elems * sizeof(float);
  const size_t b_bytes = b_elems * sizeof(float);
  const size_t c_bytes = c_elems * sizeof(float);

  std::vector<float> hA(a_elems);
  std::vector<float> hB(b_elems);
  std::vector<float> hC(c_elems, 0.0f);
  std::vector<float> hOut(c_elems);
  std::vector<float> hRef(c_elems);

  std::mt19937 rng(opts.seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : hA) x = dist(rng);
  for (float& x : hB) x = dist(rng);

  float* dA = nullptr;
  float* dB = nullptr;
  float* dC = nullptr;
  float* dRef = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), a_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), b_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), c_bytes));
  if (opts.verify) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dRef), c_bytes));
    CUDA_CHECK(cudaMemset(dRef, 0, c_bytes));
  }

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), c_bytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle = nullptr;
  if (opts.verify) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major. To compute row-major C = A * B, compute
    // C^T = B^T * A^T by swapping the operands and M/N.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, dB, N,
                             dA, K,
                             &beta, dRef, N));
  }

  GemmProblem problem{M, N, K, 1.0f, 0.0f, dA, dB, dC};

  kernel.launch(problem, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (opts.verify) {
    CUDA_CHECK(cudaMemcpy(hOut.data(), dC, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, c_bytes, cudaMemcpyDeviceToHost));

    bool passed = true;
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t first_bad = c_elems;
    double first_out = 0.0;
    double first_ref = 0.0;

    for (size_t i = 0; i < c_elems; ++i) {
      const double out = static_cast<double>(hOut[i]);
      const double ref = static_cast<double>(hRef[i]);

      if (!std::isfinite(out)) {
        passed = false;
        if (first_bad == c_elems) {
          first_bad = i;
          first_out = out;
          first_ref = ref;
        }
        continue;
      }

      const double diff = std::fabs(out - ref);
      const double rel = diff / (std::fabs(ref) + 1e-6);
      max_abs = std::max(max_abs, diff);
      max_rel = std::max(max_rel, rel);

      if (diff > kAtol + kRtol * std::fabs(ref)) {
        passed = false;
        if (first_bad == c_elems) {
          first_bad = i;
          first_out = out;
          first_ref = ref;
        }
      }
    }

    result.passed = passed;
    result.max_abs_err = max_abs;
    result.max_rel_err = max_rel;

    if (!passed) {
      std::fprintf(stderr,
                   "first mismatch at i=%zu (row=%zu, col=%zu): "
                   "out=%.6f ref=%.6f\n",
                   first_bad, first_bad / static_cast<size_t>(N),
                   first_bad % static_cast<size_t>(N), first_out, first_ref);
    }
  }

  for (int i = 0; i < opts.warmup; ++i) {
    kernel.launch(problem, 0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < opts.reps; ++i) {
    kernel.launch(problem, 0);
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  if (handle != nullptr) {
    CUBLAS_CHECK(cublasDestroy(handle));
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  if (dRef != nullptr) {
    CUDA_CHECK(cudaFree(dRef));
  }

  const double avg_ms =
      (opts.reps > 0) ? static_cast<double>(elapsed_ms) / opts.reps : 0.0;

  result.min_ms = avg_ms;
  result.median_ms = avg_ms;
  result.mean_ms = avg_ms;
  result.std_ms = 0.0;

  const double seconds = avg_ms * 1e-3;
  if (seconds > 0.0) {
    if (kernel.metric == Metric::Gflops) {
      result.value = 2.0 * M * N * K / seconds / 1e9;
    } else if (kernel.metric == Metric::Gbps) {
      result.value = 2.0 * c_elems * sizeof(float) / seconds / 1e9;
    }
  }

  const char* metric_name =
      kernel.metric == Metric::Gflops ? "gflops"
      : kernel.metric == Metric::Gbps ? "gbps"
                                      : "none";

  std::printf(
      "kernel=%-10s M=%d N=%d K=%d %s "
      "max_abs=%.3e max_rel=%.3e "
      "min_ms=%.6f median_ms=%.6f mean_ms=%.6f std_ms=%.6f "
      "%s=%.2f\n",
      kernel.name, M, N, K, result.passed ? "PASS" : "FAIL",
      result.max_abs_err, result.max_rel_err, result.min_ms,
      result.median_ms, result.mean_ms, result.std_ms, metric_name,
      result.value);
  std::fflush(stdout);

  if (csv_out != nullptr) {
    *csv_out << "1," << kernel.name << "," << kernel.config << "," << M << ","
             << N << "," << K << ",1,0," << opts.warmup << "," << opts.reps
             << "," << std::setprecision(9) << result.min_ms << ","
             << result.median_ms << "," << result.mean_ms << ","
             << result.std_ms << "," << metric_name << "," << result.value
             << "," << result.max_abs_err << "," << result.max_rel_err << ","
             << (result.passed ? "true" : "false") << "\n";
    csv_out->flush();
  }

  return result;
}
