#include "common.cuh"
#include "kernel.cuh"

void launch_cublas(const GemmProblem& p, cudaStream_t stream) {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  const float alpha = p.alpha;
  const float beta = p.beta;

  // cuBLAS is column-major: compute C^T = B^T * A^T.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           p.N, p.M, p.K,
                           &alpha, p.B, p.N,
                           p.A, p.K,
                           &beta, p.C, p.N));
}
