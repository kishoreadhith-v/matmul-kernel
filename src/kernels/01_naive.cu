#include "common.cuh"
#include "kernel.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float beta, float* __restrict__ C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    // C = α*(A@B)+β*C
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

void launch_naive(const GemmProblem& p, cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid(div_ceil(p.M, 32), div_ceil(p.N, 32));

  sgemm_naive<<<grid, block, 0, stream>>>(
      p.M, p.N, p.K, p.alpha, p.A, p.B, p.beta, p.C);

  CUDA_CHECK(cudaGetLastError());
}
