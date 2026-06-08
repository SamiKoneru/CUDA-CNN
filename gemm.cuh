#pragma once

// C (M, N) = A (M, K) @ B (K, N), all row-major device pointers.
void gemm(const float* A, const float* B, float* C, int M, int K, int N);
