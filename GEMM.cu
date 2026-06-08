// Tiled GEMM: C = A @ B
//   A is (M, K), row-major
//   B is (K, N), row-major
//   C is (M, N), row-major

#define TILE 16

__global__ void gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    // Shared-memory staging buffers for the current pair of tiles.
    __shared__ float A_shared[TILE][TILE];
    __shared__ float B_shared[TILE][TILE];

    // This thread's position in the output matrix C.
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Per-thread accumulator. Lives in a register, persists across the K-loop.
    float sum = 0.0f;

    // How many K-tiles do we need to slide through?  ceil(K / TILE).
    int num_k_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_k_tiles; t++) {
        // Cooperative load: each thread loads ONE element of A_shared and ONE of B_shared.
        // After all 256 (= TILE * TILE) threads finish, the whole tile pair is in shared memory.

        int a_col = t * TILE + threadIdx.x;     // column index into A
        int b_row = t * TILE + threadIdx.y;     // row index into B

        // Bounds-checked loads: pad with 0 if we're past the matrix edge.
        // This makes edge tiles (where M, K, or N isn't a multiple of TILE) safe.
        A_shared[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        B_shared[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Wait for every thread in the block to finish loading before we read shared memory.
        __syncthreads();

        // Inner product over the current tile.  Each thread does TILE multiply-adds,
        // all reads coming from shared memory (cheap).
        for (int k = 0; k < TILE; k++) {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        // Wait before the next iteration overwrites A_shared / B_shared.
        __syncthreads();
    }

    // One global write per thread, at the very end.
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Helper to set up launch dims and call the kernel.
// Call this from your host code; A, B, C must already be device pointers.
void gemm(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
