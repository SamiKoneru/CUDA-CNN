#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include "layer.cuh"
#include "gemm.cuh"
#include "adam.cuh"

// out[i] += bias[i % cols] — broadcast a (cols,) bias across all rows.
__global__ void dense_add_bias_kernel(float* out, const float* bias, int total, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] += bias[i % cols];
}

// out (cols, rows) = transpose of in (rows, cols), both row-major.
__global__ void dense_transpose_kernel(const float* in, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        int r = i / cols;
        int c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}

// out[j] = sum over rows of m[:, j]   (m is rows x cols, row-major).
__global__ void dense_colsum_kernel(const float* m, float* out, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float s = 0.0f;
        for (int r = 0; r < rows; r++) s += m[r * cols + j];
        out[j] = s;
    }
}

static inline int dense_blocks_for(int n, int threads) { return (n - 1) / threads + 1; }

class Dense_Layer : public Layer {
public:
    Tensor W;                       // (x_in, y_out): forward is out = x @ W
    Tensor B;                       // (y_out,)
    Adam opt_W, opt_B;
    const Tensor* last_in = nullptr;
    int x_in, y_out;
    float lrate;

    void init(int x_in, int y_out, float lrate);
    void forward(const Tensor& x, Tensor& out) override;
    void backward(const Tensor& dout, Tensor& dx) override;
    Tensor output_shape(const Tensor& x) override;
};

void Dense_Layer::init(int x_in, int y_out, float lrate) {
    this->x_in = x_in;
    this->y_out = y_out;
    this->lrate = lrate;

    W = tensor_alloc(1, 1, x_in, y_out);
    B = tensor_alloc(1, 1, 1, y_out);
    tensor_zero(B);

    // He init: std = sqrt(2 / fan_in).
    std::vector<float> hostW((size_t)x_in * y_out);
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / x_in));
    for (float& w : hostW) w = nd(rng);
    tensor_upload(W, hostW.data());

    opt_W.init(x_in * y_out, lrate);
    opt_B.init(y_out, lrate);
}

void Dense_Layer::forward(const Tensor& x, Tensor& out) {
    last_in = &x;
    int N = x.N;

    // out (N, y_out) = x (N, x_in) @ W (x_in, y_out)
    gemm(x.data, W.data, out.data, N, x_in, y_out);

    int total = N * y_out, threads = 256;
    dense_add_bias_kernel<<<dense_blocks_for(total, threads), threads>>>(out.data, B.data, total, y_out);
}

void Dense_Layer::backward(const Tensor& dout, Tensor& dx) {
    int N = last_in->N, threads = 256;

    float *Wt, *xT, *dW, *dB;
    cudaMalloc(&Wt, sizeof(float) * (size_t)y_out * x_in);
    cudaMalloc(&xT, sizeof(float) * (size_t)x_in * N);
    cudaMalloc(&dW, sizeof(float) * (size_t)x_in * y_out);
    cudaMalloc(&dB, sizeof(float) * (size_t)y_out);

    // dX (N, x_in) = dout (N, y_out) @ W^T (y_out, x_in)
    dense_transpose_kernel<<<dense_blocks_for(x_in * y_out, threads), threads>>>(W.data, Wt, x_in, y_out);
    gemm(dout.data, Wt, dx.data, N, y_out, x_in);

    // dW (x_in, y_out) = x^T (x_in, N) @ dout (N, y_out)
    dense_transpose_kernel<<<dense_blocks_for(N * x_in, threads), threads>>>(last_in->data, xT, N, x_in);
    gemm(xT, dout.data, dW, x_in, N, y_out);

    // dB (y_out,) = column sum of dout
    dense_colsum_kernel<<<dense_blocks_for(y_out, threads), threads>>>(dout.data, dB, N, y_out);

    // Adam step; both optimizers do the 1/N averaging internally.
    opt_W.step(W.data, dW, N);
    opt_B.step(B.data, dB, N);

    cudaFree(Wt);
    cudaFree(xT);
    cudaFree(dW);
    cudaFree(dB);
}

Tensor Dense_Layer::output_shape(const Tensor& x) {
    return {nullptr, x.N, 1, 1, y_out};
}
