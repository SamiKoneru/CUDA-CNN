#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include "layer.cuh"
#include "gemm.cuh"
#include "adam.cuh"

// ---- shared scalar utility kernels (mirror dense.cu's) ----

// out[i] += bias[i % cols] — broadcast bias across rows. NHWC means C_out
// is the contiguous last dim of `out`, so this works on conv output too.
__global__ void conv_add_bias_kernel(float* out, const float* bias, int total, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] += bias[i % cols];
}

__global__ void conv_transpose_kernel(const float* in, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        int r = i / cols;
        int c = i % cols;
        out[c * rows + r] = in[r * cols + c];
    }
}

__global__ void conv_colsum_kernel(const float* m, float* out, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float s = 0.0f;
        for (int r = 0; r < rows; r++) s += m[r * cols + j];
        out[j] = s;
    }
}

// ---- im2col / col2im for NHWC ----

// One thread per element of patches (shape M x K, where
// M = N*out_H*out_W, K = KH*KW*C_in). Out-of-image taps write 0 (zero pad).
__global__ void conv_im2col_kernel(
    const float* x, float* patches,
    int N, int H, int W, int C_in,
    int KH, int KW,
    int out_H, int out_W,
    int stride, int padding
) {
    int total = N * out_H * out_W * KH * KW * C_in;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ci =  idx                                       % C_in;
    int kw = (idx /  C_in)                              % KW;
    int kh = (idx / (C_in * KW))                        % KH;
    int ow = (idx / (C_in * KW * KH))                   % out_W;
    int oh = (idx / (C_in * KW * KH * out_W))           % out_H;
    int n  =  idx / (C_in * KW * KH * out_W * out_H);

    int ih = oh * stride + kh - padding;
    int iw = ow * stride + kw - padding;

    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        v = x[((n * H + ih) * W + iw) * C_in + ci];
    }

    int row = (n * out_H + oh) * out_W + ow;
    int col = (kh * KW + kw) * C_in + ci;
    int K   = KH * KW * C_in;
    patches[row * K + col] = v;
}

// Inverse of im2col: scatter dpatches back into dx with atomicAdd.
// Caller must zero dx first.
__global__ void conv_col2im_kernel(
    const float* dpatches, float* dx,
    int N, int H, int W, int C_in,
    int KH, int KW,
    int out_H, int out_W,
    int stride, int padding
) {
    int total = N * out_H * out_W * KH * KW * C_in;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ci =  idx                                       % C_in;
    int kw = (idx /  C_in)                              % KW;
    int kh = (idx / (C_in * KW))                        % KH;
    int ow = (idx / (C_in * KW * KH))                   % out_W;
    int oh = (idx / (C_in * KW * KH * out_W))           % out_H;
    int n  =  idx / (C_in * KW * KH * out_W * out_H);

    int ih = oh * stride + kh - padding;
    int iw = ow * stride + kw - padding;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) return;

    int row = (n * out_H + oh) * out_W + ow;
    int col = (kh * KW + kw) * C_in + ci;
    int K   = KH * KW * C_in;

    atomicAdd(&dx[((n * H + ih) * W + iw) * C_in + ci],
              dpatches[row * K + col]);
}

static inline int conv_blocks_for(int n, int threads) { return (n - 1) / threads + 1; }

// ----------------------------------------------------------------

class Conv_Layer : public Layer {
public:
    Tensor kernel;                  // (1, 1, K, C_out) where K = KH*KW*C_in
    Tensor bias;                    // (1, 1, 1, C_out)
    Adam opt_K, opt_B;
    const Tensor* last_in = nullptr;

    Tensor patches;                 // (1, 1, M, K), lazily sized
    int patches_cap = 0;            // capacity in floats

    int KH, KW, C_in, C_out;
    int stride, padding;
    float lrate;

    void init(int C_in, int C_out, int KH, int KW,
              int stride, int padding, float lrate);
    void forward(const Tensor& x, Tensor& out) override;
    void backward(const Tensor& dout, Tensor& dx) override;
    Tensor output_shape(const Tensor& x) override;
};

void Conv_Layer::init(int C_in_, int C_out_, int KH_, int KW_,
                     int stride_, int padding_, float lrate_) {
    C_in    = C_in_;
    C_out   = C_out_;
    KH      = KH_;
    KW      = KW_;
    stride  = stride_;
    padding = padding_;
    lrate   = lrate_;

    int K = KH * KW * C_in;

    kernel = tensor_alloc(1, 1, K, C_out);
    bias   = tensor_alloc(1, 1, 1, C_out);
    tensor_zero(bias);

    // He init: std = sqrt(2 / fan_in), fan_in = KH * KW * C_in.
    std::vector<float> hostK((size_t)K * C_out);
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / (float)K));
    for (float& w : hostK) w = nd(rng);
    tensor_upload(kernel, hostK.data());

    opt_K.init(K * C_out, lrate);
    opt_B.init(C_out, lrate);

    patches.data = nullptr;
    patches_cap  = 0;
}

void Conv_Layer::forward(const Tensor& x, Tensor& out) {
    last_in = &x;

    int N     = x.N;
    int H     = x.H;
    int W     = x.W;
    int out_H = (H + 2 * padding - KH) / stride + 1;
    int out_W = (W + 2 * padding - KW) / stride + 1;
    int M     = N * out_H * out_W;
    int K     = KH * KW * C_in;

    // (Re)allocate patches buffer if too small. Same growth policy as Pool_Layer.
    int need = M * K;
    if (need > patches_cap) {
        if (patches.data) tensor_free(patches);
        patches = tensor_alloc(1, 1, M, K);
        patches_cap = need;
    } else {
        // Reuse: keep allocation, just rewrite the logical shape.
        patches.H = 1;
        patches.W = M;
        patches.C = K;
    }

    int threads = 256;
    int im2col_total = M * K;
    conv_im2col_kernel<<<conv_blocks_for(im2col_total, threads), threads>>>(
        x.data, patches.data,
        N, H, W, C_in,
        KH, KW, out_H, out_W,
        stride, padding
    );

    // out (M, C_out) = patches (M, K) @ kernel (K, C_out)
    gemm(patches.data, kernel.data, out.data, M, K, C_out);

    int out_total = M * C_out;
    conv_add_bias_kernel<<<conv_blocks_for(out_total, threads), threads>>>(
        out.data, bias.data, out_total, C_out);
}

void Conv_Layer::backward(const Tensor& dout, Tensor& dx) {
    int N     = last_in->N;
    int H     = last_in->H;
    int W     = last_in->W;
    int out_H = dout.H;
    int out_W = dout.W;
    int M     = N * out_H * out_W;
    int K     = KH * KW * C_in;
    int threads = 256;

    float *Wt, *patchesT, *dW, *dB;
    cudaMalloc(&Wt,       sizeof(float) * (size_t)C_out * K);
    cudaMalloc(&patchesT, sizeof(float) * (size_t)K * M);
    cudaMalloc(&dW,       sizeof(float) * (size_t)K * C_out);
    cudaMalloc(&dB,       sizeof(float) * (size_t)C_out);

    // dW (K, C_out) = patches^T (K, M) @ dout (M, C_out)
    conv_transpose_kernel<<<conv_blocks_for(M * K, threads), threads>>>(
        patches.data, patchesT, M, K);
    gemm(patchesT, dout.data, dW, K, M, C_out);

    // dpatches (M, K) = dout (M, C_out) @ kernel^T (C_out, K).
    // We no longer need `patches` after computing dW, so reuse its buffer.
    conv_transpose_kernel<<<conv_blocks_for(K * C_out, threads), threads>>>(
        kernel.data, Wt, K, C_out);
    gemm(dout.data, Wt, patches.data, M, C_out, K);

    // dx = col2im(dpatches). Atomic scatter, so zero dx first.
    cudaMemset(dx.data, 0, sizeof(float) * (size_t)N * H * W * C_in);
    conv_col2im_kernel<<<conv_blocks_for(M * K, threads), threads>>>(
        patches.data, dx.data,
        N, H, W, C_in,
        KH, KW, out_H, out_W,
        stride, padding
    );

    // dB (C_out,) = column sum of dout viewed as (M, C_out).
    conv_colsum_kernel<<<conv_blocks_for(C_out, threads), threads>>>(
        dout.data, dB, M, C_out);

    // Adam averages dW/dB over the batch (divide by N, not M — the spatial
    // sum across out_H*out_W is the correct chain-rule accumulation).
    opt_K.step(kernel.data, dW, N);
    opt_B.step(bias.data,   dB, N);

    cudaFree(Wt);
    cudaFree(patchesT);
    cudaFree(dW);
    cudaFree(dB);
}

Tensor Conv_Layer::output_shape(const Tensor& x) {
    int out_H = (x.H + 2 * padding - KH) / stride + 1;
    int out_W = (x.W + 2 * padding - KW) / stride + 1;
    return {nullptr, x.N, out_H, out_W, C_out};
}
