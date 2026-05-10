#include "layer.cuh"

__global__ void pool_forward_kernel(
    const float* xdata, int* maxidxs, float* out,
    int H, int W, int C,
    int out_H, int out_W,
    int KH, int KW, int stride,
    int total
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    int c = i % C;
    int ow = (i / C) % out_W;
    int oh = i / (C * out_W) % out_H;
    int n = i / (C * out_W * out_H);
    int iw = ow * stride;
    int ih = oh * stride;
    int idx = n * H * W * C + ih * W * C + iw * C + c;

    int maxidx = idx;
    float maxval = xdata[idx];
    for (int iter_h = 0; iter_h < KH; iter_h++) {
        for (int iter_w = 0; iter_w < KW; iter_w++) {
            int patch_idx = idx + C * W * iter_h + C * iter_w;
            if (xdata[patch_idx] > maxval) {
                maxval = xdata[patch_idx];
                maxidx = patch_idx;
            }
        }
    }
    maxidxs[i] = maxidx;
    out[i] = maxval;
}

__global__ void pool_backward_kernel(const float* doutdata, const int* maxidxs, float* dxdata, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int maxidx = maxidxs[i];
        atomicAdd(&dxdata[maxidx], doutdata[i]);
    }
}

class Pool_Layer : public Layer {
public:
    int* maxes_idx = nullptr;
    int KH, KW, stride;
    int maxes_cap;

    ~Pool_Layer() override {
        if (maxes_idx) {
            cudaFree(maxes_idx);
        }
    }

    Pool_Layer(const Pool_Layer&) = delete;
    Pool_Layer& operator=(const Pool_Layer&) = delete;

    void init(int KH, int KW, int stride) {
        this->KH = KH;
        this->KW = KW;
        this->stride = stride;
        this->maxes_idx = nullptr;
        this->maxes_cap = 0;
    }

    void forward(const Tensor& x, Tensor& out) override {
        int n = out.N * out.H * out.W * out.C;
        if (n > maxes_cap) {
            if (maxes_idx) {
                cudaFree(maxes_idx);
            }
            cudaMalloc(&maxes_idx, n * sizeof(int));
            maxes_cap = n;
        }
        
        int threads = 256;
        int blocks = (n - 1) / threads + 1;
        pool_forward_kernel<<<blocks, threads>>>(x.data, maxes_idx, out.data, x.H, x.W, x.C, out.H, out.W, KH, KW, stride, n);
    }

    void backward(const Tensor& dout, Tensor& dx) override {
        cudaMemset(dx.data, 0, dx.N * dx.H * dx.W * dx.C * sizeof(float));
        int n = dout.N * dout.H * dout.W * dout.C;
        int threads = 256;
        int blocks = (n - 1) / threads + 1;
        pool_backward_kernel<<<blocks, threads>>>(dout.data, maxes_idx, dx.data, n);
    }

    Tensor output_shape(const Tensor& x) override {
        return {nullptr, x.N, (x.H - KH) / stride + 1, (x.W - KW) / stride + 1, x.C};
    }
};
