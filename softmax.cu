#include "layer.cuh"

__global__ void softmax_forward_kernel(const float* xdata, float* out, int C) {
    extern __shared__ float exps[];
    int local_i = threadIdx.x;
    int global_i = blockIdx.x * C + local_i;
    float thisval = xdata[global_i];
    exps[local_i] = thisval;
    __syncthreads();

    int half = C;
    while (half > 1) {
        if (local_i == 0 && half & 1 && exps[half - 1] > exps[local_i]) {
            exps[local_i] = exps[half - 1];
        }
        __syncthreads();
        half >>= 1;
        if (local_i < half && exps[local_i + half] > exps[local_i]) {
            exps[local_i] = exps[local_i + half];
        }
        __syncthreads();
    }
    float maxval = exps[0];
    __syncthreads();

    thisval = expf(thisval - maxval);
    exps[local_i] = thisval;
    half = C;
    __syncthreads();
    while (half > 1) {
        if (local_i == 0 && half & 1) {
            exps[local_i] += exps[half - 1];
        }
        __syncthreads();
        half >>= 1;
        if (local_i < half) {
            exps[local_i] += exps[local_i + half];
        }
        __syncthreads();
    }

    out[global_i] = thisval / exps[0];
}

__global__ void softmax_backward_kernel(const float* out, const float* dout, float* dx, int C) {
    extern __shared__ float prods[];
    int local_i = threadIdx.x;
    int global_i = blockIdx.x * C + local_i;
    prods[local_i] = dout[global_i] * out[global_i];
    __syncthreads();

    int half = C;
    while (half > 1) {
        if (local_i == 0 && half & 1) {
            prods[local_i] += prods[half - 1];
        }
        __syncthreads();
        half >>= 1;
        if (local_i < half) {
            prods[local_i] += prods[local_i + half];
        }
        __syncthreads();
    }

    dx[global_i] = out[global_i] * (dout[global_i] - prods[0]);
}

class Softmax : public Layer {
public:
    const Tensor* last_out = nullptr;

    void forward(const Tensor& x, Tensor& out) override {
        softmax_forward_kernel<<<x.N, x.C, x.C * sizeof(float)>>>(x.data, out.data, x.C);
        last_out = &out;
    }
    void backward(const Tensor& dout, Tensor& dx) override {
        softmax_backward_kernel<<<dout.N, dout.C, dout.C * sizeof(float)>>>(last_out->data, dout.data, dx.data, dout.C);
    }
    Tensor output_shape(const Tensor& x) override {
        return {nullptr, x.N, x.H, x.W, x.C};
    }
};
