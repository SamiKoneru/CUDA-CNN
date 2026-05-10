#include "layer.cuh"

__global__ void relu_forward_kernel(const float* xdata, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (xdata[i] < 0) {
            out[i] = 0;
        }
        else {
            out[i] = xdata[i];
        }
    }
}

__global__ void relu_backward_kernel(const float* xdata, const float* dout, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (xdata[i] < 0) {
            dx[i] = 0;
        }
        else {
            dx[i] = dout[i];
        }
    }
}

class ReLU : public Layer {
public:
    const Tensor* last_in = nullptr;

    void forward(const Tensor& x, Tensor& out) override {
        int n = x.N * x.H * x.W * x.C;

        last_in = &x;

        int threads = 256;
        int blocks = (n - 1) / threads + 1;
        relu_forward_kernel<<<blocks, threads>>>(x.data, out.data, n);
    }

    void backward(const Tensor& dout, Tensor& dx) override {
        int n = last_in->N * last_in->H * last_in->W * last_in->C;
        int threads = 256;
        int blocks = (n - 1) / threads + 1;

        relu_backward_kernel<<<blocks, threads>>>(last_in->data, dout.data, dx.data, n);

    }

    Tensor output_shape(const Tensor& x) override {
        return {nullptr, x.N, x.H, x.W, x.C};
    }
};
