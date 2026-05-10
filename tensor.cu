#include <cuda_runtime.h>
#include "tensor.cuh"

Tensor tensor_alloc(int N, int H, int W, int C) {
    Tensor t;
    t.N = N;
    t.H = H;
    t.W = W;
    t.C = C;
    cudaMalloc(&t.data, N*H*W*C*sizeof(float));
    return t;
}

void tensor_free(Tensor& t) {
    cudaFree(t.data);
    t.data = nullptr;
}

void tensor_upload(Tensor& t, const float* host) {
    cudaMemcpy(t.data, host, t.N * t.H * t.W * t.C * sizeof(float), cudaMemcpyHostToDevice);
}

void tensor_download(const Tensor& t, float* host) {
    cudaMemcpy(host, t.data, t.N * t.H * t.W * t.C * sizeof(float), cudaMemcpyDeviceToHost);
}

void tensor_zero(Tensor& t) {
    cudaMemset(t.data, 0, t.N * t.H * t.W * t.C * sizeof(float));
}
