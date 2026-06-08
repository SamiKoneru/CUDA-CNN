#include <cuda_runtime.h>
#include <cmath>
#include "adam.cuh"

__global__ void adam_step_kernel(
    float* param, const float* grad,
    float* m, float* v,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2, float scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grad[i] * scale;
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float mh = mi / bc1;
    float vh = vi / bc2;
    param[i] -= lr * mh / (sqrtf(vh) + eps);
}

void Adam::init(int n, float lr, float beta1, float beta2, float eps) {
    this->n = n;
    this->lr = lr;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->eps = eps;
    this->t = 0;
    cudaMalloc(&m, n * sizeof(float));
    cudaMalloc(&v, n * sizeof(float));
    cudaMemset(m, 0, n * sizeof(float));
    cudaMemset(v, 0, n * sizeof(float));
}

void Adam::step(float* param, const float* grad, int batch_size) {
    t++;
    float bc1   = 1.0f - powf(beta1, (float)t);
    float bc2   = 1.0f - powf(beta2, (float)t);
    float scale = 1.0f / (float)batch_size;

    int threads = 256;
    int blocks  = (n - 1) / threads + 1;
    adam_step_kernel<<<blocks, threads>>>(
        param, grad, m, v,
        lr, beta1, beta2, eps,
        bc1, bc2, scale, n
    );
}

Adam::~Adam() {
    cudaFree(m);
    cudaFree(v);
}
