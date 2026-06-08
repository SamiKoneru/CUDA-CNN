#pragma once

// Per-parameter Adam state. Owns two device buffers (m, v) sized to the
// parameter tensor. Construct empty, call init() once, then step() each
// backward pass.
struct Adam {
    float* m = nullptr;     // first moment, device
    float* v = nullptr;     // second moment, device
    int n = 0;
    int t = 0;              // step counter, for bias correction
    float lr, beta1, beta2, eps;

    Adam() = default;
    Adam(const Adam&) = delete;
    Adam& operator=(const Adam&) = delete;
    ~Adam();

    void init(int n, float lr,
              float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    // param -= lr * mhat / (sqrt(vhat) + eps), with grads scaled by 1/batch_size
    // so callers can pass in unaveraged batch-summed gradients (matches how
    // the dense/conv layers produce dW via GEMM).
    void step(float* param, const float* grad, int batch_size);
};
