#pragma once

struct Tensor {
    float* data;
    int N, H, W, C;
};

Tensor tensor_alloc(int N, int H, int W, int C);
void tensor_free(Tensor& t);
void tensor_upload(Tensor& t, const float* host);
void tensor_download(const Tensor& t, float* host);
void tensor_zero(Tensor& t);
