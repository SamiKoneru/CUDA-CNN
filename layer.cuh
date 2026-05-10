#pragma once
#include "tensor.cuh"

class Layer {
public:
    virtual void forward(const Tensor& x, Tensor& out) = 0;
    virtual void backward(const Tensor& dout, Tensor& dx) = 0;
    virtual Tensor output_shape(const Tensor& x) = 0;
    virtual ~Layer() {}
};
