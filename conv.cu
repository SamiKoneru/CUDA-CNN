#include "layer.cuh"

class Conv_Layer : public Layer {
public:
    Tensor kernel;
    Tensor bias;
    const Tensor* last_in = nullptr;
    Tensor patches;
    int KH, KW, C_in, C_out;
    int stride, padding;
    float lrate;

    void init(int C_in, int C_out, int KH, int KW,
              int stride, int padding, float lrate);
    void forward(const Tensor& x, Tensor& out) override;
    void backward(const Tensor& dout, Tensor& dx) override;
    Tensor output_shape(const Tensor& x) override;
};
