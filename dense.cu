#include "layer.cuh"

class Dense_Layer : public Layer {
public:
    Tensor W;
    Tensor B;
    const Tensor* last_in = nullptr;
    int x_in, y_out;
    float lrate;

    void init(int x_in, int y_out, float lrate);
    void forward(const Tensor& x, Tensor& out) override;
    void backward(const Tensor& dout, Tensor& dx) override;
    Tensor output_shape(const Tensor& x) override;
};
