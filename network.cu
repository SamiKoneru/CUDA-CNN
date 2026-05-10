#include "tensor.cuh"
#include <vector>

enum class LayerKind { Conv, Pool, Flatten, Dense, ReLU, Softmax };

struct LayerOp {
    LayerKind kind;
    void* impl;
};

class Network {
public:
    std::vector<LayerOp> layers;

    void add(LayerKind k, void* impl);
    void forward(const Tensor& x, Tensor& out);
    void backward(const Tensor& dout);
    void epoch(const Tensor& x, const int* y, int batch_size);
    void train(const Tensor& x, const int* y, int batch_size, int epochs);
};
