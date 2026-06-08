#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>

// Unity build: pull every translation unit into this one. Avoids needing
// per-layer .cuh headers. The Makefile compiles only main.cu but depends
// on each .cu so edits anywhere trigger a rebuild.
#include "tensor.cu"
#include "GEMM.cu"
#include "adam.cu"
#include "relu.cu"
#include "softmax.cu"
#include "pool.cu"
#include "dense.cu"
#include "conv.cu"
#include "mnist.cu"
#include "network.cu"

int main() {
    // ---------- Load MNIST from current directory ----------
    std::vector<float> train_imgs, test_imgs;
    std::vector<int>   train_lbls, test_lbls;
    int N_train, N_test, H, W, Nl;

    if (!load_mnist_images("train-images-idx3-ubyte", train_imgs, N_train, H, W)) {
        fprintf(stderr, "failed to load train-images-idx3-ubyte\n"); return 1;
    }
    if (!load_mnist_labels("train-labels-idx1-ubyte", train_lbls, Nl) || Nl != N_train) {
        fprintf(stderr, "failed to load train-labels-idx1-ubyte\n"); return 1;
    }
    if (!load_mnist_images("t10k-images-idx3-ubyte",  test_imgs,  N_test,  H, W)) {
        fprintf(stderr, "failed to load t10k-images-idx3-ubyte\n"); return 1;
    }
    if (!load_mnist_labels("t10k-labels-idx1-ubyte",  test_lbls,  Nl) || Nl != N_test) {
        fprintf(stderr, "failed to load t10k-labels-idx1-ubyte\n"); return 1;
    }
    printf("MNIST: %d train, %d test (%dx%d)\n", N_train, N_test, H, W);

    // ---------- Upload to GPU ----------
    Tensor train_x = tensor_alloc(N_train, H, W, 1);
    Tensor test_x  = tensor_alloc(N_test,  H, W, 1);
    tensor_upload(train_x, train_imgs.data());
    tensor_upload(test_x,  test_imgs.data());

    int *train_y_dev = nullptr, *test_y_dev = nullptr;
    cudaMalloc(&train_y_dev, (size_t)N_train * sizeof(int));
    cudaMalloc(&test_y_dev,  (size_t)N_test  * sizeof(int));
    cudaMemcpy(train_y_dev, train_lbls.data(),
               (size_t)N_train * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(test_y_dev,  test_lbls.data(),
               (size_t)N_test  * sizeof(int), cudaMemcpyHostToDevice);

    // ---------- Build network ----------
    // (N, 28, 28, 1)
    //   -> Conv 1->8  3x3 pad 1   (N, 28, 28, 8)
    //   -> ReLU
    //   -> Pool 2x2  stride 2     (N, 14, 14, 8)
    //   -> Conv 8->16 3x3 pad 1   (N, 14, 14, 16)
    //   -> ReLU
    //   -> Pool 2x2  stride 2     (N,  7,  7, 16)
    //   -> Flatten                (N,  1,  1, 784)
    //   -> Dense 784->64 -> ReLU
    //   -> Dense  64->10 -> Softmax
    float lr = 1e-3f;

    Conv_Layer  conv1;   conv1.init(1,  8, 3, 3, /*stride*/1, /*pad*/1, lr);
    ReLU        relu1;
    Pool_Layer  pool1;   pool1.init(2, 2, 2);

    Conv_Layer  conv2;   conv2.init(8, 16, 3, 3, 1, 1, lr);
    ReLU        relu2;
    Pool_Layer  pool2;   pool2.init(2, 2, 2);

    Dense_Layer dense1;  dense1.init(7 * 7 * 16, 64, lr);
    ReLU        relu3;
    Dense_Layer dense2;  dense2.init(64, 10, lr);
    Softmax     softmax;

    Network net;
    net.add(LayerKind::Conv,    &conv1);
    net.add(LayerKind::ReLU,    &relu1);
    net.add(LayerKind::Pool,    &pool1);
    net.add(LayerKind::Conv,    &conv2);
    net.add(LayerKind::ReLU,    &relu2);
    net.add(LayerKind::Pool,    &pool2);
    net.add(LayerKind::Flatten, nullptr);
    net.add(LayerKind::Dense,   &dense1);
    net.add(LayerKind::ReLU,    &relu3);
    net.add(LayerKind::Dense,   &dense2);
    net.add(LayerKind::Softmax, &softmax);

    int batch_size = 64;
    int epochs     = 20;

    printf("\ntraining: %d epochs, batch=%d, lr=%.4f\n", epochs, batch_size, lr);
    net.train(train_x, train_y_dev, batch_size, epochs);

    printf("\nevaluating on test set...\n");
    net.evaluate(test_x, test_y_dev, batch_size);

    // ---------- Cleanup ----------
    cudaFree(train_y_dev);
    cudaFree(test_y_dev);
    tensor_free(train_x);
    tensor_free(test_x);
    return 0;
}
