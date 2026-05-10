# CNN from scratch in C++/CUDA

A convolutional neural network built from the ground up in C++ and CUDA — no PyTorch, no TensorFlow, no cuDNN. Every layer's forward and backward pass is a hand-written CUDA kernel. Goal is to train a CNN to >99% on MNIST and benchmark the throughput against an equivalent PyTorch (cuDNN) model on a Colab T4.

## Status

Work in progress. The model can't train end-to-end yet — conv, dense, the network orchestrator, and the training loop are still TODO.

### Done

- ReLU — forward + backward
- Softmax — batched, numerically stable, forward + backward
- Max pooling — forward + backward

### Not started

- Convolution layer
- Dense layer
- Adam optimizer
- Network orchestrator (forward/backward pipeline across layers)
- MNIST loader
- Training loop / entry point

## Files

| File              | Status | What it is                                       |
| ----------------- | ------ | ------------------------------------------------ |
| `tensor.{cuh,cu}` | ✅      | GPU tensor struct + memory ops                   |
| `layer.cuh`       | ✅      | Abstract `Layer` interface                       |
| `relu.cu`         | ✅      | Elementwise ReLU                                 |
| `softmax.cu`      | ✅      | Batched stable softmax                           |
| `pool.cu`         | ✅      | Max pool                                         |
| `conv.cu`         | ❌      | Skeleton                                         |
| `dense.cu`        | ❌      | Skeleton                                         |
| `network.cu`      | ❌      | Skeleton                                         |
| `mnist.cu`        | ❌      | Header only                                      |
| `main.cu`         | ❌      | Empty stub                                       |

## Build

Requires NVIDIA GPU + CUDA toolkit (won't run on macOS / Apple Silicon).

```bash
make
```

The Makefile uses:

```
nvcc -O3 -std=c++17 $(SRCS) -o cnn
```

Add `-arch=sm_75` (T4) or `-arch=native` if you want to skip JIT at startup.

## Run on Google Colab (T4)

```python
# Cell 1
!nvidia-smi          # confirm you got a T4
!nvcc --version

# Cell 2 — clone and build
!git clone https://github.com/<your-username>/CNN_scratch_cuda.git
%cd CNN_scratch_cuda
!make

# Cell 3 — get MNIST
!wget -q https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
!wget -q https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
!wget -q https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
!wget -q https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
!gunzip *.gz

# Cell 4 — run
!./cnn
```

Make sure the runtime is set to GPU (Runtime → Change runtime type → T4 GPU).

## Design notes

- **Memory layout**: `(N, H, W, C)` (NHWC).
- **Layer ownership**: activation buffers are owned by the network and passed to layers as `Tensor& out` / `Tensor& dx`. Layers only own internal state (weights, biases, pool index buffers, etc.).
- **Flatten is free**: in NHWC with contiguous storage, flattening is just a reshape of the `Tensor` metadata — no kernel, no data movement.

## Performance target

End-to-end CIFAR-scale CNN on Colab T4, target throughput is ~10% of PyTorch+cuDNN running the same model. The gap is dominated by cuDNN's algorithm selection, tensor core utilization, and kernel fusion — none of which this implementation has.
