# CNN from scratch in C++/CUDA

A convolutional neural network built from the ground up in C++ and CUDA — no PyTorch, no TensorFlow, no cuDNN. Every layer's forward and backward pass is a hand-written CUDA kernel. Goal is to train a CNN to >99% on MNIST and benchmark the throughput against an equivalent PyTorch (cuDNN) model on a Colab T4.

## Status

End-to-end MNIST pipeline is wired: tensor ops, GEMM, Adam, all layers, network orchestrator, IDX loader, and `main.cu` are all in place. Development happens on macOS so nothing has been compiled or run yet — the first Colab T4 run is the real integration test.

### Done

- Tensor struct + memory ops
- Tiled, shared-memory GEMM
- Adam optimizer (reusable per-parameter helper)
- ReLU (forward + backward)
- Softmax (batched, numerically stable, forward + backward)
- Max pooling (forward + backward)
- Dense layer (im2col-free via GEMM, Adam-trained)
- Convolution layer (im2col + GEMM, Adam-trained)
- Network orchestrator (forward/backward chain, flatten-as-reshape, per-batch loss/accuracy logging)
- Fused softmax + cross-entropy gradient (skips `Softmax::backward` for numerical stability)
- MNIST IDX parser
- Entry point: load MNIST, build CNN, train, evaluate on test set

### Known limitations

- No CUDA error checks anywhere — failed kernel launches are silent and surface as NaN loss
- Dataset is not shuffled between epochs
- Dense and Conv backward `cudaMalloc`/`cudaFree` scratch buffers every step (Wt, xT/patchesT, dW, dB) — wasteful but not a correctness bug
- Pool layer assumes the input dims are exactly divisible by stride; no padding handling
- Softmax assumes `(N, 1, 1, C)` layout and uses one block per sample with `C` threads — breaks for `C > 1024`
- No regression tests; no way to save/load weights

## Files

| File                  | What it is                                       |
| --------------------- | ------------------------------------------------ |
| `tensor.{cuh,cu}`     | GPU tensor struct + memory ops                   |
| `layer.cuh`           | Abstract `Layer` interface                       |
| `gemm.cuh` / `GEMM.cu`| Tiled shared-memory GEMM                         |
| `adam.{cuh,cu}`       | Reusable per-parameter Adam optimizer            |
| `relu.cu`             | Elementwise ReLU                                 |
| `softmax.cu`          | Batched stable softmax                           |
| `pool.cu`             | Max pool                                         |
| `dense.cu`            | Dense layer (GEMM + Adam)                        |
| `conv.cu`             | Conv layer (im2col + GEMM + Adam)                |
| `network.cu`          | Sequential orchestrator + loss/metric kernels    |
| `mnist.cu`            | IDX big-endian parser                            |
| `main.cu`             | Entry point: load MNIST, build CNN, train, eval  |

## Build

Requires NVIDIA GPU + CUDA toolkit (won't run on macOS / Apple Silicon).

```bash
make
```

Project uses a unity build: only `main.cu` is compiled, but it `#include`s every other `.cu`. The Makefile lists each source as a dependency so edits anywhere trigger a rebuild.

```
nvcc -O3 -std=c++17 main.cu -o cnn
```

To target the T4 architecture explicitly (skips JIT at startup):

```bash
make NVCCFLAGS="-O3 -std=c++17 -arch=sm_75"
```

## Run on Google Colab (T4)

```python
# Cell 1 — confirm GPU and toolchain
!nvidia-smi
!nvcc --version
```

If `nvidia-smi` errors out, the runtime is CPU-only — Runtime → Change runtime type → T4 GPU → Save, then reconnect.

```python
# Cell 2 — get the code and build
!git clone https://github.com/<your-username>/CNN_scratch_cuda.git
%cd CNN_scratch_cuda
!make NVCCFLAGS="-O3 -std=c++17 -arch=sm_75"
```

```python
# Cell 3 — fetch MNIST IDX files into the same directory as ./cnn
!wget -q https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
!wget -q https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
!wget -q https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
!wget -q https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
!gunzip -f *.gz
```

```python
# Cell 4 — train and evaluate
!./cnn
```

`main.cu` opens the IDX files by relative path, so they must sit next to the `cnn` binary.

## Network architecture

Default network in `main.cu` (~58k parameters):

```
input (N, 28, 28, 1)
  -> Conv 1->8,  3x3, pad 1, stride 1 -> ReLU -> MaxPool 2x2 stride 2  (N, 14, 14,  8)
  -> Conv 8->16, 3x3, pad 1, stride 1 -> ReLU -> MaxPool 2x2 stride 2  (N,  7,  7, 16)
  -> Flatten                                                            (N, 784)
  -> Dense 784->64 -> ReLU
  -> Dense  64->10 -> Softmax
```

Trained with Adam (lr=1e-3), batch 64, 5 epochs. Should clear 98% on this config; for >99% expect to need more epochs, lr decay, and shuffled batches.

## Design notes

- **Memory layout**: `(N, H, W, C)` (NHWC).
- **Layer ownership**: activation buffers are owned by the network and passed to layers as `Tensor& out` / `Tensor& dx`. Layers only own internal state (weights, biases, pool index buffers, etc.).
- **Flatten is free**: in NHWC with contiguous storage, flattening is just a reshape of the `Tensor` metadata — no kernel, no data movement. `Network` allocates no buffer for Flatten; it aliases the previous activation in forward and the incoming gradient in backward.
- **Fused softmax + cross-entropy**: training computes `(p - y)` directly as the gradient w.r.t. the pre-softmax logits and starts backward at layer `L-2`, skipping `Softmax::backward`. Numerically stable (no divide-by-`p`). The standalone `Softmax::backward` is still correct and gets used by anyone calling `Network::backward(dout)` with a non-xent loss.
- **Gradient scaling convention**: Dense and Conv backward produce *batch-summed* gradients via GEMM; `Adam::step` divides by `batch_size` internally. Every gradient producer in the network follows this convention — including `net_softmax_xent_grad_kernel`, which emits `(p - y)`, not `(p - y) / N`.
- **Layer dispatch**: `Network` stores `void*` layer pointers and casts via `static_cast<Layer*>` in `as_layer_`. Assumes single inheritance from `Layer` (every concrete layer in this project satisfies it). Flatten has no `Layer` object — pass `nullptr` for `impl`.
- **Unity build**: layer classes are defined inline in their `.cu` files rather than in `.cuh` headers. `main.cu` `#include`s each `.cu` so the linker only sees one translation unit. Trade-off: simpler dependency graph at the cost of a single-file compile.

## Performance target

End-to-end CIFAR-scale CNN on Colab T4, target throughput is ~10% of PyTorch+cuDNN running the same model. The gap is dominated by cuDNN's algorithm selection, tensor core utilization, and kernel fusion — none of which this implementation has.
