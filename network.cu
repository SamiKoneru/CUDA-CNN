#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include "tensor.cuh"
#include "layer.cuh"

enum class LayerKind { Conv, Pool, Flatten, Dense, ReLU, Softmax };

// `impl` holds a pointer to one of the layer classes. We cast it to Layer*
// for dispatch — relies on each layer being single-inheritance from Layer
// (which all of relu/softmax/pool/dense/conv satisfy). For Flatten there's
// no Layer object; impl is ignored (pass nullptr).
struct LayerOp {
    LayerKind kind;
    void* impl;
};

// ---- Loss / metric kernels ----

// dout (N, C) = probs - one_hot(labels) — fused softmax+xent gradient.
// This is the gradient w.r.t. the *pre-softmax logits*, so when used we feed
// it directly to the layer BEFORE softmax and skip softmax.backward.
// No /N here: dense/conv backward produce batch-summed gradients via GEMM
// and Adam::step does the /N averaging at the optimizer call.
__global__ void net_softmax_xent_grad_kernel(
    const float* probs, const int* labels, float* dout,
    int N, int C
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * C) return;
    int n = i / C;
    int c = i % C;
    float y = (c == labels[n]) ? 1.0f : 0.0f;
    dout[i] = probs[i] - y;
}

__global__ void net_xent_loss_kernel(
    const float* probs, const int* labels, float* out_sum,
    int N, int C
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float p = probs[n * C + labels[n]];
    if (p < 1e-12f) p = 1e-12f;
    atomicAdd(out_sum, -logf(p));
}

__global__ void net_argmax_hit_kernel(
    const float* probs, const int* labels, int* out_hits,
    int N, int C
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = probs + n * C;
    float best = row[0];
    int   bi   = 0;
    for (int c = 1; c < C; c++) {
        if (row[c] > best) { best = row[c]; bi = c; }
    }
    if (bi == labels[n]) atomicAdd(out_hits, 1);
}

static inline int net_blocks_for(int n, int threads) { return (n - 1) / threads + 1; }

// ----------------------------------------------------------------

class Network {
public:
    std::vector<LayerOp> layers;

    // acts[0] = input (metadata only; data owned by caller).
    // acts[i+1] = output of layer i. For non-flatten layers we own the buffer;
    // for flatten we alias the previous activation (no own buffer).
    std::vector<Tensor> acts;
    std::vector<bool>   owns_act;

    // dxs[i] = gradient w.r.t. acts[i] (= input of layer i).
    // For non-flatten layers we own the buffer; for flatten we alias the dx
    // coming back from the next layer (reshaped to unflat).
    std::vector<Tensor> dxs;
    std::vector<bool>   owns_dx;

    int alloc_N = 0;

    // Scratch: pre-softmax-logit gradient, shape (N, 1, 1, C).
    // Allocated only if the final layer is Softmax (the training path).
    Tensor fused_grad{};
    bool   owns_fused = false;

    // Device scratch for per-batch loss/accuracy reductions.
    float* d_loss_sum  = nullptr;
    int*   d_hit_count = nullptr;

    Network() = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    ~Network();

    void add(LayerKind k, void* impl);
    void forward(const Tensor& x, Tensor& out);
    void backward(const Tensor& dout);
    void epoch(const Tensor& x, const int* y, int batch_size);
    void train(const Tensor& x, const int* y, int batch_size, int epochs);
    void evaluate(const Tensor& x, const int* y, int batch_size);

private:
    void allocate_buffers_(const Tensor& x);
    void free_buffers_();
    void backward_from_(int start_layer, Tensor next_dout);
    void compute_metrics_(const int* d_labels, int batch_size, int C,
                          float* out_loss, float* out_acc);

    static Layer* as_layer_(const LayerOp& op) {
        return static_cast<Layer*>(op.impl);
    }
};

Network::~Network() {
    free_buffers_();
}

void Network::add(LayerKind k, void* impl) {
    layers.push_back({k, impl});
}

void Network::free_buffers_() {
    for (size_t i = 0; i < acts.size(); i++) {
        if (owns_act[i]) tensor_free(acts[i]);
    }
    for (size_t i = 0; i < dxs.size(); i++) {
        if (owns_dx[i]) tensor_free(dxs[i]);
    }
    if (owns_fused && fused_grad.data) tensor_free(fused_grad);
    if (d_loss_sum)  { cudaFree(d_loss_sum);  d_loss_sum  = nullptr; }
    if (d_hit_count) { cudaFree(d_hit_count); d_hit_count = nullptr; }
    acts.clear();
    dxs.clear();
    owns_act.clear();
    owns_dx.clear();
    owns_fused = false;
    fused_grad = Tensor{};
    alloc_N = 0;
}

void Network::allocate_buffers_(const Tensor& x) {
    free_buffers_();
    int L = (int)layers.size();
    acts.assign(L + 1, Tensor{});
    dxs.assign(L, Tensor{});
    owns_act.assign(L + 1, false);
    owns_dx.assign(L, false);

    acts[0] = x;
    owns_act[0] = false;

    // Walk forward through the layers to compute output shapes.
    for (int i = 0; i < L; i++) {
        const LayerOp& op = layers[i];
        if (op.kind == LayerKind::Flatten) {
            int k = acts[i].H * acts[i].W * acts[i].C;
            acts[i + 1] = {acts[i].data, acts[i].N, 1, 1, k};
            owns_act[i + 1] = false;
        } else {
            Tensor s = as_layer_(op)->output_shape(acts[i]);
            acts[i + 1] = tensor_alloc(s.N, s.H, s.W, s.C);
            owns_act[i + 1] = true;
        }
    }

    // dxs[i] mirrors acts[i] shape (input shape of layer i).
    for (int i = 0; i < L; i++) {
        const LayerOp& op = layers[i];
        const Tensor& s = acts[i];
        if (op.kind == LayerKind::Flatten) {
            dxs[i] = {nullptr, s.N, s.H, s.W, s.C};
            owns_dx[i] = false;
        } else {
            dxs[i] = tensor_alloc(s.N, s.H, s.W, s.C);
            owns_dx[i] = true;
        }
    }

    if (L > 0 && layers[L - 1].kind == LayerKind::Softmax) {
        const Tensor& s = acts[L - 1];
        fused_grad = tensor_alloc(s.N, s.H, s.W, s.C);
        owns_fused = true;
    }

    cudaMalloc(&d_loss_sum,  sizeof(float));
    cudaMalloc(&d_hit_count, sizeof(int));
    alloc_N = x.N;
}

void Network::forward(const Tensor& x, Tensor& out) {
    if (acts.empty() || alloc_N != x.N) {
        allocate_buffers_(x);
    } else {
        // Same batch size: keep buffers, just retarget the input metadata.
        acts[0] = x;
    }

    int L = (int)layers.size();
    for (int i = 0; i < L; i++) {
        const LayerOp& op = layers[i];
        if (op.kind == LayerKind::Flatten) {
            // Re-alias each forward in case acts[i].data shifted (new batch slice).
            acts[i + 1].data = acts[i].data;
        } else {
            as_layer_(op)->forward(acts[i], acts[i + 1]);
        }
    }
    out = acts[L];
}

void Network::backward(const Tensor& dout_last) {
    int L = (int)layers.size();
    for (int i = L - 1; i >= 0; i--) {
        const LayerOp& op = layers[i];
        const Tensor* dout_in = (i == L - 1) ? &dout_last : &dxs[i + 1];

        if (op.kind == LayerKind::Flatten) {
            dxs[i].data = dout_in->data;
        } else {
            as_layer_(op)->backward(*dout_in, dxs[i]);
        }
    }
}

void Network::backward_from_(int start_layer, Tensor next_dout) {
    for (int i = start_layer; i >= 0; i--) {
        const LayerOp& op = layers[i];
        if (op.kind == LayerKind::Flatten) {
            dxs[i].data = next_dout.data;
            next_dout = dxs[i];
        } else {
            as_layer_(op)->backward(next_dout, dxs[i]);
            next_dout = dxs[i];
        }
    }
}

void Network::compute_metrics_(const int* d_labels, int batch_size, int C,
                               float* out_loss, float* out_acc) {
    cudaMemset(d_loss_sum,  0, sizeof(float));
    cudaMemset(d_hit_count, 0, sizeof(int));

    const Tensor& probs = acts[layers.size()];
    int threads = 256;
    net_xent_loss_kernel<<<net_blocks_for(batch_size, threads), threads>>>(
        probs.data, d_labels, d_loss_sum, batch_size, C);
    net_argmax_hit_kernel<<<net_blocks_for(batch_size, threads), threads>>>(
        probs.data, d_labels, d_hit_count, batch_size, C);

    float h_loss = 0.0f;
    int   h_hits = 0;
    cudaMemcpy(&h_loss, d_loss_sum,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_hits, d_hit_count, sizeof(int),   cudaMemcpyDeviceToHost);
    *out_loss = h_loss / (float)batch_size;
    *out_acc  = (float)h_hits / (float)batch_size;
}

void Network::epoch(const Tensor& x_full, const int* d_y_full, int batch_size) {
    int total      = x_full.N;
    int per_sample = x_full.H * x_full.W * x_full.C;
    int L          = (int)layers.size();

    float sum_loss = 0.0f, sum_acc = 0.0f;
    int   nb       = 0;

    for (int start = 0; start + batch_size <= total; start += batch_size) {
        Tensor x_batch = {
            x_full.data + (size_t)start * per_sample,
            batch_size, x_full.H, x_full.W, x_full.C
        };
        const int* y_batch = d_y_full + start;

        Tensor dummy;
        forward(x_batch, dummy);

        const Tensor& probs = acts[L];
        int C = probs.C;

        float l, a;
        compute_metrics_(y_batch, batch_size, C, &l, &a);
        sum_loss += l;
        sum_acc  += a;
        nb++;

        // Fused softmax+xent gradient → start backward at layer L-2 (skip softmax).
        int threads = 256;
        net_softmax_xent_grad_kernel<<<net_blocks_for(batch_size * C, threads), threads>>>(
            probs.data, y_batch, fused_grad.data, batch_size, C);

        backward_from_(L - 2, fused_grad);
    }

    if (nb > 0) {
        printf("  loss=%.4f  acc=%.4f  (%d batches)\n",
               sum_loss / nb, sum_acc / nb, nb);
    }
}

void Network::train(const Tensor& x_full, const int* d_y_full, int batch_size, int epochs) {
    for (int e = 0; e < epochs; e++) {
        printf("epoch %d/%d\n", e + 1, epochs);
        epoch(x_full, d_y_full, batch_size);
    }
}

void Network::evaluate(const Tensor& x_full, const int* d_y_full, int batch_size) {
    int total      = x_full.N;
    int per_sample = x_full.H * x_full.W * x_full.C;
    int L          = (int)layers.size();

    float sum_loss = 0.0f, sum_acc = 0.0f;
    int   nb       = 0;

    for (int start = 0; start + batch_size <= total; start += batch_size) {
        Tensor x_batch = {
            x_full.data + (size_t)start * per_sample,
            batch_size, x_full.H, x_full.W, x_full.C
        };
        const int* y_batch = d_y_full + start;

        Tensor dummy;
        forward(x_batch, dummy);

        float l, a;
        compute_metrics_(y_batch, batch_size, acts[L].C, &l, &a);
        sum_loss += l;
        sum_acc  += a;
        nb++;
    }

    if (nb > 0) {
        printf("  eval loss=%.4f  eval acc=%.4f  (%d batches)\n",
               sum_loss / nb, sum_acc / nb, nb);
    }
}
