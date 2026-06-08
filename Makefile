DEPS := tensor.cu GEMM.cu adam.cu relu.cu softmax.cu pool.cu dense.cu conv.cu mnist.cu network.cu \
        tensor.cuh gemm.cuh adam.cuh layer.cuh

NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17

cnn: main.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) main.cu -o $@

clean:
	rm -f cnn
