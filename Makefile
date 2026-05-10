SRCS := tensor.cu relu.cu softmax.cu dense.cu conv.cu pool.cu network.cu mnist.cu main.cu

cnn: $(SRCS)
	nvcc -O3 -std=c++17 $^ -o $@

clean:
	rm -f cnn
