#include <vector>
#include <string>

bool load_mnist_images(const std::string& path, std::vector<float>& out,
                       int& N, int& H, int& W);
bool load_mnist_labels(const std::string& path, std::vector<int>& out, int& N);
