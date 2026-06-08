#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

// IDX file format (Yann LeCun's MNIST): big-endian header, raw uint8 payload.
//   images: magic=2051, N, H, W, then N*H*W bytes in [0, 255]
//   labels: magic=2049, N,         then N bytes in [0,  9]

static uint32_t read_be32(std::ifstream& f) {
    uint8_t b[4];
    f.read((char*)b, 4);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] <<  8) |  (uint32_t)b[3];
}

bool load_mnist_images(const std::string& path, std::vector<float>& out,
                       int& N, int& H, int& W) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t magic = read_be32(f);
    if (magic != 2051) return false;

    N = (int)read_be32(f);
    H = (int)read_be32(f);
    W = (int)read_be32(f);

    size_t total = (size_t)N * H * W;
    std::vector<uint8_t> raw(total);
    f.read((char*)raw.data(), total);
    if ((size_t)f.gcount() != total) return false;

    // Normalize to [0, 1]. The CNN sees (N, H, W, 1) in NHWC, and since C=1
    // the linear ordering here is identical to NHW.
    out.resize(total);
    for (size_t i = 0; i < total; i++) out[i] = (float)raw[i] / 255.0f;
    return true;
}

bool load_mnist_labels(const std::string& path, std::vector<int>& out, int& N) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t magic = read_be32(f);
    if (magic != 2049) return false;

    N = (int)read_be32(f);

    std::vector<uint8_t> raw(N);
    f.read((char*)raw.data(), N);
    if ((int)f.gcount() != N) return false;

    out.resize(N);
    for (int i = 0; i < N; i++) out[i] = (int)raw[i];
    return true;
}
