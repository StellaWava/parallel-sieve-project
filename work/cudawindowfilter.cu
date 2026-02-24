// cudawindowfilter.cu
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::fprintf(stderr, "CUDA error: %s at %s:%d\n",       \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// ---------------- PGM writer ----------------
static void write_pgm(const char* path, const std::vector<uint8_t>& img, int W, int H) {
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::perror("fopen"); std::exit(1); }
    std::fprintf(f, "P5\n%d %d\n255\n", W, H);
    std::fwrite(img.data(), 1, (size_t)W * (size_t)H, f);
    std::fclose(f);
}

// ---------------- Small CPU prime generator ----------------
static std::vector<int> primes_upto(int n) {
    if (n < 2) return {};
    std::vector<char> is_prime((size_t)n + 1, 1);
    is_prime[0] = 0; is_prime[1] = 0;
    for (int p = 2; p * p <= n; p++) {
        if (!is_prime[p]) continue;
        for (int x = p * p; x <= n; x += p) is_prime[x] = 0;
    }
    std::vector<int> ps;
    for (int i = 2; i <= n; i++) if (is_prime[i]) ps.push_back(i);
    return ps;
}

// ---------------- Minimal RNG (no <random>) ----------------
// xorshift32 + Box-Muller for approximate Gaussian noise
static inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
static inline float u01(uint32_t& s) {
    // uniform in (0,1)
    uint32_t x = xorshift32(s);
    // avoid 0 exactly
    return (x + 1.0f) * (1.0f / 4294967297.0f);
}
static inline float gaussian01(uint32_t& s) {
    // Box-Muller
    float u1 = u01(s);
    float u2 = u01(s);
    float r = std::sqrt(-2.0f * std::log(u1));
    float t = 2.0f * 3.14159265358979323846f * u2;
    return r * std::cos(t); // N(0,1)
}

static inline float gauss(float x, float sigma) {
    return std::exp(-(x * x) / (2.0f * sigma * sigma));
}

/*
Prime intensity window kernel (radial rings):
- base Gaussian keeps it stable
- add narrow bumps at radii equal to primes
- normalize sum to 1

PCAM notes (embedded where relevant):
- Kernel design is independent per entry (PCAM-P conceptually)
- Kernel is small, reused for all pixels (memory reuse)
*/
static std::vector<float> build_prime_window_kernel(
    int Ksize,
    const std::vector<int>& primes,
    float window_sigma,
    float ring_gain,
    float base_sigma)
{
    int R = Ksize / 2;
    std::vector<float> K((size_t)Ksize * (size_t)Ksize, 0.0f);

    float sum = 0.0f;

    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            // PCAM-P (conceptual): each (dx,dy) kernel weight independent
            float r = std::sqrt((float)(dx * dx + dy * dy));
            float w = gauss(r, base_sigma);
            for (int p : primes) {
                w += ring_gain * gauss(r - (float)p, window_sigma);
            }
            K[(size_t)(dy + R) * (size_t)Ksize + (size_t)(dx + R)] = w;
            sum += w;
        }
    }

    // normalize
    if (sum > 0.0f) {
        for (float& v : K) v /= sum;
    }
    return K;
}

static std::vector<uint8_t> kernel_to_pgm(const std::vector<float>& K, int Ksize) {
    float mn = K[0], mx = K[0];
    for (float v : K) { if (v < mn) mn = v; if (v > mx) mx = v; }
    std::vector<uint8_t> img((size_t)Ksize * (size_t)Ksize);
    for (int i = 0; i < Ksize * Ksize; i++) {
        float t = (mx > mn) ? (K[i] - mn) / (mx - mn) : 0.0f;
        int v = (int)lrintf(t * 255.0f);
        if (v < 0) v = 0; if (v > 255) v = 255;
        img[(size_t)i] = (uint8_t)v;
    }
    return img;
}

/*
CUDA convolution:
PCAM embedded:
- PCAM-P: one output pixel per thread
- PCAM-C: no inter-thread comm; global reads/writes only
- PCAM-A: 2D tiling via blocks
- PCAM-M: thread->(x,y)
*/
__global__ void conv2d_kernel(const uint8_t* img, uint8_t* out, int W, int H,
                              const float* K, int Ksize)
{
    // PCAM-M
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // PCAM-P
    int R = Ksize / 2;
    float acc = 0.0f;

    for (int ky = -R; ky <= R; ky++) {
        int yy = y + ky;
        yy = (yy < 0) ? 0 : (yy >= H ? H - 1 : yy);
        for (int kx = -R; kx <= R; kx++) {
            int xx = x + kx;
            xx = (xx < 0) ? 0 : (xx >= W ? W - 1 : xx);
            float w = K[(ky + R) * Ksize + (kx + R)];
            acc += w * (float)img[yy * W + xx];
        }
    }

    // PCAM-C
    int v = (int)lrintf(acc);
    if (v < 0) v = 0; if (v > 255) v = 255;
    out[y * W + x] = (uint8_t)v;
}

int main(int argc, char** argv) {
    // Usage:
    // ./cudawindowfilter W H Ksize prime_max prime_count seed
    int W = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int H = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int Ksize = (argc > 3) ? std::atoi(argv[3]) : 31;     // odd
    int prime_max = (argc > 4) ? std::atoi(argv[4]) : 29;
    int prime_count = (argc > 5) ? std::atoi(argv[5]) : 8;
    uint32_t seed = (argc > 6) ? (uint32_t)std::strtoul(argv[6], nullptr, 10) : 123u;

    if (Ksize < 3 || (Ksize % 2) == 0) {
        std::fprintf(stderr, "Ksize must be odd and >= 3\n");
        return 1;
    }

    // Synthetic image (signal + Gaussian-ish noise), no <random>
    std::vector<uint8_t> h_img((size_t)W * (size_t)H);
    uint32_t rng = seed;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float n = 10.0f * gaussian01(rng); // stddev ~ 10
            float val = 127.0f
                + 60.0f * std::sin(2.0f * 3.14159265f * x / 64.0f)
                + 40.0f * std::sin(2.0f * 3.14159265f * y / 96.0f)
                + n;
            int v = (int)lrintf(val);
            if (v < 0) v = 0; if (v > 255) v = 255;
            h_img[(size_t)y * (size_t)W + (size_t)x] = (uint8_t)v;
        }
    }

    // Prime radii
    auto primes = primes_upto(prime_max);
    if ((int)primes.size() > prime_count) primes.resize(prime_count);

    // Kernel parameters (simple knobs for analysis)
    float window_sigma = 0.6f;
    float ring_gain = 0.15f;
    float base_sigma = 6.0f;

    // Build kernel on CPU
    // (PCAM-P conceptual: per-entry independent; small O(K^2))
    auto h_K = build_prime_window_kernel(Ksize, primes, window_sigma, ring_gain, base_sigma);
    auto kimg = kernel_to_pgm(h_K, Ksize);

    // PCAM-C: Host->Device transfers
    uint8_t *d_img = nullptr, *d_out = nullptr;
    float *d_K = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, (size_t)W * (size_t)H));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)W * (size_t)H));
    CUDA_CHECK(cudaMalloc(&d_K, (size_t)Ksize * (size_t)Ksize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), (size_t)W * (size_t)H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), (size_t)Ksize * (size_t)Ksize * sizeof(float), cudaMemcpyHostToDevice));

    // PCAM-A: Agglomeration via 2D blocks
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    conv2d_kernel<<<grid, block>>>(d_img, d_out, W, H, d_K, Ksize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_out((size_t)W * (size_t)H);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)W * (size_t)H, cudaMemcpyDeviceToHost));

    write_pgm("signal_input.pgm", h_img, W, H);
    write_pgm("prime_window_kernel.pgm", kimg, Ksize, Ksize);
    write_pgm("signal_prime_window_filtered.pgm", h_out, W, H);

    std::printf("Wrote: signal_input.pgm, prime_window_kernel.pgm, signal_prime_window_filtered.pgm\n");
    std::printf("Kernel prime radii:");
    for (int p : primes) std::printf(" %d", p);
    std::printf("\n");

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_K));
    return 0;
}
