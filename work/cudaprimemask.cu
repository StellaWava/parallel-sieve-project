//CUDA MODULO IMAGE SIEVE
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << std::endl;                                \
        std::exit(1);                                          \
    }                                                          \
} while(0)

/*
CUDA PRIME MODULO MASK (2D SIEVE MAPPING)

Concept:
    Pixel (x,y) is "active" if:
        x % p == 0 OR y % p == 0
    for any selected prime p.

This maps sieve-style elimination to 2D signal domain.

PCAM:
P: Partition by pixel (each thread handles one pixel)
C: No inter-thread communication
A: 2D blocks (tiling)
M: threadIdx/blockIdx map directly to (x,y)

Memory:
    [MEM-OPT] Coalesced row-major layout
*/

__global__ void prime_mask_kernel(const uint8_t* img,
                                  uint8_t* mask,
                                  uint8_t* out,
                                  int W, int H,
                                  const int* primes,
                                  int Pn)
{
    // -------- PCAM-M --------
    // Map thread to pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    // -------- PCAM-P --------
    // Each thread independently processes one pixel
    int active = 0;

    for (int i = 0; i < Pn; i++) {
        int p = primes[i];
        if ((x % p) == 0 || (y % p) == 0) {
            active = 1;
            break;
        }
    }

    int idx = y * W + x;

    mask[idx] = active ? 255 : 0;
    out[idx]  = active ? img[idx] : 0;
}

static void write_pgm(const char* path,
                      const std::vector<uint8_t>& img,
                      int W, int H)
{
    FILE* f = fopen(path, "wb");
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    fwrite(img.data(), 1, (size_t)W * H, f);
    fclose(f);
}

static std::vector<int> primes_upto(int n)
{
    std::vector<char> is_prime(n + 1, 1);
    is_prime[0] = is_prime[1] = 0;

    for (int p = 2; p * p <= n; p++)
        if (is_prime[p])
            for (int x = p * p; x <= n; x += p)
                is_prime[x] = 0;

    std::vector<int> primes;
    for (int i = 2; i <= n; i++)
        if (is_prime[i])
            primes.push_back(i);

    return primes;
}

int main(int argc, char** argv)
{
    // Usage: ./cuda_prime_mask_image W H prime_max prime_count seed
    int W = (argc > 1) ? atoi(argv[1]) : 1024;
    int H = (argc > 2) ? atoi(argv[2]) : 1024;
    int prime_max = (argc > 3) ? atoi(argv[3]) : 97;
    int prime_count = (argc > 4) ? atoi(argv[4]) : 10;
    int seed = (argc > 5) ? atoi(argv[5]) : 123;

    // -------- PCAM-P --------
    // Generate synthetic image (signal + noise)
    std::vector<uint8_t> h_img((size_t)W * H);

    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 10.0f);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float val =
                127.0f
                + 50.0f * sin(2.0f * 3.14159265f * x / 64.0f)
                + 40.0f * sin(2.0f * 3.14159265f * y / 96.0f)
                + noise(rng);

            int v = (int)round(val);
            v = max(0, min(255, v));
            h_img[y * W + x] = (uint8_t)v;
        }
    }

    // Generate primes
    auto all_primes = primes_upto(prime_max);
    if ((int)all_primes.size() > prime_count)
        all_primes.resize(prime_count);

    // -------- PCAM-C --------
    // Allocate device memory
    uint8_t *d_img, *d_mask, *d_out;
    int *d_primes;

    CUDA_CHECK(cudaMalloc(&d_img,  (size_t)W * H));
    CUDA_CHECK(cudaMalloc(&d_mask, (size_t)W * H));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)W * H));
    CUDA_CHECK(cudaMalloc(&d_primes, all_primes.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(),
                          (size_t)W * H,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_primes, all_primes.data(),
                          all_primes.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // -------- PCAM-A --------
    // Choose 2D thread blocks for tiling
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16,
              (H + 15) / 16);

    prime_mask_kernel<<<grid, block>>>(
        d_img, d_mask, d_out,
        W, H,
        d_primes, (int)all_primes.size());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    std::vector<uint8_t> h_mask((size_t)W * H);
    std::vector<uint8_t> h_out((size_t)W * H);

    CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask,
                          (size_t)W * H,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          (size_t)W * H,
                          cudaMemcpyDeviceToHost));

    write_pgm("signal_input.pgm",  h_img,  W, H);
    write_pgm("prime_mask.pgm",    h_mask, W, H);
    write_pgm("signal_masked.pgm", h_out,  W, H);

    std::cout << "Wrote signal_input.pgm, prime_mask.pgm, signal_masked.pgm\n";

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_primes));

    return 0;
}
