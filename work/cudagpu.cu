#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

// -------- CUDA error checking ----------
#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << std::endl;                               \
        std::exit(1);                                         \
    }                                                         \
} while(0)

// [MEM-OPT 2] Bit-packed segment bitset: 1 bit per odd
__device__ __forceinline__ void clear_bit_atomic(uint64_t* bits, uint64_t idx) {
    uint64_t w = idx >> 6;         // word index
    uint64_t b = idx & 63ULL;      // bit within word

    // -------- PCAM-C --------
    // Communication/synchronization: atomic to avoid races when multiple primes
    // clear bits in the same 64-bit word.
    atomicAnd((unsigned long long*)&bits[w], ~(1ULL << b));
}

// -------- PCAM-P --------
// Partitioning: each CUDA thread handles one base prime p,
// and clears multiples of p within a segment.
__global__ void mark_segment_kernel(uint64_t* d_bits,
                                    uint64_t seg_low,
                                    uint64_t seg_high,
                                    const uint32_t* d_primes,
                                    uint32_t primes_count) {
    // -------- PCAM-M --------
    // Mapping: thread id -> prime index
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= primes_count) return;

    uint64_t p = (uint64_t)d_primes[tid];
    if (p == 2) return; // segment stores odds only

    uint64_t p2 = p * p;
    if (p2 > seg_high) return;

    // first multiple of p in [seg_low..seg_high]
    uint64_t start = (seg_low + p - 1) / p * p;
    if (start < p2) start = p2;
    if ((start & 1ULL) == 0) start += p;  // ensure odd

    uint64_t step = 2 * p; // stay on odds

    for (uint64_t x = start; x <= seg_high; x += step) {
        // [MEM-OPT 1] Odd-only mapping: index corresponds to odd steps
        uint64_t idx = (x - seg_low) >> 1;
        clear_bit_atomic(d_bits, idx);
    }
}

// CPU base primes up to sqrt(N), odd-only + bit-packed
static std::vector<uint32_t> base_primes_odd_upto(uint64_t limit) {
    if (limit < 2) return {};
    uint64_t odds = (limit >= 3) ? ((limit - 3) / 2 + 1) : 0;

    // [MEM-OPT 1 + 2] odd-only + bit-packed
    std::vector<uint64_t> bits((odds + 63) / 64, ~0ULL);
    if (!bits.empty()) {
        uint64_t extra = bits.size() * 64 - odds;
        if (extra) bits.back() &= (~0ULL >> extra);
    }

    uint64_t r = (uint64_t)std::floor(std::sqrt((long double)limit));

    auto get_bit = [&](uint64_t i) -> bool {
        return (bits[i >> 6] >> (i & 63)) & 1ULL;
    };
    auto clear_bit = [&](uint64_t i) {
        bits[i >> 6] &= ~(1ULL << (i & 63));
    };

    for (uint64_t p = 3; p <= r; p += 2) {
        if (!get_bit((p - 3) / 2)) continue;
        for (uint64_t x = p * p; x <= limit; x += 2 * p)
            clear_bit((x - 3) / 2);
    }

    std::vector<uint32_t> primes;
    primes.push_back(2);
    for (uint64_t i = 0; i < odds; i++)
        if (get_bit(i)) primes.push_back((uint32_t)(2 * i + 3));
    return primes;
}

static inline uint64_t popcount_u64(const std::vector<uint64_t>& v) {
    uint64_t c = 0;
    for (uint64_t w : v) c += (uint64_t)__builtin_popcountll(w);
    return c;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " N segment_bytes\n";
        std::cerr << "Example: " << argv[0] << " 1000000000 67108864\n";
        return 1;
    }
    uint64_t N = strtoull(argv[1], nullptr, 10);
    uint64_t segment_bytes = strtoull(argv[2], nullptr, 10);

    if (N < 2) {
        std::cout << "primes=0 time_s=0 model=cuda\n";
        return 0;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // -------- PCAM-P --------
    // Preprocessing: base primes up to sqrt(N) on CPU, reused for all segments.
    uint64_t sqrtN = (uint64_t)std::floor(std::sqrt((long double)N));
    std::vector<uint32_t> base = base_primes_odd_upto(sqrtN);

    // -------- PCAM-C --------
    // Communication: copy base primes to GPU once.
    uint32_t* d_primes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_primes, base.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_primes, base.data(),
                          base.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // [MEM-OPT 3] Segmentation: GPU stores only one segment bitset at a time.
    uint64_t segment_bits = segment_bytes * 8ULL;   // bits
    if (segment_bits < 1024ULL * 8ULL) segment_bits = 1024ULL * 8ULL;
    uint64_t segment_odds = segment_bits;           // 1 bit per odd
    uint64_t words = (segment_odds + 63) / 64;

    uint64_t* d_bits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, words * sizeof(uint64_t)));

    std::vector<uint64_t> h_bits(words);

    uint64_t total = 1; // prime 2

    uint64_t low0 = 3;
    uint64_t high0 = (N | 1ULL);

    // odd segment covers values of span = 2*(segment_odds-1)
    uint64_t segment_span = 2 * (segment_odds - 1);
    uint64_t num_segments = (high0 >= low0) ? ((high0 - low0) / segment_span + 1) : 0;

    // -------- PCAM-A --------
    // Agglomeration: choose threads/block and segment_bytes (segment granularity).
    int threads = 256;
    int blocks = (int)((base.size() + threads - 1) / threads);

    for (uint64_t s = 0; s < num_segments; s++) {
        uint64_t seg_low = low0 + s * segment_span;
        uint64_t seg_high = seg_low + segment_span;
        if (seg_high > high0) seg_high = high0;
        if ((seg_low & 1ULL) == 0) seg_low++;
        if ((seg_high & 1ULL) == 0) seg_high--;
        if (seg_low > seg_high) continue;

        uint64_t odds_in_seg = ((seg_high - seg_low) >> 1) + 1;
        uint64_t words_in_seg = (odds_in_seg + 63) / 64;

        // Init segment bits to 1
        std::vector<uint64_t> init(words_in_seg, ~0ULL);
        uint64_t extra = words_in_seg * 64 - odds_in_seg;
        if (extra) init.back() &= (~0ULL >> extra);

        // -------- PCAM-C --------
        // Communication: copy initialized segment bitset to GPU.
        CUDA_CHECK(cudaMemcpy(d_bits, init.data(),
                              words_in_seg * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        // -------- PCAM-M --------
        // Mapping: threads -> base primes, clear multiples in this segment.
        mark_segment_kernel<<<blocks, threads>>>(d_bits, seg_low, seg_high,
                                                 d_primes, (uint32_t)base.size());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back and count on CPU (correct + simple; becomes a bottleneck discussion)
        CUDA_CHECK(cudaMemcpy(h_bits.data(), d_bits,
                              words_in_seg * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        h_bits.resize(words_in_seg);
        total += popcount_u64(h_bits);
        h_bits.resize(words); // restore
    }

    CUDA_CHECK(cudaFree(d_bits));
    CUDA_CHECK(cudaFree(d_primes));

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "primes=" << total
              << " time_s=" << secs
              << " model=cuda"
              << " N=" << N
              << " segment_bytes=" << segment_bytes
              << "\n";

    return 0;
}
