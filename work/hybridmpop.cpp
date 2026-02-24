//HYBRID MPI + OPENMP - all
#include <bits/stdc++.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

static inline bool get_bit(const vector<uint64_t>& bits, uint64_t i) {
    return (bits[i >> 6] >> (i & 63)) & 1ULL;
}
static inline void clear_bit(vector<uint64_t>& bits, uint64_t i) {
    bits[i >> 6] &= ~(1ULL << (i & 63));
}

static vector<uint32_t> base_primes_upto(uint64_t limit) {
    // [MEM-OPT 1] odd-only (store candidates 3,5,7,...)
    // [MEM-OPT 2] bit-packed (1 bit per odd candidate)
    if (limit < 2) return {};
    uint64_t odds = (limit >= 3) ? ((limit - 3) / 2 + 1) : 0;
    vector<uint64_t> bits((odds + 63) / 64, ~0ULL);

    if (!bits.empty()) {
        uint64_t extra = bits.size() * 64 - odds;
        if (extra) bits.back() &= (~0ULL >> extra);
    }

    uint64_t r = (uint64_t)floor(sqrt((long double)limit));

    for (uint64_t p = 3; p <= r; p += 2) {
        if (!get_bit(bits, (p - 3) / 2)) continue;
        for (uint64_t x = p * p; x <= limit; x += 2 * p) {
            clear_bit(bits, (x - 3) / 2);
        }
    }

    vector<uint32_t> primes;
    primes.push_back(2);
    for (uint64_t i = 0; i < odds; i++)
        if (get_bit(bits, i))
            primes.push_back((uint32_t)(2 * i + 3));
    return primes;
}

static uint64_t sieve_segment_count(uint64_t seg_low, uint64_t seg_high,
                                    const vector<uint32_t>& base) {
    // Segment stores odds only; seg_low/seg_high are odd
    uint64_t odds = (seg_high - seg_low) / 2 + 1;

    // [MEM-OPT 2] bit-packed segment bitset
    vector<uint64_t> bits((odds + 63) / 64, ~0ULL);
    if (!bits.empty()) {
        uint64_t extra = bits.size() * 64 - odds;
        if (extra) bits.back() &= (~0ULL >> extra);
    }

    // Mark composites using base primes (skip 2)
    for (size_t k = 1; k < base.size(); k++) {
        uint64_t p = base[k];
        uint64_t p2 = p * p;
        if (p2 > seg_high) break;

        uint64_t start = (seg_low + p - 1) / p * p;
        if (start < p2) start = p2;
        if ((start & 1ULL) == 0) start += p; // keep odd

        for (uint64_t x = start; x <= seg_high; x += 2 * p) {
            uint64_t idx = (x - seg_low) / 2;
            clear_bit(bits, idx);
        }
    }

    uint64_t count = 0;
    for (uint64_t w : bits) count += (uint64_t)__builtin_popcountll(w);
    return count;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np R " << argv[0]
                 << " N segment_bytes omp_threads\n";
            cerr << "Example: mpirun -np 4 " << argv[0]
                 << " 1000000000 33554432 8\n";
        }
        MPI_Finalize();
        return 1;
    }

    uint64_t N = strtoull(argv[1], nullptr, 10);
    uint64_t segment_bytes = strtoull(argv[2], nullptr, 10);
    int omp_threads = atoi(argv[3]);

#ifdef _OPENMP
    omp_set_num_threads(omp_threads);
#else
    (void)omp_threads;
#endif

    double t0 = MPI_Wtime();

    // -------- PCAM-P --------
    // Preprocessing: compute base primes up to sqrt(N) once (root rank).
    uint64_t sqrtN = (uint64_t)floor(sqrt((long double)N));
    vector<uint32_t> base;
    if (rank == 0) base = base_primes_upto(sqrtN);

    // -------- PCAM-C --------
    // Communication: broadcast base primes to all ranks.
    uint32_t base_count = (rank == 0) ? (uint32_t)base.size() : 0;
    MPI_Bcast(&base_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank != 0) base.resize(base_count);
    MPI_Bcast(base.data(), base_count, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Global odd domain [3..high0]
    uint64_t high0 = (N | 1ULL);
    uint64_t count_odds = (high0 >= 3) ? ((high0 - 3) / 2 + 1) : 0;

    // -------- PCAM-P --------
    // Partitioning: split odd index space across MPI ranks (distributed memory).
    uint64_t per = count_odds / (uint64_t)size;
    uint64_t rem = count_odds % (uint64_t)size;
    uint64_t my_start_i = (uint64_t)rank * per + (uint64_t)min<int>(rank, (int)rem);
    uint64_t my_count   = per + (rank < (int)rem ? 1ULL : 0ULL);
    uint64_t my_end_i   = (my_count == 0) ? 0 : (my_start_i + my_count - 1);

    uint64_t my_low  = 3 + 2 * my_start_i;
    uint64_t my_high = (my_count == 0) ? 1 : (3 + 2 * my_end_i);

    // [MEM-OPT 3] Segmentation: bound memory to segment_bytes per segment.
    uint64_t segment_odds = segment_bytes * 8ULL; // 1 bit per odd
    if (segment_odds < 1024) segment_odds = 1024;
    uint64_t segment_span = 2 * (segment_odds - 1);

    uint64_t local = 0;

    if (my_count > 0) {
        uint64_t num_segments = (my_high - my_low) / segment_span + 1;

        // -------- PCAM-A --------
        // Agglomeration: segments are the coarse tasks (granularity set by segment_bytes).
        // -------- PCAM-M --------
        // Mapping: OpenMP maps segments to threads inside each MPI rank.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic,1) reduction(+:local)
        #endif
        for (uint64_t s = 0; s < num_segments; s++) {
            uint64_t seg_low = my_low + s * segment_span;
            uint64_t seg_high = seg_low + segment_span;
            if (seg_high > my_high) seg_high = my_high;
            if ((seg_low & 1ULL) == 0) seg_low++;
            if ((seg_high & 1ULL) == 0) seg_high--;
            if (seg_low <= seg_high)
                local += sieve_segment_count(seg_low, seg_high, base);
        }
    }

    // Avoid double-counting prime 2: add once on rank 0 only.
    uint64_t local_total = local + ((rank == 0 && N >= 2) ? 1ULL : 0ULL);

    // -------- PCAM-C --------
    // Communication: reduce local counts to global count.
    uint64_t global_total = 0;
    MPI_Reduce(&local_total, &global_total, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    if (rank == 0) {
        cout << "primes=" << global_total
             << " time_s=" << (t1 - t0)
             << " ranks=" << size
#ifdef _OPENMP
             << " omp_threads=" << omp_threads
#endif
             << " N=" << N
             << " segment_bytes=" << segment_bytes
             << "\n";
    }

    MPI_Finalize();
    return 0;
}
 
