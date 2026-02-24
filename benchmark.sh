#!/bin/bash

mkdir -p results
echo "model,scaling,N,cores,time" > results/times.csv

# CONFIGURATION
SEG=67108864                # 64MB segments
N_STRONG=1000000000         # 1e9 (fixed for strong scaling)
PER_CORE=250000000          # 2.5e8 per core for weak scaling

THREADS=(1 2 4 8 16)
RANKS=(1 2 4 8)

# STRONG SCALING

echo "Running STRONG scaling..."

# ---- OpenMP ----
for t in "${THREADS[@]}"; do
    out=$(./openmpv4 $N_STRONG $SEG $t)
    time=$(echo $out | grep -o "time_s=[^ ]*" | cut -d= -f2)
    echo "openmp,strong,$N_STRONG,$t,$time" >> results/times.csv
done

# ---- MPI ----
for r in "${RANKS[@]}"; do
    out=$(mpirun -np $r ./mpiv4 $N_STRONG $SEG)
    time=$(echo $out | grep -o "time_s=[^ ]*" | cut -d= -f2)
    echo "mpi,strong,$N_STRONG,$r,$time" >> results/times.csv
done

# ---- Hybrid ----
for r in 1 2 4; do
    for t in 2 4 8; do
        cores=$((r*t))
        out=$(mpirun -np $r ./hybrid $N_STRONG $SEG $t)
        time=$(echo $out | grep -o "time_s=[^ ]*" | cut -d= -f2)
        echo "hybrid,strong,$N_STRONG,$cores,$time" >> results/times.csv
    done
done

# ---- CUDA ----
out=$(./cuda_sieve $N_STRONG $SEG)
time=$(echo $out | grep -o "time_s=[^ ]*" | cut -d= -f2)
echo "cuda,strong,$N_STRONG,1,$time" >> results/times.csv

# WEAK SCALING

echo "Running WEAK scaling..."

for cores in 1 2 4 8 16; do

    N_WEAK=$((PER_CORE * cores))

    # ---- OpenMP ----
    if [ $cores -le 16 ]; then
        out=$(./openmpv4 $N_WEAK $SEG $cores)
        time=$(echo $out | grep -o "time_s=[^ ]*" | cut -d= -f2)
        echo "openmp,weak,$N_WEAK,$cores,$time" >> results/times.csv
    fi

    # ---- MPI ----
    if [ $cores -le 8 ]; then
        out=$(mpirun -np $cores ./mpiv4 $N_WEAK $SEG)
        time=$(echo $out | grep -o "time_s=[^ ]*" | cut -d= -f2)
        echo "mpi,weak,$N_WEAK,$cores,$time" >> results/times.csv
    fi

done

echo "Benchmarking complete."