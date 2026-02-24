"""
run OpenMP(all) / MPI(all) /Hybrid/ CUDA (all) executables
show the following:

Create a benchmarking report comparing your implementations:
Speedup vs single-thread baseline
Strong vs weak scaling
Memory usage trends
produce results_part2.csv
make plots for each of the comparisons. 
"""

import os
import re
import csv
import statistics
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

LINE_RE = re.compile(r"time_s=([0-9]*\.?[0-9]+)")

def run_cmd(cmd: str, env=None) -> str:
    out = subprocess.check_output(cmd, shell=True, text=True, env=env)
    return out.strip()

def parse_time(output: str) -> float:
    m = LINE_RE.search(output)
    if not m:
        raise RuntimeError(f"Could not parse time from:\n{output}")
    return float(m.group(1))

def median_time(cmd: str, trials: int, env=None) -> float:
    times = []
    for _ in range(trials):
        out = run_cmd(cmd, env=env)
        times.append(parse_time(out))
    return statistics.median(times)

def ensure_exists(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing executable: {path}")

def plot_grouped(lines, xkey, ykey, group_keys, title, xlabel, ylabel, outfile):
    plt.figure()
    groups = {}
    for row in lines:
        g = tuple(row[k] for k in group_keys)
        groups.setdefault(g, []).append(row)
    for g, rows in groups.items():
        rows = sorted(rows, key=lambda r: r[xkey])
        xs = [r[xkey] for r in rows]
        ys = [r[ykey] for r in rows]
        plt.plot(xs, ys, marker="o", label=" | ".join(f"{k}={v}" for k, v in zip(group_keys, g)))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=180)

def main():
    # adjust if needed ----------
    SEG = 67108864  # 64MB
    STRONG_N = 1_000_000_000
    TRIALS = 3

    # Strong-scaling worker sets (good for your machine)
    OMP_THREADS = [1, 2, 4, 6, 8, 12, 16, 24]
    MPI_RANKS   = [1, 2, 4, 6, 8, 12]
    # For > physical cores with OpenMPI, use --use-hwthread-cpus
    MPI_EXTRA_FLAGS = "--use-hwthread-cpus"

    # Weak scaling: keep N per worker ~constant
    WEAK_BASE_N = 250_000_000
    WEAK_WORKERS = [1, 2, 4, 8, 12]

    ensure_exists("./sieve_openmp")
    ensure_exists("./sieve_mpi")
    # CUDA optional (if you built it)
    cuda_exists = Path("./sieve_cuda").exists()

    # For stability on OpenMP
    base_env = os.environ.copy()
    base_env["OMP_PROC_BIND"] = "true"
    base_env["OMP_PLACES"] = "cores"

    rows = []

    # ---------------- Strong scaling ----------------
    # OpenMP strong
    print("== Strong scaling: OpenMP ==")
    t1_omp = None
    for th in OMP_THREADS:
        env = base_env.copy()
        env["OMP_NUM_THREADS"] = str(th)
        cmd = f"./sieve_openmp {STRONG_N} {SEG} {th}"
        t = median_time(cmd, TRIALS, env=env)
        if th == 1:
            t1_omp = t
        speedup = (t1_omp / t) if t1_omp else 1.0
        eff = speedup / th
        rows.append({
            "part": "strong",
            "model": "OpenMP",
            "N": STRONG_N,
            "workers": th,
            "time_s": t,
            "speedup": speedup,
            "efficiency": eff,
        })
        print(th, t)

    # MPI strong
    print("== Strong scaling: MPI ==")
    t1_mpi = None
    for p in MPI_RANKS:
        cmd = f"mpirun {MPI_EXTRA_FLAGS} -np {p} ./sieve_mpi {STRONG_N} {SEG}"
        t = median_time(cmd, TRIALS)
        if p == 1:
            t1_mpi = t
        speedup = (t1_mpi / t) if t1_mpi else 1.0
        eff = speedup / p
        rows.append({
            "part": "strong",
            "model": "MPI",
            "N": STRONG_N,
            "workers": p,
            "time_s": t,
            "speedup": speedup,
            "efficiency": eff,
        })
        print(p, t)

    # CUDA strong (single data point; compare against CPU baselines)
    if cuda_exists:
        print("== Strong scaling: CUDA ==")
        cmd = f"./sieve_cuda {STRONG_N} {SEG}"
        t = median_time(cmd, TRIALS)
        # Compare speedup to 1-thread OpenMP baseline (common choice)
        speedup = (t1_omp / t) if t1_omp else 1.0
        rows.append({
            "part": "strong",
            "model": "CUDA",
            "N": STRONG_N,
            "workers": 1,
            "time_s": t,
            "speedup": speedup,
            "efficiency": speedup,  # 1 worker on GPU; keep simple
        })
        print("cuda", t)

    # ---------------- Weak scaling ----------------
    # OpenMP weak scaling (workers = threads, N = workers * baseN)
    print("== Weak scaling: OpenMP ==")
    for th in WEAK_WORKERS:
        N = th * WEAK_BASE_N
        env = base_env.copy()
        env["OMP_NUM_THREADS"] = str(th)
        cmd = f"./sieve_openmp {N} {SEG} {th}"
        t = median_time(cmd, TRIALS, env=env)
        rows.append({
            "part": "weak",
            "model": "OpenMP",
            "N": N,
            "workers": th,
            "time_s": t,
            "speedup": 0.0,
            "efficiency": 0.0,
        })
        print(th, N, t)

    # MPI weak scaling (workers = ranks, N = workers * baseN)
    print("== Weak scaling: MPI ==")
    for p in WEAK_WORKERS:
        N = p * WEAK_BASE_N
        cmd = f"mpirun {MPI_EXTRA_FLAGS} -np {p} ./sieve_mpi {N} {SEG}"
        t = median_time(cmd, TRIALS)
        rows.append({
            "part": "weak",
            "model": "MPI",
            "N": N,
            "workers": p,
            "time_s": t,
            "speedup": 0.0,
            "efficiency": 0.0,
        })
        print(p, N, t)

    # Write CSV
    out_csv = "results_part2.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {out_csv}")

    # Plot strong scaling metrics
    strong = [r for r in rows if r["part"] == "strong" and r["model"] != "CUDA"]
    plot_grouped(strong, "workers", "time_s", ["model"], "Strong scaling: time", "Workers", "Time (s)", "strong_time.png")
    plot_grouped(strong, "workers", "speedup", ["model"], "Strong scaling: speedup", "Workers", "Speedup", "speedup.png")
    plot_grouped(strong, "workers", "efficiency", ["model"], "Strong scaling: efficiency", "Workers", "Efficiency", "efficiency.png")

    weak = [r for r in rows if r["part"] == "weak"]
    plot_grouped(weak, "workers", "time_s", ["model"], "Weak scaling: time", "Workers", "Time (s)", "weak_time.png")

    print("Wrote plots: strong_time.png, speedup.png, efficiency.png, weak_time.png")

if __name__ == "__main__":
    main()