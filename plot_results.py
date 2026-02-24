#plot results
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results/times.csv")

# STRONG SCALING ANALYSIS

strong = df[df["scaling"] == "strong"]
#strong = df[df["type"] == "strong"].copy()

# baseline = openmp 1-core strong
baseline = strong[(strong.model=="openmp") & (strong.cores==1)]["time"].values[0]

strong["speedup"] = baseline / strong["time"]
strong["efficiency"] = strong["speedup"] / strong["cores"]

# ---- Speedup Plot ----
plt.figure()
for model in strong.model.unique():
    sub = strong[strong.model==model]
    plt.plot(sub.cores, sub.speedup, marker='o', label=model)

plt.xlabel("Cores")
plt.ylabel("Speedup")
plt.title("Strong Scaling - Speedup (N ≥ 1e9)")
plt.legend()
plt.grid()
plt.savefig("results/strong_speedup.png")

# ---- Efficiency Plot ----
plt.figure()
for model in strong.model.unique():
    sub = strong[strong.model==model]
    plt.plot(sub.cores, sub.efficiency, marker='o', label=model)

plt.xlabel("Cores")
plt.ylabel("Parallel Efficiency")
plt.title("Strong Scaling - Efficiency")
plt.legend()
plt.grid()
plt.savefig("results/strong_efficiency.png")


# WEAK SCALING ANALYSIS

weak = df[df["scaling"] == "weak"]

plt.figure()
for model in weak.model.unique():
    sub = weak[weak.model==model]
    plt.plot(sub.cores, sub.time, marker='o', label=model)

plt.xlabel("Cores")
plt.ylabel("Runtime (s)")
plt.title("Weak Scaling (constant work per core)")
plt.legend()
plt.grid()
plt.savefig("results/weak_scaling.png")


# MEMORY TREND PLOT

N_values = np.array([1e8, 5e8, 1e9, 2e9, 5e9])

mem_naive = N_values / (1024**3)
mem_odd = (N_values/2) / (1024**3)
mem_bitpacked = (N_values/16) / (1024**3)
mem_segmented = np.ones_like(N_values) * (64 / 1024)

plt.figure()
plt.plot(N_values, mem_naive, label="Naive (byte)")
plt.plot(N_values, mem_odd, label="Odd-only (byte)")
plt.plot(N_values, mem_bitpacked, label="Bit-packed")
plt.plot(N_values, mem_segmented, label="Segmented (64MB)")

plt.xlabel("N")
plt.ylabel("Memory (GB)")
plt.title("Memory Usage Trends")
plt.legend()
plt.grid()
plt.savefig("results/memory_trends.png")

print("All plots saved in results/")
