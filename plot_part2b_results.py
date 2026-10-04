import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load benchmark results
# ---------------------------------------------------------

df = pd.read_csv("part2b_results.csv")

print("Raw data:")
print(df.head())

# Calculate mean execution time for each implementation
# and thread count.
mean_results = (
    df.groupby(["implementation", "threads"])["time_seconds"]
    .mean()
    .reset_index()
)

print("\nMean execution times:")
print(mean_results)


# ---------------------------------------------------------
# Calculate speedup and efficiency
# ---------------------------------------------------------

# Get the 1-thread mean execution time for each implementation.
baseline = (
    mean_results[mean_results["threads"] == 1]
    .set_index("implementation")["time_seconds"]
)

# Speedup:
# S_p = T_1 / T_p
mean_results["speedup"] = mean_results.apply(
    lambda row: baseline[row["implementation"]] / row["time_seconds"],
    axis=1
)

# Efficiency:
# E_p = S_p / p
mean_results["efficiency"] = (
    mean_results["speedup"] / mean_results["threads"]
) * 100


# ---------------------------------------------------------
# Display calculated results
# ---------------------------------------------------------

print("\nResults:")
print(
    mean_results[
        ["implementation", "threads",
         "time_seconds", "speedup", "efficiency"]
    ].round(4)
)


# ---------------------------------------------------------
# Figure 1: Speedup vs Number of Threads
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

for implementation, data in mean_results.groupby("implementation"):
    data = data.sort_values("threads")

    plt.plot(
        data["threads"],
        data["speedup"],
        marker="o",
        linewidth=2,
        label=implementation
    )

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs. Number of OpenMP Threads")

plt.xticks([1, 4, 8, 16, 32])

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    "part2b_speedup.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# Figure 2: Efficiency vs Number of Threads
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

for implementation, data in mean_results.groupby("implementation"):
    data = data.sort_values("threads")

    plt.plot(
        data["threads"],
        data["efficiency"],
        marker="o",
        linewidth=2,
        label=implementation
    )

plt.xlabel("Number of Threads")
plt.ylabel("Efficiency (%)")
plt.title("Parallel Efficiency vs. Number of OpenMP Threads")

plt.xticks([1, 4, 8, 16, 32])

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    "part2b_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()