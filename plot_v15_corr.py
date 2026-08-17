import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load V15 correlation audit
# ============================================================

path = "v15_corr_audit.json"

with open(path, "r") as f:
    data = json.load(f)

sampled = np.asarray(data["sampled"], dtype=float)
exact = np.asarray(data["exact"], dtype=float)

print(f"Number of shards: {len(sampled)}")

# ============================================================
# Basic statistics
# ============================================================

print("\n=== Sampled correlation audit ===")
print(f"min    : {sampled.min():.6f}")
print(f"median : {np.median(sampled):.6f}")
print(f"mean   : {sampled.mean():.6f}")
print(f"max    : {sampled.max():.6f}")

print("\n=== Exact correlation audit ===")
print(f"min    : {exact.min():.6f}")
print(f"median : {np.median(exact):.6f}")
print(f"mean   : {exact.mean():.6f}")
print(f"max    : {exact.max():.6f}")

# ============================================================
# Plot 1: Distribution
# ============================================================

plt.figure(figsize=(9, 6))

bins = np.linspace(
    min(sampled.min(), exact.min()),
    max(sampled.max(), exact.max()),
    40
)

plt.hist(
    sampled,
    bins=bins,
    alpha=0.6,
    label="Sampled correlation audit"
)

plt.hist(
    exact,
    bins=bins,
    alpha=0.6,
    label="Core exact max correlation"
)

plt.axvline(
    0.95,
    linestyle="--",
    linewidth=2,
    label="Correlation threshold = 0.95"
)

plt.xlabel("Maximum absolute correlation")
plt.ylabel("Number of shards")
plt.title("V15 Correlation Audit")
plt.legend()
plt.tight_layout()

plt.savefig(
    "v15_corr_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Plot 2: Sampled vs Exact
# ============================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    sampled,
    exact,
    alpha=0.5
)

lower = min(sampled.min(), exact.min())
upper = max(sampled.max(), exact.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    linewidth=1.5,
    label="y = x"
)

plt.axvline(
    0.95,
    linestyle="--",
    linewidth=1.5,
    label="0.95 threshold"
)

plt.axhline(
    0.95,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Sampled max |correlation|")
plt.ylabel("Core exact max |correlation|")
plt.title("V15 Correlation Audit: Sampled vs Exact")
plt.legend()
plt.tight_layout()

plt.savefig(
    "v15_corr_sampled_vs_exact.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Plot 3: Sorted correlation values
# ============================================================

plt.figure(figsize=(10, 6))

sampled_sorted = np.sort(sampled)
exact_sorted = np.sort(exact)

plt.plot(
    sampled_sorted,
    label="Sampled"
)

plt.plot(
    exact_sorted,
    label="Exact"
)

plt.axhline(
    0.95,
    linestyle="--",
    linewidth=2,
    label="0.95 threshold"
)

plt.xlabel("Shard rank (sorted)")
plt.ylabel("Maximum absolute correlation")
plt.title("V15 Correlation Audit Across 508 Shards")
plt.legend()
plt.tight_layout()

plt.savefig(
    "v15_corr_sorted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Threshold statistics
# ============================================================

print("\n=== Threshold check ===")

for name, x in [
    ("Sampled", sampled),
    ("Exact", exact),
]:
    print(
        f"{name:8s}: "
        f"> 0.95 = {(x > 0.95).sum():3d} "
        f"({100 * np.mean(x > 0.95):.2f}%)"
    )