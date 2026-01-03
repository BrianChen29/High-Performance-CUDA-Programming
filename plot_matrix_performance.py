import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load the CSV file
csv_file = "final_comparison.csv"

# Check if the file exists. If not, create dummy data for testing purposes.
if not os.path.exists(csv_file):
    print(f"⚠️ {csv_file} not found. Creating dummy data...")
    data = {
        "Implementation": ["CPU", "Naïve GPU", "Optimized", "cuBLAS"],
        "N=1024": [1.2, 0.05, 0.04, 0.005],
        "N=2048": [9.6, 0.25, 0.15, 0.02],
        "N=4096": [76.8, 1.5, 0.8, 0.1]
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(csv_file)

# 2. Data Cleaning and Preparation
# Extract column names like "N=1024" and convert them to integers [1024, 2048, ...]
cols = [c for c in df.columns if "N=" in c]
x_labels = [int(c.replace("N=", "")) for c in cols]
implementations = df["Implementation"].values

# Prepare a dictionary for plotting: { "CPU": [time1, time2...], ... }
plot_data = {}
for index, row in df.iterrows():
    name = row["Implementation"]
    clean_values = []
    for c in cols:
        val = row[c]
        # FILTER LOGIC: Convert valid numbers to float, invalid/timeout to None
        try:
            float_val = float(val)
            clean_values.append(float_val)
        except (ValueError, TypeError):
            # If it's "Timeout", "Fail", or "N/A", treat as None (gap in line)
            clean_values.append(None)
    
    plot_data[name] = clean_values

# 3. Plotting Configuration
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'] # Red, Orange, Green, Blue
markers = ['o', 's', '^', 'D']

# --- Plot 1: Log Scale (Best for comparing CPU vs GPU) ---
plt.figure(figsize=(10, 6))
for i, (name, values) in enumerate(plot_data.items()):
    plt.plot(x_labels, values, marker=markers[i], label=name, color=colors[i], linewidth=2)

plt.title("Matrix Multiplication Performance (Log Scale)", fontsize=16)
plt.xlabel("Matrix Size (N)", fontsize=14)
plt.ylabel("Execution Time (seconds) - Log Scale", fontsize=14)
plt.yscale('log') # CRITICAL: Use log scale to visualize large differences
plt.xticks(x_labels, x_labels)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig("plot_log_scale.png")
print("Saved plot_log_scale.png (Best for overall comparison)")

# --- Plot 2: GPU Only (Linear Scale for analyzing GPU optimizations) ---
plt.figure(figsize=(10, 6))
# Filter out "CPU" data to focus on GPU implementations
gpu_items = [(n, v) for n, v in plot_data.items() if "CPU" not in n]

for i, (name, values) in enumerate(gpu_items):
    # Use colors starting from index 1 to match the previous plot
    plt.plot(x_labels, values, marker=markers[i+1], label=name, color=colors[i+1], linewidth=2)

plt.title("GPU Implementations Comparison (Linear Scale)", fontsize=16)
plt.xlabel("Matrix Size (N)", fontsize=14)
plt.ylabel("Execution Time (seconds)", fontsize=14)
plt.xticks(x_labels, x_labels)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.5)
plt.savefig("plot_gpu_only.png")
print("Saved plot_gpu_only.png (Best for analyzing CUDA optimizations)")

plt.show()