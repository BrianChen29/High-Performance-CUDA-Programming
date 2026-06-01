import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
csv_file = RESULTS_DIR / "final_comparison.csv"

if not csv_file.exists():
    raise FileNotFoundError(
        f"{csv_file} not found. Run `python collect_results.py` before plotting."
    )

df = pd.read_csv(csv_file)

# 2. Data Cleaning and Preparation
# Extract column names like "N=1024" and convert them to integers [1024, 2048, ...]
cols = [c for c in df.columns if "N=" in c]
x_labels = [int(c.replace("N=", "")) for c in cols]

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
RESULTS_DIR.mkdir(exist_ok=True)
log_plot = RESULTS_DIR / "plot_log_scale.png"
plt.savefig(log_plot, dpi=160, bbox_inches="tight")
print(f"Saved {log_plot} (best for overall comparison)")
plt.close()

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
gpu_plot = RESULTS_DIR / "plot_gpu_only.png"
plt.savefig(gpu_plot, dpi=160, bbox_inches="tight")
print(f"Saved {gpu_plot} (best for analyzing CUDA optimizations)")
plt.close()
