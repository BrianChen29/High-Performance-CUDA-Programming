import subprocess
import csv
import re
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BUILD_DIR = ROOT_DIR / "build"
RESULTS_DIR = ROOT_DIR / "results"
CUDA_ARCH = os.environ.get("CUDA_ARCH", "sm_75")

SIZES = [256, 512, 1024, 2048, 4096]

implementations = [
    ("CPU", "cpu/matrix_cpu.c",
     ["gcc", "cpu/matrix_cpu.c", "-O2", "-o", str(BUILD_DIR / "matrix_cpu")],
     BUILD_DIR / "matrix_cpu"),
    ("Naive GPU", "matrix_gpu.cu",
     ["nvcc", f"-arch={CUDA_ARCH}", "matrix_gpu.cu", "-o", str(BUILD_DIR / "matrix_gpu")],
     BUILD_DIR / "matrix_gpu"),
    ("Optimized", "matrix_gpu_optimized.cu",
     ["nvcc", f"-arch={CUDA_ARCH}", "matrix_gpu_optimized.cu", "-o", str(BUILD_DIR / "matrix_gpu_optimized")],
     BUILD_DIR / "matrix_gpu_optimized"),
    ("cuBLAS", "matrix_cublas.cu",
     ["nvcc", f"-arch={CUDA_ARCH}", "matrix_cublas.cu", "-lcublas", "-o", str(BUILD_DIR / "matrix_cublas")],
     BUILD_DIR / "matrix_cublas"),
]

print(f"Compiling all sources for CUDA architecture {CUDA_ARCH}...")
BUILD_DIR.mkdir(exist_ok=True)
compile_status = {}

for name, src, compile_cmd, exe in implementations:
    src_path = ROOT_DIR / src
    if src_path.exists():
        print(f"   Compiling {name} ({src})...")
        try:
            subprocess.run(compile_cmd, cwd=ROOT_DIR, check=True)
            compile_status[name] = True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            compile_status[name] = False
            print(f"Compilation failed for {name}: {exc}")
    else:
        compile_status[name] = False
        print(f"Warning: Source file '{src}' not found. Skipping compilation.")
print("Compilation process finished.\n")

results = {impl[0]: {} for impl in implementations}

for name, src, compile_cmd, exe in implementations:
    if not compile_status.get(name, False):
        print(f"Skipping tests for {name} because compilation did not succeed.")
        continue

    if not exe.exists():
        print(f"Executable '{exe}' not found. Skipping tests for {name}.")
        continue

    print(f"Running tests for {name}...")

    for N in SIZES:
        try:
            result = subprocess.run([str(exe), str(N)], capture_output=True, text=True, timeout=120)
            output = result.stdout.strip()

            match = re.search(r"([0-9]*\.[0-9]+)\s+seconds", output)

            if match:
                time_sec = match.group(1)
                results[name][N] = time_sec
                print(f"   N={N}: {time_sec} sec")
            else:
                results[name][N] = "Error"
                print(f"   N={N}: Parse Error (Could not find time in output)")
                
        except subprocess.TimeoutExpired:
            results[name][N] = "Timeout"
            print(f"   N={N}: Execution Timeout")
        except Exception as e:
            results[name][N] = "Fail"
            print(f"   N={N}: Execution Failed: {e}")
            
    print("-" * 30)

RESULTS_DIR.mkdir(exist_ok=True)
csv_filename = RESULTS_DIR / "final_comparison.csv"

with open(csv_filename, 'w', newline="") as f:
    writer = csv.writer(f)

    header = ["Implementation"] + [f"N={N}" for N in SIZES]
    writer.writerow(header)

    for name, _, _, _ in implementations:
        row = [name]
        for N in SIZES:
            row.append(results[name].get(N, "N/A"))
        writer.writerow(row)

print(f"\nAll done! Results saved to '{csv_filename}'.")
