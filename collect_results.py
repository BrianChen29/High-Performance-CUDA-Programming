import subprocess
import csv
import re
import os

# Define the matrix sizes to test (N)
sizes = [256, 512, 1024, 2048, 4096]

# Define all implementation versions to test
# Format: (Display Name, Source Filename, Compile Command, Executable Name)
implementations = [
    ("CPU",       "matrix_cpu.c",           "gcc matrix_cpu.c -o matrix_cpu -O2",               "./matrix_cpu"),
    ("Naïve GPU", "matrix_gpu.cu",          "nvcc matrix_gpu.cu -o matrix_gpu",                  "./matrix_gpu"),
    ("Optimized", "matrix_gpu_optimized.cu","nvcc matrix_gpu_optimized.cu -o matrix_gpu_optimized","./matrix_gpu_optimized"),
    ("cuBLAS",    "matrix_cublas.cu",       "nvcc matrix_cublas.cu -lcublas -o matrix_cublas",   "./matrix_cublas")
]

# Step 1: Automatically compile all source files
print("Compiling all sources...")
for name, src, compile_cmd, exe in implementations:
    # Check if the source file exists before compiling
    if os.path.exists(src):
        print(f"   Compiling {name} ({src})...")
        try:
            # Run the compilation command
            subprocess.run(compile_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
             print(f"Compilation failed for {name}. Please check the code.")
    else:
        print(f"Warning: Source file '{src}' not found. Skipping compilation.")
print("Compilation process finished.\n")

# Step 2: Execute tests and collect data
# We use a dictionary to store results: results[Implementation_Name][N] = time
results = {impl[0]: {} for impl in implementations}

for name, src, compile_cmd, exe in implementations:
    # Remove "./" to check if the executable file exists
    exe_filename = exe.replace("./", "")
    
    if not os.path.exists(exe_filename): 
        print(f"Executable '{exe}' not found. Skipping tests for {name}.")
        continue

    print(f"🚀 Running tests for {name}...")
    
    for N in sizes:
        try:
            # Execute the program with N as an argument
            # capture_output=True: Captures stdout so we can parse it
            # timeout=120: Prevents the script from hanging if the C code freezes
            result = subprocess.run([exe, str(N)], capture_output=True, text=True, timeout=120)
            output = result.stdout.strip()
            
            # Use Regex to extract the time from the output string (e.g., "1.2345 seconds")
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

# Step 3: Save results to CSV
# [cite_start]The format is designed to match the table in Part 5 of the Lab PDF [cite: 134, 148]
csv_filename = "final_comparison.csv"

with open(csv_filename, 'w', newline="") as f:
    writer = csv.writer(f)
    
    # Write Header Row: Implementation, N=512, N=1024, ...
    header = ["Implementation"] + [f"N={N}" for N in sizes]
    writer.writerow(header)
    
    # Write Data Rows
    for name, _, _, _ in implementations:
        row = [name]
        for N in sizes:
            # Get the recorded time, or "N/A" if missing
            row.append(results[name].get(N, "N/A"))
        writer.writerow(row)

print(f"\nAll done! Results saved to '{csv_filename}'.")