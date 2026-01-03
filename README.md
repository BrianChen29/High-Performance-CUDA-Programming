# High-Performance-CUDA-Programming
DSCI560 - Extra credit - Lab11

Lab 11: High-Performance CUDA Programming

Course: High Performance Computing
Environment: Google Colab (NVIDIA Tesla T4 GPU)

## Project Overview

This laboratory explores high-performance parallel computing techniques using NVIDIA CUDA through the implementation of Matrix Multiplication and Image Convolution. The experiment consists of four main parts:

1. CPU Baseline: Implementation of $O(N^3)$ matrix multiplication using C.
2. CUDA Acceleration: Implementation of a Naïve kernel and an Optimized kernel using Shared Memory Tiling.
3. cuBLAS Comparison: Performance comparison against NVIDIA's highly optimized linear algebra library.
4. Python Integration: Creation of a CUDA Shared Library (.so) and calling GPU functions via Python ctypes for high-performance Image Edge Detection.

## File Structure

1. Matrix Multiplication
   * `matrix_cpu.c`: CPU implementation (Baseline).
   * `matrix_gpu.cu`: Basic CUDA implementation (Naïve Kernel).
   * `matrix_gpu_optimized.cu`: Optimized CUDA implementation (Shared Memory Tiling).
   * `matrix_cublas.cu`: Implementation using the NVIDIA cuBLAS library.
   * `collect_results.py`: Automated compilation and testing script (generates final_comparison.csv).
   * `plot_matrix_performance.py`: Python script to read the CSV and generate performance plots (Log Scale & GPU Linear Scale).

2. Image Convolution
   * `convolution_lib.cu`: Source code containing the CUDA Kernel and C-Interface, used to compile the Shared Library.
   * `convolution_standalone.cu`: Standalone CUDA executable (used to benchmark Python overhead).
   * `test_convolution_final.py`: Main Python script responsible for downloading the test image, creating a large Mosaic, calling GPU/CPU for edge detection, and comparing performance.

## Compilation & Execution Guide

Due to the environment constraints of Google Colab, all CUDA compilation commands must include -arch=sm_75 to support the Tesla T4 architecture.

### Part A: Matrix Multiplication Performance Analysis
1. Automated Testing

Run the following Python script to automatically compile all C/CUDA programs, execute tests for matrix sizes $N=1024, 2048, 4096$, and collect data.

```python collect_results.py```

* Output: final_comparison.csv


2. Plotting Graphs
```python plot_matrix_performance.py```

* Output:
    * plot_log_scale.png (CPU vs. GPU comparison)
    * plot_gpu_only.png (Comparison between CUDA versions)

### Part B: Image Convolution (Python + CUDA)

1. Compile the Shared Library
```bash
nvcc -arch=sm_75 -Xcompiler -fPIC -shared convolution_lib.cu -o libconvolution.so
```

Note: If you update the .so file in Colab, you must perform a "Restart Session" to clear the Python cache.

2. Run Python Visualization & Performance Test
This script automatically downloads the test image (Taj Mahal), creates a large Mosaic tiling, and performs Sobel Edge Detection.

```bash
python test_convolution_final.py
```

* Output:
  * `convolution_output_demo.png` (Edge detection result image)
  * Console output showing execution times and Speedup ratio for CPU and GPU.

3. (Optional) Run Standalone CUDA Version
Used to compare the execution efficiency between "Python + CUDA" and "Pure C++ CUDA".

```bash
nvcc -arch=sm_75 convolution_standalone.cu -o convolution_standalone
./convolution_standalone
```

## Expected Results

1. Matrix Multiplication:
   * As $N$ increases, CPU time grows exponentially ($O(N^3)$), while GPU time grows much more slowly.
   * Performance Ranking: cuBLAS > Optimized CUDA > Naïve CUDA >>> CPU.
   * At $N=4096$, cuBLAS is approximately 5x-10x faster than the hand-written optimized version.

2. Image Convolution:
   * When processing large mosaic images (e.g., 4000x2665), the GPU version achieves a 20x ~ 50x speedup.
   * The overhead of the Python Interface is negligible (minimal difference compared to the Standalone version).

## Notes

* Environment: This project is designed for Google Colab (Runtime: T4 GPU).
* Toolchain Error: If you encounter an unsupported toolchain error, ensure the compilation command includes -arch=sm_75.
* Black Image Output: If the Python execution results in a black image, ensure the Numpy array has been processed with np.ascontiguousarray() to ensure contiguous memory layout, and try restarting the Runtime.