#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA Kernel: Each thread computes one element of the output matrix C
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    // Calculate global row and column index for the current thread 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: make sure that it doesn't exceed the range of matrix
    if (row < N && col < N) {
        float sum = 0.0f;
        // dot product
        for (int k = 0; k < N; k++){
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    // Setup Matrix Size
    int N = (argc > 1) ? atoi(argv[1]) : 1024;  // Use command line argument or defaut to 1024
    size_t size = N * N * sizeof(float);
    printf("Matrix Size N = %d\n", N);

    // Host Memory Allocation
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrics with random values
    // Use the same logic as the CPU version for consistency
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    // Device Memory Allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from Host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure Kernel Launch Parameters
    // Block size: 32x32 threads
    int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize);

    // Grid size: Calculate number of blocks needed to cover N
    // (N + blockSize - 1) / blockSize ensures we round up (celing division)
    dim3 dimGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    // Execute Kernel and measure time
    // Using CUDA Events for accurate GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Lauch the kernel
    matrixMultiplyGPU<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Record stop event
    cudaEventRecord(stop);

    // Wait for the GPU to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU execution time (Naïve): %f seconds\n", milliseconds / 1000.0);

    // Copy Result from Device to Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}