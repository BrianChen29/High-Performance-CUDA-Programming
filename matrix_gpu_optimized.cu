#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// define Tile size
#define TILE_WIDTH 16

// Part 4. Shared Memory Tiling Kernel
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    // Shared Memory
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // Calculate the index of current thread
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;
    for (int m = 0; m < (N + TILE_WIDTH - 1)/ TILE_WIDTH; ++m) {
        if (Row < N && (m * TILE_WIDTH + tx) < N){
            ds_A[ty][tx] = A[Row * N + (m * TILE_WIDTH + tx)];
        } 
        else {
            ds_A[ty][tx] = 0.0f;
        }

        if (Col < N && (m * TILE_WIDTH + ty) < N) {
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        }
        else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();

        if (Row < N && Col < N){
            C[Row * N + Col] = Pvalue;
        }
    }
}

int main(int argc, char **argv) {
    // Setup Matrix Size
    int N = (argc > 1) ? atoi(argv[1]) : 1024;  // Use command line argument or defaut to 1024
    size_t size = N * N * sizeof(float);
    printf("Matrix Size N = %d (Optimized Tiled CUDA)\n", N);

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
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    // Grid size: Calculate number of blocks needed to cover N
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);


    // Execute Kernel and measure time
    // Using CUDA Events for accurate GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Lauch the kernel
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Record stop event
    cudaEventRecord(stop);

    // Wait for the GPU to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU execution time (Tiled Optimization): %f seconds\n", milliseconds / 1000.0);

    // // Copy Result from Device to Host
    // cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}