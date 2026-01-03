#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void standaloneKernel(float *Image, float *Filter, float *Output, int Width, int Height, int FilterSize) {
    // Calculate global thread coordinates
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Radius of the filter (used to center the filter on the pixel)
    int r = FilterSize / 2;

    // Boundary check: Ensure thread is within image dimensions
    if (col < Width && row < Height) {
        float sum = 0.0f;
        
        // Iterate over the filter window (Multiply-Accumulate operation)
        for (int i = 0; i < FilterSize; i++) {
            for (int j = 0; j < FilterSize; j++) {
                // Calculate the corresponding pixel coordinates in the input image
                int imgRow = row - r + i;
                int imgCol = col - r + j;

                // Boundary check for the filter window (Zero Padding)
                // If the coordinate falls outside the image, treat the value as 0
                if (imgRow >= 0 && imgRow < Height && imgCol >= 0 && imgCol < Width) {
                    sum += Image[imgRow * Width + imgCol] * Filter[i * FilterSize + j];
                }
            }
        }
        // Write the computed sum to the output pixel
        Output[row * Width + col] = sum;
    }
}


int main() {
    // ==========================================
    // Dimensions Setup
    // These values match the Python "Taj Mahal" Mosaic (5x5 tiles)
    // 800px * 5 = 4000
    // 533px * 5 = 2665
    // ==========================================
    int Width = 4000;   
    int Height = 2665;  
    int FilterSize = 3;

    printf("Benchmarking Standalone CUDA...\n");
    printf("Image Size: %d x %d (Matching Python Mosaic)\n", Width, Height);

    size_t imgSize = Width * Height * sizeof(float);
    size_t filterSize_bytes = FilterSize * FilterSize * sizeof(float);

    // Host Memory Allocation
    float *h_Image = (float*)malloc(imgSize);
    float *h_Filter = (float*)malloc(filterSize_bytes);
    float *h_Output = (float*)malloc(imgSize);

    // Data Initialization
    srand(time(NULL));
    for(int i=0; i<Width*Height; i++) {
        h_Image[i] = (float)rand() / (float)RAND_MAX; // Random float 0.0 ~ 1.0
    }

    // Initialize Filter (Sobel Filter, same as Python)
    // [-1, 0, 1]
    // [-2, 0, 2]
    // [-1, 0, 1]
    float sobel[9] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
    for(int i=0; i<9; i++) h_Filter[i] = sobel[i];

    // Device Memory Allocation
    float *d_Image, *d_Filter, *d_Output;
    cudaMalloc((void**)&d_Image, imgSize);
    cudaMalloc((void**)&d_Filter, filterSize_bytes);
    cudaMalloc((void**)&d_Output, imgSize);

    // Copy Data from Host to Device
    cudaMemcpy(d_Image, h_Image, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Filter, h_Filter, filterSize_bytes, cudaMemcpyHostToDevice);

    // Grid & Block Configuration
    // Standard 16x16 threads per block
    dim3 dimBlock(16, 16);
    dim3 dimGrid((Width + 15) / 16, (Height + 15) / 16);

    // Warmup Run
    // Execute once without timing to initialize GPU resources and caches
    standaloneKernel<<<dimGrid, dimBlock>>>(d_Image, d_Filter, d_Output, Width, Height, FilterSize);
    cudaDeviceSynchronize();

    // Timed Run
    // Use CUDA Events for high-precision timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    standaloneKernel<<<dimGrid, dimBlock>>>(d_Image, d_Filter, d_Output, Width, Height, FilterSize);
    cudaEventRecord(stop);
    
    // Wait for GPU to finish
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("------------------------------------------------\n");
    printf("Direct CUDA Executable Time: %.4f seconds\n", ms / 1000.0);
    printf("------------------------------------------------\n");

    // Cleanup
    cudaFree(d_Image); cudaFree(d_Filter); cudaFree(d_Output);
    free(h_Image); free(h_Filter); free(h_Output);
    return 0;
}