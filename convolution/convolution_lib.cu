#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA Kernel: 2D Convolution
// Image: Imput image pointer (assumed single-channel grayscale)
// Filter: Convolution filter/kernel pointer (e.g., 3x3 or 5x5 matrix)
// Output: Output image pointeer
// Width, Height: Dimensions of the image
// FilterSize: Dimension of the square filter(e.g, 3 for a 3x3 filter)
__global__ void convolutionKernel(float *Image, float *Filter, float *Output, 
                                  int Width, int Height, int FilterSize) {
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

// Step 2
// C-Interface Wrapper: Exposed function for Python to call
// Handles memory allocation and data transfer between Host and Device
extern "C" void gpu_convolution(float *h_Image, float *h_Filter, float *h_Output, 
                                int Width, int Height, int FilterSize) {
    
    // Calculate memory size in bytes
    size_t imgSize = Width * Height * sizeof(float);
    size_t filterSize_bytes = FilterSize * FilterSize * sizeof(float);

    // 1. Allocate Device Memory (GPU)
    float *d_Image, *d_Filter, *d_Output;
    cudaMalloc((void**)&d_Image, imgSize);
    cudaMalloc((void**)&d_Filter, filterSize_bytes);
    cudaMalloc((void**)&d_Output, imgSize);

    // 2. Copy Data from Host to Device
    cudaMemcpy(d_Image, h_Image, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Filter, h_Filter, filterSize_bytes, cudaMemcpyHostToDevice);

    // 3. Configure Grid and Block dimensions
    // Using 16x16 threads per block is a standard choice for 2D processing
    dim3 dimBlock(16, 16);
    dim3 dimGrid((Width + 15) / 16, (Height + 15) / 16);

    // 4. Launch the CUDA Kernel
    convolutionKernel<<<dimGrid, dimBlock>>>(d_Image, d_Filter, d_Output, Width, Height, FilterSize);
    
    // Wait for the GPU to finish execution
    cudaDeviceSynchronize();

    // 5. Copy Result from Device to Host
    cudaMemcpy(h_Output, d_Output, imgSize, cudaMemcpyDeviceToHost);

    // 6. Free Device Memory
    cudaFree(d_Image);
    cudaFree(d_Filter);
    cudaFree(d_Output);
}

// CPU (Non-accelerated C program)
extern "C" void cpu_convolution(float *Image, float *Filter, float *Output, int Width, int Height, int FilterSize) {
    int r = FilterSize / 2;

    // traditional way: image scanned using double layer for loops
    for (int row = 0; row < Height; row++) {
        for (int col = 0; col < Width; col++) {
            float sum = 0.0f;

            // convolution calculation
            for(int i = 0; i < FilterSize; i++) {
                for (int j = 0; j < FilterSize; j++) {
                    int imgRow = row - r + i;
                    int imgCol = col - r + j;
                    if (imgRow >= 0 && imgRow < Height && imgCol >= 0 && imgCol < Width) {
                        sum += Image[imgRow * Width + imgCol] * Filter[i * FilterSize + j];
                    }
                }
            }
            Output[row * Width + col] = sum;
        }
    }
}