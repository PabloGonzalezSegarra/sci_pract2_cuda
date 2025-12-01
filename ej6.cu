#define N 100000000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void heavy_cpu(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float x = data[i];
        for (int j = 0; j < 10000; j++) {
            x = sinf(x) * 1.00001f + cosf(x) * 0.99999f;
        }
        data[i] = x;
    }
}

int main(int argc, char **argv){
    
    float *a; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.3f; 
    }

    float *a_cuda = NULL;

    // Allocate device memory
    cudaMalloc((void**)&a_cuda, sizeof(float) * N);

    // Copy inputs to device
    cudaMemcpy(a_cuda, a, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Create events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Main function
    // threadsPerBlock is taken from argv[1] (assume valid integer provided)
    int threadsPerBlock = atoi(argv[1]);
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Using %d blocks of %d threads\n", blocks, threadsPerBlock);
    
    // Start timing
    cudaEventRecord(start);
    // Print configuration
    heavy_cpu<<<blocks, threadsPerBlock>>>(a_cuda, N);
    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("Time elapsed: %f ms\n", milliseconds);

    cudaMemcpy(a, a_cuda, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("Success! All values are correct.\n");

    // Free memory
    cudaFree(a_cuda);
    free(a);
}
