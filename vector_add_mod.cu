#define N 100000000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv){
    
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    float *a_cuda = NULL, *b_cuda = NULL, *out_cuda = NULL;

    // Create events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Start timing
    cudaEventRecord(start);

    // Allocate device memory
    cudaMalloc((void**)&a_cuda, sizeof(float) * N);
    cudaMalloc((void**)&b_cuda, sizeof(float) * N);
    cudaMalloc((void**)&out_cuda, sizeof(float) * N);

    // Copy inputs to device
    cudaMemcpy(a_cuda, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    // threadsPerBlock is taken from argv[1] (assume valid integer provided)
    int threadsPerBlock = atoi(argv[1]);
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Print configuration
    printf("Using %d blocks of %d threads\n", blocks, threadsPerBlock);
    vector_add<<<blocks, threadsPerBlock>>>(out_cuda, a_cuda, b_cuda, N);

    cudaMemcpy(out, out_cuda, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("Time elapsed: %f ms\n", milliseconds);

    // Verify result
    const float MAX_ERR = 1e-6f;
    for(int i = 0; i < N; i++){
        if(fabsf(out[i] - 3.0f) > MAX_ERR){
            printf("Error at index %d: %f (diff=%f)\n", i, out[i], fabsf(out[i] - 3.0f));
            return -1;
        }
    }

    printf("Success! All values are correct.\n");

    // Free memory
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(out_cuda);

    free(a);
    free(b);
    free(out);
}
