#define N 100000000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Kernel principal: out[i] = a[i] * sum(b)
__global__ void vector_pro(float *out, float *a, float sum_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        out[i] = a[i] * sum_b;
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

    float *a_cuda = NULL, *out_cuda = NULL;
    
    // Allocate device memory
    cudaMalloc((void**)&a_cuda, sizeof(float) * N);
    cudaMalloc((void**)&out_cuda, sizeof(float) * N);
    
    // Copy inputs to device
    cudaMemcpy(a_cuda, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    // threadsPerBlock is taken from argv[1] (assume valid integer provided)
    int threadsPerBlock = atoi(argv[1]);
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Print configuration
    printf("Using %d blocks of %d threads\n", blocks, threadsPerBlock);

    // Create events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Start timing
    cudaEventRecord(start);
    
    // Compute sum of b on CPU using double for precision
    double sum_b = 0.0;
    for (int i = 0; i < N; i++) {
        sum_b += b[i];
    }
    // Compute out[i] = a[i] * sum_b
    vector_pro<<<blocks, threadsPerBlock>>>(out_cuda, a_cuda, sum_b, N);


    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("Time elapsed: %f ms\n", milliseconds);
    
    cudaMemcpy(out, out_cuda, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verify first result: out[0] = a[0] * sum(b)
    float expected = a[0] * sum_b;
    printf("Expected: %f, Got: %f\n", expected, out[0]);
    if (fabsf(out[0] - expected) / expected > 1e-6f) {
        printf("Error: result mismatch!\n");
        return -1;
    }

    printf("Success!\n");

    // Free memory
    cudaFree(a_cuda);
    cudaFree(out_cuda);

    free(a);
    free(b);
    free(out);
}
