#define N 100000000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vector_pro(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = 0;
        for (int j= 0; j<n; j++){
            out[i] += a[i]*b[j];
        }
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

    
    // Allocate device memory
    cudaMalloc((void**)&a_cuda, sizeof(float) * N);
    cudaMalloc((void**)&b_cuda, sizeof(float) * N);
    cudaMalloc((void**)&out_cuda, sizeof(float) * N);
    
    // Copy inputs to device
    cudaMemcpy(a_cuda, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    
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
    // Main function
    vector_pro<<<blocks, threadsPerBlock>>>(out_cuda, a_cuda, b_cuda, N);

    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("Time elapsed: %f ms\n", milliseconds);
    
    cudaMemcpy(out, out_cuda, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verify result
    // const float MAX_ERR = 1e-6f;
    // for(int i = 0; i < N; i++){
    //     if(fabsf(out[i] - 3.0f) > MAX_ERR){
    //         printf("Error at index %d: %f (diff=%f)\n", i, out[i], fabsf(out[i] - 3.0f));
    //         return -1;
    //     }
    // }

    printf("Success!\n");

    // Free memory
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(out_cuda);

    free(a);
    free(b);
    free(out);
}
