#define N 1000000000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Kernel for sum: each thread sums a portion of the vector and accumulates with atomicAdd
__global__ void sum_kernel(float *b, double *result, int n, int elements_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * elements_per_thread;
    
    // Early exit if this thread has no work to do
    if (start >= n) return;
    
    int end = start + elements_per_thread;
    if (end > n) end = n;
    
    float sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += (float)b[i];
    }
    
    // Accumulate partial sum in the global result using atomicAdd
    atomicAdd(result, sum);
}

// Main kernel: out[i] = a[i] * sum(b)
__global__ void vector_pro(float *out, float *a, double* sum_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        out[i] = a[i] * (float)(*sum_b);
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
    
    // threadsPerBlock for vector_pro from argv[1]
    int threadsPerBlock = atoi(argv[1]);
    
    // threadsPerBlock for sum kernel from argv[2]
    int sumThreadsPerBlock = atoi(argv[2]);
    
    // Calculate total threads and elements per thread for sum
    int sum_blocks = atoi(argv[3]);
    int total_sum_threads = sumThreadsPerBlock * sum_blocks;
    int elements_per_thread = (N + total_sum_threads - 1) / total_sum_threads;
    
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory for the result on GPU
    double *result_cuda = NULL;
    cudaMalloc((void**)&result_cuda, sizeof(double));
    cudaMemset(result_cuda, 0, sizeof(double)); 
    
    // Print configuration
    printf("Using %d blocks of %d threads for vector_pro\n", blocks, threadsPerBlock);
    printf("Using %d blocks of %d threads for sum kernel\n", sum_blocks, sumThreadsPerBlock);
    printf("Elements per thread: %d\n", elements_per_thread);

    // Create events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Start timing
    cudaEventRecord(start);
    
    // Compute sum of b on GPU using atomicAdd
    sum_kernel<<<sum_blocks, sumThreadsPerBlock>>>(b_cuda, result_cuda, N, elements_per_thread);
    
    // Compute out[i] = a[i] * sum_b
    vector_pro<<<blocks, threadsPerBlock>>>(out_cuda, a_cuda, result_cuda, N);

    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("Time elapsed: %f ms\n", milliseconds);
    
    cudaMemcpy(out, out_cuda, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verify result
    double sum_b = 0.0;
    for (int i = 0; i < N; i++) {
        sum_b += b[i];
    }
    // Compare with gpu calculated value
    double gpu_sum_b;
    cudaMemcpy(&gpu_sum_b, result_cuda, sizeof(double), cudaMemcpyDeviceToHost);
    printf("CPU sum: %f, GPU sum: %f\n", sum_b, gpu_sum_b);

    float expected = a[0] * sum_b;
    printf("Expected: %f, Got: %f\n", expected, out[0]);
    if (fabsf(out[0] - expected) / expected > 1e-6f) {
        printf("Error: result mismatch!\n");
        return -1;
    }
    printf("Success!\n");

    // Free memory
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(out_cuda);
    cudaFree(result_cuda);

    free(a);
    free(b);
    free(out);
}
