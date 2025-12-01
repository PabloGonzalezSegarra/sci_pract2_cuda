#define N 100000000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>

// Structure for parallel sum arguments
typedef struct {
    float *array;
    int start;
    int end;
    double partial_sum;
} SumThreadArgs;

// Thread function for parallel sum
void* parallel_sum_thread(void *arg) {
    SumThreadArgs *args = (SumThreadArgs*)arg;
    double sum = 0.0;
    
    for (int i = args->start; i < args->end; i++) {
        sum += args->array[i];
    }
    
    args->partial_sum = sum;
    return NULL;
}

// Parallel sum using pthreads (configurable number of threads)
double parallel_sum(float *array, int n, int num_threads) {
    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    SumThreadArgs *args = (SumThreadArgs*)malloc(sizeof(SumThreadArgs) * num_threads);
    
    int chunk_size = (n + num_threads - 1) / num_threads;
    
    // Launch threads
    for (int t = 0; t < num_threads; t++) {
        args[t].array = array;
        args[t].start = t * chunk_size;
        args[t].end = (t + 1) * chunk_size;
        if (args[t].end > n) args[t].end = n;
        args[t].partial_sum = 0.0;
        
        pthread_create(&threads[t], NULL, parallel_sum_thread, &args[t]);
    }
    
    // Wait for all threads and accumulate results
    double total_sum = 0.0;
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
        total_sum += args[t].partial_sum;
    }
    
    free(threads);
    free(args);
    
    return total_sum;
}

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
    
    // Number of CPU threads for parallel sum from argv[2]
    int cpu_threads = atoi(argv[2]);
    
    // Print configuration
    printf("Using %d blocks of %d threads\n", blocks, threadsPerBlock);
    printf("Using %d CPU threads for parallel sum\n", cpu_threads);

    // Create events for timing
    cudaEvent_t start, end;
    cudaEvent_t sum_start, sum_end;
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&sum_start);
    cudaEventCreate(&sum_end);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    // Start timing
    cudaEventRecord(start);
    
    // Compute sum of b on CPU using parallel threads
    cudaEventRecord(sum_start);
    double sum_b = parallel_sum(b, N, cpu_threads);
    cudaEventRecord(sum_end);
    
    // Compute out[i] = a[i] * sum_b
    cudaEventRecord(kernel_start);
    vector_pro<<<blocks, threadsPerBlock>>>(out_cuda, a_cuda, sum_b, N);
    cudaEventRecord(kernel_end);

    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float milliseconds = 0;
    float sum_milliseconds = 0;
    float kernel_milliseconds = 0;
    
    cudaEventElapsedTime(&milliseconds, start, end);
    cudaEventElapsedTime(&sum_milliseconds, sum_start, sum_end);
    cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end);
    
    printf("Time elapsed (total): %f ms\n", milliseconds);
    printf("  - Parallel sum (CPU): %f ms\n", sum_milliseconds);
    printf("  - Kernel execution (GPU): %f ms\n", kernel_milliseconds);
    
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
