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
    float partial_sum;
} SumThreadArgs;

// Thread function for parallel sum
void* parallel_sum_thread(void *arg) {
    SumThreadArgs *args = (SumThreadArgs*)arg;
    float sum = 0.0f;
    
    for (int i = args->start; i < args->end; i++) {
        sum += args->array[i];
    }
    
    args->partial_sum = sum;
    return NULL;
}

// Parallel sum using pthreads (one thread per logical CPU core)
float parallel_sum(float *array, int n) {
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);  // Get number of logical cores
    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    SumThreadArgs *args = (SumThreadArgs*)malloc(sizeof(SumThreadArgs) * num_threads);
    
    int chunk_size = (n + num_threads - 1) / num_threads;
    
    // Launch threads
    for (int t = 0; t < num_threads; t++) {
        args[t].array = array;
        args[t].start = t * chunk_size;
        args[t].end = (t + 1) * chunk_size;
        if (args[t].end > n) args[t].end = n;
        args[t].partial_sum = 0.0f;
        
        pthread_create(&threads[t], NULL, parallel_sum_thread, &args[t]);
    }
    
    // Wait for all threads and accumulate results
    float total_sum = 0.0f;
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
    // Print configuration
    printf("Using %d blocks of %d threads\n", blocks, threadsPerBlock);

    // Print number of logical CPU cores
    const int num_threads = sysconf(_SC_NPROCESSORS_ONLN);  // Get number of logical cores
    printf("Using %d CPU threads for parallel sum\n", num_threads);

    // Create events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Start timing
    cudaEventRecord(start);
    
    // Compute sum of b on CPU using parallel threads
    float sum_b = parallel_sum(b, N);
    
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
