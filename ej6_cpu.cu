#define N 1000000

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void heavy_cpu(float* data, int n) {
    for (int i = 0; i < n; i++) {
        float x = data[i];
        for (int j = 0; j < 10000; j++) {
            x = sinf(x) * 1.00001f + cosf(x) * 0.99999f;
        }
        data[i] = x;
    }
}

int main(){
    float *a; 
    struct timespec t_start, t_end;

    // Start timing the selected block (allocation + init + computation)
    
    // Allocate memory
    a = (float*)malloc(sizeof(float) * N);
    
    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.3f;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    // Main function
    heavy_cpu(a, N);

    // End timing
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1.0e6;

    printf("Elapsed time: %.6f ms\n", elapsed_ms);
    double total_ops = (double)N * 10000 * 4;
    double gflops = total_ops / (elapsed_ms * 1e6);
    printf("GFLOPS: %.6f\n", gflops);

}
