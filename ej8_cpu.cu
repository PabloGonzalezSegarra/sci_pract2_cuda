#define N 1000000

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vector_pro(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = 0;
        for (int j= 0; j<n; j++){
            out[i] += a[i]*b[j];
        }
    }
}

int main(){
    float *a, *b, *out; 
    struct timespec t_start, t_end;

    
    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    
    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
    
    // Start timing
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    // Main function
    vector_pro(out, a, b, N);

    // End timing
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1.0e6;

    printf("Elapsed time: %.6f ms\n", elapsed_ms);

    free(a);
    free(b);
    free(out);
}
