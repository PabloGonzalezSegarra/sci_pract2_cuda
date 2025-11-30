#define N 100000000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
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
    cudaMalloc((void**)&a_cuda, sizeof(float) * (size_t)N);
    cudaMalloc((void**)&b_cuda, sizeof(float) * (size_t)N);
    cudaMalloc((void**)&out_cuda, sizeof(float) * (size_t)N);

    // Copy inputs to device
    cudaMemcpy(a_cuda, a, sizeof(float) * (size_t)N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, sizeof(float) * (size_t)N, cudaMemcpyHostToDevice);

    // Main function
    vector_add<<<1,1>>>(out, a_cuda, b_cuda, N);

    cudaMemcpy(out, out_cuda, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

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
