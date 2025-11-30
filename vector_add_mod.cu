#define N 100000000

#include <stdio.h>

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

    float* a_cuda = cudaMalloc(sizeof(float) * N);
    float* b_cuda = cudaMalloc(sizeof(float) * N);
    float* out_cuda = cudaMalloc(sizeof(float) * N);

    cudaMemcpy(a_cuda, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    vector_add<<<1,1>>>(out, a_cuda, b_cuda, N);

    cudaMemcpy(out, out_cuda, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();


    // Verify result
    for(int i = 0; i < N; i++){
        if(out[i] != 3.0f){
            printf("Error at index %d: %f\n", i, out[i]);
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
