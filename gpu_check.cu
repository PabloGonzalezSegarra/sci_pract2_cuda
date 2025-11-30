#include <stdio.h>
#include <cuda_runtime.h>

// Kernel mínimo: escribe un valor conocido en memoria global
__global__ void write_kernel(int *d_val) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *d_val = 12345;
    }
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA-capable device detected.\n");
        return 1;
    }

    // Mostrar información básica del primer dispositivo
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU 0: %s\n", prop.name);

    // Reservar un entero en device, ejecutar kernel y copiar de vuelta
    int h_val = 0;
    int *d_val = NULL;
    cudaMalloc((void**)&d_val, sizeof(int));

    write_kernel<<<1, 1>>>(d_val);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Value read from kernel: %d\n", h_val);

    cudaFree(d_val);
    return 0;
}
