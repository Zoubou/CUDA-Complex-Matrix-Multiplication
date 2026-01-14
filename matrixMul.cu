#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 2048

__global__ void matrixMul(float *A, float *B, float *C, float *D, float *E, float *F, int n) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sumE = 0.0f;
        float sumF = 0.0f;

        for (int k = 0; k < n; ++k) {
            float a = A[row * n + k];
            float b = B[row * n + k];
            float c = C[k * n + col];
            float d = D[k * n + col];

            sumE += (a * c - b * d);
            sumF += (a * d + b * c);
        }

        E[row * n + col] = sumE;
        F[row * n + col] = sumF;
    }
}

int main(){
    //Host vars
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];
    float *D = new float[N*N];

    float *E = new float[N*N];
    float *F = new float[N*N];

    //Device vars
    float *dev_A;
    float *dev_B;
    float *dev_C;
    float *dev_D;

    float *dev_E;
    float *dev_F;

    for(int i=0; i<N-1; i++){
        for(int j=0; j<N; j++){
	    A[i * N + j] = (float)rand() / (float)RAND_MAX;
	    B[i * N + j] = (float)rand() / (float)RAND_MAX;
	    C[i * N + j] = (float)rand() / (float)RAND_MAX;
	    D[i * N + j] = (float)rand() / (float)RAND_MAX;
	}
    }

    std::cout << "Initializing data on GPU\n";

    cudaMalloc( (void**)&dev_A, N*N*sizeof(float) );
    cudaMalloc( (void**)&dev_B, N*N*sizeof(float) );
    cudaMalloc( (void**)&dev_C, N*N*sizeof(float) );
    cudaMalloc( (void**)&dev_D, N*N*sizeof(float) );

    cudaMalloc( (void**)&dev_E, N * N * sizeof(float) );
    cudaMalloc( (void**)&dev_F, N * N * sizeof(float) );

    cudaMemcpy(dev_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_D, D, N * N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Launching kernels on GPU\n";

    int TILE_SIZE = 32;

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    int gridDim = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 blocksPerGrid(gridDim, gridDim);

    auto start = std::chrono::high_resolution_clock::now();

    matrixMul <<< blocksPerGrid, threadsPerBlock >>>(dev_A, dev_B, dev_C, dev_D, dev_E, dev_F, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    // Wait for GPU to finish before downloading data
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();

    std::cout << "Downloadng data...\n";

    cudaMemcpy(E, dev_E, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(F, dev_F, N * N * sizeof(float), cudaMemcpyDeviceToHost);


}
