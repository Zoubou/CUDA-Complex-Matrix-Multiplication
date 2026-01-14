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

void cpu_matrixMul(float *A, float *B, float *C, float *D, float *E_cpu, float *F_cpu, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sumE = 0;
            float sumF = 0;
            for (int k = 0; k < n; ++k) {
                // (A + Bi)(C + Di) = (AC - BD) + (AD + BC)i
                sumE += A[i * n + k] * C[k * n + j] - B[i * n + k] * D[k * n + j];
                sumF += A[i * n + k] * D[k * n + j] + B[i * n + k] * C[k * n + j];
            }
            E_cpu[i * n + j] = sumE;
            F_cpu[i * n + j] = sumF;
        }
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

    std::cout << "Downloadng data...\n";

    cudaMemcpy(E, dev_E, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(F, dev_F, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    //GPU Flops/s and throughput
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();

    double N_val = static_cast<double>(N);
    double total_flops = 8.0 * N_val * N_val * N_val;

    double flops_per_second = total_flops / seconds;
    double gflops = flops_per_second / 1e9; 
    double tflops = flops_per_second / 1e12; 

    //CPU throughput
    auto start_cpu = std::chrono::high_resolution_clock::now();

    std::cout << "Calculating CPU reference..." << std::endl;
    cpu_matrixMul(A, B, C, D, cpu_E, cpu_F, N);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    double cpu_seconds = diff_cpu.count();

    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "Matrix Size N: " << N << " x " << N << std::endl;
    std::cout << "GPU Execution Time: " << seconds << " seconds" << std::endl;
    std::cout << "Total FLOPs: " << total_flops << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;
    if (tflops > 1.0) {
        std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;
    }
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "CPU Execution Time: " << seconds << " cpu_seconds" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
}
