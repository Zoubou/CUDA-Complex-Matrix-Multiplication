#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 1024

__global__ void matrixMul(float *A, float *B, float *C, float *D, float *E, float *F){

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < N && col < N) {
    		E[row][col] = A[row][col]*C[row][col] - B[row][col]*D[row][col];
		F[row][col] = A[row][col]*D[row][col] + B[row][col]*C[row][col];
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

    for(int i=0; i<N; i++){
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

    int tile_size = 32;

    dim3 nthreads(TILE_SIZE, TILE_SIZE);

    int gridDim = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 nblocks(gridDim, gridDim);

    matrixMul <<< nblocks, nthreads >>>(dev_A, dev_B, dev_C, dev_D, dev_E, dev_F);

    std::cout << "Downloadng data...\n";


}
