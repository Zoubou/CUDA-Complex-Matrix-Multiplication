# Complex Matrix Multiplication (CUDA)

A high-performance CUDA implementation of $N \times N$ complex matrix multiplication $(A+Bi)(C+Di)$ using a fused custom kernel.

## Features
* **Fused Kernel**: Calculates real ($AC - BD$) and imaginary ($AD + BC$) parts in a single pass.
* **2D Mapping**: Efficient thread-to-matrix mapping using `dim3` grid/block structures.
* **Performance**: Benchmarked for execution time and GFLOPS ($8N^3$ operations).

## How to Run
1. **Compile**:
   ```bash
   nvcc -O3 matrixMul.cu -o matrixMul
1. **Run**:
   ```bash
   ./matrixMul
