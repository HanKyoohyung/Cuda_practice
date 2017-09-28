#include <stdio.h>
#include <iostream>
#include <chrono>

/*
__global__ void VecAdd(float* A, float *B, float *C)
{
	int idx = threadIdx.x;
	C[idx] = A[idx] + B[idx];
}

// Matrix Addtion using 1 block (threadIdx has limitation about 1024)
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
	int idx1 = threadIdx.x;
	int idx2 = threadIdx.y;
	C[idx1][idx2] = A[idx1][idx2] + B[idx1][idx2];
}

// Matrix Addition using multiple blocks to solve the problem of limitation of thread
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
	int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
	C[idx1][idx2] = A[idx1][idx2] + B[idx1][idx2];
}
*/

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float *elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 32

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A to devide memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);

	auto time_memcpy1_start = std::chrono::high_resolution_clock::now();
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	auto time_memcpy1_end = std::chrono::high_resolution_clock::now();
	std::cout << "Memcpy Time : " << (double)std::chrono::duration_cast<std::chrono::microseconds>(time_memcpy1_end - time_memcpy1_start).count() / 1000000. << " seconds" << std::endl;

	// Load B to devide memory
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);

	auto time_memcpy2_start = std::chrono::high_resolution_clock::now();
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	auto time_memcpy2_end = std::chrono::high_resolution_clock::now();
	std::cout << "Memcpy Time : " << (double)std::chrono::duration_cast<std::chrono::microseconds>(time_memcpy2_end - time_memcpy2_start).count() / 1000000. << " seconds" << std::endl;

	// Allocate C in devide memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	// Kernel part
	auto time_kernel_start = std::chrono::high_resolution_clock::now();
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	auto time_kernel_end = std::chrono::high_resolution_clock::now();
	std::cout << "Kernel Time : " << (double)std::chrono::duration_cast<std::chrono::microseconds>(time_kernel_end - time_kernel_start).count() / 1000000. << " seconds" << std::endl;	

	// Read C from device memory
	auto time_memcpy3_start = std::chrono::high_resolution_clock::now();
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	auto time_memcpy3_end = std::chrono::high_resolution_clock::now();
	std::cout << "Memcpy Time : " << (double)std::chrono::duration_cast<std::chrono::microseconds>(time_memcpy3_end - time_memcpy3_start).count() / 1000000. << " seconds" << std::endl;

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	float Cval = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = 0; i < A.width; i++)
	{
		Cval += A.elements[row * A.width + i] * B.elements[i * B.width + col];
		C.elements[row * C.width + col] = Cval;
	}
}

int main()
{
	int test_dim = 1024;
	Matrix A, B, C;
	A.width = test_dim; A.height = test_dim;
	B.width = test_dim; B.height = test_dim;
	C.width = test_dim; C.height = test_dim;
	A.elements = new float[test_dim * test_dim];
	B.elements = new float[test_dim * test_dim];
	C.elements = new float[test_dim * test_dim];
	for(int i = 0; i < test_dim; i++)
	{
		for(int j = 0; j < test_dim; j++)
		{
			*(A.elements + i * test_dim + j) = 1;
			*(B.elements + i * test_dim + j) = 1;
		}
	}

	auto time_start = std::chrono::high_resolution_clock::now();
	MatMul(A, B, C);
	auto time_end = std::chrono::high_resolution_clock::now();
	std::cout << "MatMul 1 Time : " << (double)std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1000000. << " seconds" << std::endl;
 
	time_start = std::chrono::high_resolution_clock::now();
	MatMul(A, B, C);
	time_end = std::chrono::high_resolution_clock::now();
	std::cout << "MatMul 2 Time : " << (double)std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1000000. << " seconds" << std::endl;

	for(int i = 0; i < 10; i++)
	{
		std::cout << "C[0][" << i << "] = " << C.elements[i] << std::endl;
	}

	return 0;
}
