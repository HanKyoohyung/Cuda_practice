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
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Loat A to devide memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	// Loat B to devide memory
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	// Allocate C in devide memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	// Kernel part
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// Read C from device memory
	cudaMemcpy(C.elements, C_d.elements, size, cudaMemcpyDeviceToHost);

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
	Matrix A, B, C;
	A.width = 160; A.height = 160;
	B.width = 160; B.height = 160;
	C.width = 160; C.height = 160;
	A.elements = new float[160 * 160];
	B.elements = new float[160 * 160];
	C.elements = new float[160 * 160];
	for(int i = 0; i < 160; i++)
	{
		for(int j = 0; j < 160; j++)
		{
			*(A.elements + i * 160 + j) = 0.1 * (float)i + (float)j;
			*(B.elements + i * 160 + j) = 0.1 * (float)i + (float)j;
		}
	}

	MatMul(A, B, C);

	return 0;
}
