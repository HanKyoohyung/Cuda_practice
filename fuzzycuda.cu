#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <chrono>

#define NUM_BLOCK 400
#define NUM_THREAD 400

typedef struct
{
	uint64_t first;
	uint64_t second;
} 128BIT;

// elements[i * numElement + j] = i-th block's j-th element
typedef struct
{
	int numBlock; /*=120000*/
	int numElement; /*=37*/
	128BIT *elements;
} DATA;

__global__ void ChooseAndXORAndCheckKernel(const DATA, const 128BIT, bool*);

void ChooseAndXORAndCheck(const DATA Input, const 128BIT Target, bool* Result)
{
	// Load Input to GPU memory
	DATA d_Input;
	size_t size = Input.numBlock * Input.numElement * 128;

	cudaMalloc(&d_Input.elements, size);
	cudaMemcpy(d_Input.elements, Input.elements, size, cudaMemcpyHostToDevice);

	// Load Target to GPU memory
	128BIT d_Target;
	size = 128;
	cudaMalloc(&d_Target, size);
	cudaMemcpy(d_Target, Target, size, cudaMemcpyHostToDevice);

	// Allocate d_Result in GPU
	bool *Result;
	size = Input.numBlock;
	cudaMalloc(Result, size);

	// Launch Kernel
	ChooseAndXORAndCheckKernel<<<NUM_BLOCK, NUM_THREAD>>>(d_Input, d_Target, d_Result);

	// Read d_Result from GPU memory
	cudaMemcpy(Result, d_Result, size, cudaMemcpyDeviceToHost);

	// Free Memory
	cudaFree(d_Input.elements);
	cudaFree(d_Target);
	cudaFree(d_Result);
}

__global__ void ChooseAndXORAndCheckKernel(const DATA Input, const 128BIT Target, bool* Result)
{
	// blockIdx.x = i
	// threadIdx.x = j
	// (i * NUM_THREAD + j)-th BLOCK computation
	// 1. Choose 3 elements
	// 2. XOR
	// 3. Check whether Target is in or not
	// 4. Save the result in Result
}

int main()
{
	DATA input;
	128BIT target;
	bool* result;
	ChooseAndXORAndCheck(input, target, result);

	for(int i = 0; i < input.numBlock; i++)
	{
		if(result[i] == 1)
		{
			cout << "Authentication Success" << endl;
			break;
		}
		if(i == input.numBlock - 1)
		{
			cout << "Authentication Failed" << endl;
		}
	}
	return 0;
}
