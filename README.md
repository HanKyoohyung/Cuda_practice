# Cuda_practice

### Note
1. First cudaMalloc is very slow (we have to do dummy cudaMalloc in implementation)
2. cudaMemcpy from device to host is much slower than host to device (about 10 times difference)
