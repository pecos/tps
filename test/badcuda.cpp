#include<cstdio>

// cuda code which is expected to demonstrate problems with cuda-memcheck

__device__ int x;
__global__ void unaligned_kernel(void) {
    *(int*) ((char*)&x + 1) = 42;
}
__global__ void out_of_bounds_kernel(void) {
  *(int*) 0x87654320 = 42;
}

int main() {
  printf("Running unaligned_kernel\n");
  unaligned_kernel<<<1,1>>>();
  printf("Ran unaligned_kernel: %s\n",
	 cudaGetErrorString(cudaGetLastError()));
  printf("Sync: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  printf("Running out_of_bounds_kernel\n");
  out_of_bounds_kernel<<<1,1>>>();
  printf("Ran out_of_bounds_kernel: %s\n", cudaGetErrorString(cudaGetLastError()));
  printf("Sync: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  return 0;
}
