#pragma once

template <typename T>
__host__ __device__ __inline__ void swapT(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}


__device__ int atomicAdd(int* address, int val);
__device__ unsigned int atomicAdd(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicAdd(unsigned long long int* address,unsigned long long int val);
__device__ float atomicAdd(float* address, float val);
__device__ double atomicAdd(double* address, double val);
__device__ unsigned int atomicInc(unsigned int* address,unsigned int val);

__device__ int atomicAdd_block(int* address, int val);
__device__ unsigned int atomicAdd_block(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicAdd_block(unsigned long long int* address,unsigned long long int val);
__device__ float atomicAdd_block(float* address, float val);
__device__ double atomicAdd_block(double* address, double val);
__device__ unsigned int atomicInc_block(unsigned int* address,unsigned int val);


__device__ int atomicExch(int* address, int val);
__device__ unsigned int atomicExch(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicExch(unsigned long long int* address,unsigned long long int val);
__device__ float atomicExch(float* address, float val);

__device__ int atomicExch_block(int* address, int val);
__device__ unsigned int atomicExch_block(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicExch_block(unsigned long long int* address,unsigned long long int val);
__device__ float atomicExch_block(float* address, float val);

__device__ int atomicMax(int* address, int val);
__device__ unsigned int atomicMax(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicMax(unsigned long long int* address,unsigned long long int val);

__device__ int atomicMax_block(int* address, int val);
__device__ unsigned int atomicMax_block(unsigned int* address,unsigned int val);
__device__ unsigned long long int atomicMax_block(unsigned long long int* address,unsigned long long int val);
