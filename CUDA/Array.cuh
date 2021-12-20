#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "CudaMemoryManager.cuh"
#include "Swap.cuh"

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/async/reduce.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/functional.h"


template <class T>
__device__ __host__
class Array
{
private:
    int _size;
    int* _numEntries;
    T* _data= nullptr;



public:
    Array()
        : _size(0)
    {


    }

    __host__ __inline__ void init(int size)
    {

        _size = size;

        CHECK_FOR_CUDA_ERROR(cudaMalloc((void**)&_data, sizeof(T)*size));
        CHECK_FOR_CUDA_ERROR(cudaMalloc((void**)&_numEntries, sizeof(int)));

        //CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(T)*size, cudaMemcpyHostToDevice));
        //CHECK_FOR_CUDA_ERROR(cudaMemset(_data, 0, sizeof(T)*size));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numEntries, 0, sizeof(int)));
       // int* aa=new int(100);
       // CHECK_FOR_CUDA_ERROR(cudaMemcpy(_numEntries, aa, sizeof(int), cudaMemcpyHostToDevice));
       // (cudaMemcpy(aa, _numEntries, sizeof(int), cudaMemcpyDeviceToHost));

      //  printf("att");

    }

    __host__ __inline__ T* getArrayForHost(int* sizeArray) const
    {
        (cudaMemcpy(sizeArray, _numEntries, sizeof(int), cudaMemcpyDeviceToHost));
        int sizeA=*sizeArray;
        T* result=(T*)malloc(sizeof(T)*sizeA);
        (cudaMemcpy(result, _data, sizeof(T)*sizeA, cudaMemcpyDeviceToHost));

        return result;
    }

    __host__ __inline__ void getEntriesForHost(int* sizeArray) const
    {

      //  (cudaMemcpy(&result, _data, sizeof(T*), cudaMemcpyDeviceToHost));
        (cudaMemcpy(sizeArray, _numEntries, sizeof(int), cudaMemcpyDeviceToHost));

    }

    __device__ __inline__ T* getArrayForDevice() const { return *_data; }


    __host__ __inline__ void free()
    {
        T* data = nullptr;
        checkCudaErrors(cudaMemcpy(&data, _data, sizeof(T*), cudaMemcpyDeviceToHost));

        CudaMemoryManager::getInstance().freeMemory(data);
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_numEntries);
    }

    __device__ __inline__ void swapContent(Array& other)
    {
        swapT(*_numEntries, *other._numEntries);
        swapT(*_data, *other._data);
    }

    __device__ __inline__ void reset() { *_numEntries = 0; }

    int retrieveNumEntries() const
    {
        int result;
        (cudaMemcpy(&result, _numEntries, sizeof(int), cudaMemcpyDeviceToHost));
        return result;
    }

    __device__ __inline__ T* getNewSubarray(int size)
    {
        int oldIndex = atomicAdd(_numEntries, size);

        if (oldIndex + size - 1 >= _size) {
            atomicAdd(_numEntries, -size);
            printf("Not enough fixed memory!\n");
            ABORT();
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T* getNewElement()
    {
        int oldIndex = atomicAdd(_numEntries, 1);
        if (oldIndex >= _size) {
            atomicAdd(_numEntries, -1);
            printf("Not enough fixed memory!\n");
            ABORT();
        }
        return &_data[oldIndex];
    }
    __device__ __inline__ T* atPointer(int index) { return &_data[index]; }
    __device__ __inline__ T& at(int index) { return (_data)[index]; }
    __device__ __inline__ T const& at(int index) const { return (_data)[index]; }

    __device__ __inline__ int getNumEntries() const { return *_numEntries; }


    __device__ __inline__ void setNumEntries(int value) const { *_numEntries = value; }

};
