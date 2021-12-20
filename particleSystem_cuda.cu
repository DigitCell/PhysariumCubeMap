/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"


#include "particleSystem.cuh"

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }

       // cudaSetDevice(1);
        //cudaGLSetGLDevice(1);
    }
/*
    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }
*/
    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void copyArrayToDeviceS(void *device, const void *host, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device, host, size, cudaMemcpyHostToDevice));
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                             *cuda_vbo_resource));
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    void copyArrayFromDevice(void *host, const void *device,
                             struct cudaGraphicsResource **cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }


    void setParameters(SimParams *hostParams)
    {
       // copy parameters to constant memory
       checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    void InitParticleSystem(ParticleSystem* psystem )
    {
        // InitParticlesSystemKernel<<<1,5>>>(psystem->particlesList, psystem->numberGen );


         InitCatSystemKernel2<<<150,256>>>(psystem->botList, psystem->particlesList,  psystem->numberGen);
         int* pointsNumber=new int(0);
         psystem->particlesList.getEntriesForHost(pointsNumber);
         getLastCudaError("Kernel execution failed");
         free(pointsNumber);
    }


    void ClearParticleSystem(ParticleSystem* psystem )
    {
        // InitParticlesSystemKernel<<<1,5>>>(psystem->particlesList, psystem->numberGen );

         ClearSystemKernel<<<150,256>>>(psystem->botList, psystem->particlesList);
         int* pointsNumber=new int(0);
         psystem->particlesList.getEntriesForHost(pointsNumber);
         getLastCudaError("Kernel execution failed");
         free(pointsNumber);
    }

    void MoveCaterpillarsKernel(ParticleSystem* psystem, int* particlesWorldMap, int Tick )
    {
        /*
        int* catNumber=new int(0);
        psystem->botList.getEntriesForHost(catNumber);
        int* pointNumber=new int(0);
        psystem->particlesList.getEntriesForHost(pointNumber);

        MotorCatSystemKernel<<<256,2*256>>>(psystem->_T1, psystem->botList, *catNumber, psystem->particlesList, *pointNumber,  psystem->numberGen, particlesWorldMap, Tick);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
        SensorCatSystemKernel<<<256,2*256>>>(psystem->_T1, psystem->botList, *catNumber, psystem->particlesList, *pointNumber,  psystem->numberGen, particlesWorldMap, Tick);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
        DepositionCatSystemKernel<<<256,2*256>>>(psystem->_T1, psystem->botList, *catNumber, psystem->particlesList, *pointNumber,  psystem->numberGen, particlesWorldMap, Tick);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");

        free(catNumber);
        free(pointNumber);
        */

    }

    void MoveCaterpillarsKernel6(ParticleSystem* psystem, int* particlesWorldMap, int Tick )
    {
        int* catNumber=new int(0);
        psystem->botList.getEntriesForHost(catNumber);
        int* pointNumber=new int(0);
        psystem->particlesList.getEntriesForHost(pointNumber);

        //for(int face=0; face<6; face++)
        {
            MotorCatSystemKernelFace<<<256,2*256>>>(psystem->array1_gpu, psystem->botList, *catNumber, psystem->particlesList, *pointNumber,  psystem->numberGen, particlesWorldMap, Tick);
            cudaDeviceSynchronize();
            getLastCudaError("Kernel execution failed");
            SensorCatSystemKernelFace<<<256,2*256>>>(psystem->array1_gpu, psystem->botList, *catNumber, psystem->particlesList, *pointNumber,  psystem->numberGen, particlesWorldMap, Tick);
            cudaDeviceSynchronize();
            getLastCudaError("Kernel execution failed");
            DepositionCatSystemKernelFace<<<256,2*256>>>(psystem->array1_gpu, psystem->botList, *catNumber, psystem->particlesList, *pointNumber,  psystem->numberGen, particlesWorldMap, Tick);
            cudaDeviceSynchronize();
            getLastCudaError("Kernel execution failed");



        }
 /*
        SensorCatSystemKernel<<<256,2*256>>>(psystem->_T1, psystem->botList, *catNumber, psystem->particlesList, *pointNumber,  psystem->numberGen, particlesWorldMap, Tick);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
        DepositionCatSystemKernel<<<256,2*256>>>(psystem->_T1, psystem->botList, *catNumber, psystem->particlesList, *pointNumber,  psystem->numberGen, particlesWorldMap, Tick);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
*/
        free(catNumber);
        free(pointNumber);

    }

    void launch_cudaRender(ParticleSystem* psystem, unsigned int  *g_odata, int width, int height,
                           int tick,
                           int* colorMap, int numberColors, int* d_colorWorldMap,
                           int* particlesWorldMap)
    {


        dim3 threadsperBlock(32, 32);
        dim3 numBlocks1((width + threadsperBlock.x - 1) / threadsperBlock.x,
                       (height + threadsperBlock.y - 1) / threadsperBlock.y);




        cudaRenderClear<<<numBlocks1, threadsperBlock>>>( psystem->_T1, g_odata, width, height,  tick, colorMap, numberColors, d_colorWorldMap, particlesWorldMap);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");

        int* catNumber=new int(0);
        psystem->botList.getEntriesForHost(catNumber);
        int* pointNumber=new int(0);
        psystem->particlesList.getEntriesForHost(pointNumber);
/*
        MoveRenderKernel<<<256, 256>>>( psystem->_T1,
                                        g_odata,
                                        width,
                                        height,
                                        psystem->botList,
                                        *catNumber,
                                        psystem->particlesList,
                                        *pointNumber,
                                        psystem->numberGen,
                                        colorMap, numberColors, d_colorWorldMap, particlesWorldMap);
                                        */
         cudaDeviceSynchronize();
         getLastCudaError("Kernel execution failed");

         free(catNumber);
         free(pointNumber);


      }


    void launch_cudaRender6(int face, ParticleSystem* psystem, unsigned int  *g_odata, int width, int height,
                           int tick,
                           int* colorMap, int numberColors, int* d_colorWorldMap,
                           int* particlesWorldMap)
    {


        dim3 threadsperBlock(32, 32);
        dim3 numBlocks1((width + threadsperBlock.x - 1) / threadsperBlock.x,
                       (height + threadsperBlock.y - 1) / threadsperBlock.y);

        EvaporateCatSystemKernelFace<<<numBlocks1, threadsperBlock>>>( face, psystem->array1_gpu, g_odata, width, height,  tick, colorMap, numberColors, d_colorWorldMap, particlesWorldMap);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");


        cudaRenderClearFace<<<numBlocks1, threadsperBlock>>>( face, psystem->array1_gpu, g_odata, width, height,  tick, colorMap, numberColors, d_colorWorldMap, particlesWorldMap);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");

        int* catNumber=new int(0);
        psystem->botList.getEntriesForHost(catNumber);
        int* pointNumber=new int(0);
        psystem->particlesList.getEntriesForHost(pointNumber);
/*
        MoveRenderKernel<<<256, 256>>>( psystem->_T1,
                                        g_odata,
                                        width,
                                        height,
                                        psystem->botList,
                                        *catNumber,
                                        psystem->particlesList,
                                        *pointNumber,
                                        psystem->numberGen,
                                        colorMap, numberColors, d_colorWorldMap, particlesWorldMap);
                                        */
         cudaDeviceSynchronize();
         getLastCudaError("Kernel execution failed");

         free(catNumber);
         free(pointNumber);


      }


