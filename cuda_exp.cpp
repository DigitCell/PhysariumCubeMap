
#pragma once
#include "cuda_exp.hpp"

#include "particleSystem.cuh"
#include "particles_kernel.cu"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>



#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif


inline float frand()
{
    return rand() / (float) RAND_MAX;
}


Cuda_exp::Cuda_exp(const int maxnumBots, const int maxnumParticles)
{

    printf("%s Starting...\n\n", "texture Cuda  particles experimets");

    numParticles = 1000;
    numIterations = 0;

    uint gridDim = 128;
    gridSize.x = gridDim;
    gridSize.y = gridDim;

    int2 worldSize=make_int2(world_width, world_height);
    width=worldSize.x;
    height=worldSize.y;

    printf("grid: %d x %d  = %d cells\n", gridSize.x, gridSize.y, gridSize.x*gridSize.y);
    printf("particles: %d\n", numParticles);

    //bool benchmark = checkCmdLineFlag(_argc, (const char **) _argv, "benchmark") != 0;

    //cudaInit(_argc, _argv);
    //initParticleSystem(numParticles, gridSize, false);

    env=new Environment(numParticles);
    // env->agentCount=psystem->getNumParticles();
    //env->particles=psystem->m_hParticle;

    setParameters(env->parameters);

    psystem=new ParticleSystem(*env->parameters);


   // h_colorsMap=new int[255];
    /*
    h_colorsMap=(int*)malloc(sizeof(int)*colorMapNumbers);
    for(int i=0; i<colorMapNumbers;i++)
    {
        float value=(float)i/colorMapNumbers;
        const tinycolormap::Color color = tinycolormap::GetColor(value, tinycolormap::ColormapType::Cividis);
        h_colorsMap[i]=(int(colorMapNumbers*color.b()) << 16) | (int(colorMapNumbers*color.g()) << 8) | int(colorMapNumbers*color.r());
    }
     cudaMalloc(&d_colorsMap,sizeof(int)*colorMapNumbers);
     cudaMemcpy(d_colorsMap, h_colorsMap, sizeof(int)*colorMapNumbers, cudaMemcpyHostToDevice);

     */


     h_colorsMap=(int*)malloc(sizeof(int)*colorMapNumbers);
         for(int i=0; i<colorMapNumbers;i++)
         {
             float value=(float)i/colorMapNumbers;
             const tinycolormap::Color color = tinycolormap::GetColor(value, tinycolormap::ColormapType::Viridis);
            // h_colorsMap[i]=complementRGB(unsigned(colorMapNumbers*color.b()) << 16) | (unsigned(colorMapNumbers*color.g()) << 8) | unsigned(colorMapNumbers*color.r());
             h_colorsMap[i]=cRGB(unsigned(colorMapNumbers*color.r()),
                                 unsigned(colorMapNumbers*color.g()),
                                 unsigned(colorMapNumbers*color.b()),
                                 254);
         }
          cudaMalloc(&d_colorsMap,sizeof(unsigned)*colorMapNumbers);
          checkCudaCall(cudaMemcpy(d_colorsMap, h_colorsMap, sizeof(unsigned)*colorMapNumbers, cudaMemcpyHostToDevice));


     cudaMalloc(&d_colorWorldMap,sizeof(int)*width*height);
     cudaMemset(d_colorWorldMap, 0, sizeof(int)*width*height);

     cudaMalloc(&d_particlesWorldMap,sizeof(int)*width*height);
     cudaMemset( d_particlesWorldMap, 0, sizeof(int)*width*height);

}

void Cuda_exp::moveCaterpillers(int Tick)
{
   MoveCaterpillarsKernel6(psystem, d_particlesWorldMap, Tick);
}

void Cuda_exp::init()
{
       cudaDeviceProp deviceProp;
       checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
       auto version = deviceProp.major*10 + deviceProp.minor;

        h_data =(unsigned char *)std::malloc(sizeof(unsigned char) * width * height * 4);
        for (int i = 0; i < height * width * 4; ++i)
            h_data[i] = i;

        // Allocate CUDA arrays in device memory
        channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        checkCudaErrors(cudaMallocArray(&cuInputArray, &channelDesc, width, height,  cudaArraySurfaceLoadStore));
        checkCudaErrors(cudaMallocArray(&cuOutputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));

        // Set pitch of the source (the width in memory in bytes of the 2D array
        // pointed to by src, including padding), we dont have any padding
        const size_t spitch = 4 * width * sizeof(unsigned char);
        // Copy data located at address h_data in host memory to device memory
        checkCudaErrors(cudaMemcpy2DToArray(cuInputArray, 0, 0, h_data, spitch,
                       4 * width * sizeof(unsigned char), height,
                        cudaMemcpyHostToDevice));

        // Specify surface
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;

        // Create the surface objects
        resDesc.res.array.array = cuInputArray;
        checkCudaErrors(cudaCreateSurfaceObject(&inputSurfObj, &resDesc));

        resDesc.res.array.array = cuOutputArray;
        checkCudaErrors(cudaCreateSurfaceObject(&outputSurfObj, &resDesc));

      //  createGLTexturefromFile(&opengl_tex_cuda,  width, height);
        createGLTextureEmpty(&opengl_tex_cuda,  width, height);
        RegisterGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, width, height);
        initCUDABuffers();

        for(int i=0; i<6; i++)
        {
             createGLTextureEmpty(&opengl_tex_cudaList[i],  width, height);
             RegisterGLTextureForCUDA(&opengl_tex_cudaList[i], &cuda_tex_resourceList[i], width, height);
        }
        initCUDABuffersList();


       // generateCUDAImage(0);

        InitParticleSystem(psystem);

        printf("generate kernel done");


}

void Cuda_exp::surface2DCuda()
{

}


void Cuda_exp::initCUDABuffers()
{
    // set up vertex data parameters
    num_texels = width * height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(int) * num_values;
    // We don't want to use cudaMallocManaged here - since we definitely want
    checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data)); // Allocate CUDA memory for color output
}


void Cuda_exp::initCUDABuffersList()
{
    for(int i=0; i<6; i++)
    {
        // set up vertex data parameters
        num_texels = width * height;
        num_values = num_texels * 4;
        size_tex_data = sizeof(int) * num_values;
        // We don't want to use cudaMallocManaged here - since we definitely want
        checkCudaErrors(cudaMalloc(&cuda_dev_render_bufferList[i], size_tex_data)); // Allocate CUDA memory for color output
    }
}

void Cuda_exp::InitParticles()
{
    ClearParticleSystem(psystem);
    InitParticleSystem(psystem);

}


void Cuda_exp::initCUDABuffers(GLuint* cuda_dev_render_bufferTemp)
{
    // set up vertex data parameters
    num_texels = width * height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(int) * num_values;
    // We don't want to use cudaMallocManaged here - since we definitely want
    checkCudaErrors(cudaMalloc(&cuda_dev_render_bufferTemp, size_tex_data)); // Allocate CUDA memory for color output
}



void Cuda_exp::createGLTexturefromFile(GLuint *gl_tex, unsigned int size_x, unsigned int size_y)
{

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, gl_tex);
    glBindTexture(GL_TEXTURE_2D, *gl_tex);
    // set basic texture parameters

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    // set basic parameters
    /*
       glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
       glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
       glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
       glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
       */
    // Specify 2D texture
    // load and generate the texture
    int width, height, nrChannels;
    unsigned char *data = stbi_load("../assets/side3_blue512.png", &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGBA, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
       // glGenerateMipmap(GL_TEXTURE_2D);
       // glBindTexture(GL_TEXTURE_2D, 0);

    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
}

void Cuda_exp::createGLTextureEmpty(GLuint *gl_tex, unsigned int size_x, unsigned int size_y)
{
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, gl_tex); // generate 1 texture
    glBindTexture(GL_TEXTURE_2D, *gl_tex); // set it as current target
    // set basic texture parameters

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Specify 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0,  GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void Cuda_exp::RegisterGLTextureForCUDA(GLuint *gl_tex, cudaGraphicsResource **cuda_tex, unsigned int size_x, unsigned int size_y)
{
       //SDK_CHECK_ERROR_GL();
       checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

}

void Cuda_exp::generateCUDAImage(int tick)
{
    // We want to copy cuda_dev_render_buffer data to the texture
    // Map buffer objects to get CUDA device pointers

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

    int size_tex_data_w = sizeof(int) * width;
    size_t wOffset=sizeof(int)*0;
    size_t hOffset=0;

    //checkCudaErrors(cudaMemcpy2DFromArray(cuda_dev_render_buffer, size_tex_data_w, texture_ptr, wOffset, hOffset, size_tex_data_w, height, cudaMemcpyDeviceToDevice));
    launch_cudaRender(psystem, (unsigned int *)cuda_dev_render_buffer, width, height, tick, d_colorsMap, colorMapNumbers, d_colorWorldMap, d_particlesWorldMap);

    checkCudaErrors(cudaMemcpy2DToArray(texture_ptr, wOffset, hOffset, cuda_dev_render_buffer, size_tex_data_w, size_tex_data_w,height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

    //freeArray(texture_ptr);
}


void Cuda_exp::generateCUDAImageList(int tick)
{
    for(int face=0; face<6; face++)
    {
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resourceList[face], 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptrList[face], cuda_tex_resourceList[face], 0, 0));

        int size_tex_data_w = sizeof(int) * width;
        size_t wOffset=sizeof(int)*0;
        size_t hOffset=0;

        //checkCudaErrors(cudaMemcpy2DFromArray(cuda_dev_render_buffer, size_tex_data_w, texture_ptr, wOffset, hOffset, size_tex_data_w, height, cudaMemcpyDeviceToDevice));
        launch_cudaRender6(face, psystem, (unsigned int *)cuda_dev_render_bufferList[face], width, height, tick, d_colorsMap, colorMapNumbers, d_colorWorldMap, d_particlesWorldMap);

        checkCudaErrors(cudaMemcpy2DToArray(texture_ptrList[face], wOffset, hOffset, cuda_dev_render_bufferList[face], size_tex_data_w, size_tex_data_w,height, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resourceList[face], 0));

    }
}

void Cuda_exp::updateParams(SimParams *param)
{
    setParameters(param);
}



