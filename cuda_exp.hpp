#ifndef CUDA_EXP_HPP
#define CUDA_EXP_HPP

#pragma once

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "stb_image.h"
//#include "particles_kernel_impl.cuh"

//#include <helper_gl.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

//#include <rendercheck_gl.h>

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "vector_functions.h"
#include "environment.hpp"

#include "particleSystem.cuh"
#include "particles_kernel.cu"
#include "CUDA/Array.cuh"

#include "constants.hpp"



class Cuda_exp
{
public:
    Cuda_exp(const int maxnumBots, const int maxnumParticles);


    uint numParticles = 0;
    uint2 gridSize;
    int numIterations = 0; // run until exit

    Environment*  env;
    ParticleSystem* psystem;


    // fps
    int fpsCount = 0;
    int fpsLimit = 1;
    StopWatchInterface *timer = NULL;

    //ParamListGL *params;

    // Auto-Verification Code
    unsigned int frameCount = 0;
    unsigned int g_TotalErrors = 0;
    char        *g_refFile = NULL;


    void moveCaterpillers(int Tick);

    // initialize particle system
    void initParticleSystem(int numParticles, uint2 gridSize, bool bUseOpenGL)
    {

    }

    void cleanup()
    {

    }

    void computeFPS()
    {
        frameCount++;
        fpsCount++;

        if (fpsCount == fpsLimit)
        {
            char fps[256];
            float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
            sprintf(fps, "CUDA Particles (%d particles): %3.1f fps", numParticles, ifps);


            fpsCount = 0;

            fpsLimit = (int)MAX(ifps, 1.f);
            sdkResetTimer(&timer);
        }
    }

    inline float frand()
    {
        return rand() / (float) RAND_MAX;
    }


    void init();
    void surface2DCuda();

    void createGLTextureEmpty(GLuint *gl_tex, unsigned int size_x, unsigned int size_y);
    void createGLTexturefromFile(GLuint *gl_tex, unsigned int size_x, unsigned int size_y);

     void RegisterGLTextureForCUDA(GLuint *gl_tex, cudaGraphicsResource **cuda_tex, unsigned int size_x, unsigned int size_y);

    void initCUDABuffers(GLuint* cuda_dev_render_buffer);
    void initCUDABuffers();
    void generateCUDAImage(int tick);
    void generateCUDAImageList(int tick);

    void updateParams(SimParams* param);
    void initCUDABuffersList();

    void InitParticles();

    int height = 0;
    int width =  0;

    unsigned char *h_data;

    struct cudaResourceDesc resDesc;

    cudaChannelFormatDesc channelDesc;
    cudaArray_t cuInputArray;
    cudaArray_t cuOutputArray;

    cudaArray *texture_ptr;

    cudaArray *texture_ptrList[6];

    cudaSurfaceObject_t inputSurfObj = 0;
    cudaSurfaceObject_t outputSurfObj = 0;

    //  struct cudaGraphicsResource* cuda_tex_resource;
    cudaGraphicsResource* cuda_tex_resource;
    GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result
    GLuint opengl_tex;  // OpenGL Texture for cuda result

    cudaGraphicsResource* cuda_tex_resourceList[6];
    GLuint opengl_tex_cudaList[6];  // OpenGL Texture for cuda result
    GLuint opengl_texList     [6];  // OpenGL Texture for cuda result


  //  void* cuda_dev_render_buffer;
    GLuint* cuda_dev_render_buffer;
    GLuint* cuda_dev_render_bufferList[6];

    // CUDA
    size_t size_tex_data;
    unsigned int num_texels;
    unsigned int num_values;

    int colorMapNumbers=254;
    int* h_colorsMap;
    int* d_colorsMap;
    int* d_colorWorldMap;

    int* d_particlesWorldMap;

    inline unsigned cRGB(unsigned int rc, unsigned int gc, unsigned int bc, unsigned int ac)
    {
        const unsigned char r = ac;
        const unsigned char g = bc;
        const unsigned char b = gc;
        const unsigned char a = rc ;
        return (r << 24) |  (g << 16) |  (b << 8)| a ;
    }







};





#endif // CUDA_EXP_HPP
