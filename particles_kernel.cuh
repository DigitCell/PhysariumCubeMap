#pragma once


#include "CUDA/Array.cuh"
#include "CUDA/Base.cuh"
#include "constants.hpp"
//#include "constants.hpp"
#include "CUDA/tinycolormap.hpp"
#include "CUDA/checkCudaCall.hpp"

struct Caterpiller;
struct Particle;
struct ParticleSystem;


const uint const_MaxNumberBots=20e5;
const uint const_MAxNumberParticles=1;
const uint const_MaxparticlesPointers=1;
const uint const_MaxSteps=1;
const uint const_MaxfutureSteps=5;


const uint const_MAxNumberGenInit=10e6;

#define FP_PRECISION 0.00001

#define CUDA_THROW_NOT_IMPLEMENTED() printf("not implemented"); \
    while(true) {};

#define DEBUG
#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif




// simulation parameters
struct SimParams
{
    int tickTrailLenth;
    int trailDescrease;
    int imgw;
    int imgh;
    int quartSize;

    int width;
    int height;
    int faces;
    int traceStartColorIndex;
    int wormPathAlertColor;

    float wormVelocity01;
    int numSteps01;
    int numDirections01;
    float angleDivide01;


    float wormVelocity02;
    int numSteps02;
    int numDirections02;
    float angleDivide02;

    float wormVelocity03;
    int numSteps03;
    int numDirections03;
    float angleDivide03;

    int const_NumberParticles;

    float decayT;

    float FL_grad ;
    float FR_grad ;
    float RA_grad ;

    float FL ;
    float FR;
    float RA;
    float SO;
    float SS;
    float depT;

    bool blue_enable;

};


struct PointCA
{
    float value=0.0f;
    int2 direction=make_int2(0,0);
    int  color=0;
    float  status=0;
    float old_value=0.0f;
    float angle=0;
    int agent_number=0;
};

struct lbm_node{
    //velocities:
    float h=0;
    float h_new=0;

    int agent_number=0;

    float2 u;	//velocity
    float rho;	//density. aka rho

    float2 u_init;	//velocity
    float rho_init;	//density. aka rho

    float2 u_prev;	//velocity
    float rho_prev;	//density. aka rho

    float divergence=0;
    float p=0;
    float p_previous=0;

    float f[9];
    float direction=0;

    // For fast find coordinates

    int index;
    int face=0;
    int x = 180;
    int y = 92;
    int dx = 1;
    int dy = 0;

    float deltaAngle=0;

    float value=0;

};

struct SphereCoord
{
    int face = 0;
    int x = 180;
    int y = 92;
    int dx = 1;
    int dy = 0;
};


struct Particle
{
    int id;
    bool active;

    float2 relPos;
    float2 relPos_new;

    float direction;
    float direction_new;

    float h;
    float h_new;

    float2 u;	//velocity

    int face = 0;
    int x = 0;
    int y = 0;
    int dx = 0;
    int dy = 0;

    int x_new = 0;
    int y_new = 0;
    int dx_new = 0;
    int dy_new = 0;

    uint4 color;

    int3 colorInt;
    Caterpiller* Caterpille;
    float radius=1.0f;
};

struct Caterpiller
{
    int id;
    bool active;
    int length;
    float direction;
    float direction_new;
    float velocity;

    int numParticlesPointers;
    Particle* particlesPointers [const_MaxparticlesPointers];
    int numberFutureSteps=2;
    float2 futureSteps [const_MaxSteps];
    float2 futureStepsPrev [const_MaxfutureSteps];
    uint4 color;
    int3 colorInt;

};


#include <cuda_runtime.h>
#include <helper_cuda.h>

struct ParticleSystem
{

    void InitializeT( SimParams& params);

    //get 1d flat index from row and col
    inline int getIndex_cpu(int face, int x, int y, SimParams& params) {
        return face*params.width*params.width + y*params.width + x;
    }

    ParticleSystem(SimParams& params){

        botList.init(const_MaxNumberBots);
        particlesList.init(const_MAxNumberParticles*const_MaxNumberBots);
        numberGen.init(const_MAxNumberGenInit);


        uint m_texWidth=params.imgw;
        uint m_texHeight=params.imgh;

        PointCA *T = new PointCA [m_texHeight*m_texWidth];

        // initialize array on the host
        //InitializeT(params);

        cudaMalloc(&_T1,m_texHeight*m_texWidth*sizeof(PointCA));
        cudaMalloc(&_T2,m_texHeight*m_texWidth*sizeof(PointCA));

        // copy (initialized) host arrays to the GPU memory from CPU memory
        checkCudaCall(cudaMemcpy(_T1,T,m_texHeight*m_texWidth*sizeof(PointCA),cudaMemcpyHostToDevice));
        checkCudaCall(cudaMemcpy(_T2,T,m_texHeight*m_texWidth*sizeof(PointCA),cudaMemcpyHostToDevice));


        //cuda error variables:
        cudaError_t ierrAsync;
        cudaError_t ierrSync;

        int W = params.width;
        int H = params.height;
        int F=  params.faces;

        //printf("velocity is %.6f my dude\n", v);
        array1 = (lbm_node*)calloc(F*W*H, sizeof(lbm_node));
        array2 = (lbm_node*)calloc(F*W*H, sizeof(lbm_node));

        lbm_node* before = array1;

        int i;
        for(int face=0; face<6; face++)
        {
            for (int x = 0; x < params.width; x++)
            {
                for (int y = 0; y < params.height; y++)
                {
                    i = getIndex_cpu(face,x, y, params);

                    before[i].h= 0.0f;
                    before[i].h_new= 0.0f;
                    before[i].agent_number= 0;

                }
            }
        }


        ierrSync = cudaMalloc(&array1_gpu, sizeof(lbm_node)*F*W*H);
        if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
        ierrSync = cudaMalloc(&array2_gpu, sizeof(lbm_node)*F*W*H);
        if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }

        ierrSync = cudaMemcpy(array1_gpu, array1, sizeof(lbm_node)*F*W*H, cudaMemcpyHostToDevice);
        if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
        ierrSync = cudaMemcpy(array2_gpu, array1, sizeof(lbm_node)*F*W*H, cudaMemcpyHostToDevice);
        if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }


    };

    ~ParticleSystem() {};

    enum ParticleConfig
    {
        CONFIG_RANDOM,
        CONFIG_GRID,
        _NUM_CONFIGS
    };

    Array<Caterpiller> botList;
    Array<Particle> particlesList;

    CudaNumberGenerator numberGen;

    PointCA *T;          // pointer to host (CPU) memory
    PointCA *_T1, *_T2;  // pointers to device (GPU) memory


    lbm_node* array1;
    lbm_node* array2;
    lbm_node* array1_gpu;
    lbm_node* array2_gpu;



};

inline void ParticleSystem::InitializeT( SimParams& params)
{

    //cuda error variables:
    cudaError_t ierrAsync;
    cudaError_t ierrSync;

    lbm_node* before = array1;

    int W = params.width;
    int H = params.height;
    int F=  params.faces;


    int i;
    for(int face=0; face<6; face++)
    {
        for (int x = 0; x < params.width; x++)
        {
            for (int y = 0; y < params.height; y++)
            {
                i = getIndex_cpu(face,x, y, params);

                before[i].h= random(2*M_PI);
                before[i].h_new=random(2*M_PI);
                before[i].agent_number= 0;

            }
        }
    }


    ierrSync = cudaMemcpy(array1_gpu, array1, sizeof(lbm_node)*F*W*H, cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMemcpy(array2_gpu, array1, sizeof(lbm_node)*F*W*H, cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }


}
