#pragma once

//struct ENGINEGPUKERNELS_EXPORT CudaConstants
struct CudaConstants
{
    int NUM_THREADS_PER_BLOCK = 32;/* = 64*/
    int NUM_BLOCKS = 128;/* 64*/

    int MAX_CLUSTERS = 0; /*500000*/
    int MAX_CELLS = 0;/* 2000000*/
    int MAX_PARTICLES =100000;// 0;/* 2000000*/
    int MAX_TOKENS = 0;/* 500000*/
    int MAX_CELLPOINTERS = 0;/* MAX_CELLS * 10*/
    int MAX_CLUSTERPOINTERS = 0;/* MAX_CLUSTERS * 10*/
    int MAX_PARTICLEPOINTERS = 100000;/* MAX_PARTICLES * 10*/
    int MAX_TOKENPOINTERS = 0;/* MAX_TOKENS * 10*/

    int DYNAMIC_MEMORY_SIZE = 50000000;
    int METADATA_DYNAMIC_MEMORY_SIZE =  10000000;
};

struct  CudaConstantsH
{
    int NUM_THREADS_PER_BLOCK = 32;/* = 64*/
    int NUM_BLOCKS = 128;/* 64*/

    int MAX_CLUSTERS = 0; /*500000*/
    int MAX_CELLS = 0;/* 2000000*/
    int MAX_PARTICLES =100000;// 0;/* 2000000*/
    int MAX_TOKENS = 0;/* 500000*/
    int MAX_CELLPOINTERS = 0;/* MAX_CELLS * 10*/
    int MAX_CLUSTERPOINTERS = 0;/* MAX_CLUSTERS * 10*/
    int MAX_PARTICLEPOINTERS = 100000;/* MAX_PARTICLES * 10*/
    int MAX_TOKENPOINTERS = 0;/* MAX_TOKENS * 10*/

    int DYNAMIC_MEMORY_SIZE = 50000000;
    int METADATA_DYNAMIC_MEMORY_SIZE =  10000000;
};


struct SimulationParametersH
{
    float cellMinDistance = 0.0;
    float cellMaxDistance = 0.0;
};
/*
struct ENGINEGPUKERNELS_EXPORT SimulationParametersS
{
    float cellMinDistance = 0.0;
    float cellMaxDistance = 0.0;

};
*/
