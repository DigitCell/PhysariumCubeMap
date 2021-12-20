
#include "particles_kernel.cuh"


void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
void copyArrayToDevice(void *device, const void *host, int offset, int size);
void copyArrayToDeviceS(void *device, const void *host,  int size);
void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

void setParameters(SimParams *hostParams);

void InitParticleSystem(ParticleSystem* psystem);
void ClearParticleSystem(ParticleSystem* psystem );


void MoveCaterpillarsKernel(ParticleSystem* psystem, int* particlesWorldMap, int Tick);

void MoveCaterpillarsKernel6(ParticleSystem* psystem, int* particlesWorldMap, int Tick);

void launch_cudaRender(ParticleSystem* psystem, unsigned int  *g_odata, int width, int height,
                       int tick,
                       int* colorMap, int numberColors, int* d_colorWorldMap,
                       int* particlesWorldMap);

void launch_cudaRender6(int face, ParticleSystem* psystem, unsigned int  *g_odata, int width, int height,
                       int tick,
                       int* colorMap, int numberColors, int* d_colorWorldMap,
                       int* particlesWorldMap);
