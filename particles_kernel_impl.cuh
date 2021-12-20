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

/*
 * CUDA particle system kernel code.
 */

#pragma once
#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include <cooperative_groups.h>

//namespace cg = cooperative_groups;
#include "math_constants.h"
#include "particles_kernel.cu"
#include "cuda_device_runtime_api.h"

#include "constants.hpp"


// simulation parameters in constant memory
__constant__ SimParams params;

__device__
inline int32_t FloorPositiveToInt32(float x)
{
    return int32_t(x);  // for some platforms it's desirable to avoid int casts
}
__device__
/// Given a particular axis direction, how do we map (x, y, z) to (u, v, n)?
const uint8_t kSwizzleTable[3][4] =
{
    { 0, 1, 2, 0 },        // +Z
    { 1, 2, 0, 0 },        // +X
    { 2, 0, 1, 0 },        // +Y
};
__device__
const uint8_t kSwizzleTableBack[3][4] =
{
    { 0, 1, 2, 0 },        // +Z
    { 2, 0, 1, 0 },        // +X
    { 1, 2, 0, 0 },        // +Y
};





__device__
int getIndex(int face, int x, int y )
{

    return face*params.quartSize+y*params.width+x;
}



//get 1d flat index from row and col
__device__
int WrapCubeFace(int size, int* faceIn, int* xIn, int* yIn, int* dx, int* dy)
{
    int& face = *faceIn;
    int& x = *xIn;
    int& y = *yIn;

    // when wrapping to a new face, we always switch x and y.
    // when wrapping to the left/bottom (coord goes -ve):
    //   if the source face is positive,
    //   we then negate x, otherwise we negate y.
    // for wrapping to right/top (coord >= size), it's the other way around.

    // We hit one if clause roughly 4/size of the time
    // We hit both an x and y clause roughly 4/sqr(size) of the time.

    // if ((x | y) & ~(size - 1) == 0) // if size is a power of 2, this detects whether there is any wrap
    //     return;

    int s0  = face & 1;
    int s1  = 1 ^ s0;
    int ss0 = 1 - 2 * s0;
    int ss1 = -ss0;

    if (x < 0)
    {
        x += size;

        int x1(y), y1(x);

        x = ss1 * x1 + s1 * (size - 1);
        y = ss0 * y1 + s0 * (size - 1);

        face = 2 * kSwizzleTable[face >> 1][1] + s1;    // -sign(face) * swizzle(face, 0)

        if (dx)
        {
            int dx1(*dx), dy1(*dy);
            *dx = ss1 * dy1;
            *dy = ss0 * dx1;
        }

        // we just switched y to x -- we no longer have to check y, but must check x again.
        if ((x & ~(size - 1)) == 0)
            return 1;
        else
            return 1 + WrapCubeFace(size, &face, &x, &y, dx, dy);   // happens rarely (on a corner diagonal move), so just recurse
    }
    else if (x >= size)
    {
        x -= size;

        int x1(y), y1(x);

        x = ss0 * x1 + s0 * (size - 1);
        y = ss1 * y1 + s1 * (size - 1);

        face = 2 * kSwizzleTable[face >> 1][1] + s0;    // sign(face) * swizzle(face, 0)

        if (dx)
        {
            int dx1(*dx), dy1(*dy);
            *dx = ss0 * dy1;
            *dy = ss1 * dx1;
        }

        // we just switched y to x -- we no longer have to check y, but must check x again.
        if ((x & ~(size - 1)) == 0)
            return 1;
        else
            return 1 + WrapCubeFace(size, &face, &x, &y, dx, dy);   // happens rarely (on a corner diagonal move), so just recurse
    }

    if (y < 0)
    {
        y += size;

        int x1(y), y1(x);

        x = ss1 * x1 + s1 * (size - 1);
        y = ss0 * y1 + s0 * (size - 1);

        face = 2 * kSwizzleTable[face >> 1][2] + 1;    // swizzle(face, 0)

        if (dx)
        {
            int dx1(*dx), dy1(*dy);
            *dx = ss1 * dy1;
            *dy = ss0 * dx1;
        }

        return 1;
    }
    else if (y >= size)
    {
        y -= size;

        int x1(y), y1(x);

        x = ss1 * x1 + s1 * (size - 1);
        y = ss0 * y1 + s0 * (size - 1);

        face = 2 * kSwizzleTable[face >> 1][2];    // -swizzle(face, 0)

        if (dx)
        {
            int dx1(*dx), dy1(*dy);
            *dx = ss1 * dy1;
            *dy = ss0 * dx1;
        }

        return 1;
    }

    return 0;
}



__device__
lbm_node  getIndexWRPT(int face, int x, int dx, int y, int dy)
{

    const int quartSize=params.width-1;
    const int quartSizeF=params.width;

    if(face==0)
    {
        int xx=x;
        int yy=y;

        int x2=x+dx;
        int y2=y+dy;

        lbm_node tempNode;

        if(x2>=0 and x2<=quartSize and y2>=0 and y2<=quartSize)
        {

           int index=getIndex(0, x2, y2 );
           tempNode.index=index;

           tempNode.face=0;
           tempNode.x=x2;
           tempNode.y=y2;
           tempNode.dx=dx;
           tempNode.dy=dy;

           tempNode.deltaAngle=0;

           return tempNode;

        }

        if( x2>quartSize)
        {
            int dx1=x2-quartSizeF;

            int index=getIndex(2, dx1, y2);

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=2;
            tempNode2.x=dx1;
            tempNode2.y=y2;
            tempNode2.dx= dx;
            tempNode2.dy= dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if(x2<0)
        {
            int dx1=quartSizeF+x2;

            int index=getIndex(3, dx1, y2);

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=3;
            tempNode2.x=  dx1;
            tempNode2.y=  y2;
            tempNode2.dx=  dx;
            tempNode2.dy=  dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if( y2>quartSize )
        {

            int dy1=y2-quartSizeF;
            int index=getIndex(4, x2, dy1);

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=4;
            tempNode2.x= x2;
            tempNode2.y= dy1;
            tempNode2.dx=  dx;
            tempNode2.dy=  dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if( y2<0 )
        {
            int dy1=quartSizeF+y2;

            int index=getIndex(5, x2, dy1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=5;
            tempNode2.x= x2;
            tempNode2.y= dy1;
            tempNode2.dx=  dx;
            tempNode2.dy=  dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

    }

    if(face==1)
    {
        int xx=x;
        int yy=y;

        int x2=x+dx;
        int y2=y+dy;

        if(x2>=0 and x2<=quartSize and y2>=0 and y2<=quartSize)
        {

           int index=getIndex(1, x2, y2 );

           lbm_node tempNode;

           tempNode.index=index;
           tempNode.face=1;
           tempNode.x= x2;
           tempNode.y= y2;
           tempNode.dx= dx;
           tempNode.dy= dy;

           tempNode.deltaAngle=0;

           return tempNode;

        }

        if( x2<0 and y>=0)
        {
            int dx1=quartSizeF+x2;

            int index=getIndex(2, dx1, y2);

            lbm_node tempNode2;

            tempNode2.index=index;

            tempNode2.face=2;
            tempNode2.x= dx1;
            tempNode2.y=  y2;
            tempNode2.dx=   dx;
            tempNode2.dy=   dy;

            tempNode2.deltaAngle=0;


            return tempNode2;

        }


        if( x2>quartSize )
        {
            int dx1=x2-quartSizeF;

            int index=getIndex(3,  dx1, y2);

             lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=3;
            tempNode2.x= dx1;
            tempNode2.y=  y2;
            tempNode2.dx= dx;
            tempNode2.dy= dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if( y2>quartSize)
        {
            int dx1= quartSize-x2;
            int dy1=2*quartSize-y2+1;

            int index=getIndex(4,  dx1, dy1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=4;
            tempNode2.x= dx1;
            tempNode2.y= dy1;
            tempNode2.dx= -dx;
            tempNode2.dy= -dy;

            tempNode2.deltaAngle=(float)M_PI;

            return tempNode2;

        }

        if( y2<0 )
        {
            int dx1= quartSize-x2;
            int dy1=abs(y2)-1;

            int index=getIndex(5, dx1, dy1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=5;
            tempNode2.x= dx1;
            tempNode2.y= dy1;
            tempNode2.dx= -dx;
            tempNode2.dy= -dy;

             tempNode2.deltaAngle=(float)M_PI;

            return tempNode2;

        }

    }

    if(face==2)
    {
        int xx=x;
        int yy=y;

        int x2=x+dx;
        int y2=y+dy;

        lbm_node tempNode;

        if(x2>=0 and x2<=quartSize and y2>=0 and y2<=quartSize)
        {

           int index=getIndex(2, x2, y2 );
           tempNode.index=index;

           tempNode.face=2;
           tempNode.x= x2;
           tempNode.y= y2;
           tempNode.dx= dx;
           tempNode.dy= dy;

           tempNode.deltaAngle=0;
           return tempNode;

        }

        if( x2>quartSize )
        {
            int dx1=x2-quartSizeF;
            int index=getIndex(1, dx1 ,y2);

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=1;
            tempNode2.x= dx1;
            tempNode2.y= y2;
            tempNode2.dx= dx;
            tempNode2.dy= dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if( x2<0 )
        {
            int dx1=quartSize-x2;

            int index=getIndex(0, dx1, y2 );

             lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=0;
            tempNode2.x= dx1;
            tempNode2.y= y2;
            tempNode2.dx= dx;
            tempNode2.dy= dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if( y2>quartSize )
        {
            int dy1=2*quartSize-y2+1;

            int index=getIndex(4, dy1, x2 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=4;
            tempNode2.x= dy1;
            tempNode2.y= x2;
            tempNode2.dx= -dy;
            tempNode2.dy=  dx;

            tempNode2.deltaAngle=(float)M_PI_2;

            return tempNode2;

        }

        if( y2<0 )
        {
            int dy1=quartSizeF+y2;
            int dx1=quartSize-x2;

            int index=getIndex(5, dy1, dx1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=5;
            tempNode2.x= dy1;
            tempNode2.y= dx1;
            tempNode2.dx= dy;
            tempNode2.dy= -dx;

            tempNode2.deltaAngle=-(float)M_PI_2;

            return tempNode2;

        }

    }

    if(face==3)
    {
        int xx=x;
        int yy=y;

        int x2=x+dx;
        int y2=y+dy;

        lbm_node tempNode;

        if(x2>=0 and x2<=quartSize and y2>=0 and y2<=quartSize)
        {

           int index=getIndex(3, x2, y2 );

           tempNode.index=index;

           tempNode.face=3;
           tempNode.x= x2;
           tempNode.y= y2;
           tempNode.dx= dx;
           tempNode.dy= dy;
           return tempNode;

           tempNode.deltaAngle=0;

        }

        if( x2>quartSize )
        {
            int dx1=x2-quartSizeF;

            int index=getIndex(0,  dx1, y2 );

             lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=0;
            tempNode2.x= dx1;
            tempNode2.y= y2;
            tempNode2.dx= dx;
            tempNode2.dy= dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if( x2<0 )
        {
            int dx1=quartSize+x2 ;

            int index=getIndex(1, dx1, y2 );

             lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=1;
            tempNode2.x= dx1;
            tempNode2.y= y2;
            tempNode2.dx= dx;
            tempNode2.dy= dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if( y2>quartSize )
        {
            int dy1=y2-quartSizeF;
            int dx1=quartSize-x2;

            int index=getIndex(4, dy1,  dx1 );

             lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=4;
            tempNode2.x= dy1;
            tempNode2.y= dx1;
            tempNode2.dx=  dy;
            tempNode2.dy= -dx;

            tempNode2.deltaAngle=-(float)M_PI_2;

            return tempNode2;

        }

        if( y2<0 )
        {
            int dy1=abs(y2)-1;

            int index=getIndex(5, dy1, x2 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=5;
            tempNode2.x= dy1;
            tempNode2.y= x2;
            tempNode2.dx= -dy;
            tempNode2.dy=  dx;

            tempNode2.deltaAngle=(float)M_PI_2;

            return tempNode2;

        }

    }

    if(face==4)
    {
        int xx=x;
        int yy=y;

        int x2=x+dx;
        int y2=y+dy;

        lbm_node tempNode;

        if(x2>=0 and x2<=quartSize and y2>=0 and y2<=quartSize)
        {

           int index=getIndex(4, x2, y2 );

           tempNode.index=index;
           tempNode.face=4;
           tempNode.x= x2;
           tempNode.y= y2;
           tempNode.dx= dx;
           tempNode.dy= dy;

           tempNode.deltaAngle=0;
           return tempNode;

        }

        if( x2>quartSize )
        {
            int dx1=2*quartSize-x2+1;

            int index=getIndex(2,  y2, dx1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=2;
            tempNode2.x= y2;
            tempNode2.y= dx1;
            tempNode2.dx= dy;
            tempNode2.dy= -dx; ///

            tempNode2.deltaAngle=-(float)M_PI_2;

            return tempNode2;

        }

        if( x2<0 )
        {
            int dx1=quartSizeF+x2;
            int dy1=quartSize-y2;

            int index=getIndex(3, dy1, dx1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=3;
            tempNode2.x= dy1;
            tempNode2.y= dx1;
            tempNode2.dx=  -dy;
            tempNode2.dy=  dx;

            tempNode2.deltaAngle=(float)M_PI_2;

            return tempNode2;

        }
/*
        if( y2>quartSize and x2==quartSize)
        {
            int dx1=quartSize;
            int dy1=2*quartSizeF-y2+1;

            int index=getIndex(2, dx1,  dy1 );

             lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=2;
            tempNode2.x= 0;
            tempNode2.y= quartSize;
            tempNode2.dx=  -dx;
            tempNode2.dy=  dy;

            return tempNode2;

        }
        */

        if( y2>quartSize and x2<=quartSize)
        {
            int dx1=quartSize-x2;
            int dy1=2*quartSize-y2+1;

            int index=getIndex(1, dx1,  dy1 );

             lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=1;
            tempNode2.x= dx1;
            tempNode2.y= dy1;
            tempNode2.dx=  -dx;
            tempNode2.dy=  -dy;

            tempNode2.deltaAngle=-(float)M_PI;

            return tempNode2;

        }

        if( y2<0 )
        {
            int dy1=quartSizeF+y2;

            int index=getIndex(0, x2, dy1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=0;
            tempNode2.x= x2;
            tempNode2.y= dy1;
            tempNode2.dx=  dx;
            tempNode2.dy=  dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

    }

    if(face==5)
    {
        int xx=x;
        int yy=y;

        int x2=x+dx;
        int y2=y+dy;

        if(x2>=0 and x2<=quartSize and y2>=0 and y2<=quartSize)
        {
           int index=getIndex(5, x2, y2 );

           lbm_node tempNode;

           tempNode.index=index;
           tempNode.face=5;
           tempNode.x= x2;
           tempNode.y= y2;
           tempNode.dx= dx;
           tempNode.dy= dy;

           tempNode.deltaAngle=0;
           return tempNode;

        }

        if( x2>quartSize )
        {
            int dx1=x2-quartSizeF;
            int dy1=quartSize-y2;

            int index=getIndex(2, dy1, dx1);

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=2;
            tempNode2.x= dy1;
            tempNode2.y= dx1;
            tempNode2.dx=  -dy;
            tempNode2.dy=  dx;

            tempNode2.deltaAngle=(float)M_PI_2;

            return tempNode2;

        }

        if( x2<0 )
        {
            int dx1=abs(x2)-1;

            int index=getIndex(3, y2, dx1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=3;
            tempNode2.x= y2;
            tempNode2.y= dx1;
            tempNode2.dx=  dy;
            tempNode2.dy=  -dx;

            tempNode2.deltaAngle=-(float)M_PI_2;

            return tempNode2;

        }

        if( y2>quartSize )
        {
            int dy1=y2-quartSizeF;

            int index=getIndex(0, x2,  dy1 );

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=0;
            tempNode2.x= x2;
            tempNode2.y= dy1;
            tempNode2.dx=   dx;
            tempNode2.dy=   dy;

            tempNode2.deltaAngle=0;

            return tempNode2;

        }

        if( y2<0 )
        {

            int dx1=quartSize-x2;
            int dy1= abs(y2)-1;

            int index=getIndex(1, dx1, dy1);

            lbm_node tempNode2;

            tempNode2.index=index;
            tempNode2.face=1;
            tempNode2.x=  dx1;
            tempNode2.y=  dy1;
            tempNode2.dx=  -dx;
            tempNode2.dy=  -dy;

            tempNode2.deltaAngle=-(float)M_PI;

            return tempNode2;

        }

    }

    int x2=x+dx;
    int y2=y+dy;
    int dx2=dx;
    int dy2=dy;
    int face2=face;

    int result=WrapCubeFace(params.width, &face2, &x2, &y2, &dx2, &dy2);

    lbm_node tempNode;//=before[face2*params.quartSize+y2*params.width + x2];
    return tempNode;
}



__device__
int getIndexWrap(int face, int x, int& dx, int y, int& dy )
{

    int x2=x+dx;
    int y2=y+dy;
    int dx2=dx;
    int dy2=dy;
    int face2=face;

    int result=WrapCubeFace(params.width, &face2, &x2, &y2, &dx2, &dy2);

    dx=dx2;
    dy=dy2;

    return face2*params.quartSize+y2*params.width + x2;
}

__device__
int getIndexW(int face, int x, int dx, int y, int dy )
{

    int x2=x+dx;
    int y2=y+dy;
    int dx2=dx;
    int dy2=dy;
    int face2=face;

    int result=WrapCubeFace(params.width, &face2, &x2, &y2, &dx2, &dy2);

    return face2*params.quartSize+y2*params.width + x2;
}





__device__ float clamp2(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp2(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp2(r, 0.0f, 255.0f);
    g = clamp2(g, 0.0f, 255.0f);
    b = clamp2(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}


__forceinline__ __host__ __device__ unsigned texturePixelIndex(unsigned x, unsigned y)
{
    return x + params.imgw * y;
}



__device__
/// Returns the unit length vector for the given angle (in radians).
inline float2 VecFromAngle(const float a)
{
    return make_float2(cosf(a), sinf(a));
}

__device__
/// Returns the unit length vector for the given angle (in radians).
inline int2 intVecFromAngle(const float a, int dist)
{
    return make_int2((int)(cosf(a)*dist), (int)(sinf(a)*dist));
}

__device__ inline int get_index(unsigned int M, int i)
{
    /*
  if (i < 0)
    return (M + i % M) % M;
  if(i >= M)
    return i % M;
  return i;
  */
  return  (i+M)%M;
}

__device__ inline float get_indexf(unsigned int M, float i)
{
    if (i < 0)
      return M-i;
    if(i >= M)
      return i-M;
    return i;
}

__device__ inline float mapcoordf(float i)
{
    if (i < 0)
      return params.imgw-i;
    if(i >= params.imgw)
      return i-params.imgw;
    return i;
}

__device__ inline unsigned int mapPosCorrection(int x, int y, unsigned int width, unsigned int height)
{
    unsigned int xf=(x+width)  % width;
    unsigned int yf=(y+height) % height;
    return yf*height+xf;
}

__device__ inline unsigned int mapPosCorrection(int x, int y)
{
    unsigned int xf=(x+params.imgw) % params.imgw;
    unsigned int yf=(y+params.imgh) % params.imgh;
    return yf*params.imgw+xf;
}

__device__ inline unsigned int mapPosCorrection(int x)
{
    unsigned int xf=(x+params.imgw) % params.imgw;
    return xf;
}

__device__ __host__ inline unsigned int mapPosCorrection2(int x, int y, unsigned int width, unsigned int height)
{
    unsigned int xf=(x+width)  % width;
    unsigned int yf=(y+height) % height;
    if(y<0 or y>=height){
        xf=width-xf;
    }
    if(x<0 or x>=width){
        yf=height-yf;
    }
    return yf*height+xf;
}


__device__ __host__ inline unsigned int mapPos(int x, int y, unsigned int width, unsigned int height)
{
    unsigned int xf=x;
    unsigned int yf=y;

    return yf*height+xf;
}

__device__ unsigned complementRGB_d(unsigned color)
{
    const unsigned char r = 255 - ((color >> 24) & 0xff);
    const unsigned char g = 255 - ((color >> 16) & 0xff);
    const unsigned char b = 255 - ((color >> 8) & 0xff);
    const unsigned char a = color & 0xff;
    return (r << 24) + (g << 16) + (b << 8) + a;
}

__global__ void InitParticlesSystemKernel(Array<Particle> psystem, CudaNumberGenerator numberGen )
{
    //printf("Start kernel \n");
       uint index = blockIdx.x*blockDim.x + threadIdx.x;
   //{
       Particle* particle=psystem.getNewElement();
       particle->id=index;
       particle->active=true;
       particle->radius=1.0f;

       particle->relPos=make_float2(0,25-index*2);
       int a=psystem.getNumEntries();

      // printf(" add element %d ", a);
     //  printf(" \n");
     //  printf("Finish kernel \n");
 //  }
}


__global__ void OrientationParticlesSystemKernel(Array<Particle> psystem, int numParticles,  CudaNumberGenerator numberGen)
{
    //printf("Start kernel \n");
       uint index = blockIdx.x*blockDim.x + threadIdx.x;
   if(index<numParticles)
   {

      if(index!=0)
      {
          auto a=psystem.at(index-1).relPos;
          auto b=psystem.at(index).relPos;
          auto delta=a-b;
          normalize(delta);

          psystem.at(index).direction=atan2(delta.y,delta.x);
      }
      else
      {
         psystem.at(index).direction=M_PI_2+M_PI/(30.f-numberGen.random(60.f));

      }

   }
}


__global__ void InitCatSystemKernel(Array<Caterpiller> catList, Array<Particle> pointList, CudaNumberGenerator numberGen )
{
    /*
      int2 worldSize=make_int2(world_width, world_height);
    //printf("Start kernel \n");
       uint index = blockIdx.x*blockDim.x + threadIdx.x;
   //{
       Caterpiller* cat=catList.getNewElement();
       cat->id=index;
       int numberPoints=params.const_NumberParticles;
       cat->numParticlesPointers=numberPoints;
       cat->length=numberPoints;

       cat->direction=numberGen.random(2*3.1415f);//+M_PI/(60.f-numberGen.random(120.f));

       float dx=(int)(params.SS * cosf(cat->direction));
       float dy=(int)(params.SS * cosf(cat->direction));
       cat->direction_new= cat->direction;

       float2 catPlace=make_float2(0,0);
       int catFace=0;//numberGen.random(5);

       for(int i=0; i<numberPoints; i++)
       {
           Particle* particle=pointList.getNewElement();
           particle->id=index*100000+i;
           particle->Caterpille=cat;
           particle->active=true;

           particle->relPos=catPlace;
           particle->relPos_new=catPlace;

           particle->face=catFace;
           particle->x=numberGen.random(worldSize.x-1);
           particle->y=0;//numberGen.random(worldSize.y-1);

           particle->dx= -1;//dx;
           particle->dy=  1;//dy;

           particle->u.x=dx;
           particle->u.y=dy;

           particle->direction=cat->direction;
           particle->direction_new=cat->direction;
           cat->particlesPointers[i]=particle;

           particle->radius=1.0f;

       }
       */
}


__global__ void InitCatSystemKernel2(Array<Caterpiller> catList, Array<Particle> pointList, CudaNumberGenerator numberGen )
{
      int2 worldSize=make_int2(world_width, world_height);
    //printf("Start kernel \n");
       uint index = blockIdx.x*blockDim.x + threadIdx.x;
   //{
       Caterpiller* cat=catList.getNewElement();
       cat->id=index;
       int numberPoints=params.const_NumberParticles;
       cat->numParticlesPointers=numberPoints;
       cat->length=numberPoints;

       cat->direction=numberGen.random(2*3.1415f);//+M_PI/(60.f-numberGen.random(120.f));

       float dx=(int)(params.SS * cosf(cat->direction));
       float dy=(int)(params.SS * sinf(cat->direction));
       cat->direction_new= cat->direction;

       float2 catPlace=make_float2(0,0);
       int catFace=numberGen.random(5);

       for(int i=0; i<numberPoints; i++)
       {
           Particle* particle=pointList.getNewElement();
           particle->id=index*100000+i;
           particle->Caterpille=cat;
           particle->active=true;

           particle->relPos=catPlace;
           particle->relPos_new=catPlace;

           particle->face=catFace;

           particle->x=numberGen.random(worldSize.x-1);
           particle->y=numberGen.random(worldSize.y-1);

           particle->x_new=particle->x;
           particle->y_new=particle->y;

           particle->dx= dx;
           particle->dy= dy;


           particle->direction=cat->direction;
           particle->direction_new=cat->direction;

           particle->h=cat->direction;
           particle->h_new=cat->direction;

           cat->particlesPointers[i]=particle;

           particle->radius=1.0f;

       }
}

__global__ void ClearSystemKernel(Array<Caterpiller> catList, Array<Particle> pointList)
{

       uint index = blockIdx.x*blockDim.x + threadIdx.x;

       if(index==0)
       {

           catList.reset();
           pointList.reset();
       }

}



__device__
/// Returns the unit length vector for the given angle (in radians).
inline float FindDirectionMap(PointCA *T, Caterpiller* cat, Array<Particle> pointList, int numberParticles, CudaNumberGenerator numberGen,
                              float &velocity, int* particlesWorldMap, int numberSteps, int numDirections, float angleDivide, bool enableM_PI)
{

    Particle* particleH=cat->particlesPointers[0];
    auto a=particleH->relPos;
    auto h=cat->direction;

    auto h_new=h;

    float F  = T[mapPosCorrection(  roundf((a.x + params.SO * cosf(h)))
                                   ,roundf((a.y + params.SO * sinf(h))))].value;

    float FL = T[mapPosCorrection(  roundf((a.x + params.SO * cosf(h+ params.FL)))
                                   ,roundf((a.y + params.SO * sinf(h+ params.FL))))].value;

    float FR = T[mapPosCorrection(  roundf((a.x + params.SO * cosf(h+ params.FR)))
                                   ,roundf((a.y + params.SO * sinf(h+ params.FR))))].value;

    if (F > FL and F > FR){
     h_new = h;
    }
    else if (F < FL and F < FR){
     if ( numberGen.random(1000) >500){
       h_new = h + params.RA;
     }
     else{
       h_new = h - params.RA;
     }

    }
    else if (FL < FR){
     h_new= h - params.RA;
    }
    else if (FR < FL){
     h_new = h + params.RA;
    }
    else
    {
     h_new = h;
    }

    T[mapPosCorrection(a.x, a.y
                       , params.imgw
                       , params.imgw)].agent_number=1;

    return h_new;

}





__global__ void SensorCatSystemKernelFace(lbm_node* T, Array<Caterpiller> catList, int numberCat, Array<Particle> pointList, int numberParticles,  CudaNumberGenerator numberGen,
                                            int* particlesWorldMap,
                                                        int tick )
{
    uint index = blockIdx.x*blockDim.x + threadIdx.x;


   if(index<numberCat)
   {
       Caterpiller* cat=catList.atPointer(index);
       //int numberPoints=cat->numParticlesPointers;
       //Head

       float fDirection=cat->direction_new;
       cat->direction=fDirection;

       //Particle* particleH=cat->particlesPointers[0];
       Particle* p=cat->particlesPointers[0];


       //int tIndex = getIndex(particleH->face, particleH->x, particleH->y);

       int dx=(int)roundf(params.SS * cosf(p->h));
       int dy=(int)roundf(params.SS * sinf(p->h));


       int dxL=(int)roundf(params.SS * cosf(p->h+params.FL));
       int dyL=(int)roundf(params.SS * sinf(p->h+params.FL));


       int dxR=(int)roundf(params.SS * cosf(p->h+params.FR));
       int dyR=(int)roundf(params.SS * sinf(p->h+params.FR));


        lbm_node tempNodeF=getIndexWRPT(p->face, p->x, dx, p->y, dy);
        float F =T[tempNodeF.index].value;

        lbm_node tempNodeFL=getIndexWRPT(p->face, p->x,dxL, p->y, dyL);
        float FL = T[tempNodeFL.index].value;

        lbm_node tempNodeFR=getIndexWRPT(p->face, p->x, dxR, p->y, dyR);
        float FR =T[tempNodeFR.index].value;

        if (F > FL and F > FR){
          p->h_new = p->h;//+tempNodeF.deltaAngle;
        }
        else if (F < FL and F < FR){
          if ( numberGen.random(1000) >500 ){
            p->h_new =p->h + params.RA;//+tempNodeF.deltaAngle;
          }
          else{
            p->h_new =p->h - params.RA;//+tempNodeF.deltaAngle;
          }

        }
        else if (FL < FR){
          p->h_new= p->h - params.RA;//+tempNodeF.deltaAngle;
        }
        else if (FR < FL){
          p->h_new = p->h + params.RA;//+tempNodeF.deltaAngle;
        }
        else
        {
          p->h_new= p->h;//+tempNodeF.deltaAngle;
        }

        T[tempNodeF.index].agent_number=1;





    }
}
__global__ void MotorCatSystemKernelFace(lbm_node* T, Array<Caterpiller> catList, int numberCat, Array<Particle> pointList, int numberParticles,  CudaNumberGenerator numberGen,
                                    int* particlesWorldMap,
                                    int tick )
{
    uint index = blockIdx.x*blockDim.x + threadIdx.x;

   //if(index==0)
   //     printf(" number partiles %d", numberParticles);
   if(index<numberCat)
   {
       Caterpiller* cat=catList.atPointer(index);
       float fDirection=cat->direction_new;
       cat->direction=fDirection;
       Particle* particleH=cat->particlesPointers[0];

       //int tIndex = getIndex(particleH->face, particleH->x, particleH->y);

       particleH->h=particleH->h_new;

       int dx=(int)roundf(params.SS * cosf(particleH->h));
       int dy=(int)roundf(params.SS * sinf(particleH->h));

       lbm_node tempNode=getIndexWRPT(particleH->face, particleH->x, dx, particleH->y, dy);

       if( T[tempNode.index].agent_number==0)
       {
           particleH->x_new =  tempNode.x;
           particleH->y_new =  tempNode.y;

           particleH->dx = tempNode.dx;
           particleH->dy = tempNode.dy;
           particleH->face=tempNode.face;
           particleH->direction=particleH->direction+tempNode.deltaAngle;
           particleH->h=particleH->h+tempNode.deltaAngle;
       }
       else
       {
           particleH->h=numberGen.random(2.0f*3.1415f);

       }



      // T[tempNode.index].h=5;//cat->direction;


    }
}

__global__ void DepositionCatSystemKernelFace(lbm_node* T, Array<Caterpiller> catList, int numberCat, Array<Particle> pointList, int numberParticles,  CudaNumberGenerator numberGen,
                                    int* particlesWorldMap,
                                    int tick )
{
    uint index = blockIdx.x*blockDim.x + threadIdx.x;

   //if(index==0)
   //     printf(" number partiles %d", numberParticles);
   if(index<numberCat)
   {
       Caterpiller* cat=catList.atPointer(index);
       //int numberPoints=cat->numParticlesPointers;
       //Head

       float fDirection=cat->direction_new;
       cat->direction=fDirection;

       Particle* particleH=cat->particlesPointers[0];

       particleH->x=particleH->x_new;
       particleH->y=particleH->y_new;

       int tIndex = getIndex(particleH->face, particleH->x, particleH->y);

       T[tIndex].value=T[tIndex].value+params.depT;


    }
}


__global__ void EvaporateCatSystemKernelFace(int face,  lbm_node* T,
                                             unsigned int *g_odata, int imgw, int imgh,
                                             int tick,
                                             int* colorMap, int numberColors, int* d_colorWorldMap,
                                             int* particlesWorldMap)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imgw && y < imgh)
    {

        int i = getIndex(face, x, y);

        float pcolor=T[i].value;
        T[i].value= T[i].value*(1.0f-params.decayT);
    }
}


__global__ void MotorCatSystemKernelFace2(lbm_node* T, Array<Caterpiller> catList, int numberCat, Array<Particle> pointList, int numberParticles,  CudaNumberGenerator numberGen,
                                    int* particlesWorldMap,
                                    int tick )
{
    uint index = blockIdx.x*blockDim.x + threadIdx.x;

   //if(index==0)
   //     printf(" number partiles %d", numberParticles);
   if(index<numberCat)
   {
       Caterpiller* cat=catList.atPointer(index);
       int numberPoints=cat->numParticlesPointers;
       //Head

       float fDirection=cat->direction_new;
       cat->direction=fDirection;

       Particle* particleH=cat->particlesPointers[0];

       lbm_node tempNode=getIndexWRPT(particleH->face, particleH->x, particleH->dx, particleH->y, particleH->dy);

       particleH->x =  tempNode.x;
       particleH->y =  tempNode.y;

       particleH->dx = tempNode.dx;
       particleH->dy = tempNode.dy;
       particleH->face=tempNode.face;

       T[tempNode.index].h=25.0;


    }
}





__global__ void SensorCatSystemKernel(PointCA *T, Array<Caterpiller> catList, int numberCat, Array<Particle> pointList, int numberParticles,  CudaNumberGenerator numberGen,
                                    int* particlesWorldMap,
                                    int tick )
{
    uint index = blockIdx.x*blockDim.x + threadIdx.x;

   if(index<numberCat)
   {
       Caterpiller* cat=catList.atPointer(index);
       int numberPoints=cat->numParticlesPointers;

       Particle* particleH=cat->particlesPointers[0];
       auto oldPos=particleH->relPos;

       float velocity=params.wormVelocity01;
       cat->direction_new=FindDirectionMap(T, cat,pointList, numberParticles, numberGen,
                                         velocity,
                                         particlesWorldMap,
                                         params.numSteps01,
                                         params.numDirections01,
                                         params.angleDivide01,false);

    }
}


__global__ void DepositionCatSystemKernel(PointCA *T, Array<Caterpiller> catList, int numberCat, Array<Particle> pointList, int numberParticles,  CudaNumberGenerator numberGen,
                                    int* particlesWorldMap,
                                    int tick )
{
     uint index = blockIdx.x*blockDim.x + threadIdx.x;

   //if(index==0)
   //     printf(" number partiles %d", numberParticles);
   if(index<numberCat)
   {
       Caterpiller* cat=catList.atPointer(index);
       int numberPoints=cat->numParticlesPointers;

       Particle* particleH=cat->particlesPointers[0];
       auto oldPos=particleH->relPos;

       particleH->relPos=particleH->relPos_new;
       T[mapPosCorrection(roundf(particleH->relPos.x), roundf(particleH->relPos.y))].value=
                     T[mapPosCorrection(roundf(particleH->relPos.x), roundf(particleH->relPos.y))].value+params.depT;
    }
}

__global__
void MoveRenderKernel(PointCA *T,
                      unsigned int *g_odata,
                      int imgw,
                      int imgh,
                      Array<Caterpiller> catList,
                      int numberCat,
                      Array<Particle> pointList,
                      int numberParticles,
                      CudaNumberGenerator numberGen,
                      int* colorMap, int numberColors,
                      int* d_colorWorldMap,
                      int* particlesWorldMap)
{
   uint index = blockIdx.x*blockDim.x + threadIdx.x;

   if(index<numberCat)
   {
        Caterpiller* cat=catList.atPointer(index);
        int numberPoints=cat->numParticlesPointers;
        //Head
        Particle* particleH=cat->particlesPointers[0];

        auto x= (int)(particleH->relPos.x);
        auto y= (int)(particleH->relPos.y);

        const int pixelIndex=y*imgw + x;

        if(x >= imgw or y >= imgw or x<0 or y<0)
        {
        //   return;
        }
        T[pixelIndex].agent_number=0;

        float pcolor=T[pixelIndex].value;
        T[pixelIndex].value= T[pixelIndex].value*(1.0f-params.decayT);

        if(T[pixelIndex].value*25.0f>=250)
           T[pixelIndex].value=250;

        g_odata[pixelIndex] = colorMap[(int)(T[pixelIndex].value*7.0f)];


    }
}


__global__
void cudaRenderClear(PointCA *T,
                     unsigned int *g_odata, int imgw, int imgh,
                     int tick,
                     int* colorMap, int numberColors, int* d_colorWorldMap,
                     int* particlesWorldMap)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imgw && y < imgh)
    {
        // clear particle world
        const int pixelIndex=y*imgw + x;
        particlesWorldMap[pixelIndex]=0;

        T[pixelIndex].agent_number=0;

        float pcolor=T[pixelIndex].value;
        T[pixelIndex].value= T[pixelIndex].value*(1.0f-params.decayT);

        if(T[pixelIndex].value*7.0f>=250)
           T[pixelIndex].value=250/7.0f;

        g_odata[pixelIndex] = colorMap[(int)(T[pixelIndex].value*7.0f)];
    }
}



__global__
void cudaRenderClearFace(int face,  lbm_node* T,
                     unsigned int *g_odata, int imgw, int imgh,
                     int tick,
                     int* colorMap, int numberColors, int* d_colorWorldMap,
                     int* particlesWorldMap)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imgw && y < imgh)
    {
        int i = getIndex(face, x, y);

        const int pixelIndex=y*imgw + x;
        particlesWorldMap[pixelIndex]=0;

        T[i].agent_number=0;

        if(T[i].value*7.0f>=250)
           T[i].value=250/7.0f;

        g_odata[pixelIndex] = colorMap[(int)(T[i].value*7.0f)];

        if(face==0)
        {
          //  g_odata[pixelIndex] += colorMap[12];
        }
    }
}



#endif
