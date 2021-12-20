#ifndef SUPPORTBGFX_HPP
#define SUPPORTBGFX_HPP

#endif // SUPPORTBGFX_HPP


#pragma once
#include "common.h"
#include "bgfx_utils.h"
#include "imgui/imgui.h"
#include "common/imgui/imgui.h"

#include <stdio.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <bx/uint32_t.h>
#include "vector"

#include <cuda_runtime.h>
#include <helper_cuda.h>




inline void mtxFromNormalScale(float* _result, const  bx::Vec3& _normal, bx::Vec3&  _scale, const bx::Vec3& _pos)
{
    bx::Vec3 tangent;
    bx::Vec3 bitangent;
    bx::calcTangentFrame(tangent, bitangent, _normal);

    store(&_result[ 0], mul(bitangent, _scale.x) );
    store(&_result[ 4], mul(_normal,   _scale.y) );
    store(&_result[ 8], mul(tangent,   _scale.z) );

    _result[ 3] = 0.0f;
    _result[ 7] = 0.0f;
    _result[11] = 0.0f;
    _result[12] = _pos.x;
    _result[13] = _pos.y;
    _result[14] = _pos.z;
    _result[15] = 1.0f;
}

struct PosColorVertex
{
    float m_pos[3];
    float m_normal[3];
    uint32_t m_abgr;

    static void init()
    {

        ms_layout
            .begin()
            .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Normal,   3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Color0,   4, bgfx::AttribType::Uint8, true)
            .end();

    };

    inline static bgfx::VertexLayout ms_layout;
};

struct PosColorLayout
{
    float m_pos[3];
    float m_normal[3];
    float m_u;
    float m_v;
    float m_w;

    static void init()
    {

        ms_layout
            .begin()
            .add(bgfx::Attrib::Position,  3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Normal,    3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 3, bgfx::AttribType::Float)
            .end();

    };

    inline static bgfx::VertexLayout ms_layout;
};


struct InstanceLayout
{
    float data0[4];
    float data1[4];
    float data2[4];
    float data3[4];
    float color[4];
    static void init()
    {
        ms_layout
            .begin()
            .add(bgfx::Attrib::TexCoord7, 4, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord6, 4, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord5, 4, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord4, 4, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord3, 4, bgfx::AttribType::Float)
            .end();
    };

    inline static bgfx::VertexLayout ms_layout;
};

static  uint32_t tempColor1=(uint8_t(255) << 24) +
                            (uint8_t(255) << 16) +
                            (uint8_t(255) << 8)  +
                             uint8_t(255);

static  uint32_t tempColor2=(uint8_t(255) << 24) +
                            (uint8_t(155) << 16) +
                            (uint8_t(155) << 8)  +
                             uint8_t(255);

static  uint32_t tempColor3=(uint8_t(255) << 24) +
                            (uint8_t(155) << 16) +
                            (uint8_t(255) << 8)  +
                             uint8_t(255);


static InstanceLayout instanceInit[10];

static PosColorVertex s_cubeVertices[8] =
{
    {-1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,tempColor1 },
    { 1.0f,  1.0f,  1.0f,1.0f,  1.0f,  1.0f, tempColor1 },
    {-1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,tempColor1 },
    { 1.0f, -1.0f,  1.0f, 1.0f, -1.0f,  1.0f,tempColor1 },
    {-1.0f,  1.0f, -1.0f,-1.0f,  1.0f, -1.0f, tempColor1 },
    { 1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,tempColor1 },
    {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,tempColor1 },
    { 1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,tempColor1 },
};

static const uint16_t s_cubeIndices[36] =
{
    0, 1, 2, // 0
    1, 3, 2,
    4, 6, 5, // 2
    5, 6, 7,
    0, 2, 4, // 4
    4, 2, 6,
    1, 5, 3, // 6
    5, 7, 3,
    0, 4, 1, // 8
    4, 5, 1,
    2, 3, 6, // 10
    6, 3, 7,
};



static uint32_t ColorConvert2(int r, int g, int b)
{
  uint32_t c;
  c = r;
  c <<= 8;
  c |= g;
  c <<= 8;
  c |= b;
  return c;
}

static uint32_t ColorConvert(int r, int g, int b)
{
      return (uint8_t(255) << 24) +
             (uint8_t(b) << 16) +
             (uint8_t(g) << 8)  +
              uint8_t(r);

}




template<typename Ty>
inline bool intersect(DebugDrawEncoder* _dde, const Ray& _ray, const Ty& _shape)
{
   Hit hit;
   if (::intersect(_ray, _shape, &hit))
   {
       _dde->push();

       _dde->setWireframe(false);

       _dde->setColor(0xff0000ff);

       _dde->drawCone(hit.pos, bx::mad(hit.plane.normal, 0.7f, hit.pos), 0.1f);

       _dde->pop();

       return true;
   }

   return false;
}


inline bx::Vec3 arbitrary_orthogonal(bx::Vec3 vec)
{
  bool b0 = (vec.x <  vec.y) && (vec.x <  vec.z);
  bool b1 = (vec.y <= vec.x) && (vec.y <  vec.z);
  bool b2 = (vec.z <= vec.x) && (vec.z <= vec.y);

  return bx::cross(vec, bx::Vec3(b0, b1, b2));
}

inline bx::Vec3 arbitrary_orthogonal2(bx::Vec3 N)
{
    double Ax= abs(N.x), Ay= abs(N.y), Az= abs(N.z);
    if (Ax < Ay)
        return  Ax < Az ? bx::Vec3(0, -N.z, N.y) : bx::Vec3(-N.y, N.x, 0);
    else
        return  Ay < Az ? bx::Vec3(N.z, 0, -N.x) : bx::Vec3(-N.y, N.x, 0);
}
inline void circle(bx::Vec3& _out, float _angle)
{
    float sa = bx::sin(_angle);
    float ca = bx::cos(_angle);
    _out.x = sa;
    _out.y= ca;
}

inline void squircle(bx::Vec3& _out,  float _angle)
{
    float sa = bx::sin(_angle);
    float ca = bx::cos(_angle);
    _out.x = bx::sqrt(bx::abs(sa) ) * bx::sign(sa);
    _out.y = bx::sqrt(bx::abs(ca) ) * bx::sign(ca);
}

static void genSphere(uint8_t _subdiv0, PosColorVertex* pmv)
{

        struct Gen
        {
            Gen(PosColorVertex* _pmv,  uint8_t _subdiv):
              pmv2(_pmv)
            {
                static const float scale = 1.0f;
                static const float golden = 1.6180339887f;
                static const float len = bx::sqrt(golden*golden + 1.0f);
                static const float ss = 1.0f/len * scale;
                static const float ll = ss*golden;

                static const bx::Vec3 vv[] =
                {
                    { -ll, 0.0f, -ss },
                    {  ll, 0.0f, -ss },
                    {  ll, 0.0f,  ss },
                    { -ll, 0.0f,  ss },

                    { -ss,  ll, 0.0f },
                    {  ss,  ll, 0.0f },
                    {  ss, -ll, 0.0f },
                    { -ss, -ll, 0.0f },

                    { 0.0f, -ss,  ll },
                    { 0.0f,  ss,  ll },
                    { 0.0f,  ss, -ll },
                    { 0.0f, -ss, -ll },
                };

                m_numVertices = 0;

                triangle(vv[ 0], vv[ 4], vv[ 3], scale, _subdiv);
                triangle(vv[ 0], vv[10], vv[ 4], scale, _subdiv);
                triangle(vv[ 4], vv[10], vv[ 5], scale, _subdiv);
                triangle(vv[ 5], vv[10], vv[ 1], scale, _subdiv);
                triangle(vv[ 5], vv[ 1], vv[ 2], scale, _subdiv);
                triangle(vv[ 5], vv[ 2], vv[ 9], scale, _subdiv);
                triangle(vv[ 5], vv[ 9], vv[ 4], scale, _subdiv);
                triangle(vv[ 3], vv[ 4], vv[ 9], scale, _subdiv);

                triangle(vv[ 0], vv[ 3], vv[ 7], scale, _subdiv);
                triangle(vv[ 0], vv[ 7], vv[11], scale, _subdiv);
                triangle(vv[11], vv[ 7], vv[ 6], scale, _subdiv);
                triangle(vv[11], vv[ 6], vv[ 1], scale, _subdiv);
                triangle(vv[ 1], vv[ 6], vv[ 2], scale, _subdiv);
                triangle(vv[ 2], vv[ 6], vv[ 8], scale, _subdiv);
                triangle(vv[ 8], vv[ 6], vv[ 7], scale, _subdiv);
                triangle(vv[ 8], vv[ 7], vv[ 3], scale, _subdiv);

                triangle(vv[ 0], vv[11], vv[10], scale, _subdiv);
                triangle(vv[ 1], vv[10], vv[11], scale, _subdiv);
                triangle(vv[ 2], vv[ 8], vv[ 9], scale, _subdiv);
                triangle(vv[ 3], vv[ 9], vv[ 8], scale, _subdiv);
            }

            void addVert(const bx::Vec3& _v)
            {
                //bx::store(m_pos, _v);
                pmv2[m_posStride].m_pos[0]=_v.x;
                pmv2[m_posStride].m_pos[1]=_v.y;
                pmv2[m_posStride].m_pos[2]=_v.z;

                pmv2[m_posStride].m_abgr =tempColor1 ;

                m_posStride += 1;


                const bx::Vec3 normal = bx::normalize(_v);
               // bx::store(m_normals, normal);

                pmv2[m_normalStride].m_normal[0]=normal.x;
                pmv2[m_normalStride].m_normal[1]=normal.y;
                pmv2[m_normalStride].m_normal[2]=normal.z;

                m_normalStride += 1;



                m_numVertices++;
            }

            void triangle(const bx::Vec3& _v0, const bx::Vec3& _v1, const bx::Vec3& _v2, float _scale, uint8_t _subdiv)
            {
                if (0 == _subdiv)
                {
                    addVert(_v0);
                    addVert(_v1);
                    addVert(_v2);
                }
                else
                {
                    const bx::Vec3 v01 = bx::mul(bx::normalize(bx::add(_v0, _v1) ), _scale);
                    const bx::Vec3 v12 = bx::mul(bx::normalize(bx::add(_v1, _v2) ), _scale);
                    const bx::Vec3 v20 = bx::mul(bx::normalize(bx::add(_v2, _v0) ), _scale);

                    --_subdiv;
                    triangle(_v0, v01, v20, _scale, _subdiv);
                    triangle(_v1, v12, v01, _scale, _subdiv);
                    triangle(_v2, v20, v12, _scale, _subdiv);
                    triangle(v01, v12, v20, _scale, _subdiv);
                }
            }

           // float* m_pos;
           // float* m_normals;
            PosColorVertex* pmv2;
            uint16_t m_posStride=0;
            uint16_t m_normalStride=0;
            uint32_t m_numVertices;

        }
        gen(pmv,  _subdiv0);

}



static void createProceduralSphereLod(float radius1, int num,
                                    bgfx::VertexBufferHandle& m_vbh,
                                    bgfx::IndexBufferHandle&  m_ibh)
    {



    int mesh=2;
    const uint8_t  tess = uint8_t(3-mesh);
    const uint32_t numVertices = 20*3*bx::uint32_max(1, (uint32_t)bx::pow(4.0f,tess) );
    const uint32_t numIndices  = numVertices;

    auto pmv=new PosColorVertex[numVertices];
    genSphere(tess, pmv);



    uint16_t* trilist=new uint16_t[numIndices];
    int indexVertex=0;

    for (uint32_t ii = 0; ii < numIndices; ++ii)
    {
        trilist[ii] = uint16_t(ii);
    }

    uint32_t numLineListIndices = bgfx::topologyConvert(
          bgfx::TopologyConvert::TriListToLineList
        , NULL
        , 0
        , trilist
        , numIndices
        , false
        );

    uint16_t* pmi=new uint16_t[numIndices+ numLineListIndices];

    bx::memCopy(pmi , trilist, numIndices*sizeof(uint16_t) );

    bgfx::topologyConvert(
          bgfx::TopologyConvert::TriListToLineList
        , &pmi[numIndices]
        , numLineListIndices*sizeof(uint16_t)
        , trilist
        , numIndices
        , false
        );


    m_vbh=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*numVertices),
                                                  PosColorVertex::ms_layout));
    m_ibh=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices+ numLineListIndices))));

    free(trilist);

}




static void createProceduralCylinderLod(float radius1, float radius2, int num,
                                        bgfx::VertexBufferHandle& m_vbh,
                                        bgfx::IndexBufferHandle&  m_ibh)
 {

     const float step = bx::kPi * 2.0f / num;

   //  const float step = bx::kPi * 2.0f /6;

     const uint numVertices = num*2;
     const uint numIndices  = num*12;
     const uint numLineListIndices = num*6;

     auto pmv=new PosColorVertex[numVertices];
     auto pmi=new uint16_t[numIndices  + numLineListIndices];
     int indexVertex=0;

     for (uint ii = 0; ii < num; ++ii)
     {
         const float angle = step * ii;

         bx::Vec3 tempPoint;
         circle(tempPoint, angle);

         pmv[ii].m_pos[0] =(float) tempPoint.y*radius1;
         pmv[ii].m_pos[1] =(float) 0.0f;
         pmv[ii].m_pos[2] =(float) tempPoint.x*radius1;
         pmv[ii].m_abgr =(uint32_t)tempColor1 ;

         bx::Vec3 tempPointN1(0.5,1.5,0.0);

         bx::Vec3 tempPoint1(pmv[ii].m_pos[0],pmv[ii].m_pos[1],pmv[ii].m_pos[2]);
         const bx::Vec3 normal = bx::normalize(bx::sub(tempPointN1,tempPoint1));

        // bx::Vec3 tempPoint1(pmv[ii].m_pos[1],0,pmv[ii].m_pos[0]);
        // const bx::Vec3 normal = bx::normalize(tempPoint1);

         pmv[ii].m_normal[0] =normal.x;
         pmv[ii].m_normal[1] =normal.y;
         pmv[ii].m_normal[2] =normal.z;

         pmv[ii+num].m_pos[0] =(float) tempPoint.y*radius2;
         pmv[ii+num].m_pos[1] =(float) 1.0f;
         pmv[ii+num].m_pos[2] =(float) tempPoint.x*radius2;
         pmv[ii+num].m_abgr=(uint32_t)tempColor1 ;

         bx::Vec3 tempPointN2(0.5,1.5,0.0);
         bx::Vec3 tempPoint2(pmv[ii+num].m_pos[0],pmv[ii+num].m_pos[1],pmv[ii+num].m_pos[2]);
         const bx::Vec3 normal2 = bx::normalize(bx::sub(tempPointN2,tempPoint2));

        // bx::Vec3 tempPoint2(pmv[ii+num].m_pos[1],0,pmv[ii+num].m_pos[0]);
        // const bx::Vec3 normal2 = bx::normalize(tempPoint2);

         pmv[ii+num].m_normal[0] =normal2.x;
         pmv[ii+num].m_normal[1] =normal2.y;
         pmv[ii+num].m_normal[2] =normal2.z;


          pmi[ii*6+2] = uint16_t(ii+num);
          pmi[ii*6+1] = uint16_t( (ii+1)%num);
          pmi[ii*6+0] = uint16_t(ii);
          pmi[ii*6+5] = uint16_t(ii+num);
          pmi[ii*6+4] = uint16_t( (ii+1)%num+num);
          pmi[ii*6+3] = uint16_t( (ii+1)%num);

          pmi[num*6+ii*6+2] = uint16_t(0);
          pmi[num*6+ii*6+1] = uint16_t(ii);
          pmi[num*6+ii*6+0] = uint16_t( (ii+1)%num);
          pmi[num*6+ii*6+5] = uint16_t(num);
          pmi[num*6+ii*6+4] = uint16_t( (ii+1)%num+num);
          pmi[num*6+ii*6+3] = uint16_t(ii+num);

          pmi[numIndices+ii*2+0] = uint16_t(ii);
          pmi[numIndices+ii*2+1] = uint16_t(ii+num);

          pmi[numIndices+num*2+ii*2+0] = uint16_t(ii);
          pmi[numIndices+num*2+ii*2+1] = uint16_t( (ii+1)%num);

          pmi[numIndices+num*4+ii*2+0] = uint16_t(num + ii);
          pmi[numIndices+num*4+ii*2+1] = uint16_t(num + (ii+1)%num);

          indexVertex++;
    }


     m_vbh=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*numVertices),
                                                   PosColorVertex::ms_layout));

     m_ibh=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices  + numLineListIndices))));

 }



static void createProceduralCylinder(float radius1, float radius2, int num,
                                     bgfx::VertexBufferHandle& m_vbh,
                                     bgfx::IndexBufferHandle&  m_ibh)
 {

     unsigned int mNumSegBase=16;
     unsigned int mNumSegHeight=1;
     float mRadius=0.5f;
     float mHeight=1.0f;

     bool mCapped=false;

     unsigned int estimateVertexCount;
     unsigned int estimateIndexCount;

     if (mCapped)
     {
         estimateVertexCount=((mNumSegHeight+1)*(mNumSegBase+1)+2*(mNumSegBase+1)+2);
         estimateIndexCount=(mNumSegHeight*(mNumSegBase+1)*6+6*mNumSegBase);
     }
     else
     {
        estimateVertexCount=((mNumSegHeight+1)*(mNumSegBase+1));
        estimateIndexCount=(mNumSegHeight*(mNumSegBase+1)*6);
     }

     auto pmv=new PosColorVertex[estimateVertexCount];
     auto pmi=new uint16_t[estimateIndexCount];

     float deltaAngle = (2.0f*M_PI/ mNumSegBase);
     float  deltaHeight = mHeight/(float)mNumSegHeight;
     int offset = 0;


     int indexVertex=0;

     for (unsigned int i = 0; i <=mNumSegHeight; i++)
     {
         for (unsigned int j = 0; j<=mNumSegBase; j++)
         {
             float x0 = mRadius * cosf(j*deltaAngle);
             float z0 = mRadius * sinf(j*deltaAngle);


             pmv[indexVertex].m_pos[0] =x0;
             pmv[indexVertex].m_pos[1] =i*deltaHeight;
             pmv[indexVertex].m_pos[2] =z0;

             bx::Vec3 tempPoint1(x0,0,z0);
             const bx::Vec3 normal = bx::normalize(tempPoint1);

             pmv[indexVertex].m_normal[0] =normal.x;
             pmv[indexVertex].m_normal[1] =normal.y;
             pmv[indexVertex].m_normal[2] =normal.z;


             if (i != mNumSegHeight)
             {
                 pmi[indexVertex*6+2] = uint16_t(offset + mNumSegBase + 1);
                 pmi[indexVertex*6+1] = uint16_t(offset);
                 pmi[indexVertex*6+0] = uint16_t(offset + mNumSegBase);
                 pmi[indexVertex*6+5] = uint16_t(offset + mNumSegBase + 1);
                 pmi[indexVertex*6+4] = uint16_t(offset + 1);
                 pmi[indexVertex*6+3] = uint16_t(offset);
             }
             offset ++;
             indexVertex++;
         }
    }





     m_vbh=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*estimateVertexCount),
                                                   PosColorVertex::ms_layout));

     m_ibh=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(estimateIndexCount))));

 }


static void createProceduralCylinder (bgfx::VertexBufferHandle& m_vbh_cyl_t,
                               bgfx::IndexBufferHandle&  m_ibh_cyl_t,
                               float radius1, float radius2, int num)
{

        unsigned int mNumSegBase=num;
        unsigned int mNumSegHeight=1;
        float mRadius=radius1;
        float mHeight=1.0f;

        bool mCapped=true;

        unsigned int estimateVertexCount;
        unsigned int estimateIndexCount;

        if (mCapped)
        {
            estimateVertexCount=(mNumSegHeight+1)*(mNumSegBase+1)+2;
            estimateIndexCount=mNumSegHeight*(mNumSegBase+1)*6+6*mNumSegBase;
        }
        else
        {
           estimateVertexCount=((mNumSegHeight+1)*(mNumSegBase+1));
           estimateIndexCount=(mNumSegHeight*(mNumSegBase+1)*6);
        }

        auto pmv=new PosColorVertex[estimateVertexCount];
        auto pmi=new uint16_t[estimateIndexCount];

        float deltaAngle = (2.0f*M_PI/ mNumSegBase);
        float  deltaHeight = mHeight/(float)mNumSegHeight;
        int offset = 0;

        int indexI=0;
        int indexVertex=0;

        for (unsigned int i = 0; i <=mNumSegHeight; i++)
        {
            for (unsigned int j = 0; j<=mNumSegBase; j++)
            {
                float x0 = mRadius * cosf(j*deltaAngle);
                float z0 = mRadius * sinf(j*deltaAngle);


                pmv[indexVertex].m_pos[0] =x0;
                pmv[indexVertex].m_pos[1] =i*deltaHeight;
                pmv[indexVertex].m_pos[2] =z0;
                pmv[indexVertex].m_abgr =(uint32_t)tempColor1 ;

                bx::Vec3 tempPoint1(x0, 1,z0);
                const bx::Vec3 normal = bx::normalize(tempPoint1);

                pmv[indexVertex].m_normal[0] =normal.x;
                pmv[indexVertex].m_normal[1] =normal.y;
                pmv[indexVertex].m_normal[2] =normal.z;


                if (i != mNumSegHeight)
                {
                    pmi[indexVertex*6+2] = uint16_t(offset + mNumSegBase + 1);
                     indexI++;
                    pmi[indexVertex*6+1] = uint16_t(offset);
                     indexI++;
                    pmi[indexVertex*6+0] = uint16_t(offset + mNumSegBase);
                     indexI++;
                    pmi[indexVertex*6+5] = uint16_t(offset + mNumSegBase + 1);
                     indexI++;
                    pmi[indexVertex*6+4] = uint16_t(offset + 1);
                     indexI++;
                    pmi[indexVertex*6+3] = uint16_t(offset);
                    indexI++;

                }
                offset ++;
                indexVertex++;
            }
        }

        if (mCapped)
        {
            //low cap
            {
                int centerIndex = indexVertex;

                pmv[indexVertex].m_pos[0] =0;
                pmv[indexVertex].m_pos[1] =0;
                pmv[indexVertex].m_pos[2] =0;
                pmv[indexVertex].m_abgr =(uint32_t)tempColor1 ;

                bx::Vec3 tempPoint1(0, 1,0);
                const bx::Vec3 normal = bx::normalize(tempPoint1);

                pmv[indexVertex].m_normal[0] =normal.x;
                pmv[indexVertex].m_normal[1] =normal.y;
                pmv[indexVertex].m_normal[2] =normal.z;

                for (unsigned int j=0; j<mNumSegBase; j++)
                {
                    pmi[indexI] = uint16_t(j);
                     indexI++;
                    pmi[indexI] = uint16_t(indexVertex);
                     indexI++;
                    if(j+1==mNumSegBase)
                        pmi[indexI] = uint16_t(0);
                    else
                        pmi[indexI] = uint16_t(j+1);
                     indexI++;
                }

                offset++;
                indexVertex++;
            }

            //high cap
            {
                int centerIndex = indexVertex;

                pmv[indexVertex].m_pos[0] =0;
                pmv[indexVertex].m_pos[1] =mNumSegHeight*deltaHeight;
                pmv[indexVertex].m_pos[2] =0;
                pmv[indexVertex].m_abgr =(uint32_t)tempColor1 ;

                bx::Vec3 tempPoint1(0,-1,0);
                const bx::Vec3 normal = bx::normalize(tempPoint1);

                pmv[indexVertex].m_normal[0] =normal.x;
                pmv[indexVertex].m_normal[1] =normal.y;
                pmv[indexVertex].m_normal[2] =normal.z;

                for (unsigned int j=0; j<mNumSegBase; j++)
                {
                    if(j+1==mNumSegBase)
                        pmi[indexI] = uint16_t(indexVertex-mNumSegBase-2);
                    else
                        pmi[indexI] = uint16_t(indexVertex-mNumSegBase-2+j+1);
                     indexI++;
                    pmi[indexI] = uint16_t(indexVertex);
                     indexI++;

                     pmi[indexI] = uint16_t(indexVertex-mNumSegBase-2+j);
                      indexI++;

                }
            }

          }

        m_vbh_cyl_t=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*estimateVertexCount),
                                                      PosColorVertex::ms_layout));
        m_ibh_cyl_t=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(estimateIndexCount))));

}

static void createProceduralCircle(bgfx::VertexBufferHandle& m_vbh_cyl_t,
                               bgfx::IndexBufferHandle&  m_ibh_cyl_t,
                               float radius1, float radius2, int num)
{

        unsigned int mNumSegBase=num;
        unsigned int mNumSegHeight=1;
        float mRadius=radius1;
        float mHeight=1.0f;

        unsigned int estimateVertexCount;
        unsigned int estimateIndexCount;

        estimateVertexCount=(mNumSegBase)+1;
        estimateIndexCount= (mNumSegBase)*3;

        auto pmv=new PosColorVertex[estimateVertexCount];
        auto pmi=new uint16_t[estimateIndexCount];

        float deltaAngle = (2.0f*M_PI/ mNumSegBase);
        float  deltaHeight = mHeight/(float)mNumSegHeight;
        int offset = 0;

        int indexI=0;
        int indexVertex=0;


            for (unsigned int j = 0; j<mNumSegBase; j++)
            {
                float x0 = mRadius * cosf(j*deltaAngle);
                float z0 = mRadius * sinf(j*deltaAngle);

                pmv[indexVertex].m_pos[0] =x0;
                pmv[indexVertex].m_pos[1] =0;
                pmv[indexVertex].m_pos[2] =z0;
                pmv[indexVertex].m_abgr =(uint32_t)tempColor1 ;

                bx::Vec3 tempPoint1(x0, 1,z0);
                const bx::Vec3 normal = bx::normalize(tempPoint1);

                pmv[indexVertex].m_normal[0] =normal.x;
                pmv[indexVertex].m_normal[1] =normal.y;
                pmv[indexVertex].m_normal[2] =normal.z;

                indexVertex++;
            }

            //low cap
            {
                int centerIndex = indexVertex;

                pmv[indexVertex].m_pos[0] =0;
                pmv[indexVertex].m_pos[1] =0;
                pmv[indexVertex].m_pos[2] =0;
                pmv[indexVertex].m_abgr =(uint32_t)tempColor1 ;

                bx::Vec3 tempPoint1(0, 1,0);
                const bx::Vec3 normal = bx::normalize(tempPoint1);

                pmv[indexVertex].m_normal[0] =normal.x;
                pmv[indexVertex].m_normal[1] =normal.y;
                pmv[indexVertex].m_normal[2] =normal.z;

                for (unsigned int j=0; j<mNumSegBase; j++)
                {
                    pmi[indexI] = uint16_t(indexVertex);
                     indexI++;
                    pmi[indexI] = uint16_t(j);
                     indexI++;
                    if(j+1==mNumSegBase)
                        pmi[indexI] = uint16_t(0);
                    else
                        pmi[indexI] = uint16_t(j+1);
                     indexI++;
                }

                offset++;
                indexVertex++;
            }


        m_vbh_cyl_t=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*estimateVertexCount),
                                                       PosColorVertex::ms_layout));
        m_ibh_cyl_t=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(estimateIndexCount))));

      //  free(pmv);
      //  free(pmi);

}


static void createProceduralCircleWH(bgfx::VertexBufferHandle& m_vbh_cyl_t,
                                     bgfx::IndexBufferHandle&  m_ibh_cyl_t,
                                     float radius1, int num)
{

        unsigned int mNumSegBase=num;
        unsigned int mNumSegHeight=1;
        float mRadius=radius1;
        float mHeight=1.0f;

        unsigned int estimateVertexCount;
        unsigned int estimateIndexCount;

        estimateVertexCount=2*(mNumSegBase);
        estimateIndexCount= (mNumSegBase)*6;

        auto pmv=new PosColorVertex[estimateVertexCount];
        auto pmi=new uint16_t[estimateIndexCount];

        float deltaAngle = (2.0f*M_PI/ mNumSegBase);
        float  deltaHeight = mHeight/(float)mNumSegHeight;
        int offset = 0;


        int indexVertex=0;

            //outer circle
            for (unsigned int j = 0; j<mNumSegBase; j++)
            {
                float x0 = mRadius * cosf(j*deltaAngle);
                float z0 = mRadius * sinf(j*deltaAngle);

                pmv[indexVertex].m_pos[0] =x0;
                pmv[indexVertex].m_pos[1] =0;
                pmv[indexVertex].m_pos[2] =z0;
                pmv[indexVertex].m_abgr =(uint32_t)tempColor1 ;

                bx::Vec3 tempPoint1(x0, 1,z0);
                const bx::Vec3 normal = bx::normalize(tempPoint1);

                pmv[indexVertex].m_normal[0] =normal.x;
                pmv[indexVertex].m_normal[1] =normal.y;
                pmv[indexVertex].m_normal[2] =normal.z;

                indexVertex++;
            }

            //inner circle
            for (unsigned int j = 0; j<mNumSegBase; j++)
            {

                float mRadius2=mRadius/4.0f;
                float x0 = mRadius/2.0f+mRadius2 * cosf(j*deltaAngle);
                float z0 = mRadius2 * sinf(j*deltaAngle);

                pmv[indexVertex].m_pos[0] =x0;
                pmv[indexVertex].m_pos[1] =0;
                pmv[indexVertex].m_pos[2] =z0;
                pmv[indexVertex].m_abgr =(uint32_t)tempColor1 ;

                bx::Vec3 tempPoint1(x0, 1,z0);
                const bx::Vec3 normal = bx::normalize(tempPoint1);

                pmv[indexVertex].m_normal[0] =normal.x;
                pmv[indexVertex].m_normal[1] =normal.y;
                pmv[indexVertex].m_normal[2] =normal.z;

                indexVertex++;
            }

            //fill by index
            int indexI=0;
            {

                for (unsigned int j=0; j<mNumSegBase; j++)
                {

                    //first trianle
                    pmi[indexI] = uint16_t(j+mNumSegBase);
                      indexI++;
                    pmi[indexI] = uint16_t(j);
                      indexI++;
                    if(j+1==mNumSegBase)
                        pmi[indexI] = uint16_t(0);
                    else
                        pmi[indexI] = uint16_t(j+1);
                      indexI++;

                    //second trianle
                    pmi[indexI] = uint16_t(j+mNumSegBase);
                      indexI++;

                    if(j+1==mNumSegBase)
                        pmi[indexI] = uint16_t(0);
                    else
                        pmi[indexI] = uint16_t(j+1);
                      indexI++;

                    if(j+1==mNumSegBase)
                        pmi[indexI] = uint16_t(0+mNumSegBase);
                    else
                        pmi[indexI] = uint16_t(j+mNumSegBase+1);
                      indexI++;
                }

                offset++;
                indexVertex++;
            }


        m_vbh_cyl_t=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*estimateVertexCount),
                                                       PosColorVertex::ms_layout));
        m_ibh_cyl_t=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(estimateIndexCount))));

      //  free(pmv);
      //  free(pmi);

}

static void createProceduralFace(bgfx::VertexBufferHandle& m_vbh_ref,
                          bgfx::IndexBufferHandle&  m_ibh_ref,
                          int N)
    {

        vector<PosColorVertex> position;
        vector<int> elements;

        float d = 1.0f/N;
        int R = 2*N+2;
        for (int row = 0; row <= N; row++)
        {
            for (int col = 0; col <= N; col++)
            {
                float x = -1.0f + d * col;
                float y = -1.0f + d * row;
                int S = position.size();

                {
                    PosColorVertex tempPos;
                    tempPos.m_pos[0]=x;
                    tempPos.m_pos[1]=y;
                    tempPos.m_pos[2]=0;

                    tempPos.m_normal[0]=1;
                    tempPos.m_normal[1]=0;
                    tempPos.m_normal[2]=1;

                   // tempPos.m_u=x;
                   // tempPos.m_v=y;
                   // tempPos.m_w=0;
                    position.push_back(tempPos);
                }
                {

                    PosColorVertex tempPos;

                    tempPos.m_pos[0]=x+d/2;
                    tempPos.m_pos[1]=y-d/2;
                    tempPos.m_pos[2]=0;

                    tempPos.m_normal[0]=1;
                    tempPos.m_normal[1]=0;
                    tempPos.m_normal[2]=1;

                    //tempPos.m_u=x+d/2;
                    //tempPos.m_v=y-d/2;
                    //tempPos.m_w=0;
                    position.push_back(tempPos);
                }

                //position.push([x, y], [x+d/2, y-d/2]);
                if (row > 0 && col < N)
                {
                     elements.push_back(S+2);
                     elements.push_back(S);
                     elements.push_back(S+1);

                     elements.push_back(S);
                     elements.push_back(S-R);
                     elements.push_back(S+1);

                     elements.push_back(S-R);
                     elements.push_back(S-R+2);
                     elements.push_back(S+1);

                     elements.push_back(S-R+2);
                     elements.push_back(S+2);
                     elements.push_back(S+1);
                    //elements.push_back([S+2, S, S+1], [S, S-R, S+1], [S-R, S-R+2, S+1], [S-R+2, S+2, S+1]);

                }
            }
        }


        auto pmv=position.data();

        uint32_t numVertices =position.size();
        uint32_t numIndices  =elements.size();


        uint16_t* pmi=new uint16_t[numIndices];
        for (uint32_t ii = 0; ii < numIndices; ++ii)
        {
            pmi[ii] = uint16_t(elements[ii]);
        }

        m_vbh_ref=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*numVertices),
                                                      PosColorVertex::ms_layout));
        m_ibh_ref=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices))));

        //free(trilist);

 }

static void createProceduralFace2(bgfx::VertexBufferHandle& m_vbh_ref,
                               bgfx::IndexBufferHandle&  m_ibh_ref,
                               int N)
    {

        vector<PosColorVertex> position;
        vector<uint16_t> elements;

        float d = 2.0f/N;
        int R = 2*N+1;
        for (int row = 0; row <= N; row++)
        {
            for (int col = 0; col <= N; col++)
            {
                float x = -1.0f + d * col;
                float y = -1.0f + d * row;

                float x1 = 0.0f + d * col;
                float y1 = 0.0f + d * row;

                {
                    PosColorVertex tempPos;
                    tempPos.m_pos[0]=x;
                    tempPos.m_pos[1]=y;
                    tempPos.m_pos[2]=0;

                    tempPos.m_normal[0]=0;
                    tempPos.m_normal[1]=0;
                    tempPos.m_normal[2]=1;

                    //tempPos.m_u=(1.0f+x)/2.0f;
                    //tempPos.m_v=(1.0f+y)/2.0f;
                    //tempPos.m_w=0;
                    position.push_back(tempPos);
                }
            }
            if(row<N)
            for (int col = 0; col < N; col++)
            {
                {
                    float x = -1.0f + d * col;
                    float y = -1.0f + d * row;

                    float x1 = 0.0f + d * col;
                    float y1 = 0.0f + d * row;

                    PosColorVertex tempPos;

                    tempPos.m_pos[0]=x+d/2;
                    tempPos.m_pos[1]=y+d/2;
                    tempPos.m_pos[2]=0;

                    tempPos.m_normal[0]=0;
                    tempPos.m_normal[1]=0;
                    tempPos.m_normal[2]=1;

                    //tempPos.m_u=(1.0f+x+d/2)/2.0f;
                    //tempPos.m_v=(1.0f+y+d/2)/2.0f;
                    //tempPos.m_w=0;
                    position.push_back(tempPos);
                }
            }

        }

        for (uint16_t row = 1; row <= N; row++)
        {
            uint16_t sr=row*R;

            for (uint16_t col = 0; col < N; col++)
            {
                uint16_t s=sr+col;
                uint16_t sd=s-R;
                uint16_t sc=s-N;

                elements.push_back(s+1);
                elements.push_back(s);
                elements.push_back(sc);

                elements.push_back(s);
                elements.push_back(sd);
                elements.push_back(sc);

                elements.push_back(sd);
                elements.push_back(sd+1);
                elements.push_back(sc);

                elements.push_back(sd+1);
                elements.push_back(s+1);
                elements.push_back(sc);
            }
        }



        uint32_t numVertices =position.size();
        uint32_t numIndices  =elements.size();

        PosColorVertex* pmv=new PosColorVertex[numVertices];


        for(uint16_t ii = 0; ii < numVertices; ++ii)
        {
            pmv[ii] = position[ii];
        }

        uint16_t* pmi=new uint16_t[numIndices];
        //uint16_t* pmi=elements.data();


        for( uint16_t ii = 0; ii < numIndices; ++ii)
        {
            pmi[ii] =elements[ii];
        }


        m_vbh_ref=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*numVertices),PosColorVertex::ms_layout));
        m_ibh_ref=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices))));

        //free(trilist);

 }

static void createProceduralFaceT(bgfx::VertexBufferHandle& m_vbh_ref,
                                  bgfx::IndexBufferHandle&  m_ibh_ref,
                                  int N)
       {

           vector<PosColorLayout> position;
           vector<uint16_t> elements;

           float d = 1.0f/N;

           for (int row = 0; row <=N; row++)
           {
               for (int col = 0; col <=N; col++)
               {
                   float x = -1.0f + d * col;
                   float y = -1.0f + d * row;
                   {
                       PosColorLayout tempPos;
                       tempPos.m_pos[0]=x;
                       tempPos.m_pos[1]=y;
                       tempPos.m_pos[2]=0;

                       tempPos.m_normal[0]=0;
                       tempPos.m_normal[1]=0;
                       tempPos.m_normal[2]=1;

                       tempPos.m_u=(1.0f+x)/1.0f;
                       tempPos.m_v=(1.0f+y)/1.0f;
                       tempPos.m_w=0;
                       position.push_back(tempPos);
                   }
               }

           }

           for (int row = 1; row <= N; row++)
           {
               int sr=(N+1)*row;

               for (int col = 1; col <= N; col++)
               {
                   int s=sr+col;
                   int sd=s-(N+2);

                   elements.push_back(s);
                   elements.push_back(s-1);
                   elements.push_back(sd);

                   elements.push_back(s);
                   elements.push_back(sd);
                   elements.push_back(sd+1);


               }
           }


           uint32_t numVertices =position.size();
           uint32_t numIndices  =elements.size();

           PosColorLayout* pmv=new PosColorLayout[numVertices];

           for (uint32_t ii = 0; ii < numVertices; ++ii)
           {
               pmv[ii] = position[ii];
           }

           uint16_t* pmi=new uint16_t[numIndices];
           //uint16_t* pmi=elements.data();


           for (uint32_t ii = 0; ii < numIndices; ++ii)
           {
               pmi[ii] = uint16_t(elements[ii]);
           }


           m_vbh_ref=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorLayout)*numVertices),PosColorLayout::ms_layout));
           m_ibh_ref=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices))));

           //free(trilist);

    }

static  void createProceduralSimpleFace(bgfx::VertexBufferHandle& m_vbh_ref,
                                    bgfx::IndexBufferHandle&  m_ibh_ref,
                                    int N)
    {

        vector<PosColorVertex> position;
        vector<uint16_t> elements;

        float d = 1.0f/N;

        for (int row = 0; row <=N; row++)
        {
            for (int col = 0; col <=N; col++)
            {
                float x = -0.5f + d * col;
                float y = -0.5f + d * row;
                {
                    PosColorVertex tempPos;
                    tempPos.m_pos[0]=x;
                    tempPos.m_pos[1]=0;
                    tempPos.m_pos[2]=y;

                    tempPos.m_normal[0]=0;
                    tempPos.m_normal[1]=1;
                    tempPos.m_normal[2]=0;

                    //tempPos.m_u=(1.0f+x)/2.0f;
                    //tempPos.m_v=(1.0f+y)/2.0f;
                    //tempPos.m_w=0;
                    position.push_back(tempPos);
                }
            }

        }

        for (int row = 1; row <= N; row++)
        {
            int sr=(N+1)*row;

            for (int col = 1; col <= N; col++)
            {
                int s=sr+col;
                int sd=s-(N+2);

                elements.push_back(s);
                elements.push_back(s-1);
                elements.push_back(sd);

                elements.push_back(s);
                elements.push_back(sd);
                elements.push_back(sd+1);


            }
        }


        uint32_t numVertices =position.size();
        uint32_t numIndices  =elements.size();

        PosColorVertex* pmv=new PosColorVertex[numVertices];

        for (uint32_t ii = 0; ii < numVertices; ++ii)
        {
            pmv[ii] = position[ii];
        }

        uint16_t* pmi=new uint16_t[numIndices];
        //uint16_t* pmi=elements.data();


        for (uint32_t ii = 0; ii < numIndices; ++ii)
        {
            pmi[ii] = uint16_t(elements[ii]);
        }


        m_vbh_ref=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*numVertices),PosColorVertex::ms_layout));
        m_ibh_ref=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices))));

        //free(trilist);

 }

static  void createProceduralSimpleFace2(bgfx::VertexBufferHandle& m_vbh_ref,
                                    bgfx::IndexBufferHandle&  m_ibh_ref,
                                    int N)
    {

        vector<PosColorLayout> position;
        vector<uint16_t> elements;
        float d = 2.0f/N;

        for (int row = 0; row <=N; row++)
        {
            for (int col = 0; col <=N; col++)
            {
                float x = -1.0f + d * col;
                float y = -1.0f + d * row;
                {
                    PosColorLayout tempPos;
                    tempPos.m_pos[0]=x;
                    tempPos.m_pos[1]=y;
                    tempPos.m_pos[2]=0;

                    tempPos.m_normal[0]=0;
                    tempPos.m_normal[1]=0;
                    tempPos.m_normal[2]=1;

                    tempPos.m_u=(1.0f+x)/2.0f;
                    tempPos.m_v=(1.0f+y)/2.0f;
                    tempPos.m_w=0;
                    position.push_back(tempPos);
                }
            }

        }

        for (int row = 1; row <= N; row++)
        {
            int sr=(N+1)*row;

            for (int col = 1; col <= N; col++)
            {
                int s=sr+col;
                int sd=s-(N+2);

                elements.push_back(s);
                elements.push_back(s-1);
                elements.push_back(sd);

                elements.push_back(s);
                elements.push_back(sd);
                elements.push_back(sd+1);


            }
        }


        uint32_t numVertices =position.size();
        uint32_t numIndices  =elements.size();

        PosColorLayout* pmv=new PosColorLayout[numVertices];

        for (uint32_t ii = 0; ii < numVertices; ++ii)
        {
            pmv[ii] = position[ii];
        }

        uint16_t* pmi=new uint16_t[numIndices];
        //uint16_t* pmi=elements.data();


        for (uint32_t ii = 0; ii < numIndices; ++ii)
        {
            pmi[ii] = uint16_t(elements[ii]);
        }


        m_vbh_ref=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorLayout)*numVertices),PosColorLayout::ms_layout));
        m_ibh_ref=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices))));

        //free(trilist);

 }

static  void createProceduralSimpleFaceY(bgfx::VertexBufferHandle& m_vbh_ref,
                                         bgfx::IndexBufferHandle&  m_ibh_ref,
                                         bx::Vec3 point1,
                                         bx::Vec3 point2,
                                         int N)
    {

        vector<PosColorVertex> position;
        vector<uint16_t> elements;

        float d = 1.0f/N;
/*
        float dist=bx::length(bx::sub(bpoint2,bpoint1));
        bx::Vec3 normal = bx::normalize(bx::sub(bpoint2,bpoint1));

        bx::Vec3 targetPoint1=bpoint2;
        bx::Vec3 distNorm3=normal;
        bx::Vec3 targetPoint3=bx::sub(targetPoint1,bx::mul(normal,dist/2.0f));
*/
        for (int row = 0; row <=N; row++)
        {
            for (int col = 0; col <=N; col++)
            {
                float x = -0.5f + d * col;
                float y = -0.5f + d * row;
                {
                    PosColorVertex tempPos;
                    tempPos.m_pos[0]=0;
                    tempPos.m_pos[1]=x;
                    tempPos.m_pos[2]=y;

                    tempPos.m_normal[0]=0.3;
                    tempPos.m_normal[1]=0.3;
                    tempPos.m_normal[2]=0;

                    //tempPos.m_u=(1.0f+x)/2.0f;
                    //tempPos.m_v=(1.0f+y)/2.0f;
                    //tempPos.m_w=0;
                    position.push_back(tempPos);
                }
            }

        }

        for (int row = 1; row <= N; row++)
        {
            int sr=(N+1)*row;

            for (int col = 1; col <= N; col++)
            {
                int s=sr+col;
                int sd=s-(N+2);

                elements.push_back(s);
                elements.push_back(s-1);
                elements.push_back(sd);

                elements.push_back(s);
                elements.push_back(sd);
                elements.push_back(sd+1);


            }
        }


        uint32_t numVertices =position.size();
        uint32_t numIndices  =elements.size();

        PosColorVertex* pmv=new PosColorVertex[numVertices];

        for (uint32_t ii = 0; ii < numVertices; ++ii)
        {
            pmv[ii] = position[ii];
        }

        uint16_t* pmi=new uint16_t[numIndices];
        //uint16_t* pmi=elements.data();


        for (uint32_t ii = 0; ii < numIndices; ++ii)
        {
            pmi[ii] = uint16_t(elements[ii]);
        }


        m_vbh_ref=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*numVertices),PosColorVertex::ms_layout));
        m_ibh_ref=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices))));

        //free(trilist);

 }

inline int RandomInt(int a, int b)
{
    float  random = rand();
    float rd=random/(float)(RAND_MAX);
    int  diff = b - a;
    int  r = (int)(rd * diff);
    return a + r;
}

inline int32_t FloorToI32(float f)
{
    return int32_t(floorf(f));
}

inline uint8_t FloatToU8(float f)
{
    int i = int(f * 256.0f);
    return uint8_t(i - (i >> 8));
}

static void CreateTestHMap16(int size, uint16_t* dst, int meshSize, float heightcoeff)
{
    int w = size;
    int h = size;

    float step=(float)meshSize / size;
    float s = 4*M_PI * ((float)meshSize / size);
    float2 point=make_float2(20,20);

    for (int y = 0; y < h; y++)
    {
        uint16_t* dstSpan = dst + y * w;
        for (int x = 0; x < w; x++)
        {
            dstSpan[x] = FloorToI32(heightcoeff*65000.0f * (powf(sinf(s * x),2) * powf(sinf(s * y),2)));
            //uint8_t dstTemp=FloorToI32(heightcoeff*175.0f * (sqr(sinf(s * x)) * sqr(sinf(s * y))));
           /*
            dstSpan[x]=0;
            if(x>0 and y>0)
            {
                Vec2 tempPoint(x,y);
                //if(point.x!=tempPoint.x and point.y!=tempPoint.y)
                {
                    float distance=Length(point-tempPoint);
                    if(distance<25.0f)
                        dstSpan[x]=heightcoeff*65000.0f*(25.0f-distance);
                }
            }
*/
            //dstSpan[x] =dstTemp;

        }
    }
}

static void CreateTestHMap8(int size, uint8_t* dst, int meshSize, float heightcoeff)
{
    int w = size;
    int h = size;

    float step=(float)meshSize / size;
    float s = 4*M_PI * ((float)meshSize / size);
    float2 point=make_float2(20,20);

    for (int y = 0; y < h; y++)
    {
        uint8_t* dstSpan = dst + y * w;
        for (int x = 0; x < w; x++)
        {
            uint8_t dstTemp=FloorToI32(heightcoeff*175.0f * (powf(sinf(s * x),2) * powf(sinf(s * y),2)));
            /*
            uint8_t dstTemp=0;
            dstTemp=0;
            if(x>0 and y>0)
            {
                Vec2 tempPoint(x, y);
                //if(point.x!=tempPoint.x and point.y!=tempPoint.y)
                {
                    float distance=Length(point-tempPoint);
                    if(distance<25.0f)
                        dstTemp=heightcoeff*175.0f*(25.0f-distance);
                }
            }
*/
            dstSpan[x] =dstTemp;
        }
    }
}


