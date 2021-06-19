#ifndef GRAPHMODULE_HPP
#define GRAPHMODULE_HPP

#pragma once
#include "common/common.h"
#include "common/bgfx_utils.h"
#include "common/imgui/imgui.h"
#include "bounds.h"

#include <stdio.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include "camera.h"
#include <bx/uint32_t.h>
#include <debugdraw/debugdraw.h>

#include "bgfx_utils.h"

#include <bgfx/bgfx.h>
#include <bx/commandline.h>
#include <bx/endian.h>
#include <bx/math.h>
#include <bx/readerwriter.h>
#include <bx/string.h>

#include <cuda_runtime.h>
#include "algorithm"
#include <cmath>

#include "SupportBGFX.hpp"

#define SPRITE_TEXTURE_SIZE 1024

class GraphModule: public entry::AppI
{
public:
    GraphModule(const char* _name, const char* _description, const char* _url)
        : entry::AppI(_name, _description, _url),
          m_pt(0)
    {

    }

    void init(int32_t _argc, const char* const* _argv,  uint32_t _width, uint32_t _height) override
    {
        Args args(_argc, _argv);

        m_width  = _width;
        m_height = _height;
        m_debug  = BGFX_DEBUG_TEXT;
        m_reset  = BGFX_RESET_VSYNC;

        bgfx::Init init;
        init.type     = bgfx::RendererType::OpenGL; //args.m_type;
        init.vendorId = args.m_pciId;
        init.resolution.width  = m_width;
        init.resolution.height = m_height;
        init.resolution.reset  = m_reset;

        bgfx::init(init);

        // Enable debug text.
        bgfx::setDebug(m_debug);

        // Set view 0 clear state.
        bgfx::setViewClear(0
            , BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH
            , 0x303030ff
            , 1.0f
            , 0
            );

        // Get renderer capabilities info.
        const bgfx::Caps* caps = bgfx::getCaps();
        m_instancingSupported = 0 != (caps->supported & BGFX_CAPS_INSTANCING);

        // Create vertex stream declaration.
        PosColorVertex::init();
        InstanceLayout::init();

        instanceBuffer= bgfx::createDynamicVertexBuffer(
                            // Static data can be passed with bgfx::makeRef
                              bgfx::makeRef(instanceInit, sizeof(instanceInit) )
                            , InstanceLayout::ms_layout, BGFX_BUFFER_ALLOW_RESIZE
                            );
        instanceBufferLink= bgfx::createDynamicVertexBuffer(
                            // Static data can be passed with bgfx::makeRef
                              bgfx::makeRef(instanceInit, sizeof(instanceInit) )
                            , InstanceLayout::ms_layout, BGFX_BUFFER_ALLOW_RESIZE
                            );

        instanceBufferLink2= bgfx::createDynamicVertexBuffer(
                            // Static data can be passed with bgfx::makeRef
                              bgfx::makeRef(instanceInit, sizeof(instanceInit) )
                            , InstanceLayout::ms_layout, BGFX_BUFFER_ALLOW_RESIZE
                            );


        m_numLights = 4;
        u_lightPosRadius = bgfx::createUniform("u_lightPosRadius", bgfx::UniformType::Vec4, m_numLights);
        u_lightRgbInnerR = bgfx::createUniform("u_lightRgbInnerR", bgfx::UniformType::Vec4, m_numLights);


        // Create program from shaders.
        m_program = loadProgram("vs_instancing", "fs_instancing");

        createProceduralSphereLod(1.0f,16);
        //createProceduralCylinderLod(0.85f,0.97f,12);

        createProceduralCylinder(1.0f, 1.0f,32);


        for(int icolor=0; icolor<255;icolor++)
        {

            int baseColorR=(int)(145*(((float) rand() / (RAND_MAX))));
            int baseColorG=(int)(145*(((float) rand() / (RAND_MAX))));
            int baseColorB=(int)(145*(((float) rand() / (RAND_MAX))));

        }

        m_timeOffset = bx::getHPCounter();

        cameraCreate();
        cameraSetPosition({ 0.0f, 195.f, 90.0f });
        cameraSetVerticalAngle(bx::toRad(-90));
        cameraSetHorizontalAngle(bx::toRad(0));

        ddInit();

        imguiCreate();
        ImGui::LoadIniSettingsFromDisk("tempImgui.ini");
         //ImGui::SaveIniSettingsToDisk("tempImgui.ini");
    }



    uint32_t ColorConvert2(int r, int g, int b)
    {
      uint32_t c;
      c = r;
      c <<= 8;
      c |= g;
      c <<= 8;
      c |= b;
      return c;
    }

    uint32_t ColorConvert(int r, int g, int b)
    {
          return (uint8_t(255) << 24) +
                 (uint8_t(b) << 16) +
                 (uint8_t(g) << 8)  +
                  uint8_t(r);

    }

    int shutdown() override
    {
        ImGui::SaveIniSettingsToDisk("tempImgui.ini");

        imguiDestroy();

        //experiment.freeMemory();
        // Cleanup.
        //bgfx::destroy(m_ibh);
       // bgfx::destroy(m_vbh);
        bgfx::destroy(m_program);
        //bgfx::destroy(m_textureColor);
        //bgfx::destroy(m_textureNormal);
        //bgfx::destroy(s_texColor);
       //bgfx::destroy(s_texNormal);
        bgfx::destroy(u_lightPosRadius);
        bgfx::destroy(u_lightRgbInnerR);

        // Shutdown bgfx.
        bgfx::shutdown();

        return 0;
    }

    template<typename Ty>
    bool intersect(DebugDrawEncoder* _dde, const Ray& _ray, const Ty& _shape)
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

    bx::Vec3 arbitrary_orthogonal(bx::Vec3 vec)
       {
         bool b0 = (vec.x <  vec.y) && (vec.x <  vec.z);
         bool b1 = (vec.y <= vec.x) && (vec.y <  vec.z);
         bool b2 = (vec.z <= vec.x) && (vec.z <= vec.y);

         return bx::cross(vec, bx::Vec3(b0, b1, b2));
       }

       bx::Vec3 arbitrary_orthogonal2(bx::Vec3 N)
       {
           double Ax= abs(N.x), Ay= abs(N.y), Az= abs(N.z);
           if (Ax < Ay)
               return  Ax < Az ? bx::Vec3(0, -N.z, N.y) : bx::Vec3(-N.y, N.x, 0);
           else
               return  Ay < Az ? bx::Vec3(N.z, 0, -N.x) : bx::Vec3(-N.y, N.x, 0);
       }
      void circle(bx::Vec3& _out, float _angle)
       {
           float sa = bx::sin(_angle);
           float ca = bx::cos(_angle);
           _out.x = sa;
           _out.y= ca;
       }

       void squircle(bx::Vec3& _out,  float _angle)
       {
           float sa = bx::sin(_angle);
           float ca = bx::cos(_angle);
           _out.x = bx::sqrt(bx::abs(sa) ) * bx::sign(sa);
           _out.y = bx::sqrt(bx::abs(ca) ) * bx::sign(ca);
       }

       void genSphere(uint8_t _subdiv0, PosColorVertex* pmv)
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



       void createProceduralSphereLod(float radius1, int num)
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




    void createProceduralCylinderLod(float radius1, float radius2, int num)
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


            m_vbh_cyl=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*numVertices),
                                                          PosColorVertex::ms_layout));

            m_ibh_cyl=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(numIndices  + numLineListIndices))));

        }



    void createProceduralCylinder(float radius1, float radius2, int num)
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

            m_vbh_cyl=(bgfx::createVertexBuffer(bgfx::makeRef(pmv, sizeof(PosColorVertex)*estimateVertexCount),
                                                          PosColorVertex::ms_layout));
            m_ibh_cyl=(bgfx::createIndexBuffer(bgfx::makeRef(pmi, sizeof(uint16_t)*(estimateIndexCount))));

    }

    bool update(Environment* env) override
    {
        if (!entry::processEvents(m_width, m_height, m_debug, m_reset, &m_mouseState) )
        {
            imguiBeginFrame(m_mouseState.m_mx
                ,  m_mouseState.m_my
                , (m_mouseState.m_buttons[entry::MouseButton::Left  ] ? IMGUI_MBUT_LEFT   : 0)
                | (m_mouseState.m_buttons[entry::MouseButton::Right ] ? IMGUI_MBUT_RIGHT  : 0)
                | (m_mouseState.m_buttons[entry::MouseButton::Middle] ? IMGUI_MBUT_MIDDLE : 0)
                ,  m_mouseState.m_mz
                , uint16_t(m_width)
                , uint16_t(m_height)
                );

            //showExampleDialog(this);

            ImGui::Begin("Experiment commands");

                ImGui::SliderInt("Mesh size", &meshSize, 1, 9);

                ImGui::Checkbox("Run solver", &runSolver);

                ImGui::Text("number of particles %i", env->agents.size());
                if(maxBuffer)
                     ImGui::Text("maxBuffer Limit over");
                if(ImGui::Button("Step Simulation"))
               {
                   stepClick=true;
               }
               ImGui::Checkbox("Start-Pause simulation", &runSimulation);

            ImGui::End();


            imguiEndFrame();


            // Set view 0 default viewport.
            //bgfx::setViewRect(0, 0, 0, uint16_t(m_width), uint16_t(m_height) );

            // This dummy draw call is here to make sure that view 0 is cleared
            // if no other draw calls are submitted to view 0.

            // Set view 0 default viewport.
            bgfx::setViewRect(0, 0, 0, uint16_t(m_width), uint16_t(m_height) );

            // This dummy draw call is here to make sure that view 0 is cleared
            // if no other draw calls are submitted to view 0.
            bgfx::touch(0);

            float time = (float)( (bx::getHPCounter()-m_timeOffset)/double(bx::getHPFrequency() ) );

            int64_t now = bx::getHPCounter() - m_timeOffset;
            static int64_t last = now;
            const int64_t frameTime = now - last;
            last = now;
            const double freq = double(bx::getHPFrequency() );
            const float deltaTime = float(10.0f*frameTime/freq);

            // Update camera.
            cameraUpdate(deltaTime, m_mouseState);

            float view[16];
            cameraGetViewMtx(view);

            float proj[16];

            // Set view and projection matrix for view 0.
            {
               bx::mtxProj(proj, 60.0f, float(m_width)/float(m_height), 0.1f, 1750.0f, bgfx::getCaps()->homogeneousDepth);

               bgfx::setViewTransform(0, view, proj);
               bgfx::setViewRect(0, 0, 0, uint16_t(m_width), uint16_t(m_height) );
            }

            float mtxVp[16];
            bx::mtxMul(mtxVp, view, proj);

            float mtxInvVp[16];
            bx::mtxInverse(mtxInvVp, mtxVp);

            const bx::Vec3 at  = { 0.0f,  0.0f, 0.0f };
            const bx::Vec3 eye = { 5.0f, 10.0f, 5.0f };
            bx::mtxLookAt(view, eye, at);
            bx::mtxProj(proj, 45.0f, float(m_width)/float(m_height), 1.0f, 15.0f, bgfx::getCaps()->homogeneousDepth);
            bx::mtxMul(mtxVp, view, proj);


            Ray ray = makeRay(
                               (float(m_mouseState.m_mx)/float(m_width)  * 2.0f - 1.0f)
                            , -(float(m_mouseState.m_my)/float(m_height) * 2.0f - 1.0f)
                            , mtxInvVp
                            );

           // bgfx::touch(0);



            float lightPosRadius[4][4];
            for (uint32_t ii = 0; ii < m_numLights; ++ii)
            {
                lightPosRadius[ii][0] = -50.0f+ii*50.0f;
                lightPosRadius[ii][1] = 30.0f*ii;
                lightPosRadius[ii][2] = -50.0f+57.5f+3*ii;
                lightPosRadius[ii][3] = 555.0f;
            }

            bgfx::setUniform(u_lightPosRadius, lightPosRadius, m_numLights);

            float lightRgbInnerR[4][4] =
            {
                { 1.0f, 1.0f, 1.0f, 0.8f },
                { 1.0f, 1.0f, 1.0f, 0.8f },
                { 1.0f, 1.0f, 1.0f, 0.8f },
                { 1.0f, 1.0f, 1.0f, 0.8f },
            };

            bgfx::setUniform(u_lightRgbInnerR, lightRgbInnerR, m_numLights);


            constexpr uint32_t kSelected = 0xff80ffff;
            constexpr uint32_t kOverlapA = 0xff0000ff;
            constexpr uint32_t kOverlapB = 0xff8080ff;

            DebugDrawEncoder dde;

            dde.begin(0);
            dde.drawAxis(-0.0f, 0.0f, -0.0f,15.0f);
           // dde.drawAxis(0.0f, 0.0f, 0.0f,5);


            {
                const bx::Vec3 normal = { 0.0f,  1.0f, 0.0f };
                const bx::Vec3 pos    = { 0.0f, 0.0f, 100.0f };

                dde.drawGrid(Axis::Y, pos, 20, 10.0f);
            }

            dde.setColor(0xffffffff);
           // const uint16_t instanceStride =128;

            // 80 bytes stride = 64 bytes for 4x4 matrix + 16 bytes for RGBA color.
            const uint16_t instanceStride = 64;
            const uint16_t numInstances   = 10;


          bool IntersectSphere=false;
          int numberOfSphere=0;

/*
          for(auto pos:model->ps())
          {
            dde.push();


            // dde.lineTo(seg);
            bx::Vec3 temp=(bx::Vec3(pos.x,pos.y,pos.z));
            Sphere sphere1 = { temp, model->RadiusOfInfluence()/3.0f };

            float dist=bx::length(bx::Vec3(pos.x,pos.y,pos.z));
            int cR=(int)(dist*25);
            int cG=5;
            int cB=5;
            if(cR>255){
                cR=255-(int)(dist*15.0f);
                cG=(int)(dist*25);
                if(cG>255)
                {
                    cG=255-(int)(dist*15.0f);
                    cB=(int)(dist);
                }

            }
            uint32_t colorDist=ColorConvert(cR,cG,cB);
            dde.setColor(colorDist);
            dde.setWireframe(false);
            dde.setLod(1);
            dde.draw(sphere1);


            dde.pop();
          }
*/
            //


          // Get renderer capabilities info.
          const bgfx::Caps* caps = bgfx::getCaps();

          // Check if instancing is supported.
          if (0 == (BGFX_CAPS_INSTANCING & caps->supported) )
          {
              // When instancing is not supported by GPU, implement alternative
              // code path that doesn't use instancing.
              bool blink = uint32_t(time*3.0f)&1;
              bgfx::dbgTextPrintf(0, 0, blink ? 0x4f : 0x04, " Instancing is not supported by GPU. ");
              shutdown();

          }



          {

              int numberOfSpheres=0;
              int numberOfCylinders=0;
              for(auto agent:env->agents)
              {
                  numberOfSpheres++;

              }


              // 80 bytes stride = 64 bytes for 4x4 matrix + 16 bytes for RGBA color.
              const uint16_t instanceStride = 80;
              // number of instances
              const uint32_t numInstances= numberOfSpheres;
              auto instanceData=new InstanceLayout[numInstances];

              int index=0;
              for(auto agent:env->agents)
              {
                       float mtx[16];
                       // bx::mtxRotateXY(mtx, time , time );
                     //   bx::mtxRotateZ(mtx, 0);
                        //bx::mtxScale(mtx, agent.radius
                        //                 ,agent.radius
                        //                 ,agent.radius);

                       // mtx[12] = agent.p.x;
                       // mtx[13] = 0;
                       // mtx[14] = agent.p.y;

                        float dist=bx::length(bx::Vec3(agent.p.x,
                                                       0,
                                                       agent.p.y));
                        int cR=55;
                        int cG=155;
                        int cB=115;

                        if(agent.mat_index==1)
                             cG=75;

                        bx::mtxSRT(mtx,
                                    agent.radius,0.5,agent.radius,
                                    0, agent.a, 0,
                                    agent.p.x,0,agent.p.y
                                    );

                        uint32_t colorDist=ColorConvert(cR,cG,cB);

                        instanceData[index].color[0] = cR/255.0f;
                        instanceData[index].color[1] = cG/255.0f;
                        instanceData[index].color[2] = cB/255.0f;
                        instanceData[index].color[3] = 1.0f;

                        instanceData[index].data0[0]=mtx[0];
                        instanceData[index].data0[1]=mtx[1];
                        instanceData[index].data0[2]=mtx[2];
                        instanceData[index].data0[3]=mtx[3];

                        instanceData[index].data1[0]=mtx[4];
                        instanceData[index].data1[1]=mtx[5];
                        instanceData[index].data1[2]=mtx[6];
                        instanceData[index].data1[3]=mtx[7];

                        instanceData[index].data2[0]=mtx[8];
                        instanceData[index].data2[1]=mtx[9];
                        instanceData[index].data2[2]=mtx[10];
                        instanceData[index].data2[3]=mtx[11];

                        instanceData[index].data3[0]=mtx[12];
                        instanceData[index].data3[1]=mtx[13];
                        instanceData[index].data3[2]=mtx[14];
                        instanceData[index].data3[3]=mtx[15];

                        index++;

                }


            const bgfx::Memory* mem = bgfx::copy(instanceData, numInstances*sizeof(InstanceLayout));
            bgfx::update(instanceBuffer, 0, mem);

            // Set vertex and index buffer.
            bgfx::setVertexBuffer(0, m_vbh_cyl);
            bgfx::setIndexBuffer(m_ibh_cyl);

            // Set instance data buffer.
            bgfx::setInstanceDataBuffer(instanceBuffer,0,numInstances);

            // Set render states.
            bgfx::setState(BGFX_STATE_DEFAULT);

            // Submit primitive for rendering to view 0.
            bgfx::submit(0, m_program);

            free(instanceData);

          }



          //Debug draw
          for(auto agent:env->agents)
          {
              for(int i=0; i<agent.numVertex; i++)
              {
                    bx::Vec3 tempVect=bx::Vec3(agent.subp[i].x,0, agent.subp[i].y);
                    dde.push();
                        float spRadius=0.5;
                        dde.setColor(tempColor1);
                        if(i==0)
                            dde.setColor(tempColor2);
                        Sphere sphere2 = {tempVect, spRadius};

                        dde.setWireframe(false);
                        dde.setLod(2);
                        dde.draw(sphere2);

                    dde.pop();
              }

          }

          //Debug draw contsraints


          {
              int numberOfSpheres=0;
              int numberOfCylinders=0;
              for(auto agent:env->agents)
              {
                  numberOfSpheres++;

              }

              if(numberOfSpheres>0)
              {
                  // 80 bytes stride = 64 bytes for 4x4 matrix + 16 bytes for RGBA color.
                  const uint16_t instanceStride = 80;
                  // number of instances
                  const uint32_t numInstances= numberOfSpheres;
                  auto instanceData=new InstanceLayout[numInstances];

                  int index=0;

                  for(auto agent:env->agents)
                  {
                          float mtx[16];

                          bx::Vec3 bpoint1=bx::Vec3(0,0,200);

                          bx::Vec3 bpoint2=bx::Vec3(agent.subp[0].x,
                                                    0,
                                                    agent.subp[0].y);


                          //bx::Vec3 bpoint2=bx::add(bpoint1,bx::mul(bx::Vec3(segment.direction.x,segment.direction.y,segment.direction.z),TentacleSPACING));

                          float dist=bx::length(bx::sub(bpoint2,bpoint1));
                          bx::Vec3 normal = bx::normalize(bx::sub(bpoint2,bpoint1));

                          bx::Vec3 targetPoint1=bpoint1;
                          bx::Vec3 distNorm3=normal;
                          bx::Vec3 targetPoint3=targetPoint1;

                          bx::Vec3 Scale=bx::Vec3( 0.3f,1.0f*dist,0.3f);
                          mtxFromNormalScale(mtx, normal, Scale, targetPoint3);


                          int cR=55;
                          int cG=55;
                          int cB=75;

                            uint32_t colorDist=ColorConvert(cR,cG,cB);

                            instanceData[index].color[0] = cR/255.0f;
                            instanceData[index].color[1] = cG/255.0f;
                            instanceData[index].color[2] = cB/255.0f;
                            instanceData[index].color[3] = 1.0f;

                            instanceData[index].data0[0]=mtx[0];
                            instanceData[index].data0[1]=mtx[1];
                            instanceData[index].data0[2]=mtx[2];
                            instanceData[index].data0[3]=mtx[3];

                            instanceData[index].data1[0]=mtx[4];
                            instanceData[index].data1[1]=mtx[5];
                            instanceData[index].data1[2]=mtx[6];
                            instanceData[index].data1[3]=mtx[7];

                            instanceData[index].data2[0]=mtx[8];
                            instanceData[index].data2[1]=mtx[9];
                            instanceData[index].data2[2]=mtx[10];
                            instanceData[index].data2[3]=mtx[11];

                            instanceData[index].data3[0]=mtx[12];
                            instanceData[index].data3[1]=mtx[13];
                            instanceData[index].data3[2]=mtx[14];
                            instanceData[index].data3[3]=mtx[15];

                            index++;
                }





                const bgfx::Memory* mem3 = bgfx::copy(instanceData, numInstances*sizeof(InstanceLayout));
                bgfx::update(instanceBufferLink2, 0, mem3);

                // Set vertex and index buffer.
                bgfx::setVertexBuffer(0, m_vbh_cyl);
                bgfx::setIndexBuffer(m_ibh_cyl);

                // Set instance data buffer.
                bgfx::setInstanceDataBuffer(instanceBufferLink2,0,numInstances);

                // Set render states.
                bgfx::setState(BGFX_STATE_DEFAULT);

                // Submit primitive for rendering to view 0.
                bgfx::submit(0, m_program);

                free(instanceData);
              }

          }




          //Debug draw contsraints

/*
          {
              int numberOfSpheres=0;
              for(auto contraint:env->constraints)
              {
                  if(contraint.active)
                  {
                    numberOfSpheres++;
                  }

              }

              if(numberOfSpheres>0)
              {
                  // 80 bytes stride = 64 bytes for 4x4 matrix + 16 bytes for RGBA color.
                  const uint16_t instanceStride = 80;
                  // number of instances
                  const uint32_t numInstances= numberOfSpheres;
                  auto instanceData=new InstanceLayout[numInstances];

                  int index=0;
                  for(auto contraint:env->constraints)
                  {
                      if(contraint.active)
                      {
                          float mtx[16];

                          bx::Vec3 bpoint1=bx::Vec3(env->agents[contraint.body1Index].subp[contraint.body1SubPointIndex].x, 0,
                                                    env->agents[contraint.body1Index].subp[contraint.body1SubPointIndex].y);

                          bx::Vec3 bpoint2=bx::Vec3(env->agents[contraint.body2Index].subp[contraint.body2SubPointIndex].x, 0,
                                                    env->agents[contraint.body2Index].subp[contraint.body2SubPointIndex].y);


                          //bx::Vec3 bpoint2=bx::add(bpoint1,bx::mul(bx::Vec3(segment.direction.x,segment.direction.y,segment.direction.z),TentacleSPACING));

                          float dist=bx::length(bx::sub(bpoint2,bpoint1));
                          bx::Vec3 normal = bx::normalize(bx::sub(bpoint2,bpoint1));

                          bx::Vec3 targetPoint1=bpoint1;
                          bx::Vec3 distNorm3=normal;
                          bx::Vec3 targetPoint3=targetPoint1;

                          bx::Vec3 Scale=bx::Vec3( 0.3f,1.0f*dist,0.3f);
                          mtxFromNormalScale(mtx, normal, Scale, targetPoint3);


                          int cR=55;
                          int cG=155;
                          int cB=115;

                            uint32_t colorDist=ColorConvert(cR,cG,cB);

                            instanceData[index].color[0] = cR/255.0f;
                            instanceData[index].color[1] = cG/255.0f;
                            instanceData[index].color[2] = cB/255.0f;
                            instanceData[index].color[3] = 1.0f;

                            instanceData[index].data0[0]=mtx[0];
                            instanceData[index].data0[1]=mtx[1];
                            instanceData[index].data0[2]=mtx[2];
                            instanceData[index].data0[3]=mtx[3];

                            instanceData[index].data1[0]=mtx[4];
                            instanceData[index].data1[1]=mtx[5];
                            instanceData[index].data1[2]=mtx[6];
                            instanceData[index].data1[3]=mtx[7];

                            instanceData[index].data2[0]=mtx[8];
                            instanceData[index].data2[1]=mtx[9];
                            instanceData[index].data2[2]=mtx[10];
                            instanceData[index].data2[3]=mtx[11];

                            instanceData[index].data3[0]=mtx[12];
                            instanceData[index].data3[1]=mtx[13];
                            instanceData[index].data3[2]=mtx[14];
                            instanceData[index].data3[3]=mtx[15];

                            index++;
                      }

                    }



                const bgfx::Memory* mem2 = bgfx::copy(instanceData, numInstances*sizeof(InstanceLayout));
                bgfx::update(instanceBufferLink, 0, mem2);

                // Set vertex and index buffer.
                bgfx::setVertexBuffer(0, m_vbh_cyl);
                bgfx::setIndexBuffer(m_ibh_cyl);

                // Set instance data buffer.
                bgfx::setInstanceDataBuffer(instanceBufferLink,0,numInstances);

                // Set render states.
                bgfx::setState(BGFX_STATE_DEFAULT);

                // Submit primitive for rendering to view 0.
                bgfx::submit(0, m_program);

                free(instanceData);
              }

          }

*/





          bgfx::frame();
          dde.end();

          int colorIndex=-1;
          int colorParticlesNumber=0;
          int colorParticlesNumberStore=0;


          tick++;

          return true;
       }

        return false;
    }

    entry::MouseState m_mouseState;

    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_debug;
    uint32_t m_reset;

    bgfx::VertexBufferHandle m_vbh;
    bgfx::IndexBufferHandle  m_ibh;

    bgfx::VertexBufferHandle m_vbh_cyl;
    bgfx::IndexBufferHandle  m_ibh_cyl;

    bgfx::DynamicVertexBufferHandle  instanceBuffer;
    bgfx::DynamicVertexBufferHandle  instanceBufferLink;
    bgfx::DynamicVertexBufferHandle  instanceBufferLink2;

  //  std::vector<bgfx::VertexBufferHandle> m_vbhList;
  //  std::vector<bgfx::IndexBufferHandle>  m_ibhList;

    bgfx::ProgramHandle m_program;
/*
    bgfx::UniformHandle s_texColor;
    bgfx::UniformHandle s_texNormal;
    bgfx::TextureHandle m_textureColor;
    std::vector<bgfx::TextureHandle> m_textureColorList;
    bgfx::TextureHandle m_textureNormal;
    */
    bgfx::UniformHandle u_lightPosRadius;
    bgfx::UniformHandle u_lightRgbInnerR;


    uint16_t m_numLights;
    bool m_instancingSupported;


    std::vector<bx::Vec3> scaleVector;

    int32_t m_pt;

    int numberVertex=0;
    int64_t m_timeOffset;

    int colorMesh=0;
    int meshSize=1;

    int tick=0;
    bool runSolver=true;
    bool maxBuffer=false;


    bool stepClick=false;
    bool runSimulation=false;


};

#endif // GRAPHMODULE_HPP
