#ifndef BGFXAPPL_HPP
#define BGFXAPPL_HPP

#pragma once

#include "bgfx_utils.h"

#include <bgfx/bgfx.h>
#include <bx/commandline.h>
#include <bx/endian.h>
#include <bx/math.h>
#include <bx/readerwriter.h>
#include "bx/timer.h"

#include <bx/bx.h>
#include <bx/file.h>
#include <bx/sort.h>
#include <bgfx/bgfx.h>
#include "bgfx/platform.h"

#include "algorithm"
#include <cmath>

#include "common/entry/entry.h"
#include "common/camera.h"
#include "common/debugdraw/debugdraw.h"
#include "common/imgui/imgui.h"
#include "environment.hpp"

#include "constants.hpp"

#include <stdint.h>
#include <bx/rng.h>
//#include "SupportBGFX.hpp"



#include "CubeMapLib.h"

using namespace CML;

static const int numberRunners=35;
static const uint16_t kTextureSide   =256;//512;
static const uint16_t kTextureSideSmall   = kTextureSide;//512;
static const uint32_t kTexture2dSize = kTextureSide*kTextureSide;//256;
static const uint16_t kSize =  kTextureSide;

static const uint16_t kMeshSide   = kTextureSide;

namespace entry
{

class AppIs
{
public:
    ///
    AppIs(const char* _name, const char* _description, const char* _url = "https://bkaradzic.github.io/bgfx/index.html");

    ///
     ~AppIs();

    ///
    //virtual
    void init(AppIs* _app,int32_t _argc, const char* const* _argv, uint32_t _width, uint32_t _height);

    ///
    int shutdown();

    ///
    bool update(bool tex_update, uintptr_t _tx_pointer[6], Environment *env);

    ///
    const char* getName() const;

    ///
    const char* getDescription() const;

    ///
    const char* getUrl() const;

    ///
    //AppI* getNext();

    //AppI* m_next;

    void    LogDraw(const char* title);
    void    LogClear();
    void    LogAddLog(const char* fmt, ...) IM_FMTARGS(2)
    {
        int old_size = Buf.size();
        va_list args;
        va_start(args, fmt);
        Buf.appendfv(fmt, args);
        va_end(args);
        for (int new_size = Buf.size(); old_size < new_size; old_size++)
            if (Buf[old_size] == '\n')
                LineOffsets.push_back(old_size + 1);


    }

    void WrapCubeFace(int size, int face, int x, int v, int *faceOut, int *xOut, int *yOut);
    void updatetexturePreview(unsigned char *dataColor, int preview);

    void updatetextureTexture(unsigned char *dataColor, int preview);


private:
    const char* m_name;
    const char* m_description;
    const char* m_url;

public:
    entry::MouseState m_mouseState;

    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_debug;
    uint32_t m_reset;

    uint32_t m_numTextures3d;
    bool m_texture3DSupported;
    bool m_blitSupported;
    bool m_computeSupported;

    bgfx::VertexBufferHandle m_vbh;
    bgfx::IndexBufferHandle  m_ibh;

    bgfx::VertexBufferHandle m_vbh_cyl;
    bgfx::IndexBufferHandle  m_ibh_cyl;

    bgfx::VertexBufferHandle m_vbh_cyl_s;
    bgfx::IndexBufferHandle  m_ibh_cyl_s;

    bgfx::VertexBufferHandle m_vbh_sq;
    bgfx::IndexBufferHandle  m_ibh_sq;

    bgfx::VertexBufferHandle m_vbh_seg;
    bgfx::IndexBufferHandle  m_ibh_seg;

    bgfx::DynamicVertexBufferHandle  instanceBuffer;
    bgfx::DynamicVertexBufferHandle  instanceBufferPtr [7];
    bgfx::DynamicVertexBufferHandle  instanceBufferPtrSQ [3];

    bgfx::TextureHandle m_texture2d;
    bgfx::TextureHandle m_texture2d_2p;

    bgfx::UniformHandle uh_simpleTexture;
    bgfx::TextureHandle th_simpleTexture;
    bgfx::UniformHandle s_texColor;


  //  std::vector<bgfx::VertexBufferHandle> m_vbhList;
  //  std::vector<bgfx::IndexBufferHandle>  m_ibhList;

    bgfx::ProgramHandle m_program;
    bgfx::ProgramHandle m_program_face;
    bgfx::ProgramHandle m_program_face_cube;

/*
    bgfx::UniformHandle s_texColor;
    bgfx::UniformHandle s_texNormal;
    bgfx::TextureHandle m_textureColor;
    std::vector<bgfx::TextureHandle> m_textureColorList;
    bgfx::TextureHandle m_textureNormal;
    */
    bgfx::UniformHandle u_lightPosRadius;
    bgfx::UniformHandle u_lightRgbInnerR;




    //-----------------------CubeMap


    int dxa[8] {-1, 1, 0, 0, 1,-1, 1, -1};
    int dya[8] { 0, 0,-1, 1,-1, 1, 1, -1};

    float m_time;
    float m_timeScale=0.2f;

    bx::RngMwc m_rng;

    const bgfx::Memory* mem;
    const bgfx::Memory* mem2;


    uint8_t* m_texture2dData;
    uint8_t* m_texture2dData0;
    uint8_t* m_texture2dData1;

    uint32_t m_hit;
    uint32_t m_miss;

    uint8_t m_rr;
    uint8_t m_gg;
    uint8_t m_bb;

    bgfx::TextureHandle m_textures[12];
    bgfx::TextureHandle m_textures3d[3];


    bgfx::TextureHandle m_texture2d_0;
    bgfx::TextureHandle m_texture2d_1;
    bgfx::TextureHandle m_texture2d_2;
    bgfx::TextureHandle m_texture2d_3;
    bgfx::TextureHandle m_texture2d_4;
    bgfx::TextureHandle m_texture2d_5;
    vector<bgfx::TextureHandle> m_texture2dList;


    bgfx::TextureHandle m_texture2dArrayPreview[6];

    bool needUpdate_m_texture2dArrayPreview=false;

    bool needUpdate_m_textureHeightMap_Normals=true;
    bool needUpdate_m_textureHeightMap_Normals_FS=true;

    bool needUpdate_Barrier=true;

    bgfx::TextureHandle m_texture2dArrayPreviewVel[6];
    bgfx::TextureHandle m_texture2dArrayPreviewRho[6];
    bgfx::TextureHandle m_texture2dArray[6];
    bgfx::TextureHandle m_texture2dArrayHeight[6];
    bgfx::TextureHandle m_texture2dArrayNormals[6];

    bgfx::TextureHandle m_textureCube[4];
    bgfx::FrameBufferHandle m_textureCubeFaceFb[6];

    bgfx::TextureHandle m_textureCubeSimple;



    bgfx::VertexBufferHandle m_vbh_face;
    bgfx::IndexBufferHandle  m_ibh_face;

    bgfx::VertexBufferHandle m_vbh_faceXZ;
    bgfx::IndexBufferHandle  m_ibh_faceXZ;


    bgfx::UniformHandle u_time;
    bgfx::UniformHandle s_texCube;
    bgfx::UniformHandle s_texNormal;
    bgfx::UniformHandle s_texHeight;

    bgfx::TextureHandle cubeTexture;

    const float kR = 10.0f;     ///< Radius of sphere's zero height.
    const float kH = 1.0f;      ///< Scale of heightmap.

    float p_deform=1.0;
    bgfx::UniformHandle u_deform;

    bool drawGrid=false;

    //64;//256;
    uint8_t htex[6][kSize * kSize];
    uint8_t htex_new[6][kSize * kSize];
    uint8_t hmap[6][kSize * kSize];
    uint16_t hmap16[6][kSize * kSize];
    //cPixel4U8 nmap[kSize * kSize];
    cPixel4U8 nmapsData[6][kSize * kSize];

    vector<uint8_t> hmaps8v;
    uint8_t*   hmaps8[6] ={ hmap[0], hmap[1], hmap[2], hmap[3], hmap[4], hmap[5] };
    uint8_t*   htex8[6] = { htex[0], htex[1], htex[2], htex[3], htex[4], htex[5] };
    uint8_t*   htex8_new[6] = { htex_new[0], htex_new[1], htex_new[2], htex_new[3], htex_new[4], htex_new[5] };

    uint16_t*  hmaps[6] = { hmap16[0], hmap16[1], hmap16[2], hmap16[3], hmap16[4], hmap16[5] };
    cPixel4U8* nmaps[6] = { nmapsData[0],  nmapsData[1], nmapsData[2],  nmapsData[3],  nmapsData[4], nmapsData[5] };

    //---------------------------------


    uint16_t m_numLights;
    bool m_instancingSupported;

    int32_t m_pt;

    int numberVertex=0;
    int64_t m_timeOffset;

    int colorMesh=0;
    int meshSize=1;

    int tick=0;
    bool runSolver=true;
    bool maxBuffer=false;


    int sim_width= 0;
    int sim_height=0;


    bool showTrails=false;
    bool showParticles=false;
    bool showDebugGrid=false;
    bool showSensors=false;

    uintptr_t tx_pointer;

    void DrawGrid(Environment *env);
    void DrawSegments(Environment *env);
    void DrawBots(Environment *env);

    ImGuiTextBuffer     Buf;
    ImGuiTextFilter     Filter;
    ImVector<int>       LineOffsets; // Index to lines offset. We maintain this with AddLog() calls.
    bool                AutoScroll=true;  // Keep scrolling if already at the bottom.

};

}

#endif // BGFXAPPL_HPP
