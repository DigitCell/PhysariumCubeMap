#include "bgfxappl.hpp"
#include "SupportBGFX.hpp"

namespace entry
{

AppIs::AppIs(const char *_name, const char *_description, const char *_url)
{
    m_name        = _name;
    m_description = _description;
    m_url         = _url;

    //s_apps = this;
    //s_numApps++;
}

const char* AppIs::getName() const
{
    return m_name;
}

const char* AppIs::getDescription() const
{
    return m_description;
}

const char* AppIs::getUrl() const
{
    return m_url;
}

void AppIs::init(AppIs* _app,int32_t _argc, const char * const *_argv, uint32_t _width, uint32_t _height)
{
    //Args args(_argc, _argv);

    m_width  = _width;
    m_height = _height;
    m_debug  = BGFX_DEBUG_TEXT;
    m_reset  = BGFX_RESET_VSYNC;

    bgfx::Init init;
    init.type     = bgfx::RendererType::OpenGL; //args.m_type;
   // init.vendorId = args.m_pciId;
    init.resolution.width  = m_width;
    init.resolution.height = m_height;
    init.resolution.reset  = m_reset;

    bgfx::init(init);



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
    PosColorLayout::init();

    u_time = bgfx::createUniform("u_time", bgfx::UniformType::Vec4);

    u_deform   = bgfx::createUniform("u_deform",  bgfx::UniformType::Vec4);

    for(int i=0; i<7;i++)
    {
        instanceBufferPtr[i]=bgfx::createDynamicVertexBuffer(
                    // Static data can be passed with bgfx::makeRef
                      bgfx::makeRef(instanceInit, sizeof(instanceInit) )
                    , InstanceLayout::ms_layout, BGFX_BUFFER_ALLOW_RESIZE
                    );
    }

    for(int i=0; i<3;i++)
    {
        instanceBufferPtrSQ[i]=bgfx::createDynamicVertexBuffer(
                    // Static data can be passed with bgfx::makeRef
                      bgfx::makeRef(instanceInit, sizeof(instanceInit) )
                    , InstanceLayout::ms_layout, BGFX_BUFFER_ALLOW_RESIZE
                    );
    }

    instanceBuffer= bgfx::createDynamicVertexBuffer(
                    // Static data can be passed with bgfx::makeRef
                      bgfx::makeRef(instanceInit, sizeof(instanceInit) )
                    , InstanceLayout::ms_layout, BGFX_BUFFER_ALLOW_RESIZE
                    );

   instanceBuffer= bgfx::createDynamicVertexBuffer(
                      // Static data can be passed with bgfx::makeRef
                        bgfx::makeRef(instanceInit, sizeof(instanceInit) )
                      , InstanceLayout::ms_layout, BGFX_BUFFER_ALLOW_RESIZE
                      );


    m_numLights = 4;
    u_lightPosRadius = bgfx::createUniform("u_lightPosRadius", bgfx::UniformType::Vec4, m_numLights);
    u_lightRgbInnerR = bgfx::createUniform("u_lightRgbInnerR", bgfx::UniformType::Vec4, m_numLights);


    // Create program from shaders.
    m_program = loadProgram("vs_instancing", "fs_instancing");
    m_program_face = loadProgram("vs_cubes4", "fs_cubes4");
    m_program_face_cube = loadProgram("vs_cubes2", "fs_cubes2");


    createProceduralCircleWH(m_vbh_cyl,m_ibh_cyl,1.0f, 16);
    createProceduralCylinder(m_vbh_seg,m_ibh_seg,1.0f, 1.0f,16);

    //createProceduralFaceT(m_vbh_sq,m_ibh_sq,12);

  //  createProceduralSphereLod(1.0f,64);
   // createProceduralFace2(kMeshSide);
    createProceduralSimpleFace2(m_vbh_sq,m_ibh_sq,128);


    cameraCreate();
    cameraSetPosition({ 150.0f, 575.f, 150.0f });
    cameraSetVerticalAngle(bx::toRad(-90));
    cameraSetHorizontalAngle(bx::toRad(0));

    // Enable debug text.
    bgfx::setDebug(m_debug);
    ddInit();

    imguiCreate();
    ImGui::LoadIniSettingsFromDisk("tempImgui.ini");
    //ImGui::SaveIniSettingsToDisk("tempImgui.ini");

    int2 worldSize=make_int2(world_width, world_height);
    sim_width=worldSize.x;
    sim_height=worldSize.y;

    {

       // int kTextureSide=sim_width;
        const uint32_t pitch = worldSize.x*4;
        const uint16_t tw = worldSize.x;
        const uint16_t th = worldSize.y;
        const uint16_t tx = 0;
        const uint16_t ty = 0;

        bgfx::TextureInfo ti;
        bgfx::calcTextureSize(ti, sim_width, sim_height, 1, false, false, 1, bgfx::TextureFormat::BGRA8);

        auto mem = bgfx::alloc(ti.storageSize);

        uint8_t* data = (uint8_t*)mem->data;

        int dataColorIndex=0;

       for (int32_t yy = th-1; yy >=0; yy--)
       {
           for (uint32_t xx = 0; xx < tw; xx++)
           {

               data[0] = (uint8_t)(70);
               data[1] = (uint8_t)(70);
               data[2] = (uint8_t)(10);
               data[3] =  0xff;

               data+= 4;
           }
       }

       m_texture2d= bgfx::createTexture2D(   sim_width
                                           , sim_height
                                           , false
                                           , 1
                                           , bgfx::TextureFormat::BGRA8
                                           , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT,
                                           mem
                                           );



    }

    {
        const uint32_t pitch = worldSize.x*4;
        const uint16_t tw = worldSize.x;
        const uint16_t th = worldSize.y;
        const uint16_t tx = 0;
        const uint16_t ty = 0;

        bgfx::TextureInfo ti;
        bgfx::calcTextureSize(ti, sim_width, sim_height, 1, false, false, 1, bgfx::TextureFormat::RGBA8);

        auto mem2 = bgfx::alloc(ti.storageSize);

        uint8_t* data = (uint8_t*)mem2->data;

        int dataColorIndex=0;

       for (int32_t yy = th-1; yy >=0; yy--)
       {
           for (uint32_t xx = 0; xx < tw; xx++)
           {

               data[0] = (uint8_t)(90);
               data[1] = (uint8_t)(190);
               data[2] = (uint8_t)(10);
               data[3] =  0xff;

               data+= 4;
           }
       }

       m_texture2d_2= bgfx::createTexture2D( sim_width
                                           , sim_height
                                           , false
                                           , 1
                                           , bgfx::TextureFormat::RGBA8
                                           , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT,
                                           mem2
                                           );



    }

   // th_simpleTexture = m_texture2d_2;
    uh_simpleTexture  = bgfx::createUniform("s_texColor",  bgfx::UniformType::Sampler);

     //  ss->tx_pointer=bgfx::getInternal(ss->m_texture2d);
    //   auto res=bgfx::overrideInternal(ss->m_texture2d,ss->tx_pointer);

      //bgfx::overrideInternal(m_texture2d,tx_pointer);
     // printf("pointer %d \n",tx_pointer);


    s_texColor  = bgfx::createUniform("s_texColor", bgfx::UniformType::Sampler);
    s_texNormal  = bgfx::createUniform("s_texNormal", bgfx::UniformType::Sampler);
    s_texCube   = bgfx::createUniform("s_texCube",  bgfx::UniformType::Sampler);
    s_texHeight   = bgfx::createUniform("s_texHeight",  bgfx::UniformType::Sampler);
/*
       for(uint32_t ii = 0; ii<BX_COUNTOF(m_textureCube ); ++ii)
       {
             m_textureCube[ii].idx = bgfx::kInvalidHandle;
       }

       m_textureCube[0] = bgfx::createTextureCube(
             kTextureSide
           , false
           , 1
           , bgfx::TextureFormat::BGRA8
           , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
           );

       m_textureCubeSimple.idx = bgfx::kInvalidHandle;
       m_textureCubeSimple= bgfx::createTextureCube(
                   kTextureSide
                 , false
                 , 1
                 , bgfx::TextureFormat::BGRA8
                 , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                 );

*/
       m_texture2d_0 = bgfx::createTexture2D(
                     kTextureSide
                   , kTextureSide
                   , false
                   , 1
                   , bgfx::TextureFormat::BGRA8
                   , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                   );

       m_texture2d_1 = bgfx::createTexture2D(
                     kTextureSide
                   , kTextureSide
                   , false
                   , 1
                   , bgfx::TextureFormat::BGRA8
                   , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                   );


       for(uint16_t ii = 0; ii<6; ++ii)
           m_texture2dArrayPreview[ii].idx= bgfx::kInvalidHandle;

       for(uint16_t ii = 0; ii<6; ++ii)
       {
           //bgfx::TextureHandle texture2d_temp
            m_texture2dArrayPreview[ii]       = bgfx::createTexture2D(
                                                     kTextureSide
                                                   , kTextureSide
                                                   , false
                                                   , 1
                                                   , bgfx::TextureFormat::BGRA8
                                                   , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                                                   );


             //m_texture2dList.push_back(texture2d_temp);
       }


       for(uint16_t ii = 0; ii<6; ++ii)
           m_texture2dArrayPreviewVel[ii].idx= bgfx::kInvalidHandle;

       for(uint16_t ii = 0; ii<6; ++ii)
       {
           //bgfx::TextureHandle texture2d_temp
            m_texture2dArrayPreviewVel[ii]       = bgfx::createTexture2D(
                                                     kTextureSide
                                                   , kTextureSide
                                                   , false
                                                   , 1
                                                   , bgfx::TextureFormat::BGRA8
                                                   , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                                                   );


             //m_texture2dList.push_back(texture2d_temp);
       }

       for(uint16_t ii = 0; ii<6; ++ii)
           m_texture2dArrayPreviewRho[ii].idx= bgfx::kInvalidHandle;

       for(uint16_t ii = 0; ii<6; ++ii)
       {
           //bgfx::TextureHandle texture2d_temp
            m_texture2dArrayPreviewRho[ii]       = bgfx::createTexture2D(
                                                     kTextureSide
                                                   , kTextureSide
                                                   , false
                                                   , 1
                                                   , bgfx::TextureFormat::BGRA8
                                                   , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                                                   );


             //m_texture2dList.push_back(texture2d_temp);
       }

       for(uint16_t ii = 0; ii<6; ++ii)
           m_texture2dArray[ii].idx= bgfx::kInvalidHandle;

       for(uint16_t ii = 0; ii<6; ++ii)
       {
           //bgfx::TextureHandle texture2d_temp
            m_texture2dArray[ii]       = bgfx::createTexture2D(
                                                     kTextureSide
                                                   , kTextureSide
                                                   , false
                                                   , 1
                                                   , bgfx::TextureFormat::BGRA8
                                                   , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                                                   );


             //m_texture2dList.push_back(texture2d_temp);
       }

       for(uint16_t ii = 0; ii<6; ++ii)
           m_texture2dArrayHeight[ii].idx= bgfx::kInvalidHandle;

       for(uint16_t ii = 0; ii<6; ++ii)
       {
           //bgfx::TextureHandle texture2d_temp
            m_texture2dArrayHeight[ii]       = bgfx::createTexture2D(
                                                     kMeshSide
                                                   , kMeshSide
                                                   , false
                                                   , 1
                                                   , bgfx::TextureFormat::BGRA8
                                                   , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                                                   );


             //m_texture2dList.push_back(texture2d_temp);
       }

       for(uint16_t ii = 0; ii<6; ++ii)
           m_texture2dArrayNormals[ii].idx= bgfx::kInvalidHandle;

       for(uint16_t ii = 0; ii<6; ++ii)
       {
           //bgfx::TextureHandle texture2d_temp
            m_texture2dArrayNormals[ii]       = bgfx::createTexture2D(
                                                     kTextureSide
                                                   , kTextureSide
                                                   , false
                                                   , 1
                                                   , bgfx::TextureFormat::BGRA8
                                                   , BGFX_SAMPLER_MIN_POINT|BGFX_SAMPLER_MAG_POINT|BGFX_SAMPLER_MIP_POINT
                                                   );


             //m_texture2dList.push_back(texture2d_temp);
       }




       m_texture2dData = (uint8_t*)malloc(kTextureSide*kTextureSide*4);
       m_texture2dData0 = (uint8_t*)malloc(kTextureSide*kTextureSide*4);
       m_texture2dData1 = (uint8_t*)malloc(kTextureSide*kTextureSide*4);

       m_rr = m_rng.gen()%255;
       m_gg = m_rng.gen()%255;
       m_bb = m_rng.gen()%255;


       for(int icolor=0; icolor<255;icolor++)
       {

           int baseColorR=(int)(145*(((float) rand() / (RAND_MAX))));
           int baseColorG=(int)(145*(((float) rand() / (RAND_MAX))));
           int baseColorB=(int)(145*(((float) rand() / (RAND_MAX))));

       }

       //---------------------------------------CubeMapLib--------------------------------------------
       hmaps8v.resize(kSize*kSize*6);
       for(int t=0; t<6; t++)
       {
            //CreateTestHMap8(kSize, hmaps8[t], t);

           int w = kSize;
           int h = kSize;

           float s = bx::kPi * (1.0f / kSize);

           for (uint16_t yy = 0; yy < kSize; yy++)
           {
               for (uint16_t xx = 0; xx < kSize; xx++)
               {
                   uint16_t dstSpan =kSize* yy + xx;

                   int dstTemp=(uint8_t)(170.0f * (sqr(sinf(s * xx)) * sqr(sinf(s * yy))));
                   hmaps8[t][dstSpan]=dstTemp;//dstTemp;//dstTemp;
                   hmaps8v[kSize*kSize*t+kSize* yy + xx]=dstTemp;


               }
           }
       }


       for(int t=0; t<6;t++)
       {
            CreateTestHMap16(kSize, hmaps[t], 1, m_timeScale);
       }


       for(int t=0; t<6;t++)
       {
            CreateNormalMap(kSize, kH / kR, hmaps[t], nmaps[t]);
       }
       for(int t=0; t<6;t++)
       {
            CreateSphereNormalMap(kSize, kR, kH, hmaps[t], nmaps[t]);
       }


      // memset(nmapsData, 0x80, sizeof(nmapsData));

      // uint8_t*   hmaps8[6] = { hmap, hmap, hmap, hmap, hmap, hmap };
      // uint16_t*  hmaps[6] = { hmap16, hmap16, hmap16, hmap16, hmap16, hmap16 };
      // cPixel4U8* nmaps[6] = { nmapsData[0],  nmapsData[1], nmapsData[2],  nmapsData[3],  nmapsData[4], nmapsData[5] };


       const uint16_t tw = kTextureSide;
       const uint16_t th = kTextureSide;
       int counter=0;

       for(int t=0; t<6;t++)
       {
           for (uint16_t yy = 0; yy < kSize; yy++)
           {
               for (uint16_t xx = 0; xx < kSize; xx++)
               {
                   htex8[t][counter]=0;
                   htex8_new[t][counter]=0;
                   counter++;
               }
           }
       }





}

int AppIs::shutdown()
{

    ImGui::SaveIniSettingsToDisk("tempImgui.ini");
    imguiDestroy();

    // Cleanup.
    //bgfx::destroy(m_ibh);
    //bgfx::destroy(m_vbh);

    bgfx::destroy(m_vbh_cyl);
    bgfx::destroy(m_ibh_cyl);

    //bgfx::destroy(m_vbh_cyl_s);
    //bgfx::destroy(m_ibh_cyl_s);

    //bgfx::destroy(instanceBuffer);
    //bgfx::destroy(instanceBufferLink);
    //bgfx::destroy(instanceBufferLink2);

    bgfx::destroy(m_program);
    bgfx::destroy(m_texture2d);
    bgfx::destroy(u_lightPosRadius);
    bgfx::destroy(u_lightRgbInnerR);


    bgfx::destroy(instanceBuffer);
    // Shutdown bgfx.
    bgfx::shutdown();
}

bool AppIs::update(bool tex_update, uintptr_t _tx_pointer[6], Environment *env )
{

    if(tex_update){
        //auto tx_pointer2=bgfx::getInternal(m_texture2d_2);
       // auto res1=bgfx::overrideInternal(m_texture2d,tx_pointer2);

        //auto tx_pointer1=bgfx::getInternal(m_texture2d);
        auto res2=bgfx::overrideInternal(m_texture2d_2, _tx_pointer[0]);

        for(int i=0; i<6; i++)
        {
           auto res1=bgfx::overrideInternal(m_texture2dArray[i], _tx_pointer[i]);
        }

    }

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

                ImGui::Text("number of particles %i", env->numberParticles);
                if(maxBuffer)
                     ImGui::Text("maxBuffer Limit over");
                if(ImGui::Button("Step Simulation"))
                {
                   env->stepClick=true;
                }
                ImGui::Checkbox("Start-Pause simulation", &env->runSimulation);

                if(ImGui::Button("Reset simulation"))
                {
                   env->resetSimulation=true;
                }

               // needUpdate_m_textureHeightMap_Normals=false;

                needUpdate_m_textureHeightMap_Normals+= ImGui::SliderFloat("Height scale", &m_timeScale, 0.0f, 1.0f);
                ImGui::SliderFloat("U_deform", &p_deform, 0.0f, 1.0f);


                ImGui::SliderFloat("FL_grad ", &env->parameters->FL_grad, -180.0f , 180.0f);
                ImGui::SliderFloat("FR_grad ", &env->parameters->FR_grad , -180.0f , 180.0f);
                ImGui::SliderFloat("RA_grad ", &env->parameters->RA_grad, -180.0f , 180.0f);

                env->parameters->FL = (env->parameters->FL_grad *   (float)(M_PI)) / 180.0f; // FL sensor angle from forward position (degrees, 22.5 - 45)
                env->parameters->FR = (env->parameters->FR_grad *   (float)M_PI) / 180.0f; // FR sensor angle from forward position (degrees, 22.5 - 45)
                env->parameters->RA = (env->parameters->RA_grad *   (float)M_PI) / 180.0f; // Agent rotation angle (degrees)

                ImGui::SliderFloat("Sensor offset distance (pixels)", &env->parameters->SO, 0.3f , 27.0f);
                ImGui::SliderFloat("Step moves per step (pixels)", &env->parameters->SS, 0.15f , 15.5f);
                ImGui::SliderFloat("Chemoattractant deposition per step ", &env->parameters->depT, 0.1f , 35.0f);
                ImGui::SliderFloat("diffusion decay factor ", &env->parameters->decayT, 0.001f , 0.3f);



                ImGui::Checkbox("Show Particles", &showParticles);
                ImGui::Checkbox("Show Sensors", &showSensors);
                ImGui::Checkbox("Show Trails", &showTrails);
                ImGui::Checkbox("Show DebugGrid", &showDebugGrid);


                ImGui::End();

                ImGui::Begin("Preview host texture 2 temp");
                         ImGui::Image(m_texture2d_2, ImVec2(sim_width,sim_height));
                ImGui::End();


                bgfx::setUniform(u_deform, &p_deform);

                ImGui::Begin("Image Show texture2dArray");
                for(int ii=0; ii<6;ii++)
                {
                    ImGui::Image(m_texture2dArray[ii], ImVec2(kTextureSide/4,kTextureSide/4));
                    ImGui::SameLine();
                }

                ImGui::End();
                ImGui::Begin("Image Show texture2dArrayHeight");
                for(int ii=0; ii<6;ii++)
                {
                    ImGui::Image(m_texture2dArrayHeight[ii], ImVec2(kTextureSide/4,kTextureSide/4));
                    ImGui::SameLine();
                }

                ImGui::End();

                ImGui::Begin("Image Show texture2dArrayNormals");
                for(int ii=0; ii<6;ii++)
                {
                    ImGui::Image(m_texture2dArrayNormals[ii], ImVec2(kTextureSide/4,kTextureSide/4));
                    ImGui::SameLine();
                }

                ImGui::End();



                //where 0 is +X, 1 is -X, 2 is +Y, 3 is -Y, 4 is +Z, and 5 is -Z.


                  ///
                  ///                  +----------+
                  ///                  |-z       4|
                  ///                  | ^  +y    |
                  ///                  | |        |    Unfolded cube:
                  ///                  | +---->+x |
                  ///       +----------+----------+----------+----------+
                  ///       |+y       3|+y       0|+y       2|+y       1|
                  ///       | ^  -x    | ^  +z    | ^  +x    | ^  -z    |
                  ///       | |        | |        | |        | |        |
                  ///       | +---->+z | +---->+x | +---->-z | +---->-x |
                  ///       +----------+----------+----------+----------+
                  ///                  |+z       5|
                  ///                  | ^  -y    |
                  ///                  | |        |
                  ///                  | +---->+x |
                  ///                  +----------+


                int texSize=kTextureSide;
                ImGui::Begin("Velocity Show Unfolded cube");
                    ImGui::Image(m_texture2d_1, ImVec2(texSize,texSize));
                    ImGui::SameLine();
                    ImGui::Image(m_texture2dArray[4], ImVec2(texSize,texSize), ImVec2(0.0,1.0),ImVec2(1.0,0.0));
                    ImGui::Image(m_texture2dArray[3], ImVec2(texSize,texSize), ImVec2(0.0,1.0),ImVec2(1.0,0.0));
                    ImGui::SameLine();
                    ImGui::Image(m_texture2dArray[0], ImVec2(texSize,texSize), ImVec2(0.0,1.0),ImVec2(1.0,0.0));
                    ImGui::SameLine();
                    ImGui::Image(m_texture2dArray[2], ImVec2(texSize,texSize), ImVec2(0.0,1.0),ImVec2(1.0,0.0));
                    ImGui::SameLine();
                    ImGui::Image(m_texture2dArray[1], ImVec2(texSize,texSize), ImVec2(0.0,1.0),ImVec2(1.0,0.0));
                    ImGui::Image(m_texture2d_1, ImVec2(texSize,texSize));
                    ImGui::SameLine();
                    ImGui::Image(m_texture2dArray[5], ImVec2(texSize,texSize), ImVec2(0.0,1.0),ImVec2(1.0,0.0));
                ImGui::End();

                imguiEndFrame();

              //  if(needUpdate_m_textureHeightMap_Normals or needUpdate_m_textureHeightMap_Normals_FS)
              if(needUpdate_m_textureHeightMap_Normals)
              {
                    for(int t=0; t<6;t++)
                    {
                         CreateTestHMap8(kMeshSide, hmaps8[t], meshSize, m_timeScale);
                        // CreateTestHMap8(kSize, htex8[t], meshSize);
                         CreateTestHMap16(kSize, hmaps[t], meshSize, m_timeScale);
                         CreateSphereNormalMap(kSize, kR, kH, hmaps[t], nmaps[t]);
                    }


               }





              if(needUpdate_m_textureHeightMap_Normals)
              {
                  UpdateFaceNormalMap(kR, kH, kSize, hmaps, nmaps, kFace_PZ);
                  UpdateFaceNormalMap(kR, kH, kSize, hmaps, nmaps, kFace_NZ);

                  UpdateFaceNormalMap(kR, kH, kSize, hmaps, nmaps, kFace_PX);
                  UpdateFaceNormalMap(kR, kH, kSize, hmaps, nmaps, kFace_NX);

                  UpdateFaceNormalMap(kR, kH, kSize, hmaps, nmaps, kFace_PY);
                  UpdateFaceNormalMap(kR, kH, kSize, hmaps, nmaps, kFace_NY);


              }

                vector<int> orderFaces {0,1,2,3,4,5};
                //where 0 is +X, 1 is -X, 2 is +Y, 3 is -Y, 4 is +Z, and 5 is -Z.


                  ///
                  ///                  +----------+
                  ///                  |-z       4|
                  ///                  | ^  +y    |
                  ///                  | |        |    Unfolded cube:
                  ///                  | +---->+x |
                  ///       +----------+----------+----------+----------+
                  ///       |+y       3|+y       0|+y       2|+y       1|
                  ///       | ^  -x    | ^  +z    | ^  +x    | ^  -z    |
                  ///       | |        | |        | |        | |        |
                  ///       | +---->+z | +---->+x | +---->-z | +---->-x |
                  ///       +----------+----------+----------+----------+
                  ///                  |+z       5|
                  ///                  | ^  -y    |
                  ///                  | |        |
                  ///                  | +---->+x |
                  ///                  +----------+
                  ///
                  ///   kFace_PZ,   ///< (u, v) = ( x, y)
                  ///   kFace_NZ,   ///< (u, v) = (-x, y)
                  ///   kFace_PX,   ///< (u, v) = ( y, z)
                  ///   kFace_NX,   ///< (u, v) = (-y, z)
                  ///   kFace_PY,   ///< (u, v) = ( z, x)
                  ///   kFace_NY,   ///< (u, v) = (-z, x)



                //---------------------------------Textures HeightMap-----------------------------------------------------------------//

                if(needUpdate_m_textureHeightMap_Normals or needUpdate_m_textureHeightMap_Normals_FS)
                {

                   needUpdate_m_textureHeightMap_Normals=false;
                   needUpdate_m_textureHeightMap_Normals_FS=false;

                for(int t=0; t<6;t++)
                {

                    const uint32_t pitch = kMeshSide*4;
                    const uint16_t tw = kMeshSide;
                    const uint16_t th = kMeshSide;
                    const uint16_t tx = 0;
                    const uint16_t ty = 0;

                    bgfx::TextureInfo ti2;
                    bgfx::calcTextureSize(ti2, kMeshSide, kMeshSide,0, false, false, 1, bgfx::TextureFormat::BGRA8);

                   // const bgfx::Memory* mem = bgfx::alloc(ti.storageSize);
                    mem2 = bgfx::alloc(ti2.storageSize);
                    uint8_t* data = (uint8_t*)mem2->data;

                    if(t==0) //+x
                    {
                        for (int32_t yy = th-1; yy >=0; yy--)
                        {
                            for (uint32_t xx = 0; xx < tw; xx++)
                            {
                                    data[0] = 0.0;
                                    data[1] = hmaps8[t][yy*kMeshSide+xx];
                                    data[2] = 0.0;
                                    data[3] = 0xff;
                                    data+= 4;
                            }
                        }


                    }
                    if(t==1) //-x
                    {

                        for (uint32_t yy = 0; yy < th; yy++)
                        {
                            for (int32_t xx = tw-1; xx >=0; xx--)
                            {
                                data[0] = 0.0;
                                data[1] =  hmaps8[t][yy*kMeshSide+xx];
                                data[2] = 0.0;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }

                    if(t==2) //+y
                    {

                        for (int32_t yy = th-1; yy >=0; yy--)
                        {
                            for (int32_t xx = tw-1; xx >=0; xx--)
                            {
                                data[0] = 0.0;
                                data[1] =  hmaps8[t][yy*kMeshSide+xx];
                                data[2] = 0.0;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }
                    if(t==3) //-y
                    {
                        for (int32_t yy = th-1; yy >=0; yy--)
                        {
                            for (int32_t xx = tw-1; xx >=0; xx--)
                            {
                                data[0] = 0.0;
                                data[1] = hmaps8[t][yy*kMeshSide+xx];
                                data[2] = 0.0;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }
                    if(t==4) //+z
                    {
                        for (uint32_t xx = 0; xx < tw; xx++)
                        {
                            for (uint32_t yy = 0; yy < th; yy++)
                            {
                                data[0] = 0.0;
                                data[1] =  hmaps8[t][yy*kMeshSide+xx];
                                data[2] = 0.0;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }
                    if(t==5) //-z
                    {
                        for (int32_t yy = th-1; yy >=0; yy--)
                        {
                            for (int32_t xx = tw-1; xx >=0; xx--)
                            {
                                data[0] = 0.0;
                                data[1] =  hmaps8[t][xx*kMeshSide+yy];
                                data[2] = 0.0;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }

                    bgfx::updateTexture2D(m_texture2dArrayHeight[t], 0, 0, tx, ty, tw, th, mem2, pitch );
                }

                //---------------------------------Textures Normal Maps Spherical-----------------------------------------------------------------//

                for(int t=0; t<6;t++)
                {

                    const uint32_t pitch = kTextureSide*4;
                    const uint16_t tw = kTextureSide;
                    const uint16_t th = kTextureSide;
                    const uint16_t tx = 0;
                    const uint16_t ty = 0;

                    bgfx::TextureInfo ti;
                    bgfx::calcTextureSize(ti, kTextureSide, kTextureSide, 1, false, false, 1, bgfx::TextureFormat::BGRA8);

                    //const bgfx::Memory* mem = bgfx::alloc(ti.storageSize);
                    mem = bgfx::alloc(ti.storageSize);
                   // const bgfx::Memory* mem2 = bgfx::copy(mem,ti.storageSize);
                    uint8_t* data = (uint8_t*)mem->data;

                    if(t==0) //+x
                    {
                        for (int32_t yy = th-1; yy >=0; yy--)
                        {
                            for (uint32_t xx = 0; xx < tw; xx++)
                            {
                                    data[0] = nmaps[t][th*yy+xx].mX;
                                    data[1] = nmaps[t][th*yy+xx].mY;
                                    data[2] = nmaps[t][th*yy+xx].mZ;
                                    data[3] = 0xff;
                                    data+= 4;
                            }
                        }


                    }
                    else if(t==1) //-x
                    {
                        for (uint32_t yy = 0; yy < th; yy++)
                        {
                            for (int32_t xx = tw-1; xx >=0; xx--)
                            {
                                data[0] = nmaps[t][th*yy+xx].mX;
                                data[1] = nmaps[t][th*yy+xx].mY;
                                data[2] = nmaps[t][th*yy+xx].mZ;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }

                    else if(t==2) //+y
                    {
                        for (int32_t yy = th-1; yy >=0; yy--)
                        {
                            for (int32_t xx = tw-1; xx >=0; xx--)
                            {

                                 data[0] = nmaps[t][th*yy+xx].mX;
                                 data[1] = nmaps[t][th*yy+xx].mY;
                                 data[2] = nmaps[t][th*yy+xx].mZ;
                                 data[3] = 0xff;
                                 data+= 4;
                            }
                        }
                    }
                    else if(t==3) //-y
                    {

                        for (int32_t yy = th-1; yy >=0; yy--)
                        {
                            for (int32_t xx = tw-1; xx >=0; xx--)
                            {

                                data[0] = nmaps[t][th*yy+xx].mX;
                                data[1] = nmaps[t][th*yy+xx].mY;
                                data[2] = nmaps[t][th*yy+xx].mZ;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }
                    else if(t==4) //+z
                    {
                        for (uint32_t xx = 0; xx < tw; xx++)
                        {
                            for (uint32_t yy = 0; yy < th; yy++)
                            {
                                data[0] = nmaps[t][th*yy+xx].mX;
                                data[1] = nmaps[t][th*yy+xx].mY;
                                data[2] = nmaps[t][th*yy+xx].mZ;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }
                    else if(t==5) //-z
                    {
                        for (int32_t yy = th-1; yy >=0; yy--)
                        {
                            for (int32_t xx = tw-1; xx >=0; xx--)
                            {
                                data[0] = nmaps[t][th*xx+yy].mX;
                                data[1] = nmaps[t][th*xx+yy].mY;
                                data[2] = nmaps[t][th*xx+yy].mZ;
                                data[3] = 0xff;
                                data+= 4;
                            }
                        }
                    }

                    bgfx::updateTexture2D(m_texture2dArrayNormals[t], 0, 0, tx, ty, tw, th, mem, pitch);
                    //free(mem);
                    }
            }

          }

           needUpdate_m_textureHeightMap_Normals=false;
          {

              int index=0;

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


                DebugDrawEncoder dde;

                dde.begin(0);
                dde.drawAxis(-0.0f, 0.0f, -0.0f,15.0f);
                // dde.drawAxis(0.0f, 0.0f, 0.0f,5);
                if(showDebugGrid)
                {
                   const bx::Vec3 normal = { 0.0f,  1.0f, 0.0f };
                   const bx::Vec3 pos    = { 150.0f, 0.0f, 150.0f };

                   dde.drawGrid(Axis::Y, pos, 30, 10.0f);
                }
/*
                dde.push();
                    Aabb aabb =
                    {
                        { 0.0f, 1.0f, 0.0f },
                        { (float)sim_width, 1.0f, (float)sim_height },
                    };
                    dde.setWireframe(true);
                    dde.setColor( 0xff00ff00);
                    dde.draw(aabb);
               dde.pop();
*/
               // tx_pointer=bgfx::getInternal(m_texture2d);
              //  if(tex_update)
                   // auto res=bgfx::overrideInternal(m_texture2d,_tx_pointer);




                if(showParticles)
                {
                    DrawBots(env);
                }
                if(showTrails)
                {
                      float mtx[16];
                       //  bx::Quaternion a=rotateAxis(bx::Vec3(1,0,0), bx::toRad(180));
                       //  bx::Quaternion b=rotateAxis(bx::Vec3(0,1,0), bx::toRad(0));
                       //  auto aa=bx::toEuler(bx::mul(a,b));
                        // auto aa=bx::toEuler(a);
                      bx::mtxSRT(mtx,
                                   sim_width/1.0f,0.0f,sim_width/1.0f,
                                   0.0f,0.0f,0.0f,
                                   sim_width,-1.0f,sim_height
                                   );

                      bgfx::setTransform(mtx);
                      // Set vertex and index buffer.
                      bgfx::setVertexBuffer(0, m_vbh_sq);
                      bgfx::setIndexBuffer(m_ibh_sq);

                      // Set render states.
                      //bgfx::setState(BGFX_STATE_DEFAULT);

                      bgfx::setTexture(0, uh_simpleTexture ,   m_texture2dArray[0]);

                    //  tx_pointer=getInternal(m_texture2d); //(uintptr_t)



                      bgfx::setState(BGFX_STATE_DEFAULT);
                      // Submit primitive for rendering to view 0.
                      bgfx::submit(0, m_program_face);
               }

// First cubemap face
                bx::Vec3 pointsSphere=bx::Vec3(sim_height/2,sim_height/2,sim_height/2);
                bx::Vec3 pos(0+sim_height/2,0,0+sim_width/2);
                {
                     float mtx[16];

                     bx::Quaternion a=rotateAxis(bx::Vec3(1,0,0), bx::toRad(0));
                     auto aa=bx::toEuler(a);

                     bx::mtxSRT(mtx,
                               pointsSphere.x,pointsSphere.y , pointsSphere.z,
                               aa.x,aa.y,aa.z,
                               pos.x,pos.y,pos.z
                               );


                  bgfx::setTransform(mtx);
                  // Set vertex and index buffer.
                  bgfx::setVertexBuffer(0, m_vbh_sq);
                  bgfx::setIndexBuffer(m_ibh_sq);

                  // Set render states.
                  //bgfx::setState(BGFX_STATE_DEFAULT);

                  bgfx::setTexture(0, uh_simpleTexture ,  m_texture2dArray[0]);
                  bgfx::setTexture(1, s_texNormal, m_texture2dArrayNormals[4]);
                  bgfx::setTexture(2, s_texHeight, m_texture2dArrayHeight[4] );



                  bgfx::setState(BGFX_STATE_DEFAULT);
                  // Submit primitive for rendering to view 0.
                  bgfx::submit(0, m_program_face_cube);

                }


      // -Y 5

                {
                    float mtx[16];
                     bx::Quaternion a=rotateAxis(bx::Vec3(1,0,0), bx::toRad(180));
                     bx::Quaternion b=rotateAxis(bx::Vec3(0,1,0), bx::toRad(180));
                     auto aa=bx::toEuler(bx::mul(a,b));
                    // auto aa=bx::toEuler(a);
                     bx::mtxSRT(mtx,
                               pointsSphere.x,pointsSphere.y , pointsSphere.z,
                               aa.x,aa.y,aa.z,
                               pos.x,pos.y,pos.z
                               );

                  bgfx::setTransform(mtx);
                  // Set vertex and index buffer.
                  bgfx::setVertexBuffer(0, m_vbh_sq);
                  bgfx::setIndexBuffer(m_ibh_sq);

                  // Set render states.
                  //bgfx::setState(BGFX_STATE_DEFAULT);

                  bgfx::setTexture(0, uh_simpleTexture ,  m_texture2dArray[1]);
                  bgfx::setTexture(1, s_texNormal, m_texture2dArrayNormals[5]);
                  bgfx::setTexture(2, s_texHeight, m_texture2dArrayHeight[5] );




                  bgfx::setState(BGFX_STATE_DEFAULT);
                  // Submit primitive for rendering to view 0.
                  bgfx::submit(0, m_program_face_cube);

                }

                // -X 3
                          {
                              float mtx[16];
                              bx::Quaternion a=rotateAxis(bx::Vec3(0,0,1), bx::toRad(90));
                             //  bx::Quaternion a=rotateAxis(bx::Vec3(1,0,0), bx::toRad(90));
                               auto aa=bx::toEuler(a);
                               bx::mtxSRT(mtx,
                                         pointsSphere.x,pointsSphere.y , pointsSphere.z,
                                         aa.x,aa.y,aa.z,
                                         pos.x,pos.y,pos.z
                                         );

                            bgfx::setTransform(mtx);
                            // Set vertex and index buffer.
                            bgfx::setVertexBuffer(0, m_vbh_sq);
                            bgfx::setIndexBuffer(m_ibh_sq);

                            // Set render states.
                            //bgfx::setState(BGFX_STATE_DEFAULT);

                            bgfx::setTexture(0, uh_simpleTexture ,  m_texture2dArray[2]);
                            bgfx::setTexture(1, s_texNormal, m_texture2dArrayNormals[2]);
                            bgfx::setTexture(2, s_texHeight, m_texture2dArrayHeight[2] );


                            bgfx::setState(BGFX_STATE_DEFAULT);
                            // Submit primitive for rendering to view 0.
                            bgfx::submit(0, m_program_face_cube);

                          }

                // +X 2
                          {
                              float mtx[16];
                              bx::Quaternion a=rotateAxis(bx::Vec3(0,0,1), bx::toRad(-90));
                              //a=mul(rotateAxis(bx::Vec3(0,0,1), bx::toRad(0)),a);
                              auto aa=bx::toEuler(a);
                              bx::mtxSRT(mtx,
                                         pointsSphere.x,pointsSphere.y , pointsSphere.z,
                                         aa.x,aa.y,aa.z,
                                         pos.x,pos.y,pos.z
                                         );

                            bgfx::setTransform(mtx);
                            // Set vertex and index buffer.
                            bgfx::setVertexBuffer(0, m_vbh_sq);
                            bgfx::setIndexBuffer(m_ibh_sq);

                            // Set render states.
                            //bgfx::setState(BGFX_STATE_DEFAULT);

                            bgfx::setTexture(0, uh_simpleTexture ,  m_texture2dArray[3]);
                            bgfx::setTexture(1, s_texNormal, m_texture2dArrayNormals[3]);
                            bgfx::setTexture(2, s_texHeight, m_texture2dArrayHeight[3] );


                            bgfx::setState(BGFX_STATE_DEFAULT);
                            // Submit primitive for rendering to view 0.
                            bgfx::submit(0, m_program_face_cube);

                          }
                // -Z 4

                          {
                              float mtx[16];
                               bx::Quaternion a=rotateAxis(bx::Vec3(1,0,0), bx::toRad(90));

                               auto aa=bx::toEuler(a);
                               bx::mtxSRT(mtx,
                                         pointsSphere.x,pointsSphere.y , pointsSphere.z,
                                         aa.x,aa.y,aa.z,
                                         pos.x,pos.y,pos.z
                                         );

                            bgfx::setTransform(mtx);
                            // Set vertex and index buffer.
                            bgfx::setVertexBuffer(0, m_vbh_sq);
                            bgfx::setIndexBuffer(m_ibh_sq);

                            // Set render states.
                            //bgfx::setState(BGFX_STATE_DEFAULT);

                            bgfx::setTexture(0, uh_simpleTexture ,  m_texture2dArray[5]);
                            bgfx::setTexture(1, s_texNormal, m_texture2dArrayNormals[1]);
                            bgfx::setTexture(2, s_texHeight, m_texture2dArrayHeight[1] );



                            bgfx::setState(BGFX_STATE_DEFAULT);
                            // Submit primitive for rendering to view 0.
                            bgfx::submit(0, m_program_face_cube);

                          }

                // +Z 0
                          {
                              float mtx[16];
                              bx::Quaternion a=rotateAxis(bx::Vec3(1,0,0), bx::toRad(-90));
                               auto aa=bx::toEuler(a);
                               bx::mtxSRT(mtx,
                                         pointsSphere.x,pointsSphere.y , pointsSphere.z,
                                         aa.x,aa.y,aa.z,
                                         pos.x,pos.y,pos.z
                                         );

                            bgfx::setTransform(mtx);
                            // Set vertex and index buffer.
                            bgfx::setVertexBuffer(0, m_vbh_sq);
                            bgfx::setIndexBuffer(m_ibh_sq);

                            // Set render states.
                            //bgfx::setState(BGFX_STATE_DEFAULT);

                            bgfx::setTexture(0, uh_simpleTexture ,  m_texture2dArray[4]);
                            bgfx::setTexture(1, s_texNormal, m_texture2dArrayNormals[0]);
                            bgfx::setTexture(2, s_texHeight, m_texture2dArrayHeight[0]);


                            bgfx::setState(BGFX_STATE_DEFAULT);
                            // Submit primitive for rendering to view 0.
                            bgfx::submit(0, m_program_face_cube);

                          }



                bgfx::frame();
                dde.end();

    }

     return false;
}

void AppIs::DrawBots(Environment *env)
{
    {

        int numberOfSpheres=0;
        int numberOfCylinders=0;

            if(env->numberParticles==0)
                numberOfSpheres=0;
            else
            numberOfSpheres=env->numberParticles;


        if(numberOfSpheres>0)
        {

                // env_dataTO->particles=(ParticleAccessTO*)malloc(*env_dataTO->numParticles*sizeof(ParticleAccessTO));

                int deltaCount=25000;

                int numberCycles=(int)(trunc(numberOfSpheres/deltaCount));

                int secondNumbers=numberOfSpheres-deltaCount*numberCycles;

                int numberDelta=deltaCount;


            for(int ibuf=0; ibuf<numberCycles+1; ibuf++)
            {

                // 80 bytes stride = 64 bytes for 4x4 matrix + 16 bytes for RGBA color.
                const uint16_t instanceStride = 80;
                // number of instances


                if(ibuf==numberCycles)
                    numberDelta=secondNumbers;
                const uint32_t numInstances= numberDelta;
                auto instanceData=new InstanceLayout[numInstances];

                int index=0;

                    for(int i=ibuf*deltaCount;i<ibuf*deltaCount+numberDelta;i++)
                    {
                            float mtx[16];

                            int cR=115;
                            int cG=215;
                            int cB=115+2*(env->particles[i].id %30);

                            float particleRadius=env->particles[i].radius/2.0f;
                            bx::mtxSRT(mtx,
                                        particleRadius,
                                        particleRadius,
                                        particleRadius,
                                        0, env->particles[i].direction,  0,
                                        env->particles[i].relPos.x,0.1f,env->particles[i].relPos.y
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
                bgfx::update(instanceBufferPtr[ibuf], 0, mem);
                // Set vertex and index buffer.
                bgfx::setVertexBuffer(0, m_vbh_cyl);
                bgfx::setIndexBuffer(m_ibh_cyl);
                // Set instance data buffer.
                bgfx::setInstanceDataBuffer(instanceBufferPtr[ibuf],0,numInstances);
                // Set render states.
                bgfx::setState(BGFX_STATE_DEFAULT);
                // Submit primitive for rendering to view 0.
                bgfx::submit(0, m_program);
                free(instanceData);



            }
        }


    }
}


void AppIs::WrapCubeFace(int size, int face, int x, int v, int* faceOut, int* xOut, int* yOut)
{
    *faceOut = face;
    *xOut = x;
    *yOut = v;

    CML::WrapCubeFace(size, faceOut, xOut, yOut);
}



void AppIs::updatetexturePreview(unsigned char* dataColor, int preview)
{

    for(int t=0; t<6;t++)
    {

        const uint32_t pitch = kTextureSide*4;
        const uint16_t tw = kTextureSide;
        const uint16_t th = kTextureSide;
        const uint16_t tx = 0;
        const uint16_t ty = 0;

        bgfx::TextureInfo ti;
        bgfx::calcTextureSize(ti, kTextureSide, kTextureSide, 1, false, false, 1, bgfx::TextureFormat::BGRA8);

        mem = bgfx::alloc(ti.storageSize);

        uint8_t* data = (uint8_t*)mem->data;

        int dataColorIndex=0;

        if(t==0) //+z
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;
            for (int32_t yy = th-1; yy >=0; yy--)
            {
                for (uint32_t xx = 0; xx < tw; xx++)
                {

                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;
                }
            }


        }
        else if(t==1) //-z
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;

            for (int32_t yy = th-1; yy >=0; yy--)
             {
            for (uint32_t xx = 0; xx < tw; xx++)
            {



                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;
                }
            }
        }

        else if(t==2) //+x
        {

            dataColorIndex=4*t*kTextureSide*kTextureSide;

            for (int32_t xx = tw-1; xx >=0; xx--)
            {
                for (int32_t yy = th-1; yy >=0; yy--)
                {

                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;

                }
            }
        }
        else if(t==3) //-x
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;
            for (uint32_t xx = 0; xx < tw; xx++)
            {
                for (uint32_t yy = 0; yy < th; yy++)
                {
                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;

                }
            }
        }
        else if(t==4) //+y
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;

            for (uint32_t xx = 0; xx < tw; xx++)
            {
               for (uint32_t yy = 0; yy < th; yy++)
               {

                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;

                }
            }
        }
        else if(t==5) //-y
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;
            for (uint32_t xx = 0; xx < tw; xx++)
            {
                for (uint32_t yy = 0; yy < th; yy++)
                {
                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;

                }
            }
        }



        if(preview==0)
             bgfx::updateTexture2D(m_texture2dArrayPreviewVel[t], 0, 0, tx, ty, tw, th, mem, pitch);
        else
            bgfx::updateTexture2D(m_texture2dArrayPreviewRho[t], 0, 0, tx, ty, tw, th, mem, pitch);
     }
}

void AppIs::updatetextureTexture(unsigned char* dataColor, int preview)
{

    for(int t=0; t<6;t++)
    {

        const uint32_t pitch = kTextureSide*4;
        const uint16_t tw = kTextureSide;
        const uint16_t th = kTextureSide;
        const uint16_t tx = 0;
        const uint16_t ty = 0;

        bgfx::TextureInfo ti;
        bgfx::calcTextureSize(ti, kTextureSide, kTextureSide, 1, false, false, 1, bgfx::TextureFormat::BGRA8);

        mem = bgfx::alloc(ti.storageSize);

        uint8_t* data = (uint8_t*)mem->data;

        int dataColorIndex=0;

        if(t==0) //+z
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;
            for (int32_t yy = th-1; yy >=0; yy--)
            {
                for (int32_t xx = tw-1; xx >=0; xx--)
                {

                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;
                }
            }


        }
        else if(t==1) //-z
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;

            for (uint32_t yy = 0; yy < th; yy++)
             {
            for (uint32_t xx = 0; xx < tw; xx++)
            {

                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;
                }
            }
        }

        else if(t==2) //+x
        {

            dataColorIndex=4*t*kTextureSide*kTextureSide;


            for (uint32_t yy = 0; yy < th; yy++)
            {
                for (uint32_t xx = 0; xx < tw; xx++)
                {

                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;

                }
            }
        }
        else if(t==3) //-x
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;

                for (uint32_t yy = 0; yy < th; yy++)
                {
                    for (uint32_t xx = 0; xx < tw; xx++)
                    {
                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;

                }
            }
        }
        else if(t==4) //+y
        {
           dataColorIndex=4*t*kTextureSide*kTextureSide;

                for (uint32_t xx = 0; xx < tw; xx++)
                {
                    for (int32_t yy = th-1; yy >=0; yy--)
                    {

                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;

                }
            }
        }
        else if(t==5) //-y
        {
            dataColorIndex=4*t*kTextureSide*kTextureSide;
            for (uint32_t xx = 0; xx < tw; xx++)
            {
                for (int32_t yy = th-1; yy >=0; yy--)
                {
                    data[0] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+2];
                    data[1] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+1];
                    data[2] = (uint8_t)dataColor[dataColorIndex+4*yy*kTextureSide+4*xx+0];
                    data[3] =  0xff;

                    data+= 4;

                }
            }
        }



          bgfx::updateTexture2D(m_texture2dArray[t], 0, 0, tx, ty, tw, th, mem, pitch);
     }
}

void    AppIs::LogClear()
{
    Buf.clear();
    LineOffsets.clear();
    LineOffsets.push_back(0);
}



void    AppIs::LogDraw(const char* title)
{
    if (!ImGui::Begin(title))
    {
        ImGui::End();
        return;
    }

    // Options menu
    if (ImGui::BeginPopup("Options"))
    {
        ImGui::Checkbox("Auto-scroll", &AutoScroll);
        ImGui::EndPopup();
    }

    // Main window
    if (ImGui::Button("Options"))
        ImGui::OpenPopup("Options");
    ImGui::SameLine();
    bool clear = ImGui::Button("Clear");
    ImGui::SameLine();
    bool copy = ImGui::Button("Copy");
    ImGui::SameLine();
    Filter.Draw("Filter", -100.0f);

    ImGui::Separator();
    ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    if (clear)
        LogClear();
    if (copy)
        ImGui::LogToClipboard();

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    const char* buf = Buf.begin();
    const char* buf_end = Buf.end();
    if (Filter.IsActive())
    {
        // In this example we don't use the clipper when Filter is enabled.
        // This is because we don't have a random access on the result on our filter.
        // A real application processing logs with ten of thousands of entries may want to store the result of
        // search/filter.. especially if the filtering function is not trivial (e.g. reg-exp).
        for (int line_no = 0; line_no < LineOffsets.Size; line_no++)
        {
            const char* line_start = buf + LineOffsets[line_no];
            const char* line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end;
            if (Filter.PassFilter(line_start, line_end))
                ImGui::TextUnformatted(line_start, line_end);
        }
    }
    else
    {
        // The simplest and easy way to display the entire buffer:
        //   ImGui::TextUnformatted(buf_begin, buf_end);
        // And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward
        // to skip non-visible lines. Here we instead demonstrate using the clipper to only process lines that are
        // within the visible area.
        // If you have tens of thousands of items and their processing cost is non-negligible, coarse clipping them
        // on your side is recommended. Using ImGuiListClipper requires
        // - A) random access into your data
        // - B) items all being the  same height,
        // both of which we can handle since we an array pointing to the beginning of each line of text.
        // When using the filter (in the block of code above) we don't have random access into the data to display
        // anymore, which is why we don't use the clipper. Storing or skimming through the search result would make
        // it possible (and would be recommended if you want to search through tens of thousands of entries).
        ImGuiListClipper clipper;
        clipper.Begin(LineOffsets.Size);
        while (clipper.Step())
        {
            for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
            {
                const char* line_start = buf + LineOffsets[line_no];
                const char* line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end;
                ImGui::TextUnformatted(line_start, line_end);
            }
        }
        clipper.End();
    }
    ImGui::PopStyleVar();

    if (AutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();
    ImGui::End();
}


}
