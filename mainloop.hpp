#ifndef MAINLOOP_HPP
#define MAINLOOP_HPP

#pragma once

#include "common/entry/entry_p.h"
#include "common/entry/entry.h"
#include "common/entry/entry_sdl.h"
//#include "common/entry/entry_sdl.cpp"

#include <chrono>
#include <memory>

#include "cuda_exp.hpp"
#include <cuda_gl_interop.h>

//using namespace entry;

class MainLoop
{
public:

    MainLoop(int _argc, char** _argv, entry::Context* _context):
        cuda_exp(1500,1500*100)
    {


        auto fullscr_flag=_context->m_fullscreen;
        if(fullscr_flag)
            printf("full screen mode \n");
        else
            printf("window mode \n");

         _context->init_graph(_argc, _argv);

         cudaInit(_argc, _argv);
         cuda_exp.init();



    };

    Cuda_exp cuda_exp;
    bool run(entry::Context* _context);
    bool runMainLoop=true;
    bool step(entry::Context* _context, int Tick);
    bool ResetSimulation(int Tick);
};


#endif // MAINLOOP_HPP
