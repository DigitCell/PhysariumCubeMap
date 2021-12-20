#include "mainloop.hpp"
//#include "common/entry/entry_sdl.cpp"

bool MainLoop::step(entry::Context* _context, int Tick)
{
    cuda_exp.moveCaterpillers(Tick);
    cudaDeviceSynchronize();
    return true;
}

bool MainLoop::ResetSimulation(int Tick)
{

    cuda_exp.psystem->InitializeT(*cuda_exp.env->parameters);
    cuda_exp.InitParticles();

    return true;
}

bool MainLoop::run(entry::Context* _context)
{

    //cuda_exp.env->particles=cuda_exp.psystem->m_hParticle;
    //cuda_exp.env->segments=cuda_exp.psystem->m_hSegment;

    int Tick=0;
    while(!_context->m_exit)
    {
        cuda_exp.updateParams(cuda_exp.env->parameters);
        cuda_exp.env->particles=cuda_exp.psystem->particlesList.getArrayForHost(&cuda_exp.env->numberParticles);
        cuda_exp.env->catList=cuda_exp.psystem->botList.getArrayForHost(&cuda_exp.env->numberCaterpiller);

        if(cuda_exp.env->runSimulation)
        {
           step(_context,Tick);
        }

        if(cuda_exp.env->resetSimulation)
        {
           cuda_exp.env->resetSimulation=false;
           ResetSimulation(Tick);
        }

        if(cuda_exp.env->stepClick)
        {
           step(_context, Tick);
           cuda_exp.env->stepClick=false;
        }


         _context->step_graph(cuda_exp.env);

         free(cuda_exp.env->catList);
         free(cuda_exp.env->particles);

      // cuda_exp.surface2DCuda();

       Tick++;

       if(Tick==1)
       {
           _context->updateTextureFlag=true;
          // _context->tx_pointer[0]=cuda_exp.opengl_tex_cuda;
           for(int i=0; i<6; i++)
           {
             _context->tx_pointer[i]=cuda_exp.opengl_tex_cudaList[i];
           }

       }
       else
       {
           _context->updateTextureFlag=false;
       }

       if(Tick>1 and (cuda_exp.env->runSimulation))
       {
            //cuda_exp.generateCUDAImage(Tick);
            cuda_exp.generateCUDAImageList(Tick);
       }
    }

    _context->quit_graph();

    return true;// entry::shutdownApp(&graphModule);
}
