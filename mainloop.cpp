#include "mainloop.hpp"


bool MainLoop::run()
{
    int Tick=0;

    while(runMainLoop)
    {

    //   graphModule.CreateMesh(env);
       bool engineState= entry::stepApp(&graphModule, env);

       if(!engineState)
            runMainLoop=false;

       if(graphModule.stepClick)
       {
           UpdateStep();
           Tick++;
           graphModule.stepClick=false;
       }

       if(graphModule.runSimulation)
       {
            UpdateStep();
            Tick++;
       }

    }

    return entry::shutdownApp(&graphModule);
}
