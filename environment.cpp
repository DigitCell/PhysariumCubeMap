#include "environment.hpp"

Environment::Environment(int _numParticles)
{
    parameters=new SimParams();
    parameters->tickTrailLenth=1;
    parameters->trailDescrease=1;
    parameters->imgw=world_width;
    parameters->imgh=world_height;

    parameters->width=world_width;
    parameters->height=world_height;
    parameters->quartSize=world_width*world_width;
    parameters->faces=6;

    parameters->traceStartColorIndex=250;
    parameters->wormPathAlertColor=245;

    parameters->wormVelocity01=0.75f;
    parameters->numSteps01=10;
    parameters->numDirections01=7;
    parameters->angleDivide01=27.0f;

    parameters->const_NumberParticles=const_MAxNumberParticles;

    parameters->decayT= 0.05f;

    parameters->FL_grad = 22.5f ; // FL sensor angle from forward position (degrees, 22.5 - 45)
    parameters->FR_grad = -22.5f ; // FR sensor angle from forward position (degrees, 22.5 - 45)
    parameters->RA_grad = 45.0f ;

    parameters->FL = (parameters->FL_grad *  M_PI) / 180.0f; // FL sensor angle from forward position (degrees, 22.5 - 45)
    parameters->FR = (parameters->FR_grad *  M_PI) / 180.0f; // FR sensor angle from forward position (degrees, 22.5 - 45)
    parameters->RA = (parameters->RA_grad *  M_PI) / 180.0f; // Agent rotation angle (degrees)
    parameters->SO = 2.0f; // Sensor offset distance (pixels)
    parameters->SS = 1.0f; // Step size—how far agent moves per step (pixels)
    parameters->depT = 5.0f; // Chemoattractant deposition per step

    parameters->blue_enable=true;


}
