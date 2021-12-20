#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

#pragma once

#include "cuda.h"
#include <helper_cuda.h>
#include "helper_math.h"
#include <vector>
#include "constants.hpp"
//#include "CUDA/Definitions.cuh"
#include "particles_kernel.cuh"
//#include "cuda_exp.hpp"

using namespace std;


struct Environment
{

    Environment(int _numParticles);
    Particle* particles;
    int numberParticles;
    Caterpiller* catList;
    int numberCaterpiller;

    SimParams* parameters;

    bool stepClick=false;
    bool runSimulation=true;
    bool resetSimulation=false;

};


#endif // ENVIRONMENT_HPP
