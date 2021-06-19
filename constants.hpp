#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#pragma once

#include <cstdint>
#include "math.h"

#include "vec3.h"
#include "maths.h"


#include <chrono>
#include <random>

#define c_window_width 1900
#define c_window_height 900

#define c_world_width 1400
#define c_world_height 800

#define c_ground_level 50

inline uint32_t seed1=1153421111;
inline uint32_t seed2=3444991111;

using namespace std;

inline Vec3 arbitrary_orthogonal2(Vec3 N)
{
    double Ax= abs(N.x), Ay= abs(N.y), Az= abs(N.z);
    if (Ax < Ay)
        return  Ax < Az ? Vec3(0, -N.z, N.y) : Vec3(-N.y, N.x, 0);
    else
        return  Ay < Az ? Vec3(N.z, 0, -N.x) : Vec3(-N.y, N.x, 0);
}

//World

const Vec2 WorldSize=Vec2(100,200);

//Gravity

const Vec2 Env_gravity=Vec2(0,-9.8f);
const float Env_gravityFloat=-9.8f;
//DNATentacle

//DNABrain


//Body

const float BodyMASS_PER_AREA=1.0f;

//Agent

const float AgentFRICTION = 0.798f;
const float AgentTORQUE = 0.18f;
const float AgentIMPULSE = 1.10;

//Environment


const int EnvironmentDEFAULT_AGENT_COUNT =41;

const int EnvSubPoint_Count =8;

const float EnvironmentFRAME_TIME = 0.015f;//1 / 60.0f;
const float EnvironmentWARP_STEP = EnvironmentFRAME_TIME * 10;


const float EnvironmentConstraintLenth =15.0f;


#endif // CONSTANTS_HPP
