#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

//#pragma once

#include "constants.hpp"
#include "particle/body.hpp"
#include "particle/constraint.hpp"

class Environment
{
public:
    Environment(float _radius):
        radius(_radius),
        agentCount(EnvironmentDEFAULT_AGENT_COUNT)
    {

       stepAngle = M_PI * 2.0f / agentCount;
       initialize(agentCount);
       initConstraints();

    };

    float radius;
    float stepAngle;
    int agentCount;
    vector<Body> agents;
    vector<Constraint> constraints;

    int RandomInt(int min, int max)
    {
        return std::rand()%((max - min) + 1) + min;
    }


    Vec3 getInitialPosition(int index) {
        Vec2 tempPoint;
        tempPoint=AngleToVector(stepAngle*index);
        return  Vec3(tempPoint.x*radius,5,tempPoint.y*radius+70);
    };

    Vec3 getInitialDirection(int index) {
        Vec2 tempPoint;
        tempPoint=AngleToVector(stepAngle*index);
        return  Vec3(0, 0.0, 0);
        //return  Vec3(tempPoint.x,15,tempPoint.y);
    };


    bool initialize(int count)
    {


      //  Vec3 direction=UniformSampleHemisphere();
      //  agents.push_back( Squid(0, DNA(direction), Vec3(0,125,0), direction));
      //   direction=UniformSampleHemisphere();
      //  agents.push_back( Squid(0, DNA(direction), Vec3(55,125,0), direction));

        for (int agent = 0; agent < count; ++agent)
        {
             Vec3 direction=UniformSampleHemisphere();
             float radius=2.5f;
             if(agent%5==0)
                 radius=5.5f;
             agents.push_back( Body(agent,  Vec2(getInitialPosition(agent).x,getInitialPosition(agent).z) , 0, radius));
        }
    }


    bool CreateConstraint(int b1Index, int b2Index )
    {
        auto ptrBody1=make_unique<Body>(agents[b1Index]);
        auto ptrBody2=make_unique<Body>(agents[b2Index]);
        Constraint constraintTemp=Constraint(agents[b1Index],agents[b1Index]);
        constraintTemp.body1Index=b1Index;
        constraintTemp.body2Index=b2Index;

        constraints.push_back(constraintTemp);

        return true;

    }

    bool initConstraints()
    {
        constraints.clear();
        constraints.resize(EnvSubPoint_Count*EnvSubPoint_Count*EnvironmentDEFAULT_AGENT_COUNT*EnvironmentDEFAULT_AGENT_COUNT);

        for (auto itA = agents.begin(); itA!= agents.end(); ++itA) {
            int indexA = std::distance(agents.begin(), itA);

            for (auto itB = agents.begin(); itB!= agents.end(); ++itB) {
                int indexB = std::distance(agents.begin(), itB);

               // if(indexA!=indexB)
                {
                   for(int sa=0; sa<EnvSubPoint_Count; sa++)
                   {
                       for(int sb=0; sb<EnvSubPoint_Count; sb++)
                       {
                           // auto ptrBody1=make_unique<Body>(agents[indexA]);
                           // auto ptrBody2=make_unique<Body>(agents[indexB]);
                            Constraint constraintTemp=Constraint(agents[indexA],agents[indexB]);
                            constraintTemp.body1Index=indexA;
                            constraintTemp.body2Index=indexB;

                            constraintTemp.body1SubPointIndex=sa;
                            constraintTemp.body2SubPointIndex=sb;

                            constraintTemp.active=false;

                            constraints[(sa*EnvSubPoint_Count+sb)*EnvironmentDEFAULT_AGENT_COUNT*EnvironmentDEFAULT_AGENT_COUNT+
                                        indexA*EnvironmentDEFAULT_AGENT_COUNT+indexB]=constraintTemp;
                       }
                   }

                }

             }

          }

        return true;

    }

};


#endif // ENVIRONMENT_HPP
