#ifndef MAINLOOP_HPP
#define MAINLOOP_HPP

#pragma once

#include <chrono>
#include "GraphModule.hpp"
#include "environment.hpp"

class MainLoop
{
public:
    MainLoop(int _argc, char** _argv):

        graphModule("particle exp","Squid simulation","some info")

    {
        env=new Environment(50.0f);


        entry::initApp(&graphModule, _argc, _argv );
    };

    bool run();
    bool runMainLoop=true;
    GraphModule graphModule;

    Environment* env;


    //Collision

    std::vector<std::pair<int,int>> collisionVector;

    bool CollisionResponse(Body& a, Body& b)
    {
        Vec2 direction_norm=Norm(a.position-b.position);
        Vec2 a_r= -a.radius*direction_norm;
        Vec2 a_ro= Vec2(-a_r.y,a_r.x);
        Vec2 b_r= b.radius*direction_norm;
        Vec2 b_ro=Vec2(-b_r.y,b_r.x);


        Vec2 vp = a.velocity + Vec2(-a.angle_velocity*a_r.y, a.angle_velocity*a_r.x)
                - b.velocity - Vec2(-b.angle_velocity*b_r.y, b.angle_velocity*b_r.x);
        float vp_p=Dot(vp, direction_norm);

        if (vp_p >= 0) { // do they move apart?
            return false;
        }

        float e=1.55f; //elastity between 1 and 2
/*
        float j = - e * vp_p / (
                    1.0f/a.mass + pow(Dot(a_r,direction_norm),2) / a.momentInertia +
                    1.0f/b.mass + pow(Dot(b_r,direction_norm),2) / b.momentInertia
                );

        */

        float j = - e * vp_p / (
                    1.0f/a.mass + pow(Cross(a_ro,direction_norm),2) / a.momentInertia +
                    1.0f/b.mass + pow(Cross(b_ro,direction_norm),2) / b.momentInertia
                );

        Vec2 jn=j*direction_norm;

        a.velocity=a.velocity+jn/a.mass;
        //a.angle_velocity=a.angle_velocity+Cross(a_r,jn) / a.momentInertia;
        a.angle_velocity=a.angle_velocity+j/ a.momentInertia;

        Vec2 jnN=-jn;
        b.velocity=b.velocity+jnN/b.mass;
        b.angle_velocity=b.angle_velocity-j / b.momentInertia;



         return true;

    }

    bool CollisionResponse2(Body& a, Body& b)
    {
        Vec2 direction_norm=Norm(a.position-b.position);
        Vec2 p= a.position-a.radius*direction_norm;
        //Vec2 a_ro= Vec2(a_r.y, -a_r.x);
        //Vec2 b_r=b.position -b.radius*direction_norm;
        //Vec2 b_ro= Vec2(b_r.y, -b_r.x);


        float jj;
        Vec2 v, vv1, vv2, ap, bp;

        ap =  p - Vec2(a.position.x, a.position.y);  /////// p is point where two triangles collide  // v is edge of collision
        bp =  p - Vec2(b.position.x, b.position.y);;  //cm is center of mass

        Vec2 ap_o=Vec2(ap.y, -ap.x);
        Vec2 bp_o=Vec2(bp.y, -bp.x);

        vv2 = a.velocity + Vec2(-ap.y, ap.x)*a.angle_velocity;  //omega is angular speed
        vv1 = b.velocity - Vec2(-bp.y, bp.x)*b.angle_velocity;

        v = direction_norm;    /////rotate vector clokwise 90 degrees // v is now normal vector
        ap = ap_o;
        bp = bp_o;

        //impulse magnitude
        float e=0.25f; //elastity between 1 and 2

        jj = Dot(vv2 - vv1,-(1.0f+e)*v) / ((Dot(v,v)*(1.0f/ a.mass+ 1.0f/ b.mass))
            + pow(Dot(ap,v),2) / a.momentInertia + pow(Dot(bp,v),2) / b.momentInertia);

        //i.speed = i.speed + v*((jj) / i.mass);
       // j.speed = j.speed + v*((-jj) / j.mass);

        //i.omega = i.omega + ap*(v*((jj) / i.angmass));  // angmass is angular mass , Moment of inertia
       // j.omega = j.omega + bp*(v*((-jj) / j.angmass));

        a.velocity=a.velocity+ v*((jj) / a.mass);
        a.angle_velocity=a.angle_velocity+ Dot(ap,v*jj) / a.momentInertia;

        b.velocity=b.velocity+ v*((-jj) / b.mass);
        b.angle_velocity=b.angle_velocity+ Dot(bp,v*(-jj)) / b.momentInertia;



         return true;

    }

    bool CollisionResponse3(Body& a, Body& b)
    {
        float eps = 1e-7f;
        float mu=0.3175f, cor=0.97f;

        Vec2 direction_norm=Norm(b.position-a.position);

        Vec2 n = direction_norm;
        Vec2 dv = a.velocity +  a.radius*a.angle_velocity*Vec2(-n.y, n.x)
                 -(b.velocity - b.radius*b.angle_velocity*Vec2(-n.y, n.x));

        // normal impulse
        float fn = -(1.0+cor)*(Dot(n,dv))/(1.0/a.mass  + 1.0/b.mass );
        // tangential unit vector
        Vec2 t = dv -Dot(n,dv)*n;
        float t_n = Length(t);
        t = t_n > eps ? t/t_n : t;

        float ft_match_velocity = -1.0*Dot(dv,t)/(//1.0/a.mass+1.0/b.mass
                                                   + a.radius*a.radius/a.momentInertia
                                                   + b.radius*b.radius/b.momentInertia);
        float sign_ft_match_velocity = ft_match_velocity > 0.0 ? 1.0 : -1.0;
        // tangential impulse. track sign of the impulse here since we use fabs

        float ft = fabs (ft_match_velocity) < mu*fabs (fn) ?
                    ft_match_velocity : sign_ft_match_velocity*fabs (fn)*mu;
        Vec2 f = fn*n + ft*t;

       // translational velocity
       //aparticle.velocity+= f/aparticle.mass;
       //bparticle.velocity-= f/bparticle.mass;
       // angular velocity

       //aparticle.anglevelocity+=  aparticle.radius*Cross(n,f)/aparticle.Iw;
       //bparticle.anglevelocity+=  bparticle.radius*Cross(n,f)/bparticle.Iw;


        a.velocity=a.velocity+ f/ a.mass;
        a.angle_velocity=a.angle_velocity+  a.radius*Cross(n,f) / a.momentInertia;

        b.velocity=b.velocity- f / b.mass;
        b.angle_velocity=b.angle_velocity+  b.radius*Cross(n,f) / b.momentInertia;

        return true;

    }

    bool CollisionResponseWall3(Body& a, Vec2 n)
    {
        float eps = 1e-9f;
        float mu=0.075f, cor=0.95f;

        //Vec2 n = direction_norm;
        Vec2 dv = a.velocity +  a.radius*a.angle_velocity*Vec2(-n.y, n.x);
        // normal impulse
        float fn = -(1.0+cor)*(Dot(n,dv))/(1.0/a.mass );
        // tangential unit vector
        Vec2 t = dv -Dot(n,dv)*n;
        float t_n = Length(t);
        t = t_n > eps ? t/t_n : t;

        float ft_match_velocity = -1.0*Dot(dv,t)/(//1.0/a.mass+1.0/b.mass
                                                   + a.radius*a.radius/a.momentInertia);
        float sign_ft_match_velocity = ft_match_velocity > 0.0 ? 1.0 : -1.0;
        // tangential impulse. track sign of the impulse here since we use fabs

        float ft = fabs (ft_match_velocity) < mu*fabs (fn) ?
                    ft_match_velocity : sign_ft_match_velocity*fabs (fn)*mu;
        Vec2 f = fn*n + ft*t;


        a.velocity=a.velocity+ f/ a.mass;
        a.angle_velocity=a.angle_velocity+  a.radius*Cross(n,f) / a.momentInertia;

        return true;
    }


    bool UpdateStep()
    {
        for(auto& agent:env->agents)
        {
           agent.update();

           agent.collide=false;
        }
        /*
        for(auto& agentA:env->agents)
        {
           for(auto& agentB:env->agents)
           {
               auto delta=agentA.position-agentB.position;
               float sumRadius=agentA.radius+agentB.radius;

               if(sumRadius>abs(delta.x)+abs(delta.y))
               {
                   float deltaLenth=Length(delta);
                   if(sumRadius>deltaLenth)
                   {

                   }


               }
           }
         }
           */
       float deltaR=0.000000001f;//0.0000000001f;
        collisionVector.clear();
        for (auto itA = env->agents.begin(); itA!= env->agents.end(); ++itA) {
            int indexA = std::distance(env->agents.begin(), itA);

            for (auto itB = env->agents.begin(); itB!= env->agents.end(); ++itB) {
                int indexB = std::distance(env->agents.begin(), itB);


                if(indexA!=indexB)
                {

                    auto&& agentA=env->agents[indexA];
                    auto&& agentB=env->agents[indexB];

                    bool activeDistance=false;

                    auto delta=agentA.position-agentB.position;
                    float sumRadius=agentA.radius+agentB.radius;
                    float deltaLenth=Length(delta);
                   // if(sumRadius>2*abs(delta.x)+abs(delta.y))
                    {

                        if(sumRadius>deltaLenth)
                        {
                            activeDistance=true;
                            collisionVector.push_back(std::make_pair(indexA,indexB));

                            auto delta=agentA.position-agentB.position;
                            float sumRadius=agentA.radius+agentB.radius;
                            float deltaLenth=(abs(sumRadius-Length(delta))+deltaR)/2.0f;
                            auto norm=Normalize(delta);

                            agentA.position=agentA.position+deltaLenth*norm;
                            agentB.position=agentB.position-deltaLenth*norm;

                            agentA.collide=true;
                            agentB.collide=true;
                        }


                    }


                    //if(abs(delta.x)+abs(delta.y)<EnvironmentConstraintLenth )

                    {


                            if(deltaLenth<2*EnvironmentConstraintLenth )
                            {
                                 int indexAS=indexA;
                                 int indexBS=indexB;
                                 if(indexA>indexB)
                                 {
                                    // indexAS=indexB;
                                    // indexBS=indexA;


                                 }

                                 bool findConstraint=false;
                                 int saSub=0;
                                 int sbSub=0;
                                 float minLength=EnvironmentConstraintLenth*EnvironmentConstraintLenth ;
/*
                                 for(int sa=0; sa<EnvSubPoint_Count; sa++)
                                 {
                                     for(int sb=0; sb<EnvSubPoint_Count; sb++)
                                     {
                                         int mixIndex=(sa*EnvSubPoint_Count+sb)*EnvironmentDEFAULT_AGENT_COUNT*EnvironmentDEFAULT_AGENT_COUNT;
                                         env->constraints[mixIndex+indexA*EnvironmentDEFAULT_AGENT_COUNT+indexB].active=false;
                                     }
                                 }
*/
                                 for(int sa=0; sa<agentA.numVertex; sa++)
                                 {
                                     for(int sb=0; sb<agentB.numVertex; sb++)
                                     {


                                         Vec2 deltaSub=agentA.subPosition[sa]-agentB.subPosition[sb];
                                         float deltaLengthSub=Length(deltaSub);//Dot(agentA.subPosition[sa],agentB.subPosition[sb]);// //compare quard also normal
                                         if(deltaLengthSub<minLength)// and !env->agents[indexA].subBodyArray[sa].use and !env->agents[indexB].subBodyArray[sb].use)
                                         {
                                           findConstraint=true;
                                           saSub=sa;
                                           sbSub=sb;
                                           minLength=deltaLengthSub;
                                         }

                                         int mixIndex=(sa*EnvSubPoint_Count+sb)*EnvironmentDEFAULT_AGENT_COUNT*EnvironmentDEFAULT_AGENT_COUNT;
                                         env->constraints[mixIndex+indexA*EnvironmentDEFAULT_AGENT_COUNT+indexB].active=false;
                                         //env->agents[indexA].subBodyArray[saSub].use=true;
                                        // env->agents[indexB].subBodyArray[sbSub].use=true;


                                        // int mixIndex=(sa*EnvSubPoint_Count+sb)*EnvironmentDEFAULT_AGENT_COUNT*EnvironmentDEFAULT_AGENT_COUNT;
                                        // env->constraints[mixIndex+indexA*EnvironmentDEFAULT_AGENT_COUNT+indexB].active=false;
                                     }
                                 }

                                 if(findConstraint)
                                 {
                                     int mixIndex=(saSub*EnvSubPoint_Count+sbSub)*EnvironmentDEFAULT_AGENT_COUNT*EnvironmentDEFAULT_AGENT_COUNT;
                                     env->constraints[mixIndex+indexA*EnvironmentDEFAULT_AGENT_COUNT+indexB].active=true;
                                     env->agents[indexA].subBodyArray[saSub].use=true;
                                     env->agents[indexB].subBodyArray[sbSub].use=true;
                                 }


                            }
                            else
                            {
                                int indexAS=indexA;
                                int indexBS=indexB;
                                if(indexA>indexB)
                                {
                                   // indexAS=indexB;
                                  //  indexBS=indexA;
                                }

                                for(int sa=0; sa<EnvSubPoint_Count; sa++)
                                {
                                    for(int sb=0; sb<EnvSubPoint_Count; sb++)
                                    {
                                        int mixIndex=(sa*EnvSubPoint_Count+sb)*EnvironmentDEFAULT_AGENT_COUNT*EnvironmentDEFAULT_AGENT_COUNT;
                                        env->constraints[mixIndex+indexA*EnvironmentDEFAULT_AGENT_COUNT+indexB].active=false;
                                        env->agents[indexA].subBodyArray[sa].use=false;
                                        env->agents[indexB].subBodyArray[sb].use=false;
                                    }
                                }
                            }


                    }
                }

            }

        }


        for(auto collide_pair:collisionVector)
        {
            auto&& agentA=env->agents[collide_pair.first];
            auto&& agentB=env->agents[collide_pair.second];
/*
            auto delta=agentA.position-agentB.position;
            float sumRadius=agentA.radius+agentB.radius;
            float deltaLenth=abs(Length(delta)-sumRadius)/2.0f+0.0001f;
            auto norm=Normalize(delta);

            agentA.position=agentA.position+deltaLenth*norm;
            agentB.position=agentB.position-deltaLenth*norm;
*/
            CollisionResponse3(agentA,agentB);
            agentA.collide=false;
            agentB.collide=false;
        }

        return true;


    }



};


#endif // MAINLOOP_HPP
