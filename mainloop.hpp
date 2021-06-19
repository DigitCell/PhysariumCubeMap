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
        Vec2 direction_norm=Norm(a.p-b.p);
        Vec2 a_r= -a.radius*direction_norm;
        Vec2 a_ro= Vec2(-a_r.y,a_r.x);
        Vec2 b_r= b.radius*direction_norm;
        Vec2 b_ro=Vec2(-b_r.y,b_r.x);


        Vec2 vp = a.v + Vec2(-a.w*a_r.y, a.w*a_r.x)
                - b.v - Vec2(-b.w*b_r.y, b.w*b_r.x);
        float vp_p=Dot(vp, direction_norm);

        if (vp_p >= 0) { // do they move apart?
            return false;
        }

        float e=1.55f; //elastity between 1 and 2
/*
        float j = - e * vp_p / (
                    1.0f/a.m + pow(Dot(a_r,direction_norm),2) / a.i +
                    1.0f/b.m + pow(Dot(b_r,direction_norm),2) / b.i
                );

        */

        float j = - e * vp_p / (
                    1.0f/a.m + pow(Cross(a_ro,direction_norm),2) / a.i +
                    1.0f/b.m + pow(Cross(b_ro,direction_norm),2) / b.i
                );

        Vec2 jn=j*direction_norm;

        a.v=a.v+jn/a.m;
        //a.w=a.w+Cross(a_r,jn) / a.i;
        a.w=a.w+j/ a.i;

        Vec2 jnN=-jn;
        b.v=b.v+jnN/b.m;
        b.w=b.w-j / b.i;



         return true;

    }

    bool CollisionResponse2(Body& a, Body& b)
    {
        Vec2 direction_norm=Norm(a.p-b.p);
        Vec2 p= a.p-a.radius*direction_norm;
        //Vec2 a_ro= Vec2(a_r.y, -a_r.x);
        //Vec2 b_r=b.p -b.radius*direction_norm;
        //Vec2 b_ro= Vec2(b_r.y, -b_r.x);


        float jj;
        Vec2 v, vv1, vv2, ap, bp;

        ap =  p - Vec2(a.p.x, a.p.y);  /////// p is point where two triangles collide  // v is edge of collision
        bp =  p - Vec2(b.p.x, b.p.y);;  //cm is center of mass

        Vec2 ap_o=Vec2(ap.y, -ap.x);
        Vec2 bp_o=Vec2(bp.y, -bp.x);

        vv2 = a.v + Vec2(-ap.y, ap.x)*a.w;  //omega is angular speed
        vv1 = b.v - Vec2(-bp.y, bp.x)*b.w;

        v = direction_norm;    /////rotate vector clokwise 90 degrees // v is now normal vector
        ap = ap_o;
        bp = bp_o;

        //impulse magnitude
        float e=0.25f; //elastity between 1 and 2

        jj = Dot(vv2 - vv1,-(1.0f+e)*v) / ((Dot(v,v)*(1.0f/ a.m+ 1.0f/ b.m))
            + pow(Dot(ap,v),2) / a.i + pow(Dot(bp,v),2) / b.i);

        //i.speed = i.speed + v*((jj) / i.m);
       // j.speed = j.speed + v*((-jj) / j.m);

        //i.omega = i.omega + ap*(v*((jj) / i.angmass));  // angmass is angular mass , Moment of inertia
       // j.omega = j.omega + bp*(v*((-jj) / j.angmass));

        a.v=a.v+ v*((jj) / a.m);
        a.w=a.w+ Dot(ap,v*jj) / a.i;

        b.v=b.v+ v*((-jj) / b.m);
        b.w=b.w+ Dot(bp,v*(-jj)) / b.i;



         return true;

    }

    bool CollisionResponse3(Body& a, Body& b)
    {
        float eps = 1e-7f;
        float mu=0.3175f, cor=0.97f;

        Vec2 direction_norm=Norm(b.p-a.p);

        Vec2 n = direction_norm;
        Vec2 dv = a.v +  a.radius*a.w*Vec2(-n.y, n.x)
                 -(b.v - b.radius*b.w*Vec2(-n.y, n.x));

        // normal impulse
        float fn = -(1.0+cor)*(Dot(n,dv))/(1.0/a.m  + 1.0/b.m );
        // tangential unit vector
        Vec2 t = dv -Dot(n,dv)*n;
        float t_n = Length(t);
        t = t_n > eps ? t/t_n : t;

        float ft_match_v = -1.0*Dot(dv,t)/(1.0/a.m+1.0/b.m
                                                   + a.radius*a.radius/a.i
                                                   + b.radius*b.radius/b.i);
        float sign_ft_match_v = ft_match_v > 0.0 ? 1.0 : -1.0;
        // tangential impulse. track sign of the impulse here since we use fabs

        float ft = fabs (ft_match_v) < mu*fabs (fn) ?
                    ft_match_v : sign_ft_match_v*fabs (fn)*mu;
        Vec2 f = fn*n + ft*t;

       // translational v
       //aparticle.v+= f/aparticle.m;
       //bparticle.v-= f/bparticle.m;
       // angular v

       //aparticle.anglev+=  aparticle.radius*Cross(n,f)/aparticle.Iw;
       //bparticle.anglev+=  bparticle.radius*Cross(n,f)/bparticle.Iw;


        a.v=a.v+ f/ a.m;
        a.w=a.w+  a.radius*Cross(n,f) / a.i;

        b.v=b.v- f / b.m;
        b.w=b.w+  b.radius*Cross(n,f) / b.i;

        return true;

    }

    bool CollisionResponseWall3(Body& a, Vec2 n)
    {
        float eps = 1e-9f;
        float mu=0.0175f, cor=0.95f;

        //Vec2 n = direction_norm;
        Vec2 dv = a.v +  a.radius*a.w*Vec2(-n.y, n.x);
        // normal impulse
        float fn = -(1.0+cor)*(Dot(n,dv))/(1.0/a.m );
        // tangential unit vector
        Vec2 t = dv -Dot(n,dv)*n;
        float t_n = Length(t);
        t = t_n > eps ? t/t_n : t;

        float ft_match_v = -1.0*Dot(dv,t)/(//1.0/a.m+1.0/b.m
                                                   + a.radius*a.radius/a.i);
        float sign_ft_match_v = ft_match_v > 0.0 ? 1.0 : -1.0;
        // tangential impulse. track sign of the impulse here since we use fabs

        float ft = fabs (ft_match_v) < mu*fabs (fn) ?
                    ft_match_v : sign_ft_match_v*fabs (fn)*mu;
        Vec2 f = fn*n + ft*t;


        a.v=a.v+ f/ a.m;
        a.w=a.w+  a.radius*Cross(n,f) / a.i;

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
               auto delta=agentA.p-agentB.p;
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

                    auto delta=agentA.p-agentB.p;
                    float sumRadius=agentA.radius+agentB.radius;
                    float deltaLenth=Length(delta);
                   // if(sumRadius>2*abs(delta.x)+abs(delta.y))
                    {

                        if(sumRadius>deltaLenth)
                        {
                            activeDistance=true;
                            collisionVector.push_back(std::make_pair(indexA,indexB));

                            auto delta=agentA.p-agentB.p;
                            float sumRadius=agentA.radius+agentB.radius;
                            float deltaLenth=(abs(sumRadius-Length(delta))+deltaR)/2.0f;
                            auto norm=Normalize(delta);

                            agentA.p=agentA.p+deltaLenth*norm;
                            agentB.p=agentB.p-deltaLenth*norm;

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


                                         Vec2 deltaSub=agentA.subp[sa]-agentB.subp[sb];
                                         float deltaLengthSub=Length(deltaSub);//Dot(agentA.subp[sa],agentB.subp[sb]);// //compare quard also normal
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
            auto delta=agentA.p-agentB.p;
            float sumRadius=agentA.radius+agentB.radius;
            float deltaLenth=abs(Length(delta)-sumRadius)/2.0f+0.0001f;
            auto norm=Normalize(delta);

            agentA.p=agentA.p+deltaLenth*norm;
            agentB.p=agentB.p-deltaLenth*norm;
*/
            CollisionResponse3(agentA,agentB);
            agentA.collide=false;
            agentB.collide=false;
        }

        return true;


    }



};


#endif // MAINLOOP_HPP
