#ifndef BODY_HPP
#define BODY_HPP

#include "../constants.hpp"
#include "vec3.h"
#include "vec2.h"
#include "cpTransform.h"
#include "maths.h"
#include "array"
#include <memory>


struct SubBody
{
    int index=0;
    int indexParent=0;
    bool use=false;


};

class Body
{
public:
    Body( int _index, Vec2 _p, float  _a, float _radius):
        index(_index),
        p(_p),
        pPrevious(_p),
        a(_a),
        aPrevious(_a),
        radius(_radius)
    {
        if(index%9==0)
        {
            mat_index=0;
            numVertex=6;
            radius=7.5f;
        }
        else
        {
            mat_index=1;
            numVertex=4;
            radius=2.5f;
        }

        //a=Vec3(0, 0.0, 0);

        deltaAngle = (2.0f*M_PI/numVertex);

        m=getm();
        i=geti();

        if(m>0)
            m_inv=1.0f/m;
        if(i>0)
            i_inv=1.0f/i;

        SetTransform(this, p, a);


        for(int i=0; i<numVertex;i++)
        {
            vertex[i].x = (0.5+radius) * cosf(a+i*deltaAngle);
            vertex[i].y = (0.5+radius) * sinf(a+i*deltaAngle);

            subp[i]=p+vertex[i];
        }

    };

    int index;
    int mat_index=0;

    Vec2  p=Vec2(0,0);
    Vec2  pPrevious=Vec2(0,0);

    Vec2  v=Vec2(0.0,0);
    Vec2  acceleration=Vec2(0,0);

    Vec2 f=Vec2(0.0,0);
    Vec2 v_bias=Vec2(0.0,0);

    Vec2 cog=Vec2(0.0,0);


    std::array<Vec2, EnvSubPoint_Count> vertex;
    std::array<Vec2, EnvSubPoint_Count> subp;
    std::array<SubBody, EnvSubPoint_Count> subBodyArray;

    cpTransform transform;


    float a=0;
    float aPrevious=0;

    float w=0;//2.0f*M_PI/25.0f;
    float w_bias=0;
    float t=0;

    float m;
    float m_inv;
    float i;
    float i_inv;

    float deltaAngle;

    int numVertex=6;

    bool inverseXmin=false;
    bool inverseXmax=false;

    bool inverseYmin=false;
    bool inverseYmax=false;

    bool collide=false;




    float radius;

    float tick=0.0f;
    float tickStep=0.025f;



    void update(Vec2 & impulse, Vec2  _pos, float  _a)
    {
       p=_pos;
       a=_a;
       tick+=tickStep;
    }

    void SetTransform(Body* body, Vec2 p, float a)
    {
        Vec2 rot = Vec2forangle(a);
        Vec2 c = body->cog;

        body->transform = cpTransformNewTranspose(
            rot.x, -rot.y, p.x - (c.x*rot.x - c.y*rot.y),
            rot.y,  rot.x, p.y - (c.x*rot.y + c.y*rot.x)
        );
    }

    void Setp(Body* body, Vec2 pos)
    {
        //cpBodyActivate(body);
        Vec2  p = p = cpTransformVect(body->transform, body->cog)+ pos;
        //cpAssertSaneBody(body);

        SetTransform(body, p, body->a);
    }



    void UpdateVelocity(Body* body, Vec2 gravity, float damping, float dt)
    {
        // Skip kinematic bodies.
        //if(cpBodyGetType(body) == CP_BODY_TYPE_KINEMATIC) return;

        //cpAssertSoft(body->m > 0.0f && body->i > 0.0f, "Body's m and moment must be positive to simulate. (m: %f Moment: %f)", body->m, body->i);

        body->v = Vec2add(Vec2mult(body->v, damping), Vec2mult(Vec2add(gravity, Vec2mult(body->f, body->m_inv)), dt));
        body->w = body->w*damping + body->t*body->i_inv*dt;

        // Reset forces.
        body->f = Vec2(0,0);
        body->t = 0.0f;

        //cpAssertSaneBody(body);
    }

    void UpdatePosition(Body* body, float dt)
    {
        Vec2 p = body->p = Vec2add(body->p, Vec2mult(Vec2add(body->v, body->v_bias), dt));
        float a = body->a= body->a + (body->w + body->w_bias)*dt;
        SetTransform(body, p, a);

        body->v_bias = Vec2(0,0);
        body->w_bias = 0.0f;

        //cpAssertSaneBody(body);
    }

    Vec2
    BodyLocalToWorld(Body *body, const Vec2 point)
    {
        return cpTransformPoint(body->transform, point);
    }

    Vec2
    BodyWorldToLocal(Body *body, const Vec2 point)
    {
        return cpTransformPoint(cpTransformRigidInverse(body->transform), point);
    }

    void
    BodyApplyForceAtWorldPoint(Body *body, Vec2 force, Vec2 point)
    {
        //cpBodyActivate(body);
        body->f = Vec2add(body->f, force);

        Vec2 r = Vec2sub(point, cpTransformPoint(body->transform, body->cog));
        body->t += Vec2cross(r, force);
    }

    inline void
    apply_impulse(Body *body, Vec2 j, Vec2 r){
        body->v = Vec2add(body->v, Vec2mult(j, body->m_inv));
        body->w += body->i_inv*Vec2cross(r, j);
    }

    inline void
    apply_bias_impulse(Body *body, Vec2 j, Vec2 r)
    {
        body->v_bias = Vec2add(body->v_bias, Vec2mult(j, body->m_inv));
        body->w_bias += body->i_inv*Vec2cross(r, j);
    }

    void
    BodyApplyForceAtLocalPoint(Body *body, Vec2 force, Vec2 point)
    {
        BodyApplyForceAtWorldPoint(body, cpTransformVect(body->transform, force), cpTransformPoint(body->transform, point));
    }

    void
    BodyApplyImpulseAtWorldPoint(Body *body, Vec2 impulse, Vec2 point)
    {
        //cpBodyActivate(body);

        Vec2 r = Vec2sub(point, cpTransformPoint(body->transform, body->cog));
        apply_impulse(body, impulse, r);
    }

    void
    BodyApplyImpulseAtLocalPoint(Body *body, Vec2 impulse, Vec2 point)
    {
        BodyApplyImpulseAtWorldPoint(body, cpTransformVect(body->transform, impulse), cpTransformPoint(body->transform, point));
    }

    bool CollisionResponseWall3(Body* a, Vec2 n)
    {
        float eps = 1e-9f;
        float mu=0.7575f, cor=0.75f;

        //Vec2 n = a_norm;
        Vec2 dv = a->v +  a->radius*a->w*Vec2(-n.y, n.x);
        // normal impulse
        float fn = -(1.0+cor)*(Dot(n,dv))/(1.0/a->m );
        // tangential unit vector
        Vec2 t = dv -Dot(n,dv)*n;
        float t_n = Length(t);
        t = t_n > eps ? t/t_n : t;

        float ft_match_v = -1.0*Dot(dv,t)/(//1.0/a.m+1.0/b.m
                                                   + a->radius*a->radius/a->i);
        float sign_ft_match_v = ft_match_v > 0.0 ? 1.0 : -1.0;
        // tangential impulse. track sign of the impulse here since we use fabs

        float ft = fabs (ft_match_v) < mu*fabs (fn) ?
                    ft_match_v : sign_ft_match_v*fabs (fn)*mu;
        Vec2 f = fn*n + ft*t;


        a->v=a->v+ f/a->m;
        a->w=a->w*0.995f+a->radius*Cross(n,f)/a->i;

        return true;
    }

    void UpdateWorldPosition(float dt)
    {
        float dump=0.97f;
        float deltaR=0.0000000001f;//0.0000001f;//
        if(p.y<0.0f+radius and inverseYmin==false)
        {
            p.y=deltaR+radius;
            //v.y =-v.y*dump;

            CollisionResponseWall3(this, Vec2(0,-1));
            inverseYmin=true;
        }
        else
        {
            inverseYmin=false;
        }

        if(p.y>WorldSize.y-radius and inverseYmax==false)
        {
            p.y=WorldSize.y-deltaR-radius;
            CollisionResponseWall3(this, Vec2(0,1));
           // v.y =-v.y*dump;
            inverseYmax=true;
        }
        else
        {
            inverseYmax=false;
        }


        if(p.x<-WorldSize.x+radius and inverseXmin==false)
        {
            p.x=deltaR-WorldSize.x+radius;
            CollisionResponseWall3(this, Vec2(1,0));
           // v.x =-v.x*dump;
            inverseXmin=true;
        }
        else
        {
            inverseXmin=false;
        }

        if(p.x>WorldSize.x-radius and inverseXmax==false)
        {
             p.x=WorldSize.x-radius-deltaR;
             CollisionResponseWall3(this, Vec2(-1,0));
             //v.x =-v.x*dump;
             inverseXmax=true;
        }
        else
        {
            inverseXmax=false;
        }



       //v = v+Env_gravity * dt;
       //p = p + v* dt;
    }

    void update()
    {

       Vec2 constrDirection=Vec2(0,200)-this->subp[0];
       float clength=Length(constrDirection);

       Vec2 cd_norm=Normalize(constrDirection);

       float deltaLength=clength-125;
       if(deltaLength<0)
       {
          cd_norm=-cd_norm;
          deltaLength=0;//deltaLength/2.0f;
       }
       float dotDirection=Dot(cd_norm, Normalize(this->p-subp[0]));


       deltaLength=abs(deltaLength);

      // if(dotDirection<0)
      //     deltaLength+=1.3f+dotDirection;
       if(deltaLength>5)
           deltaLength=5;

       //BodyApplyForceAtWorldPoint(this,100.0f*cd_norm*deltaLength,subp[0]);
       BodyApplyForceAtWorldPoint(this,70*this->m*cd_norm*deltaLength,subp[0]);




       //UpdateVelocity(this, Env_gravity ,1.0f, EnvironmentFRAME_TIME);
       //UpdatePosition(this,EnvironmentFRAME_TIME);

       UpdateWorldPosition(EnvironmentFRAME_TIME);

       UpdateVelocity(this, Env_gravity ,0.99f, EnvironmentFRAME_TIME);
       UpdatePosition(this,EnvironmentFRAME_TIME);
       //a=a+w*EnvironmentFRAME_TIME; //M_PI/70.f;
       //a=Vec3(0, 0.0, 0);

       for(int i=0; i<numVertex;i++)
       {
           vertex[i].x = (0.5+radius) * cosf(a+i*deltaAngle);
           vertex[i].y = (0.5+radius) * sinf(a+i*deltaAngle);

           subp[i]=p+vertex[i];
       }



       tick+=tickStep;
    }



    float getm()
    {
        float appendagesm = 0;

       // for (auto& appendage :appendages)
       //     appendagesm += appendage.getm();

        return M_PI* radius * radius * BodyMASS_PER_AREA + appendagesm;
    }

    float geti()
    {
        float appendagesm = 0;

       // for (auto& appendage :appendages)
       //     appendagesm += appendage.getm();

        return  pow(radius,2) * m/2.0f;
    }


};

#endif // BODY_HPP
