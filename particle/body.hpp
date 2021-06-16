#ifndef BODY_HPP
#define BODY_HPP

#include "../constants.hpp"
#include "vec3.h"
#include "vec2.h"
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
    Body( int _index, Vec2 _position, float  _direction, float _radius):
        index(_index),
        position(_position),
        positionPrevious(_position),
        direction(_direction),
        directionPrevious(_direction),
        radius(_radius)
    {
        if(index%9==0)
        {
            mat_index=0;
            numVertex=8;
            radius=7.5f;
        }
        else
        {
            mat_index=1;
            numVertex=4;
            radius=2.5f;
        }

        //direction=Vec3(0, 0.0, 0);

        deltaAngle = (2.0f*M_PI/numVertex);

        mass=getMass();
        momentInertia=getMomentInertia();


        for(int i=0; i<numVertex;i++)
        {
            vertex[i].x = (0.5+radius) * cosf(direction+i*deltaAngle);
            vertex[i].y = (0.5+radius) * sinf(direction+i*deltaAngle);

            subPosition[i]=position+vertex[i];
        }




    };

    int index;
    int mat_index=0;

    Vec2  position=Vec2(0,0);
    Vec2  positionPrevious=Vec2(0,0);

    Vec2  velocity=Vec2(-45.0,0);
    Vec2  acceleration=Vec2(0,0);


    std::array<Vec2, EnvSubPoint_Count> vertex;
    std::array<Vec2, EnvSubPoint_Count> subPosition;
    std::array<SubBody, EnvSubPoint_Count> subBodyArray;


    float direction=0;
    float directionPrevious=0;

    float angle_velocity=0;//2.0f*M_PI/25.0f;
    float angle_acc=0;

    float mass;
    float momentInertia;

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



    void update(Vec2 & impulse, Vec2  _pos, float  _direction)
    {
       position=_pos;
       direction=_direction;
       tick+=tickStep;
    }

    bool CollisionResponseWall3(Body* a, Vec2 n)
    {
        float eps = 1e-9f;
        float mu=0.3575f, cor=0.95f;

        //Vec2 n = direction_norm;
        Vec2 dv = a->velocity +  a->radius*a->angle_velocity*Vec2(-n.y, n.x);
        // normal impulse
        float fn = -(1.0+cor)*(Dot(n,dv))/(1.0/a->mass );
        // tangential unit vector
        Vec2 t = dv -Dot(n,dv)*n;
        float t_n = Length(t);
        t = t_n > eps ? t/t_n : t;

        float ft_match_velocity = -1.0*Dot(dv,t)/(//1.0/a.mass+1.0/b.mass
                                                   + a->radius*a->radius/a->momentInertia);
        float sign_ft_match_velocity = ft_match_velocity > 0.0 ? 1.0 : -1.0;
        // tangential impulse. track sign of the impulse here since we use fabs

        float ft = fabs (ft_match_velocity) < mu*fabs (fn) ?
                    ft_match_velocity : sign_ft_match_velocity*fabs (fn)*mu;
        Vec2 f = fn*n + ft*t;


        a->velocity=a->velocity+ f/a->mass;
        a->angle_velocity=a->angle_velocity*0.995f+a->radius*Cross(n,f)/a->momentInertia;

        return true;
    }

    void UpdateWorldPosition(float dt)
    {
        float dump=0.97f;
        float deltaR=0.0000000001f;//0.0000001f;//
        if(position.y<0.0f+radius and inverseYmin==false)
        {
            position.y=deltaR+radius;
            //velocity.y =-velocity.y*dump;

            CollisionResponseWall3(this, Vec2(0,-1));
            inverseYmin=true;
        }
        else
        {
            inverseYmin=false;
        }

        if(position.y>WorldSize.y-radius and inverseYmax==false)
        {
            position.y=WorldSize.y-deltaR-radius;
            CollisionResponseWall3(this, Vec2(0,1));
           // velocity.y =-velocity.y*dump;
            inverseYmax=true;
        }
        else
        {
            inverseYmax=false;
        }


        if(position.x<-WorldSize.x+radius and inverseXmin==false)
        {
            position.x=deltaR-WorldSize.x+radius;
            CollisionResponseWall3(this, Vec2(1,0));
           // velocity.x =-velocity.x*dump;
            inverseXmin=true;
        }
        else
        {
            inverseXmin=false;
        }

        if(position.x>WorldSize.x-radius and inverseXmax==false)
        {
             position.x=WorldSize.x-radius-deltaR;
             CollisionResponseWall3(this, Vec2(-1,0));
             //velocity.x =-velocity.x*dump;
             inverseXmax=true;
        }
        else
        {
            inverseXmax=false;
        }



        velocity = velocity+Env_gravity * dt;
        position = position + velocity* dt;
    }

    void update()
    {


       UpdateWorldPosition(EnvironmentFRAME_TIME);
       direction=direction+angle_velocity*EnvironmentFRAME_TIME; //M_PI/70.f;
       //direction=Vec3(0, 0.0, 0);

       for(int i=0; i<numVertex;i++)
       {
           vertex[i].x = (0.5+radius) * cosf(direction+i*deltaAngle);
           vertex[i].y = (0.5+radius) * sinf(direction+i*deltaAngle);

           subPosition[i]=position+vertex[i];
       }

       tick+=tickStep;
    }



    float getMass()
    {
        float appendagesMass = 0;

       // for (auto& appendage :appendages)
       //     appendagesMass += appendage.getMass();

        return M_PI* radius * radius * BodyMASS_PER_AREA + appendagesMass;
    }

    float getMomentInertia()
    {
        float appendagesMass = 0;

       // for (auto& appendage :appendages)
       //     appendagesMass += appendage.getMass();

        return  pow(radius,2) * mass/2.0f;
    }


};

#endif // BODY_HPP
