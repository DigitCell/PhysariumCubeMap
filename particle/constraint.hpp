#ifndef CONSTRAIN_H
#define CONSTRAIN_H


#include "../constants.hpp"
#include "vec3.h"
#include "vec2.h"
#include "maths.h"
#include "array"
#include <memory>
#include "body.hpp"


class Constraint
{
public:

    Constraint():
            body1(),
            body2(),
            m_length(0.0f),
            m_broken(false),
            m_strength(1.0f)
        {}

        Constraint(Body& b1, Body& b2, float length = 0.0f, float resistance = 1000.0f) :

            m_length(length),
            m_broken(false),
            m_resistance(resistance),
            m_strength(0.995f),
            active(false)
        {

            body1=make_shared<Body>(b1);
            body2=make_shared<Body>(b2);
            if (!m_length)
            {
              //  Vec2 dir = b1->position() - b2->position();
              //  m_length = dir.length();
            }
        }

    int index;
    bool active=false;
    int mat_index=0;

    std::shared_ptr<Body> body1;
    std::shared_ptr<Body> body2;

    int body1Index=0;
    int body2Index=0;


    int body1SubPointIndex=0;
    int body2SubPointIndex=0;

    float m_length;
    bool m_broken;
    float m_resistance;
    float m_strength;

};




#endif // CONSTRAIN_H
