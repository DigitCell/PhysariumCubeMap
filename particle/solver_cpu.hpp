#ifndef SOLVER_CPU_HPP
#define SOLVER_CPU_HPP


#include "../constants.hpp"
#include "body.hpp"
#include "constraint.hpp"

class Solver_CPU
{
public:
    Solver_CPU();


    vector<Body> bodies;
    vector<Constraint> constraints;
};

#endif // SOLVER_CPU_H
