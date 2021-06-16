#ifndef UTILS_HPP
#define UTILS_HPP


#pragma once

#include "Common_eigen.hpp"


namespace Utils {


void circlePointRotate(Vec2& _out, float _angle);
/// Returns the unit length vector for the given angle (in radians).
Vec2 vectorFromAngle(const float a);
/// Returns the angular direction v is pointing in (in radians).
float vectorToAngle(const Vec2 v);

/// Returns a perpendicular vector. (-90 degree rotation)
Vec2 perpendicularClowise(const Vec2 v);

/// Returns a perpendicular vector. (-90 degree rotation)
Vec2 perpendicularCounterClowise(const Vec2 v);

float randomFloat(float min = 0.0f, float max = 1.0f);
Vec3 randomVector(const Vec3 &min, const Vec3 &max);

Vec3 randomDirection();
Vec3 randomDirectionCircle(const Vec3 &n);
Vec3 randomDirectionHalfSphere(const Vec3 &n);



} // namespace Utils



#endif // UTILS_HPP
