#include "utils.hpp"

#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <iostream>
#include <map>
#include <random>



typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
typedef Tr::Geom_traits GT;
typedef GT::Sphere_3 Sphere_3;
typedef GT::Point_3 Point_3;
typedef GT::FT FT;

template <typename function>
using Surface_3 = CGAL::Implicit_surface_3<GT, function>;


namespace Utils {


void circlePointRotate(Vec2& _out, float _angle)
 {
     float sa = sin(_angle);
     float ca = cos(_angle);
     _out.x = sa;
     _out.y= ca;
 }

/// Returns the unit length vector for the given angle (in radians).
Vec2 vectorFromAngle(const float a)
{
    return Vec2(cos(a), sin(a));
}
/// Returns the angular direction v is pointing in (in radians).
float vectorToAngle(const Vec2 v)
{
    return atan2(v.y, v.x);
}

/// Returns a perpendicular vector. (-90 degree rotation)
Vec2 perpendicularClowise(const Vec2 v)
{
    return Vec2(-v.y, v.x);
}

/// Returns a perpendicular vector. (-90 degree rotation)
Vec2 perpendicularCounterClowise(const Vec2 v)
{
    return Vec2(v.y, -v.x);
}


Vec3 safeNormal(const Vec3 &a, const Vec3 &b) {
    Vec3 n = b - a;
    float d = Length(n);
    const float eps = 1e-8f;
    if (d < eps)
        n = Utils::randomDirection();
    else
        n = n / d;

    return n;
}


float randomFloat(float min, float max) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist(min, max);
    float r = dist(e2);
    return r;
}

Vec3 randomVector(const Vec3 &min, const Vec3 &max) {
    float x = randomFloat(min.x, max.x);
    float y = randomFloat(min.y, max.y);
    float z = randomFloat(min.z, max.z);
    return Vec3(x, y, z);
}

float clamp(float n, float min, float max) {
    if (n < min)
        return min;
    if (n > max)
        return max;
    return n;
}



Vec3 randomDirection() {
    float t = randomFloat(0.0f, 2.0f * M_PIf);
    float z = randomFloat(-1.0f, 1.0f);
    float a = sqrt(1.f - z * z);
    return Vec3(a * cos(t), a * sin(t), z);
}


Vec3 randomDirectionHalfSphere(const Vec3 &n) {
    Vec3 v = randomDirection();
    float p = Dot(v,n);
    if (p < 0.f)
        v -= 2.f * p * n;
    return v;
}


float triArea2D(float x1, float y1, float x2, float y2, float x3, float y3) {
    return (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);
}

} // namespace Utils

