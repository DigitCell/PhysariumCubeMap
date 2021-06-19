// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2013-2016 NVIDIA Corporation. All rights reserved.

#pragma once
#include "cuda_runtime.h"
#include "math.h"

#if defined(_WIN32) && !defined(__CUDACC__)
#if defined(_DEBUG)

#define VEC2_VALIDATE() {	assert(_finite(x));\
	assert(!_isnan(x));\
	\
	assert(_finite(y));\
	assert(!_isnan(y));\
						 }
#else

#define VEC2_VALIDATE() {\
	assert(isfinite(x));\
	assert(isfinite(y)); }\

#endif // _WIN32

#else
#define VEC2_VALIDATE()
#endif

#ifdef _DEBUG
#define FLOAT_VALIDATE(f) { assert(_finite(f)); assert(!_isnan(f)); }
#else
#define FLOAT_VALIDATE(f)
#endif

#define float_MIN 0.00001f
#define INFINITY (1e1000)


// vec2
template <typename T>
class XVector2
{
public:

	typedef T value_type;

         __host__ __device__ XVector2() : x(0.0f), y(0.0f) { VEC2_VALIDATE(); }
         __host__ __device__ XVector2(T _x) : x(_x), y(_x) { VEC2_VALIDATE(); }
         __host__ __device__ XVector2(T _x, T _y) : x(_x), y(_y) { VEC2_VALIDATE(); }
         __host__ __device__ XVector2(const T* p) : x(p[0]), y(p[1]) {}

         template <typename U>
	 explicit XVector2(const XVector2<U>& v) : x(v.x), y(v.y) {}

         __host__ __device__ operator T* () { return &x; }
         __host__ __device__ operator const T* () const { return &x; };

         __host__ __device__ void Set(T x_, T y_) { VEC2_VALIDATE(); x = x_; y = y_; }

         __host__ __device__ XVector2<T> operator * (T scale) const { XVector2<T> r(*this); r *= scale; VEC2_VALIDATE(); return r; }
         __host__ __device__ XVector2<T> operator / (T scale) const { XVector2<T> r(*this); r /= scale; VEC2_VALIDATE(); return r; }
         __host__ __device__ XVector2<T> operator + (const XVector2<T>& v) const { XVector2<T> r(*this); r += v; VEC2_VALIDATE(); return r; }
         __host__ __device__ XVector2<T> operator - (const XVector2<T>& v) const { XVector2<T> r(*this); r -= v; VEC2_VALIDATE(); return r; }

         __host__ __device__ XVector2<T>& operator *=(T scale) {x *= scale; y *= scale; VEC2_VALIDATE(); return *this;}
         __host__ __device__ XVector2<T>& operator /=(T scale) {T s(1.0f/scale); x *= s; y *= s; VEC2_VALIDATE(); return *this;}
         __host__ __device__ XVector2<T>& operator +=(const XVector2<T>& v) {x += v.x; y += v.y; VEC2_VALIDATE(); return *this;}
         __host__ __device__ XVector2<T>& operator -=(const XVector2<T>& v) {x -= v.x; y -= v.y; VEC2_VALIDATE(); return *this;}

         __host__ __device__ XVector2<T>& operator *=(const XVector2<T>& scale) {x *= scale.x; y *= scale.y; VEC2_VALIDATE(); return *this;}

	// negate
         __host__ __device__ XVector2<T> operator -() const { VEC2_VALIDATE(); return XVector2<T>(-x, -y); }

	// returns this vector
         __host__ __device__ void Normalize() { *this /= Length(*this); }
         __host__ __device__ void SafeNormalize(const XVector2<T>& v=XVector2<T>(0.0f,0.0f))
	{
		T length = Length(*this);
		*this = (length==0.00001f)?v:(*this /= length);
	}

	T x;
	T y;
};

typedef XVector2<float> Vec2;
typedef XVector2<float> Vector2;

typedef XVector2<float> Vec2fc;
typedef XVector2<float> Vector2f;

typedef XVector2<int> Vec2i;
typedef XVector2<int> Vector2i;

// lhs scalar scale
template <typename T>
__host__ __device__  XVector2<T> operator *(T lhs, const XVector2<T>& rhs)
{
	XVector2<T> r(rhs);
	r *= lhs;
	return r;
}

template <typename T>
__host__ __device__  XVector2<T> operator*(const XVector2<T>& lhs, const XVector2<T>& rhs)
{
	XVector2<T> r(lhs);
	r *= rhs;
	return r;
}

template <typename T>
__host__ __device__  bool operator==(const XVector2<T>& lhs, const XVector2<T>& rhs)
{
	return (lhs.x == rhs.x && lhs.y == rhs.y);
}


template <typename T>
__host__ __device__  T Dot(const XVector2<T>& v1, const XVector2<T>& v2)
{
	return v1.x * v2.x + v1.y * v2.y; 
}

template <typename T>
__host__ __device__  T Length(const XVector2<T>& v)
{
        return sqrt(v.x * v.x + v.y * v.y);
}

template <typename T>
__host__ __device__  XVector2<T> Norm(const XVector2<T>& v)
{
        float len=Length(v);
        if(len>0)
            return v/Length(v);
        else
            return XVector2<T>(0,0);
}

// returns the ccw perpendicular vector 
template <typename T>
__host__ __device__  XVector2<T> PerpCCW(const XVector2<T>& v)
{
	return XVector2<T>(-v.y, v.x);
}

template <typename T>
__host__ __device__  XVector2<T> PerpCW(const XVector2<T>& v)
{
	return XVector2<T>(v.y, -v.x);
}

// component wise min max functions
template <typename T>
__host__ __device__  XVector2<T> Max(const XVector2<T>& a, const XVector2<T>& b)
{
	return XVector2<T>(Max(a.x, b.x), Max(a.y, b.y));
}

template <typename T>
__host__ __device__  XVector2<T> Min(const XVector2<T>& a, const XVector2<T>& b)
{
	return XVector2<T>(Min(a.x, b.x), Min(a.y, b.y));
}

// 2d cross product, treat as if a and b are in the xy plane and return magnitude of z
template <typename T>
__host__ __device__  T Cross(const XVector2<T>& a, const XVector2<T>& b)
{
	return (a.x*b.y - a.y*b.x);
}

template <typename T>
__host__ __device__  XVector2<T> Vec2forangle(const float a)
{
    return XVector2<T>(cos(a), sin(a));
}

/*
inline Vec2 Vec2forangle(float& a)
{
    return Vec2(cos(a), sin(a));
}
*/

// Return the max of two cpFloats.
 inline  float cpfmax( float a,  float b)
{
    return (a > b) ? a : b;
}

/// Return the min of two cpFloats.
 inline  float cpfmin( float a,  float b)
{
    return (a < b) ? a : b;
}

/// Return the absolute value of a cpFloat.
 inline  float cpfabs( float f)
{
    return (f < 0) ? -f : f;
}

/// Clamp @c f to be between @c min and @c max.
 inline  float cpfclamp( float f,  float min,  float max)
{
    return cpfmin(cpfmax(f, min), max);
}

/// Clamp @c f to be between 0 and 1.
 inline  float cpfclamp01( float f)
{
    return cpfmax(0.0f, cpfmin(f, 1.0f));
}



/// Linearly interpolate (or extrapolate) between @c f1 and @c f2 by @c t percent.
 inline  float cpflerp( float f1,  float f2,  float t)
{
    return f1*(1.0f - t) + f2*t;
}

/// Linearly interpolate from @c f1 to @c f2 by no more than @c d.
 inline  float cpflerpconst( float f1,  float f2,  float d)
{
    return f1 + cpfclamp(f2 - f1, -d, d);
}


/// Add two vectors
 inline  Vec2 Vec2add(const  Vec2 v1, const  Vec2 v2)
{
    return Vec2(v1.x + v2.x, v1.y + v2.y);
}

/// Subtract two vectors.
 inline  Vec2 Vec2sub(const  Vec2 v1, const  Vec2 v2)
{
    return Vec2(v1.x - v2.x, v1.y - v2.y);
}

/// Negate a vector.
 inline  Vec2 Vec2neg(const  Vec2 v)
{
    return Vec2(-v.x, -v.y);
}

/// Scalar multiplication.
 inline  Vec2 Vec2mult(const  Vec2 v, const float s)
{
    return Vec2(v.x*s, v.y*s);
}

/// Vector dot product.
 inline float Vec2dot(const  Vec2 v1, const  Vec2 v2)
{
    return v1.x*v2.x + v1.y*v2.y;
}

/// 2D vector cross product analog.
/// The cross product of 2D vectors results in a 3D vector with only a z component.
/// This function returns the magnitude of the z value.
 inline float Vec2cross(const  Vec2 v1, const  Vec2 v2)
{
    return v1.x*v2.y - v1.y*v2.x;
}

/// Returns a perpendicular vector. (90 degree rotation)
 inline  Vec2 Vec2perp(const  Vec2 v)
{
    return Vec2(-v.y, v.x);
}

/// Returns a perpendicular vector. (-90 degree rotation)
 inline  Vec2 Vec2rperp(const  Vec2 v)
{
    return Vec2(v.y, -v.x);
}

/// Returns the vector projection of v1 onto v2.
 inline  Vec2 Vec2project(const  Vec2 v1, const  Vec2 v2)
{
    return Vec2mult(v2, Vec2dot(v1, v2)/Vec2dot(v2, v2));
}

/// Returns the unit length vector for the given angle (in radians).
 inline  Vec2 Vec2forangle(const float a)
{
    return Vec2(cos(a), sin(a));
}

/// Returns the angular direction v is pointing in (in radians).
 inline float Vec2toangle(const  Vec2 v)
{
    return atan2(v.y, v.x);
}

/// Uses complex number multiplication to rotate v1 by v2. Scaling will occur if v1 is not a unit vector.
 inline  Vec2 Vec2rotate(const  Vec2 v1, const  Vec2 v2)
{
    return Vec2(v1.x*v2.x - v1.y*v2.y, v1.x*v2.y + v1.y*v2.x);
}

/// Inverse of Vec2rotate().
 inline  Vec2 Vec2unrotate(const  Vec2 v1, const  Vec2 v2)
{
    return Vec2(v1.x*v2.x + v1.y*v2.y, v1.y*v2.x - v1.x*v2.y);
}

/// Returns the squared length of v. Faster than Vec2length() when you only need to compare lengths.
 inline float Vec2lengthsq(const  Vec2 v)
{
    return Vec2dot(v, v);
}

/// Returns the length of v.
 inline float Vec2length(const  Vec2 v)
{
    return sqrt(Vec2dot(v, v));
}

/// Linearly interpolate between v1 and v2.
 inline  Vec2 Vec2lerp(const  Vec2 v1, const  Vec2 v2, const float t)
{
    return Vec2add(Vec2mult(v1, 1.0f - t), Vec2mult(v2, t));
}

/// Returns a normalized copy of v.
 inline  Vec2 Vec2normalize(const  Vec2 v)
{
    // Neat trick I saw somewhere to avoid div/0.
    return Vec2mult(v, 1.0f/(Vec2length(v) + float_MIN));
}

/// Spherical linearly interpolate between v1 and v2.
 inline  Vec2
Vec2slerp(const  Vec2 v1, const  Vec2 v2, const float t)
{
    float dot = Vec2dot(Vec2normalize(v1), Vec2normalize(v2));
    float omega = acos(cpfclamp(dot, -1.0f, 1.0f));

    if(omega < 1e-3){
        // If the angle between two vectors is very small, lerp instead to avoid precision issues.
        return Vec2lerp(v1, v2, t);
    } else {
        float denom = 1.0f/sin(omega);
        return Vec2add(Vec2mult(v1, sin((1.0f - t)*omega)*denom), Vec2mult(v2, sin(t*omega)*denom));
    }
}

/// Spherical linearly interpolate between v1 towards v2 by no more than angle a radians
 inline  Vec2
Vec2slerpconst(const  Vec2 v1, const  Vec2 v2, const float a)
{
    float dot = Vec2dot(Vec2normalize(v1), Vec2normalize(v2));
    float omega =acos(cpfclamp(dot, -1.0f, 1.0f));

    return Vec2slerp(v1, v2, cpfmin(a, omega)/omega);
}

/// Clamp v to length len.
 inline  Vec2 Vec2clamp(const  Vec2 v, const float len)
{
    return (Vec2dot(v,v) > len*len) ? Vec2mult(Vec2normalize(v), len) : v;
}

/// Linearly interpolate between v1 towards v2 by distance d.
 inline  Vec2 Vec2lerpconst( Vec2 v1,  Vec2 v2, float d)
{
    return Vec2add(v1, Vec2clamp(Vec2sub(v2, v1), d));
}

/// Returns the distance between v1 and v2.
 inline float Vec2dist(const  Vec2 v1, const  Vec2 v2)
{
    return Vec2length(Vec2sub(v1, v2));
}

/// Returns the squared distance between v1 and v2. Faster than Vec2dist() when you only need to compare distances.
 inline float Vec2distsq(const  Vec2 v1, const  Vec2 v2)
{
    return Vec2lengthsq(Vec2sub(v1, v2));
}

/// Returns true if the distance between v1 and v2 is less than dist.
 inline bool Vec2near(const  Vec2 v1, const  Vec2 v2, const float dist)
{
    return Vec2distsq(v1, v2) < dist*dist;
}
