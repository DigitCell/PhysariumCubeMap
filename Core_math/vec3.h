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


#if 0 //_DEBUG
#define VEC3_VALIDATE() {	\
	assert(_finite(x));\
	assert(!_isnan(x));\
	\
	assert(_finite(y));\
	assert(!_isnan(y));\
	\
	assert(_finite(z));\
	assert(!_isnan(z));\
						 }
#else
#define VEC3_VALIDATE()
#endif

template <typename T=float>
class XVector3
{
public:

	typedef T value_type;

	__host__ __device__ inline XVector3() : x(0.0f), y(0.0f), z(0.0f) {}
	__host__ __device__ inline XVector3(T a) : x(a), y(a), z(a) {}
	__host__ __device__ inline XVector3(const T* p) : x(p[0]), y(p[1]), z(p[2]) {}
	__host__ __device__ inline XVector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_)
	{
		VEC3_VALIDATE();
	}

	__host__ __device__ inline operator T* () { return &x; }
	__host__ __device__ inline operator const T* () const { return &x; };

	__host__ __device__ inline void Set(T x_, T y_, T z_) { VEC3_VALIDATE(); x = x_; y = y_; z = z_;}

	__host__ __device__ inline XVector3<T> operator * (T scale) const { XVector3<T> r(*this); r *= scale; return r; VEC3_VALIDATE();}
	__host__ __device__ inline XVector3<T> operator / (T scale) const { XVector3<T> r(*this); r /= scale; return r; VEC3_VALIDATE();}
	__host__ __device__ inline XVector3<T> operator + (const XVector3<T>& v) const { XVector3<T> r(*this); r += v; return r; VEC3_VALIDATE();}
	__host__ __device__ inline XVector3<T> operator - (const XVector3<T>& v) const { XVector3<T> r(*this); r -= v; return r; VEC3_VALIDATE();}
	__host__ __device__ inline XVector3<T> operator /(const XVector3<T>& v) const { XVector3<T> r(*this); r /= v; return r; VEC3_VALIDATE();}
	__host__ __device__ inline XVector3<T> operator *(const XVector3<T>& v) const { XVector3<T> r(*this); r *= v; return r; VEC3_VALIDATE();}

	__host__ __device__ inline XVector3<T>& operator *=(T scale) {x *= scale; y *= scale; z*= scale; VEC3_VALIDATE(); return *this;}
	__host__ __device__ inline XVector3<T>& operator /=(T scale) {T s(1.0f/scale); x *= s; y *= s; z *= s; VEC3_VALIDATE(); return *this;}
	__host__ __device__ inline XVector3<T>& operator +=(const XVector3<T>& v) {x += v.x; y += v.y; z += v.z; VEC3_VALIDATE(); return *this;}
	__host__ __device__ inline XVector3<T>& operator -=(const XVector3<T>& v) {x -= v.x; y -= v.y; z -= v.z; VEC3_VALIDATE(); return *this;}
	__host__ __device__ inline XVector3<T>& operator /=(const XVector3<T>& v) {x /= v.x; y /= v.y; z /= v.z; VEC3_VALIDATE(); return *this; }
	__host__ __device__ inline XVector3<T>& operator *=(const XVector3<T>& v) {x *= v.x; y *= v.y; z *= v.z; VEC3_VALIDATE(); return *this; }

	__host__ __device__ inline bool operator != (const XVector3<T>& v) const { return (x != v.x || y != v.y || z != v.z); }

	// negate
	__host__ __device__ inline XVector3<T> operator -() const { VEC3_VALIDATE(); return XVector3<T>(-x, -y, -z); }

	__host__ __device__ void Validate()
	{
		VEC3_VALIDATE();
	}

	T x,y,z;
};

typedef XVector3<float> Vec3;
typedef XVector3<float> Vector3;

// lhs scalar scale
template <typename T>
__host__ __device__ XVector3<T> operator *(T lhs, const XVector3<T>& rhs)
{
	XVector3<T> r(rhs);
	r *= lhs;
	return r;
}

template <typename T>
__host__ __device__ bool operator==(const XVector3<T>& lhs, const XVector3<T>& rhs)
{
	return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

template <typename T>
__host__ __device__ typename T::value_type Dot3(const T& v1, const T& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z*v2.z; 
}

__host__ __device__ inline float Dot3(const float* v1, const float * v2)
{
	return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]; 
}

template <typename T>
__host__ __device__ inline T Abs(const XVector3<T>& v1)
{
	return Vec3(fabs(v1.x),fabs(v1.y), fabs(v1.z));
}

template <typename T>
__host__ __device__ inline T Dot(const XVector3<T>& v1, const XVector3<T>& v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

template <typename T>
__host__ __device__ inline T Lenght(const XVector3<T>& v1)
{
	return sqrt(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
}



template <typename T>
inline T SelfNormalize(const T& v, const T& fallback=T())
{
	float l = LengthSq(v);
	if (l > 0.0f)
	{
		return v * (1.0f/sqrt(l));
	}
	else
		return fallback;
}

__host__ __device__ inline Vec3 Cross(const Vec3& b, const Vec3& c)
{
	return Vec3(b.y*c.z - b.z*c.y,
			    b.z*c.x - b.x*c.z,
				b.x*c.y - b.y*c.x);
}

template <typename T>
__host__ __device__ inline T cMin(T a, T b)
{
        return a < b ? a : b;
}

template <typename T>
__host__ __device__ inline T cMax(T a, T b)
{
        return a > b ? a : b;
}



// component wise min max functions
template <typename T>
__host__ __device__ inline XVector3<T> Max(const XVector3<T>& a, const XVector3<T>& b)
{
        return XVector3<T>(cMax(a.x, b.x), cMax(a.y, b.y), cMax(a.z, b.z));
}

template <typename T>
__host__ __device__ inline XVector3<T> Min(const XVector3<T>& a, const XVector3<T>& b)
{
        return XVector3<T>(cMin(a.x, b.x), cMin(a.y, b.y), cMin(a.z, b.z));
}

