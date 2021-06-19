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

#include <cassert>

struct Matrix33;

template <typename T>
class XQuat
{
public:

	typedef T value_type;

	 XQuat() : x(0), y(0), z(0), w(1.0) {}
	 XQuat(const T* p) : x(p[0]), y(p[1]), z(p[2]), w(p[3]) {}
	 XQuat(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) { 	}
	 XQuat(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) { }
	 XQuat(const Matrix33& m);

	 operator T* () { return &x; }
	 operator const T* () const { return &x; };

	 void Set(T x_, T y_, T z_, T w_) {  x = x_; y = y_; z = z_; w = w_; }

	 XQuat<T> operator * (T scale) const { XQuat<T> r(*this); r *= scale;  return r;}
	 XQuat<T> operator / (T scale) const { XQuat<T> r(*this); r /= scale;  return r; }
	 XQuat<T> operator + (const XQuat<T>& v) const { XQuat<T> r(*this); r += v;  return r; }
	 XQuat<T> operator - (const XQuat<T>& v) const { XQuat<T> r(*this); r -= v;  return r; }
	 XQuat<T> operator * (XQuat<T> q) const 
	{
		// quaternion multiplication
		return XQuat<T>(w * q.x + q.w * x + y * q.z - q.y * z, w * q.y + q.w * y + z * q.x - q.z * x,
		            w * q.z + q.w * z + x * q.y - q.x * y, w * q.w - x * q.x - y * q.y - z * q.z);		
	}

	 XQuat<T>& operator *=(T scale) {x *= scale; y *= scale; z*= scale; w*= scale;  return *this;}
	 XQuat<T>& operator /=(T scale) {T s(1.0f/scale); x *= s; y *= s; z *= s; w *=s;  return *this;}
	 XQuat<T>& operator +=(const XQuat<T>& v) {x += v.x; y += v.y; z += v.z; w += v.w;  return *this;}
	 XQuat<T>& operator -=(const XQuat<T>& v) {x -= v.x; y -= v.y; z -= v.z; w -= v.w;  return *this;}

	 bool operator != (const XQuat<T>& v) const { return (x != v.x || y != v.y || z != v.z || w != v.w); }

	// negate
	 XQuat<T> operator -() const {  return XQuat<T>(-x, -y, -z, -w); }

	T x,y,z,w;
};

typedef XQuat<float> Quat;

// lhs scalar scale
template <typename T>
 XQuat<T> operator *(T lhs, const XQuat<T>& rhs)
{
	XQuat<T> r(rhs);
	r *= lhs;
	return r;
}

template <typename T>
 bool operator==(const XQuat<T>& lhs, const XQuat<T>& rhs)
{
	return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}

template <typename T>
 inline XQuat<T> QuatFromAxisAngle(const Vec3& axis, float angle)
{
	Vec3 v = Normalize(axis);

	float half = angle*0.5f;
	float w = cosf(half);

	const float sin_theta_over_two = sinf(half);
	v *= sin_theta_over_two;

	return XQuat<T>(v.x, v.y, v.z, w);
}


// rotate vector by quaternion (q, w)
 inline Vec3 Rotate(const Quat& q, const Vec3& x)
{
	return x*(2.0f*q.w*q.w-1.0f) + Cross(Vec3(q), x)*q.w*2.0f + Vec3(q)*Dot(Vec3(q), x)*2.0f;
}

// rotate vector by inverse transform in (q, w)
 inline Vec3 RotateInv(const Quat& q, const Vec3& x)
{
	return x*(2.0f*q.w*q.w-1.0f) - Cross(Vec3(q), x)*q.w*2.0f + Vec3(q)*Dot(Vec3(q), x)*2.0f;
}

 inline Quat Inverse(const Quat& q)
{
	return Quat(-q.x, -q.y, -q.z, q.w);
}

 inline Quat Normalize(const Quat& q)
{
	float lSq = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;

	if (lSq > 0.0f)
	{
		float invL = 1.0f / sqrtf(lSq);

		return q*invL;
	}
	else
		return Quat();
}