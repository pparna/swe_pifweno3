/*
Supplementary source code for:
"GPU Driven Finite Difference WENO Scheme for Real Time Solution of the Shallow Water Equations"
P. Parna, K. Meyer, R. Falconer

From: phys-gfx.net/swe_pifweno3
*/

#ifdef __cplusplus
#pragma once
#endif

#define PRECISION SINGLE_PRECISION

#define DOUBLE_PRECISION 1
#define SINGLE_PRECISION 2

#if PRECISION == DOUBLE_PRECISION
	#define Real double
#else
	#define Real float
#endif

// proxy HLSL types
#ifndef __cplusplus

#if PRECISION == DOUBLE_PRECISION
	#define Real2 double2
	#define Real3 double3
	#define Real4 double4
	#define Real3x3 double3x3
#else
	#define Real2 float2
	#define Real3 float3
	#define Real4 float4
	#define Real3x3 float3x3
#endif

#endif