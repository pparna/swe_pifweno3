/*
Supplementary source code for:
"GPU Driven Finite Difference WENO Scheme for Real Time Solution of the Shallow Water Equations"
P. Parna, K. Meyer, R. Falconer

From: phys-gfx.net/swe_pifweno3
*/

#ifdef __cplusplus
#pragma once
#endif

#include "Precision.h"

#ifdef __cplusplus
#define uint unsigned int
namespace PIFWENO {
#endif

	// simulation constants
	static const uint nWidth = 1600;
	static const uint nHeight = 1152;
	static const Real nDx = 0.2;
	static const Real nDy = 0.2;
	//static const Real CFL = 0.7;
	static const Real g = (Real)9.81L;

	// packed value given to boundary cells regarding their BC copy destinations
	#define BC_WRITE_OFFSET 1.99414051f

	// PIFWENO3 and PIFWENO3C constants
#ifdef __cplusplus
	namespace PIFWENO3 {
#endif
		static const uint pifweno3_bcCount = 4;

		// threading configurations
		static const uint PIFWENO3SP_THREADS_X = 32;
		static const uint PIFWENO3SP_THREADS_Y = 24;

		static const uint PIFWENO3DP_THREADS_X = 24;
		static const uint PIFWENO3DP_THREADS_Y = 16;
		
		static const uint PIFWENO3CSP_THREADS_X = 24;
		static const uint PIFWENO3CSP_THREADS_Y = 24;

		static const uint PIFWENO3CDP_THREADS_X = 20;
		static const uint PIFWENO3CDP_THREADS_Y = 20;

		static const Real pifweno3_v_epsilon = 0.01;
		static const Real pifweno3_WENO_epsilon = nDx * nDx;
#ifdef __cplusplus
	}
#endif

	// rendering constants
	static const float gDx = 0.01f;
	static const float gDy = 0.01f;
	static const uint gTHREADS_X = 16; // normal calculation array size X
	static const uint gTHREADS_Y = 18; // normal calculation array size Y

#ifdef __cplusplus
}
#endif

// HLSL helper functions
#ifndef __cplusplus
#if PRECISION == DOUBLE_PRECISION
// double load from uint4 texture
void LoadFromMemory(Texture2D<uint4> src, uint i, uint j, out Real data1, out Real data2) {
	uint4 uData = src.Load(int3(i, j, 0));
	data1 = asdouble(uData.y, uData.x);
	data2 = asdouble(uData.w, uData.z);
}

// double load from uint2 texture
void LoadFromMemory(Texture2D<uint2> src, uint i, uint j, out Real data) {
	uint2 uData = src.Load(int3(i, j, 0));
	data = asdouble(uData.y, uData.x);
}

// packs a double into a uint2
uint2 PackDouble(double d) {
	uint low, high;
	asuint(d, low, high);
	return uint2(high, low);
}
#endif // double precision
#endif