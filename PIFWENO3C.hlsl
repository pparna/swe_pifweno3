/*
Supplementary source code for:
"GPU Driven Finite Difference WENO Scheme for Real Time Solution of the Shallow Water Equations"
P. Parna, K. Meyer, R. Falconer

From: phys-gfx.net/swe_pifweno3
*/

#include "CommonPIFWENO.h"

// shader resources
cbuffer cbBCCount : register(b0) {
	uint bcCount;
};

#if PRECISION == DOUBLE_PRECISION
Texture2D<uint4> simData_1 : register(t0);
Texture2D<uint4> simData_2 : register(t1);
Texture2D<uint2> bathymetry : register(t2);
RWTexture2D<uint4> results_1 : register(u0);
RWTexture2D<uint4> results_2 : register(u1);
#else
Texture2D<float4> simData : register(t0);
Texture2D<float> bathymetry : register(t2);
RWTexture2D<float4> results : register(u0);
#endif

#if PRECISION == DOUBLE_PRECISION
#define THREADS_X PIFWENO3CDP_THREADS_X
#define THREADS_Y PIFWENO3CDP_THREADS_Y
#else
#define THREADS_X PIFWENO3CSP_THREADS_X
#define THREADS_Y PIFWENO3CSP_THREADS_Y
#endif

#define dx nDx
#define dy nDy
#define dt 0.016
#define epsilon pifweno3_WENO_epsilon
#define desingularization_epsilon pifweno3_v_epsilon
#define REV_CONST 1e-37f

#define SRC1(b) (b * b)
#define SRC2(b) (b)

groupshared Real f1Array0[THREADS_Y][THREADS_X];
groupshared Real3 f3Array0[THREADS_Y][THREADS_X];
groupshared Real3 f3Array1[THREADS_Y][THREADS_X];
groupshared Real3 f3Array2[THREADS_Y][THREADS_X];

void WENO3PlusX(Real3 v_im1, Real3 v_i, Real3 v_ip1,
		Real3 s1_im1, Real3 s1_i, Real3 s1_ip1,
		Real3 s2_im1, Real3 s2_i, Real3 s2_ip1,
		out Real3 flux, out Real3 s1, out Real3 s2) {

	// find the small stencil approximations
	Real3 f0 = 0.5 * v_i + 0.5 * v_ip1;
	Real3 f1 = -0.5 * v_im1 + 1.5 * v_i;

	Real3 s1_0 = 0.5 * s1_i + 0.5 * s1_ip1;
	Real3 s1_1 = -0.5 * s1_im1 + 1.5 * s1_i;

	Real3 s2_0 = 0.5 * s2_i + 0.5 * s2_ip1;
	Real3 s2_1 = -0.5 * s2_im1 + 1.5 * s2_i;

	// define the linear weights
	Real d0 = 2.0 / 3.0;
	Real d1 = 1.0 / 3.0;

	// find the smoothness indicators
	Real3 b0 = (v_ip1 - v_i) * (v_ip1 - v_i);
	Real3 b1 = (v_i - v_im1) * (v_i - v_im1);

	// find the non-linear weights
	Real3 a0 = d0 / ((epsilon + b0) * (epsilon + b0));
	Real3 a1 = d1 / ((epsilon + b1) * (epsilon + b1));

	Real3 aSum = a0 + a1;

	Real3 w0 = a0 / aSum;
	Real3 w1 = a1 / aSum;

	flux = w0 * f0 + w1 * f1;
	s1 = w0 * s1_0 + w1 * s1_1;
	s2 = w0 * s2_0 + w1 * s2_1;
}

void WENO3MinusX(Real3 v_i, Real3 v_ip1, Real3 v_ip2,
		 Real3 s1_i, Real3 s1_ip1, Real3 s1_ip2,
		 Real3 s2_i, Real3 s2_ip1, Real3 s2_ip2,
		 out Real3 flux, out Real3 s1, out Real3 s2) {

	// find the small stencil approximations
	Real3 f0 = 0.5 * v_ip1 + 0.5 * v_i;
	Real3 f1 = -0.5 * v_ip2 + 1.5 * v_ip1;

	Real3 s1_0 = 0.5 * s1_ip1 + 0.5 * s1_i;
	Real3 s1_1 = -0.5 * s1_ip2 + 1.5 * s1_ip1;

	Real3 s2_0 = 0.5 * s2_ip1 + 0.5 * s2_i;
	Real3 s2_1 = -0.5 * s2_ip2 + 1.5 * s2_ip1;

	// define the linear weights
	Real d0 = 2.0 / 3.0;
	Real d1 = 1.0 / 3.0;

	// find the smoothness indicators
	Real3 b0 = (v_ip1 - v_i) * (v_ip1 - v_i);
	Real3 b1 = (v_ip2 - v_ip1) * (v_ip2 - v_ip1);

	// find the non-linear weights
	Real3 a0 = d0 / ((epsilon + b0) * (epsilon + b0));
	Real3 a1 = d1 / ((epsilon + b1) * (epsilon + b1));

	Real3 aSum = a0 + a1;

	Real3 w0 = a0 / aSum;
	Real3 w1 = a1 / aSum;

	flux = w0 * f0 + w1 * f1;
	s1 = w0 * s1_0 + w1 * s1_1;
	s2 = w0 * s2_0 + w1 * s2_1;
}

void WENO3PlusY(Real3 v_im1, Real3 v_i, Real3 v_ip1,
		Real3 s1_im1, Real3 s1_i, Real3 s1_ip1,
		Real3 s2_im1, Real3 s2_i, Real3 s2_ip1,
		out Real3 flux, out Real3 s1, out Real3 s2) {

	// find the small stencil approximations
	Real3 f0 = 0.5 * v_i + 0.5 * v_ip1;
	Real3 f1 = -0.5 * v_im1 + 1.5 * v_i;

	Real3 s1_0 = 0.5 * s1_i + 0.5 * s1_ip1;
	Real3 s1_1 = -0.5 * s1_im1 + 1.5 * s1_i;

	Real3 s2_0 = 0.5 * s2_i + 0.5 * s2_ip1;
	Real3 s2_1 = -0.5 * s2_im1 + 1.5 * s2_i;

	// define the linear weights
	Real d0 = 2.0 / 3.0;
	Real d1 = 1.0 / 3.0;

	// find the smoothness indicators
	Real3 b0 = (v_ip1 - v_i) * (v_ip1 - v_i);
	Real3 b1 = (v_i - v_im1) * (v_i - v_im1);

	// find the non-linear weights
	Real3 a0 = d0 / ((epsilon + b0) * (epsilon + b0));
	Real3 a1 = d1 / ((epsilon + b1) * (epsilon + b1));

	Real3 aSum = a0 + a1;

	Real3 w0 = a0 / aSum;
	Real3 w1 = a1 / aSum;

	flux = w0 * f0 + w1 * f1;
	s1 = w0 * s1_0 + w1 * s1_1;
	s2 = w0 * s2_0 + w1 * s2_1;
}

void WENO3MinusY(Real3 v_i, Real3 v_ip1, Real3 v_ip2,
		 Real3 s1_i, Real3 s1_ip1, Real3 s1_ip2,
		 Real3 s2_i, Real3 s2_ip1, Real3 s2_ip2,
		 out Real3 flux, out Real3 s1, out Real3 s2) {

	// find the small stencil approximations
	Real3 f0 = 0.5 * v_ip1 + 0.5 * v_i;
	Real3 f1 = -0.5 * v_ip2 + 1.5 * v_ip1;

	Real3 s1_0 = 0.5 * s1_ip1 + 0.5 * s1_i;
	Real3 s1_1 = -0.5 * s1_ip2 + 1.5 * s1_ip1;

	Real3 s2_0 = 0.5 * s2_ip1 + 0.5 * s2_i;
	Real3 s2_1 = -0.5 * s2_ip2 + 1.5 * s2_ip1;

	// define the linear weights
	Real d0 = 2.0 / 3.0;
	Real d1 = 1.0 / 3.0;

	// find the smoothness indicators
	Real3 b0 = (v_ip1 - v_i) * (v_ip1 - v_i);
	Real3 b1 = (v_ip2 - v_ip1) * (v_ip2 - v_ip1);

	// find the non-linear weights
	Real3 a0 = d0 / ((epsilon + b0) * (epsilon + b0));
	Real3 a1 = d1 / ((epsilon + b1) * (epsilon + b1));

	Real3 aSum = a0 + a1;

	Real3 w0 = a0 / aSum;
	Real3 w1 = a1 / aSum;

	flux = w0 * f0 + w1 * f1;
	s1 = w0 * s1_0 + w1 * s1_1;
	s2 = w0 * s2_0 + w1 * s2_1;
}


void UnpackBCData(float src, out int2 offsets) {
	// reinterpret the bit sequence as a signed integer
	int i32 = asint(src);

	// extract the offsets
	offsets.x = i32 >> 16;
	offsets.y = (i32 << 16) >> 16;
}


// each group invocation deals with (THREADS_X - 2 * bcCount) * (THREADS_Y - 2 * bcCount) amount of internal data
[numthreads(THREADS_X, THREADS_Y, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint3 threadID : SV_GroupThreadID) {
	// define the global index
	int i_global = groupID.x * (THREADS_X - 2 * bcCount) + threadID.x;
	int j_global = groupID.y * (THREADS_Y - 2 * bcCount) + threadID.y;

	// shorthand for the local index
	int i = threadID.x;
	int j = threadID.y;

	// load the simulation data for the previous timestep
#if PRECISION == DOUBLE_PRECISION
	Real simData1, simData2, simData3, simData4;
	LoadFromMemory(simData_1, i_global, j_global, simData1, simData2);
	LoadFromMemory(simData_2, i_global, j_global, simData3, simData4);
	Real3 U_n = Real3(simData1, simData2, simData3);
	// boundary information in the lower 32bits of the final double entry, i.e. the 4th channel
	uint uBCData = simData_2.Load(int3(i_global, j_global, 0)).w;
	float bcData = asfloat(uBCData);
#else
	Real4 simData_n = simData.Load(int3(i_global, j_global, 0));
	Real3 U_n = simData_n.xyz;
	float bcData = simData_n.w;
#endif
	int2 bcOffsets;
	UnpackBCData(bcData, bcOffsets);

#if PRECISION == DOUBLE_PRECISION
	Real b;
	LoadFromMemory(bathymetry, i_global, j_global, b);
#else
	Real b = bathymetry.Load(int3(i_global, j_global, 0));
#endif
	
	Real h = U_n.x;
	Real u = (h != 0.0) ? U_n.y / h : 0.0;
	Real v = (h != 0.0) ? U_n.z / h : 0.0;

	// find the eigenvalues
	Real3 e_x = abs(Real3(u - sqrt((float)(g * h)), u, u + sqrt((float)(g * h))));
	Real3 e_y = abs(Real3(v - sqrt((float)(g * h)), v, v + sqrt((float)(g * h))));

	// store b in f1Array0
	f1Array0[j][i] = b;
	// store U_n in f3Array0
	f3Array0[j][i] = U_n;
	// store the x-directional eigenvalues in f3Array1
	f3Array1[j][i] = e_x;
	// store the y-directional eigenvalues in f3Array2
	f3Array2[j][i] = e_y;

	GroupMemoryBarrierWithGroupSync();

	// find alpha for the local Lax-Friedrichs flux splitting
	Real3 alpha_x = Real3(-10000, -10000, -10000);
	Real3 alpha_y = Real3(-10000, -10000, -10000);
	[unroll]for (int k = -1; k <= 2; ++k) {
		alpha_x = max(alpha_x, f3Array1[j][i + k]);
		alpha_y = max(alpha_y, f3Array2[j + k][i]);
	}

	// find the average states
	Real3 U_iph_j = 0.5 * (U_n + f3Array0[j][i + 1]);
	Real3 U_i_jph = 0.5 * (U_n + f3Array0[j + 1][i]);

	GroupMemoryBarrierWithGroupSync();
	
	// find the 2nd parts of the low-order LF fluxes for the h component
	Real lowAlphaX = max(alpha_x.x, max(alpha_x.y, alpha_x.z));
	Real lowAlphaY = max(alpha_y.x, max(alpha_y.y, alpha_y.z));
	Real hLowFlux_x = -0.5 * lowAlphaX * (f3Array0[j][i + 1].x - U_n.x);
	Real hLowFlux_y = -0.5 * lowAlphaY * (f3Array0[j + 1][i].x - U_n.x);

	// evaluate the fluxes
	Real3 flux_f = Real3(h * u, h * u * u + g * h * h / 2.0, h * u * v);
	Real3 flux_g = Real3(h * v, h * u * v, h * v * v + g * h * h / 2.0);

	// evaluate the Jacobian matrices
	Real3x3 dfdU = Real3x3(0.0, 1.0, 0.0,
				-u * u + g * h, 2.0 * u, 0.0,
				-u * v, v, u);
	Real3x3 dgdU = Real3x3(0.0, 0.0, 1.0,
				-u * v, v, u,
				-v * v + g * h, 0.0, 2.0 * v);

	// store flux_f in f3Array1
	f3Array1[j][i] = flux_f;
	// store flux_g in f3Array2
	f3Array2[j][i] = flux_g;

	GroupMemoryBarrierWithGroupSync();

	// find the 1st parts of the low-order LF fluxes for the h component
	hLowFlux_x += 0.5 * (flux_f.x + f3Array1[j][i + 1].x);
	hLowFlux_y += 0.5 * (flux_g.x + f3Array2[j + 1][i].x);

	// find the 2nd order flux derivatives
	Real3 dfdx = (1.0 / (2.0 * dx)) * (f3Array1[j][i + 1] - f3Array1[j][i - 1]);
	Real3 dgdy = (1.0 / (2.0 * dy)) * (f3Array2[j + 1][i] - f3Array2[j - 1][i]);

	// find the 2nd order source term
	Real3 S_low = Real3(0.0,
			    0.5 * g * (1.0 / (2.0 * dx))
			    * (SRC1(f1Array0[j][i + 1]) - SRC1(f1Array0[j][i - 1]))
			    - g * (h + b) * (1.0 / (2.0 * dx))
			    * (SRC2(f1Array0[j][i + 1]) - SRC2(f1Array0[j][i - 1])),

			    0.5 * g * (1.0 / (2.0 * dy))
			    * (SRC1(f1Array0[j + 1][i]) - SRC1(f1Array0[j - 1][i]))
			    - g * (h + b) * (1.0 / (2.0 * dy))
			    * (SRC2(f1Array0[j + 1][i]) - SRC2(f1Array0[j - 1][i])));

	// find the time-averaged fluxes
	Real3 F_tilde = flux_f + (dt / 2.0) * mul(dfdU, S_low - dfdx - dgdy);
	Real3 G_tilde = flux_g + (dt / 2.0) * mul(dgdU, S_low - dfdx - dgdy);

	// store the low order flux in x- and y-directions in f3Array0 x- and y-component
	f3Array0[j][i].xy = Real2(hLowFlux_x, hLowFlux_y);

	GroupMemoryBarrierWithGroupSync();

	// find GAMMA
	Real GAMMA = -(h - (dt / dx) * (hLowFlux_x - f3Array0[j][i - 1].x) - (dt / dy) * (hLowFlux_y - f3Array0[j - 1][i].y));

	// =========================== BEGIN WENO RECONSTRUCTION IN X-DIRECTION =========================== //

	// store F_tilde in f3Array1
	f3Array1[j][i] = F_tilde;
	// store the conserved quantities for the flux splitting in f3Array2
	f3Array2[j][i] = Real3(h + b, U_n.y, U_n.z);

	GroupMemoryBarrierWithGroupSync();

	// set the flux boundaries
	if (((i + bcOffsets.x) >= 0) && ((i + bcOffsets.x) < (int)THREADS_X)) {
		f3Array1[j][i + bcOffsets.x] = F_tilde * Real3(-1.0, 1.0, 1.0);
	}

	GroupMemoryBarrierWithGroupSync();

	// define some shorthands based on the x-directional average state
	Real u_avg = U_iph_j.y / U_iph_j.x;
	Real v_avg = U_iph_j.z / U_iph_j.x;
	Real c = sqrt((float)(g * U_iph_j.x));

	// find the left and right eigenvector matrices for the x-directional flux Jacobian
	Real3x3 R_x = Real3x3(1.0, 0.0, 1.0,
			      u_avg - c, 0.0, u_avg + c,
			      v_avg, 1.0, v_avg);

	Real3x3 L_x = Real3x3((u_avg + c) / (2.0 * c), -1.0 / (2.0 * c), 0.0,
			      -v_avg, 0.0, 1.0,
			      -(u_avg - c) / (2.0 * c), 1.0 / (2.0 * c), 0.0);

	R_x = (U_iph_j.x < REV_CONST) ? Real3x3(1.0, 0.0, 0.0,
					    	0.0, 1.0, 0.0,
						0.0, 0.0, 1.0) : R_x;

	L_x = (U_iph_j.x < REV_CONST) ? Real3x3(1.0, 0.0, 0.0,
						0.0, 1.0, 0.0,
						0.0, 0.0, 1.0) : L_x;

	// do the flux splitting
	Real3 f_plus_x_im1_j = 0.5 * (mul(L_x, f3Array1[j][i - 1]) + alpha_x * mul(L_x, f3Array2[j][i - 1]));
	Real3 f_plus_x_i_j =   0.5 * (mul(L_x, f3Array1[j][i])     + alpha_x * mul(L_x, f3Array2[j][i]    ));
	Real3 f_plus_x_ip1_j = 0.5 * (mul(L_x, f3Array1[j][i + 1]) + alpha_x * mul(L_x, f3Array2[j][i + 1]));

	Real3 s1_x_im1_j = mul(L_x, Real3(0.0, SRC1(f1Array0[j][i - 1]), 0.0));
	Real3 s1_x_i_j   = mul(L_x, Real3(0.0, SRC1(f1Array0[j][i]), 0.0));
	Real3 s1_x_ip1_j = mul(L_x, Real3(0.0, SRC1(f1Array0[j][i + 1]), 0.0));
	Real3 s1_x_ip2_j = mul(L_x, Real3(0.0, SRC1(f1Array0[j][i + 2]), 0.0));

	Real3 s2_x_im1_j = mul(L_x, Real3(0.0, SRC2(f1Array0[j][i - 1]), 0.0));
	Real3 s2_x_i_j   = mul(L_x, Real3(0.0, SRC2(f1Array0[j][i]), 0.0));
	Real3 s2_x_ip1_j = mul(L_x, Real3(0.0, SRC2(f1Array0[j][i + 1]), 0.0));
	Real3 s2_x_ip2_j = mul(L_x, Real3(0.0, SRC2(f1Array0[j][i + 2]), 0.0));

	Real3 f_iph_plus_x, f_iph_minus_x;
	Real3 s1_iph_plus_x, s2_iph_plus_x, s1_iph_minus_x, s2_iph_minus_x;

	// do the reconstruction
	WENO3PlusX(f_plus_x_im1_j, f_plus_x_i_j, f_plus_x_ip1_j,
		   s1_x_im1_j, s1_x_i_j, s1_x_ip1_j,
		   s2_x_im1_j, s2_x_i_j, s2_x_ip1_j,
		   f_iph_plus_x, s1_iph_plus_x, s2_iph_plus_x);

	// do the flux splitting
	Real3 f_minus_x_i_j   = 0.5 * (mul(L_x, f3Array1[j][i])	    - alpha_x * mul(L_x, f3Array2[j][i]    ));
	Real3 f_minus_x_ip1_j = 0.5 * (mul(L_x, f3Array1[j][i + 1]) - alpha_x * mul(L_x, f3Array2[j][i + 1]));
	Real3 f_minus_x_ip2_j = 0.5 * (mul(L_x, f3Array1[j][i + 2]) - alpha_x * mul(L_x, f3Array2[j][i + 2]));

	// do the reconstruction
	WENO3MinusX(f_minus_x_i_j, f_minus_x_ip1_j, f_minus_x_ip2_j,
		    s1_x_i_j, s1_x_ip1_j, s1_x_ip2_j,
		    s2_x_i_j, s2_x_ip1_j, s2_x_ip2_j,
		    f_iph_minus_x, s1_iph_minus_x, s2_iph_minus_x);

	Real3 f_iph_x = mul(R_x, f_iph_plus_x + f_iph_minus_x);
	Real s1_iph_x = mul(R_x, 0.5 * s1_iph_plus_x + 0.5 * s1_iph_minus_x).y;
	Real s2_iph_x = mul(R_x, 0.5 * s2_iph_plus_x + 0.5 * s2_iph_minus_x).y;

	// store G_tilde f3Array0
	f3Array0[j][i] = G_tilde;

	GroupMemoryBarrierWithGroupSync();

	// set the flux boundaries
	if (((j + bcOffsets.y) >= 0) && ((j + bcOffsets.y) < (int)THREADS_Y)) {
		f3Array0[j + bcOffsets.y][i] = G_tilde * Real3(-1.0, 1.0, 1.0);
	}

	GroupMemoryBarrierWithGroupSync();
	
	// =========================== BEGIN WENO RECONSTRUCTION IN Y-DIRECTION =========================== //

	// update the shorthands for the y-directional average state
	u_avg = U_i_jph.y / U_i_jph.x;
	v_avg = U_i_jph.z / U_i_jph.x;
	c = sqrt((float)(g * U_i_jph.x));

	// find the left and right eigenvector matrices for the y-directional flux Jacobian
	Real3x3 R_y = Real3x3(1.0, 0.0, 1.0,
			      u_avg, 1.0, u_avg,
			      v_avg - c, 0.0, v_avg + c);

	Real3x3 L_y = Real3x3((v_avg + c) / (2.0 * c), 0.0, -1.0 / (2.0 * c),
			      -u_avg, 1.0, 0.0,
			      -(v_avg - c) / (2.0 * c), 0.0, 1.0 / (2.0 * c));

	R_y = (U_i_jph.x < REV_CONST) ? Real3x3(1.0, 0.0, 0.0,
						0.0, 1.0, 0.0,
						0.0, 0.0, 1.0) : R_y;

	L_y = (U_i_jph.x < REV_CONST) ? Real3x3(1.0, 0.0, 0.0,
						0.0, 1.0, 0.0,
						0.0, 0.0, 1.0) : L_y;

	// do the flux splitting
	Real3 f_plus_y_i_jm1 = 0.5 * (mul(L_y, f3Array0[j - 1][i]) + alpha_y * mul(L_y, f3Array2[j - 1][i]));
	Real3 f_plus_y_i_j   = 0.5 * (mul(L_y, f3Array0[j][i])	   + alpha_y * mul(L_y, f3Array2[j][i]    ));
	Real3 f_plus_y_i_jp1 = 0.5 * (mul(L_y, f3Array0[j + 1][i]) + alpha_y * mul(L_y, f3Array2[j + 1][i]));

	Real3 s1_y_i_jm1 = mul(L_y, Real3(0.0, 0.0, SRC1(f1Array0[j - 1][i])));
	Real3 s1_y_i_j   = mul(L_y, Real3(0.0, 0.0, SRC1(f1Array0[j][i])));
	Real3 s1_y_i_jp1 = mul(L_y, Real3(0.0, 0.0, SRC1(f1Array0[j + 1][i])));
	Real3 s1_y_i_jp2 = mul(L_y, Real3(0.0, 0.0, SRC1(f1Array0[j + 2][i])));

	Real3 s2_y_i_jm1 = mul(L_y, Real3(0.0, 0.0, SRC2(f1Array0[j - 1][i])));
	Real3 s2_y_i_j   = mul(L_y, Real3(0.0, 0.0, SRC2(f1Array0[j][i])));
	Real3 s2_y_i_jp1 = mul(L_y, Real3(0.0, 0.0, SRC2(f1Array0[j + 1][i])));
	Real3 s2_y_i_jp2 = mul(L_y, Real3(0.0, 0.0, SRC2(f1Array0[j + 2][i])));

	Real3 f_iph_plus_y, f_iph_minus_y;
	Real3 s1_iph_plus_y, s2_iph_plus_y, s1_iph_minus_y, s2_iph_minus_y;

	// do the reconstruction
	WENO3PlusY(f_plus_y_i_jm1, f_plus_y_i_j, f_plus_y_i_jp1,
		   s1_y_i_jm1, s1_y_i_j, s1_y_i_jp1,
		   s2_y_i_jm1, s2_y_i_j, s2_y_i_jp1,
		   f_iph_plus_y, s1_iph_plus_y, s2_iph_plus_y);

	// do the flux splitting
	Real3 f_minus_y_i_j   = 0.5 * (mul(L_y, f3Array0[j][i])	  - alpha_y * mul(L_y, f3Array2[j][i]));
	Real3 f_minus_y_i_jp1 = 0.5 * (mul(L_y, f3Array0[j + 1][i]) - alpha_y * mul(L_y, f3Array2[j + 1][i]));
	Real3 f_minus_y_i_jp2 = 0.5 * (mul(L_y, f3Array0[j + 2][i]) - alpha_y * mul(L_y, f3Array2[j + 2][i]));

	// do the reconstruction
	WENO3MinusY(f_minus_y_i_j, f_minus_y_i_jp1, f_minus_y_i_jp2,
		    s1_y_i_j, s1_y_i_jp1, s1_y_i_jp2,
		    s2_y_i_j, s2_y_i_jp1, s2_y_i_jp2,
		    f_iph_minus_y, s1_iph_minus_y, s2_iph_minus_y);

	Real3 f_iph_y = mul(R_y, f_iph_plus_y + f_iph_minus_y);
	Real s1_iph_y = mul(R_y, 0.5 * s1_iph_plus_y + 0.5 * s1_iph_minus_y).z;
	Real s2_iph_y = mul(R_y, 0.5 * s2_iph_plus_y + 0.5 * s2_iph_minus_y).z;

	// =========================== BEGIN POSITIVITY PRESERVATION =========================== //

	Real F_i = (dt / dx) * (f_iph_x.x - hLowFlux_x);
	Real F_j = (dt / dy) * (f_iph_y.x - hLowFlux_y);

	// store F_i in f3Array1 x-component and F_j in f3Array1 y-component
	f3Array1[j][i].xy = Real2(F_i, F_j);

	GroupMemoryBarrierWithGroupSync();

	Real F_iph_j = -F_i;
	Real F_imh_j =  f3Array1[j][i - 1].x;
	Real F_i_jph = -F_j;
	Real F_i_jmh =  f3Array1[j - 1][i].y;

	int alpha = F_iph_j < 0;
	int beta = F_imh_j < 0;
	int gamma = F_i_jph < 0;
	int delta = F_i_jmh < 0;
	Real R = (alpha * F_iph_j + beta * F_imh_j + gamma * F_i_jph + delta * F_i_jmh);
	Real Q = min(1, (R != 0) ? GAMMA / R : 0.0);

	Real L_r = (1 - alpha) + alpha * Q;
	Real L_l = (1 - beta) + beta * Q;
	Real L_u = (1 - gamma) + gamma * Q;
	Real L_d = (1 - delta) + delta * Q;

	// store L_l in f3Array0 x-component, L_d in f3Array0 y-component
	f3Array0[j][i].xy = Real2(L_l, L_d);

	GroupMemoryBarrierWithGroupSync();

	Real theta_i = min(L_r, f3Array0[j][i + 1].x);
	Real theta_j = min(L_u, f3Array0[j + 1][i].y);

	// find the final fluxes for the h component
	Real f_tilde = theta_i * (f_iph_x.x - hLowFlux_x) + hLowFlux_x;
	Real g_tilde = theta_j * (f_iph_y.x - hLowFlux_y) + hLowFlux_y;

	// =========================== BEGIN TIME INTEGRATION =========================== //

	// store the final fluxes in x- and y-directions in f3Array1 and f3Array2 respectively
	Real3 finalFlux_x = Real3(f_tilde, f_iph_x.yz);
	Real3 finalFlux_y = Real3(g_tilde, f_iph_y.yz);
	f3Array1[j][i] = finalFlux_x;
	f3Array2[j][i] = finalFlux_y;

	GroupMemoryBarrierWithGroupSync();

	Real3 U_new = U_n - (dt / dx) * (finalFlux_x - f3Array1[j][i - 1]) - (dt / dy) * (finalFlux_y - f3Array2[j - 1][i]);
	
	// store s1_iph_x in f1Array0
	f1Array0[j][i] = s1_iph_x;
	// store s1_iph_y, s2_iph_x, s2_iph_y in f3Array0
	f3Array0[j][i] = Real3(s1_iph_y, s2_iph_x, s2_iph_y);

	GroupMemoryBarrierWithGroupSync();

	U_new += 0.5 * g * Real3(0.0,
				 (dt / dx) * (s1_iph_x - f1Array0[j][i - 1]),
				 (dt / dy) * (s1_iph_y - f3Array0[j - 1][i].x));

	U_new -= g * (h + b) * Real3(0.0,
				     (dt / dx) * (s2_iph_x - f3Array0[j][i - 1].y),
				     (dt / dy) * (s2_iph_y - f3Array0[j - 1][i].z));

	// find the velocities for the current cell
	Real h4 = U_new.x * U_new.x;
	h4 *= h4;
	// desingularize the velocities if needed
	Real u_des = (U_new.x < desingularization_epsilon) ? sqrt(2.0) * U_new.x * U_new.y / sqrt((float)(h4 + max(h4, desingularization_epsilon)))
							   : U_new.y / U_new.x;
	Real v_des = (U_new.x < desingularization_epsilon) ? sqrt(2.0) * U_new.x * U_new.z / sqrt((float)(h4 + max(h4, desingularization_epsilon)))
							   : U_new.z / U_new.x;

	// consistency requirement
	U_new.x = max(0.0, U_new.x);
	U_new.y = U_new.x * u_des;
	U_new.z = U_new.x * v_des;

	// check if dealing with an inner domain cell
	bool isDomainInner = (i_global >= (int)bcCount) && (i_global < (int)(nWidth + bcCount)) && (j_global >= (int)bcCount) && (j_global < (int)(nHeight + bcCount));
	// check if in the inner domain of the current grid patch
	bool isPatchInner = (i >= (int)bcCount) && (i < (int)(THREADS_X - bcCount)) && (j >= (int)bcCount) && (j < (int)(THREADS_Y - bcCount));

	// store the result
	if (isDomainInner && isPatchInner) {
#if PRECISION == DOUBLE_PRECISION
		uint u_bc = asuint(bcData);
		double d_bc = asdouble(u_bc, (uint)0);

		results_1[int2(i_global, j_global)] = uint4(PackDouble(U_new.x), PackDouble(U_new.y));
		results_2[int2(i_global, j_global)] = uint4(PackDouble(U_new.z), PackDouble(d_bc));
#else
		results[int2(i_global, j_global)] = Real4(U_new, bcData);
#endif

		// set the boundary conditions
#if PRECISION == DOUBLE_PRECISION
		float f_b = BC_WRITE_OFFSET;
		uint u_b = asuint(f_b);
		double d_b = asdouble(u_b, (uint)0);
		
		Real4 boundaryValue = Real4(U_new * Real3(1, -1, -1), d_b);

		results_1[int2(i_global + bcOffsets.x, j_global)] = uint4(PackDouble(boundaryValue.x), PackDouble(boundaryValue.y));
		results_2[int2(i_global + bcOffsets.x, j_global)] = uint4(PackDouble(boundaryValue.z), PackDouble(boundaryValue.w));

		results_1[int2(i_global, j_global + bcOffsets.y)] = uint4(PackDouble(boundaryValue.x), PackDouble(boundaryValue.y));
		results_2[int2(i_global, j_global + bcOffsets.y)] = uint4(PackDouble(boundaryValue.z), PackDouble(boundaryValue.w));

#else
		results[int2(i_global + bcOffsets.x, j_global)] = Real4(U_new * Real3(1.0, -1.0, -1.0), BC_WRITE_OFFSET);
		results[int2(i_global, j_global + bcOffsets.y)] = Real4(U_new * Real3(1.0, -1.0, -1.0), BC_WRITE_OFFSET);
#endif
	}
}
