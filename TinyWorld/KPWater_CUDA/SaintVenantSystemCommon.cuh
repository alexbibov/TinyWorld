#ifndef SVS__COMMON__
#define SVS__COMMON__


#include <cuda_runtime.h>

#include "SaintVenantSystemCompileParams.cuh"


namespace SaintVenantSystem
{
#if defined SVS_PRECISION_DOUBLE && defined SVS_PRECISION_FLOAT
#undef SVS_PRECISION_FLOAT
#endif
#if !defined SVS_PRECISION_DOUBLE && !defined SVS_PRECISION_FLOAT
#define SVS_PRECISION_FLOAT
#endif
#ifdef SVS_PRECISION_FLOAT
	typedef float Numeric;
	typedef float4 Numeric4;
	typedef float3 Numeric3;
#define _min(a,b)fminf(a,b)
#define _max(a,b)fmaxf(a,b)
#define _sqrt(x)__fsqrt_rn(x)
#define _pow(x,y) powf(x,y)
#endif
#ifdef SVS_PRECISION_DOUBLE
	typedef double Numeric;
	typedef double4 Numeric4;
	typedef double3 Numeric3;
#define _min(a,b)fmin(a,b)
#define _max(a,b)fmax(a,b)
#define _sqrt(x)__dsqrt_rn(x)
#define _pow(x,y) pow(x,y)
#endif


	//Types used to store parameters of Saint-Venant system and to represent the boundary values
	typedef struct{
		Numeric *w;
		Numeric *hu;
		Numeric *hv;
	}SVSVarU;

	typedef struct{
		Numeric *w;
		Numeric *w_edge;
		Numeric *hu;
		Numeric *hu_edge;
		Numeric *hv;
		Numeric *hv_edge;
	}SVSBoundary;

	typedef struct{
		unsigned int grid_width;	//Horizontal dimension of the grid
		unsigned int grid_height;	//Vertical dimension of the grid
		Numeric dx;					//Horizontal discretization step
		Numeric dy;					//Vertical discretization step
		const Numeric *B;           //(2*grid_height+1)-by-(2*grid_width+1) matrix representing bilinear  
		//approximation of the bottom topography
		Numeric g;					//Gravity constant
		Numeric theta;				//Nonlinear limiter parameter. Must lie between 1 and 2
		Numeric eps;				//Desingularization parameter for the forces (u,v)
	}SVSParameters;
}


#endif
