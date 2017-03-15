#version 430 core

#define pi 3.1415926535897932384626433832795f

layout(local_size_x = 8, local_size_y = 8, local_size_z = 16) in;


layout(rgba32f) uniform restrict imageBuffer ibGradientGrid;	//buffer texture receiving the gradient field required for creation of the Perlin noise
uniform uvec3 uv3GradientGridSize;	//resolution of the gradient grid at the current scale level of the noise
uniform uvec4 uv4Seed;	//seed value used to instantiate xorshift128+ random number generator
uniform bool bFirstCall;	//equals 'true' when the gradient grid gets generated for the first time at given scale level of the noise
uniform bool bUpdateGradients;	//if equals 'true' the new gradient field gets generated based on the previous values. The value of this flag is ignored when bFirstCall=true
uniform uint uOffset;	//offset from the beginning of the buffer containing the gradient grid. This offset is used when data is read from and written into the buffer
uniform float fGradientUpdateRate;	//has effect only if bUpdateGradients=true. Describes how rapid should the gradient grid evolve



//Treats input 'v' as a 64-bit value and shifts its bitwise representation leftwards for n bit positions
uvec2 lbs(uvec2 v, uint n)
{
	uint p1 = v.x << n;
	uint aux = uint(n<=32)*(v.y>>32-n) + uint(n>32)*(v.y<<n-32);
	p1 |= aux;
	uint p2 = v.y << n;

	return uvec2(p1, p2);
}


//Treats input 'v' as a 64-bit value and shifts its bitwise representation rightwards for n bit positions
uvec2 rbs(uvec2 v, uint n)
{
	uint p2 = v.y >> n;
	uint aux = uint(n<=32)*(v.x<<32-n) + uint(n>32)*(v.x>>n-32);
	p2 |= aux;
	uint p1 = v.x >> n;

	return uvec2(p1, p2);
}


//Implements a "light" version of xorshift128+ PRNG as suggested by S.Vigna
uvec2 xorshift128plus(uvec4 uv4State)
{
	uvec2 v2S1 = uv4State.xy;
	const uvec2 v2S0 = uv4State.zw;

	v2S1 ^= lbs(v2S1, 23);
	v2S1 = v2S1 ^ v2S0 ^ rbs(v2S1, 18) ^ rbs(v2S0, 5);

	return v2S1 + v2S0;
}


//Implements the Box-Muller transform. The components of the input vector 'v2U' must be uniformly distributed on segment [0,1].
//The ouput is a two-component vector with elements having standard normal distribution. These elements of the ouput vector are guaranteed to be independent.
vec2 BoxMuller(vec2 v2U)
{
	float R = sqrt(-2.0f*log(v2U.x));
	float theta = 2*pi*v2U.y;

	return R*vec2(cos(theta), sin(theta));
}


//Modified signum function:
//msgn(x) =  1, if x >= 0
//msgn(x) = -1, if x <  1
float msgn(float x)
{
	return (float(x >= 0) - float(x < 0));
}


void main()
{
	//Check if the current executing thread is out of the gradient grid bounds
	if(any(greaterThanEqual(gl_GlobalInvocationID, vec3(uv3GradientGridSize)))) return;

	uint uAux = gl_GlobalInvocationID.z*gl_NumWorkGroups.x*gl_WorkGroupSize.x*gl_NumWorkGroups.y*gl_WorkGroupSize.y + gl_GlobalInvocationID.y*gl_NumWorkGroups.x*gl_WorkGroupSize.x + gl_GlobalInvocationID.x + 1;
	uAux ^= uAux << 13;
	vec2 v2Aux = vec2(xorshift128plus(uv4Seed*uAux)) * 0.00000000023283064365386962890625f;

	//Write gradient data into the grid
	if(!bFirstCall && bUpdateGradients)
	{
		vec3 v3OldGradient = imageLoad(ibGradientGrid, int(uOffset + gl_GlobalInvocationID.z*uv3GradientGridSize.x*uv3GradientGridSize.y + gl_GlobalInvocationID.y*uv3GradientGridSize.x + gl_GlobalInvocationID.x)).rgb;
		float fOldZonalAngle = asin(v3OldGradient.z);
		float fMeridionalVectorLength = sqrt(1 - pow(v3OldGradient.z, 2.0f));
		vec2 v2MeridionalVector = float(fMeridionalVectorLength>0)*clamp(v3OldGradient.xy/fMeridionalVectorLength, -1, 1) + float(fMeridionalVectorLength==0)*vec2(cos(dot(v2Aux, vec2(0.5f))), sin(dot(v2Aux, vec2(0.5f))));
		float fOldMeridionalAngle = acos(v2MeridionalVector.x*msgn(v2MeridionalVector.y)) + float(v2MeridionalVector.y<0)*pi;
		vec2 v2NewSphericalCoords = vec2(fOldZonalAngle, fOldMeridionalAngle) + fGradientUpdateRate*BoxMuller(v2Aux);
		imageStore(ibGradientGrid, int(uOffset + gl_GlobalInvocationID.z*uv3GradientGridSize.x*uv3GradientGridSize.y + gl_GlobalInvocationID.y*uv3GradientGridSize.x + gl_GlobalInvocationID.x), 
			vec4(cos(v2NewSphericalCoords.x)*cos(v2NewSphericalCoords.y), cos(v2NewSphericalCoords.x)*sin(v2NewSphericalCoords.y), sin(v2NewSphericalCoords.x), 0));
	}
	else
	{
		float fZonalAngle = pi*(v2Aux.x - 0.5f);
		float fMeridionalAngle = 2*pi*v2Aux.y;
		imageStore(ibGradientGrid, int(uOffset + gl_GlobalInvocationID.z*uv3GradientGridSize.x*uv3GradientGridSize.y + gl_GlobalInvocationID.y*uv3GradientGridSize.x + gl_GlobalInvocationID.x), 
			vec4(cos(fZonalAngle)*cos(fMeridionalAngle), cos(fZonalAngle)*sin(fMeridionalAngle), sin(fZonalAngle), 0));
	}
}