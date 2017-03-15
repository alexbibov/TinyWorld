#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 16) in;

layout(rgba32f) uniform restrict readonly imageBuffer ibGradientGrid;	//buffer texture receiving the gradient field required for creation of the Perlin noise
uniform uint uOffset;	//offset from the beginning of the buffer containing the gradient grid. This offset is used when data is read from and written into the buffer
layout(r32f) uniform restrict image3D i3dOutNoise;	//texture receiving the output values of the noise
uniform uvec3 uv3GradientGridSize;	//resolution of the gradient grid at the current scale level of the noise
uniform bool bFirstCall;	//equals 'true' whilst i3dOutNoise does not yet contain any output noise values
uniform uint uNumScaleLevels;	//number of scale levels employed to produce the noise map
uniform bool isPeriodic;	//equals 'true' if the noise map being generated should be periodic; equals 'false' otherwise


//Modified ceil function defined as mceil(x)=floor(x)+1 (for normal ceil the latter does not hold when x is an integer)
vec3 mceil(vec3 v3x){ return floor(v3x) + 1; }


shared vec3 v3Gradients[1024];	//buffer receiving information about the gradient vectors

void main()
{
	uvec3 uv3OutNoiseSize = imageSize(i3dOutNoise);	//retrieve size of the output noise map
	if(any(greaterThanEqual(gl_GlobalInvocationID, vec3(uv3OutNoiseSize)))) return;	//drop the dummy threads
	uvec3 uv3CorrectedGradientGridSize = uv3GradientGridSize + uint(isPeriodic);


	uvec3 uv3BlockCoverAreaLLF = uvec3(min(floor(gl_WorkGroupID*gl_WorkGroupSize*vec3(uv3CorrectedGradientGridSize-1)/(uv3OutNoiseSize-1.0f)), uv3CorrectedGradientGridSize-2));	//lower-left-far corner of the cubic area occupied by the block
	uvec3 uv3BlockCoverAreaURN = uvec3(min(floor(((gl_WorkGroupID+1)*gl_WorkGroupSize-1)*vec3(uv3CorrectedGradientGridSize-1)/(uv3OutNoiseSize-1.0f)), uv3CorrectedGradientGridSize-2));	//upper-right-near corner of the cubic area occupied by the block
	
	//Load the gradient vectors accessed by the current block into the shared memory
	if(all(lessThanEqual(gl_LocalInvocationID, uv3BlockCoverAreaURN-uv3BlockCoverAreaLLF+1)))
	{
		v3Gradients[gl_LocalInvocationID.z*gl_WorkGroupSize.x*gl_WorkGroupSize.y + gl_LocalInvocationID.y*gl_WorkGroupSize.x + gl_LocalInvocationID.x] = 
			imageLoad(ibGradientGrid, 
			int(uOffset + 
			mod((uv3BlockCoverAreaLLF.z+gl_LocalInvocationID.z), uv3GradientGridSize.z)*uv3GradientGridSize.x*uv3GradientGridSize.y + 
			mod((uv3BlockCoverAreaLLF.y+gl_LocalInvocationID.y), uv3GradientGridSize.y)*uv3GradientGridSize.x +
			mod(uv3BlockCoverAreaLLF.x+gl_LocalInvocationID.x, uv3GradientGridSize.x))).rgb;
	}
	barrier();
	memoryBarrierShared();



	//Obtain position of the currently evaluated spatial point within the gradient grid
	vec3 v3CurrentPoint = gl_GlobalInvocationID/vec3(uv3OutNoiseSize-1)*(uv3CorrectedGradientGridSize-1);


	//Get the neighboring points from the gradient grid
	uvec3 uv3LLF = uvec3(floor(v3CurrentPoint));	//lower-left-far vertex of the cell containing the currently processed pixel of the noise map
	uvec3 uv3URN = uvec3(mceil(v3CurrentPoint));	//lower-left-near vertex of the cell containing the currently processed pixel of the noise map


	//Compute the distance vectors
	vec3 v3XmYmZm_distance = v3CurrentPoint - uv3LLF;
	vec3 v3XpYmZm_distance = v3CurrentPoint - uvec3(uv3URN.x, uv3LLF.yz);
	vec3 v3XmYpZm_distance = v3CurrentPoint - uvec3(uv3LLF.x, uv3URN.y, uv3LLF.z);
	vec3 v3XpYpZm_distance = v3CurrentPoint - uvec3(uv3URN.xy, uv3LLF.z);

	vec3 v3XmYmZp_distance = v3CurrentPoint - uvec3(uv3LLF.xy, uv3URN.z);
	vec3 v3XpYmZp_distance = v3CurrentPoint - uvec3(uv3URN.x, uv3LLF.y, uv3URN.z);
	vec3 v3XmYpZp_distance = v3CurrentPoint - uvec3(uv3LLF.x, uv3URN.yz);
	vec3 v3XpYpZp_distance = v3CurrentPoint - uv3URN;


	//Compute the dot products
	float v3XmYmZm_dp = dot(v3XmYmZm_distance, v3Gradients[(uv3LLF.z-uv3BlockCoverAreaLLF.z)*gl_WorkGroupSize.x*gl_WorkGroupSize.y + (uv3LLF.y-uv3BlockCoverAreaLLF.y)*gl_WorkGroupSize.x + (uv3LLF.x-uv3BlockCoverAreaLLF.x)]);
	float v3XpYmZm_dp = dot(v3XpYmZm_distance, v3Gradients[(uv3LLF.z-uv3BlockCoverAreaLLF.z)*gl_WorkGroupSize.x*gl_WorkGroupSize.y + (uv3LLF.y-uv3BlockCoverAreaLLF.y)*gl_WorkGroupSize.x + (uv3URN.x-uv3BlockCoverAreaLLF.x)]);
	float v3XmYpZm_dp = dot(v3XmYpZm_distance, v3Gradients[(uv3LLF.z-uv3BlockCoverAreaLLF.z)*gl_WorkGroupSize.x*gl_WorkGroupSize.y + (uv3URN.y-uv3BlockCoverAreaLLF.y)*gl_WorkGroupSize.x + (uv3LLF.x-uv3BlockCoverAreaLLF.x)]);
	float v3XpYpZm_dp = dot(v3XpYpZm_distance, v3Gradients[(uv3LLF.z-uv3BlockCoverAreaLLF.z)*gl_WorkGroupSize.x*gl_WorkGroupSize.y + (uv3URN.y-uv3BlockCoverAreaLLF.y)*gl_WorkGroupSize.x + (uv3URN.x-uv3BlockCoverAreaLLF.x)]);

	float v3XmYmZp_dp = dot(v3XmYmZp_distance, v3Gradients[(uv3URN.z-uv3BlockCoverAreaLLF.z)*gl_WorkGroupSize.x*gl_WorkGroupSize.y + (uv3LLF.y-uv3BlockCoverAreaLLF.y)*gl_WorkGroupSize.x + (uv3LLF.x-uv3BlockCoverAreaLLF.x)]);
	float v3XpYmZp_dp = dot(v3XpYmZp_distance, v3Gradients[(uv3URN.z-uv3BlockCoverAreaLLF.z)*gl_WorkGroupSize.x*gl_WorkGroupSize.y + (uv3LLF.y-uv3BlockCoverAreaLLF.y)*gl_WorkGroupSize.x + (uv3URN.x-uv3BlockCoverAreaLLF.x)]);
	float v3XmYpZp_dp = dot(v3XmYpZp_distance, v3Gradients[(uv3URN.z-uv3BlockCoverAreaLLF.z)*gl_WorkGroupSize.x*gl_WorkGroupSize.y + (uv3URN.y-uv3BlockCoverAreaLLF.y)*gl_WorkGroupSize.x + (uv3LLF.x-uv3BlockCoverAreaLLF.x)]);
	float v3XpYpZp_dp = dot(v3XpYpZp_distance, v3Gradients[(uv3URN.z-uv3BlockCoverAreaLLF.z)*gl_WorkGroupSize.x*gl_WorkGroupSize.y + (uv3URN.y-uv3BlockCoverAreaLLF.y)*gl_WorkGroupSize.x + (uv3URN.x-uv3BlockCoverAreaLLF.x)]);


	//Perform interpolation
	float fInterpYmZm = (v3CurrentPoint.x - uv3LLF.x)*v3XpYmZm_dp + (uv3URN.x - v3CurrentPoint.x)*v3XmYmZm_dp;
	float fInterpYpZm = (v3CurrentPoint.x - uv3LLF.x)*v3XpYpZm_dp + (uv3URN.x - v3CurrentPoint.x)*v3XmYpZm_dp;
	float fInterpZm = (v3CurrentPoint.y - uv3LLF.y)*fInterpYpZm + (uv3URN.y - v3CurrentPoint.y)*fInterpYmZm;

	float fInterpYmZp = (v3CurrentPoint.x - uv3LLF.x)*v3XpYmZp_dp + (uv3URN.x - v3CurrentPoint.x)*v3XmYmZp_dp;
	float fInterpYpZp = (v3CurrentPoint.x - uv3LLF.x)*v3XpYpZp_dp + (uv3URN.x - v3CurrentPoint.x)*v3XmYpZp_dp;
	float fInterpZp = (v3CurrentPoint.y - uv3LLF.y)*fInterpYpZp + (uv3URN.y - v3CurrentPoint.y)*fInterpYmZp;

	float fNoiseValue = (v3CurrentPoint.z - uv3LLF.z)*fInterpZp + (uv3URN.z - v3CurrentPoint.z)*fInterpZm;


	//Write the output data
	float fPrecedingNoiseScale = 0;
	if(!bFirstCall) fPrecedingNoiseScale = imageLoad(i3dOutNoise, ivec3(gl_GlobalInvocationID)).r;
	float fOutputNoiseValue = fPrecedingNoiseScale + 0.5f*(fNoiseValue+1.0f)/uNumScaleLevels;
	imageStore(i3dOutNoise, ivec3(gl_GlobalInvocationID), vec4(fOutputNoiseValue, 0, 0 ,0));
}