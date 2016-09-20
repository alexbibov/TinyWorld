#version 430 core

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(rg32f) uniform restrict readonly imageBuffer ibGradientGrid;	//buffer texture receiving the gradient field required for creation of the Perlin noise
uniform uint uOffset;	//offset from the beginning of the buffer containing the gradient grid. This offset is used when data is read from and written into the buffer
layout(r32f) uniform restrict image2D i2dOutNoise;	//texture receiving the output values of the noise
uniform uvec2 uv2GradientGridSize;	//resolution of the gradient grid at the current scale level of the noise
uniform bool bFirstCall;	//equals 'true' whilst i2dOutNoise does not yet contain any output noise values
uniform uint uNumScaleLevels;	//number of scale levels employed to produce the noise map
uniform bool isPeriodic;	//equals 'true' if the noise map being generated should be periodic; equals 'false' otherwise


//Modified ceil function defined as mceil(x)=floor(x)+1 (for normal ceil the latter does not hold when x is an integer)
vec2 mceil(vec2 v2x){ return floor(v2x) + 1; }


shared vec2 v2Gradients[1024];	//buffer receiving information about the gradient vectors

void main()
{
	uvec2 uv2OutNoiseSize = imageSize(i2dOutNoise);	//retrieve size of the output noise map
	if(any(greaterThanEqual(gl_GlobalInvocationID.xy, vec2(uv2OutNoiseSize)))) return;	//drop the dummy threads
	uvec2 uv2CorrectedGradientGridSize = uv2GradientGridSize + uint(isPeriodic);	//compute gradient grid size taking into account periodicity if applicable


	uvec2 uv2BlockCoverAreaLL = uvec2(min(floor(gl_WorkGroupID.xy*gl_WorkGroupSize.xy*vec2(uv2CorrectedGradientGridSize-1)/(uv2OutNoiseSize-1.0f)), uv2CorrectedGradientGridSize-2));
	uvec2 uv2BlockCoverAreaUR = uvec2(min(floor(((gl_WorkGroupID.xy+1)*gl_WorkGroupSize.xy-1)*vec2(uv2CorrectedGradientGridSize-1)/(uv2OutNoiseSize-1.0f)), uv2CorrectedGradientGridSize-2));
	
	//Load the gradient vectors accessed by the current block into the shared memory
	if(all(lessThanEqual(gl_LocalInvocationID.xy, uv2BlockCoverAreaUR - uv2BlockCoverAreaLL + 1)))
	{
		v2Gradients[gl_LocalInvocationID.y*gl_WorkGroupSize.x+gl_LocalInvocationID.x] = 
			imageLoad(ibGradientGrid, int(uOffset + mod(uv2BlockCoverAreaLL.y+gl_LocalInvocationID.y, uv2GradientGridSize.y)*uv2GradientGridSize.x + mod(uv2BlockCoverAreaLL.x+gl_LocalInvocationID.x, uv2GradientGridSize.x))).rg;
	}
	barrier();
	memoryBarrierShared();


	//Obtain position of the currently evaluated spatial point within the gradient grid
	vec2 v2CurrentPoint = gl_GlobalInvocationID.xy/vec2(uv2OutNoiseSize-1)*(uv2CorrectedGradientGridSize-1);

	//Get the neighboring points from the gradient grid
	uvec2 uv2LL = uvec2(floor(v2CurrentPoint));	//lower-left corner of the cell containing the currently processed point of the noise map
	uvec2 uv2UR = uvec2(mceil(v2CurrentPoint));	//upper-right corner of the cell containing the currently processed point of the noise map

	//Compute the distance vectors
	vec2 v2XmYm_distance = v2CurrentPoint - uv2LL;
	vec2 v2XpYm_distance = v2CurrentPoint - uvec2(uv2UR.x, uv2LL.y);
	vec2 v2XmYp_distance = v2CurrentPoint - uvec2(uv2LL.x, uv2UR.y);
	vec2 v2XpYp_distance = v2CurrentPoint - uv2UR;

	//Compute the dot products
	float v2XmYm_dp = dot(v2XmYm_distance, v2Gradients[(uv2LL.y-uv2BlockCoverAreaLL.y)*gl_WorkGroupSize.x+(uv2LL.x-uv2BlockCoverAreaLL.x)]);
	float v2XpYm_dp = dot(v2XpYm_distance, v2Gradients[(uv2LL.y-uv2BlockCoverAreaLL.y)*gl_WorkGroupSize.x+(uv2UR.x-uv2BlockCoverAreaLL.x)]);
	float v2XmYp_dp = dot(v2XmYp_distance, v2Gradients[(uv2UR.y-uv2BlockCoverAreaLL.y)*gl_WorkGroupSize.x+(uv2LL.x-uv2BlockCoverAreaLL.x)]);
	float v2XpYp_dp = dot(v2XpYp_distance, v2Gradients[(uv2UR.y-uv2BlockCoverAreaLL.y)*gl_WorkGroupSize.x+(uv2UR.x-uv2BlockCoverAreaLL.x)]);

	//Perform interpolation
	float fYm = (v2CurrentPoint.x - uv2LL.x)*v2XpYm_dp + (uv2UR.x - v2CurrentPoint.x)*v2XmYm_dp;
	float fYp = (v2CurrentPoint.x - uv2LL.x)*v2XpYp_dp + (uv2UR.x - v2CurrentPoint.x)*v2XmYp_dp;
	float fNoiseValue = (v2CurrentPoint.y - uv2LL.y)*fYp + (uv2UR.y - v2CurrentPoint.y)*fYm;

	//Write the output data
	float fPrecedingNoiseScale = 0;
	if(!bFirstCall) fPrecedingNoiseScale = imageLoad(i2dOutNoise, ivec2(gl_GlobalInvocationID.xy)).r;
	float fOutputNoiseValue = fPrecedingNoiseScale + 0.5f*(fNoiseValue+1.0f)/uNumScaleLevels;
	imageStore(i2dOutNoise, ivec2(gl_GlobalInvocationID.xy), vec4(fOutputNoiseValue, 0, 0 ,0));
}