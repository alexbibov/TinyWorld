#version 430 core

const unsigned int max_sample_points = 256;	//maximal number of sample points that can be used in SSAO kernel

uniform float fFocalDistance;	//distance between the observer and the near plane of the view frustum
uniform vec4 v4FocalPlane;		//contains left, right, bottom, and top boundaries in this order of the focal plane 
uniform vec4 v4Viewport;		//contains viewport coordinates x and y and viewport dimensions width and height in this order
uniform mat4 m4ProjMat;			//projection transform (it must be the same transform as the one used during formation of the normal and linear depth buffers)

uniform float fKernelRadius;	 //radius of SSAO kernel
uniform float fOcclusionRange;	 //range of occlusion
uniform unsigned int numSamples; //factual number of samples used in computations


layout(std140) uniform SSAO_extra_params
{
	vec3 samples[max_sample_points];	//sample point locations
}extra_params;


uniform sampler2D s2dNoiseTexture;	//sampler object used to extract kernel random rotations



//source0 — screen-space normal map
//source1 — screen-space linear depth buffer
vec4 do_filtering(in sampler2D source0, in sampler2D source1)
{
	float L = v4FocalPlane.x;	float Vx = v4Viewport.x;
	float R = v4FocalPlane.y;	float Vy = v4Viewport.y;
	float B = v4FocalPlane.z;	float Vw = v4Viewport.z;
	float T = v4FocalPlane.w;	float Vh = v4Viewport.w;

	//Compute position of the current fragment on the focal plane
	vec2 v2FocalPlanePosition = vec2((gl_FragCoord.x - Vx)/Vw*(R-L)+L, (gl_FragCoord.y - Vy)/Vh*(T-B)+B);
	
	//Get linear depth of the current fragment
	float fDepth = texelFetch(source1, ivec2(gl_FragCoord.xy-0.5f), 0).r;

	//Compute position of the current fragment in viewer space
	vec3 v3FragmentPosition = vec3(-v2FocalPlanePosition.xy*fDepth/fFocalDistance, fDepth);

	//Extract normal vectors corresponding to the fragment
	vec3 v3Normal = normalize(texelFetch(source0, ivec2(gl_FragCoord.xy-0.5f), 0).rgb);

	//Z-axis of the kernel should be aligned with the currently selected normal vector.
	//We also perform bootstrapping by rotating the kernel randomly around its Z-axis.
	float fNoiseSize = textureSize(s2dNoiseTexture, 0).x;
	vec3 v3RotationVector = vec3(texture(s2dNoiseTexture, (gl_FragCoord.xy-0.5f)/fNoiseSize).rg, 1.0f);
	v3RotationVector = v3RotationVector - dot(v3RotationVector, v3Normal)*v3Normal;
	v3RotationVector=normalize(v3RotationVector);

	vec3 v3Binormal = cross(v3Normal, v3RotationVector);

	mat3 m3TBN = mat3(v3RotationVector, v3Binormal, v3Normal);		//kernel transformation matrix

	//Compute occlusion factor
	float fOcclusionFactor = 0.0f;
	for(unsigned int i = 0; i < numSamples; ++i)
	{
		float fScale = float(i)/(numSamples-1.0f);
		fScale = fKernelRadius*mix(0.1f, 1.0f, fScale*fScale);

		vec3 v3CurrentSample = m3TBN*extra_params.samples[i]*fScale;

		v3CurrentSample += v3FragmentPosition;

		vec4 v4CurrentSampleProjected = m4ProjMat*vec4(v3CurrentSample, 1.0f);
		v4CurrentSampleProjected.xy /= v4CurrentSampleProjected.w;
		v4CurrentSampleProjected.xy = v4CurrentSampleProjected.xy*0.5f + 0.5f;

		float fSampleDepth = texture(source1, v4CurrentSampleProjected.xy).r;
		float fRangeCheck = abs(fDepth - fSampleDepth) <= fOcclusionRange ? 1.0f : 0.0f;

		fOcclusionFactor += (fSampleDepth > v3CurrentSample.z ? 1.0f : 0.0f)*fRangeCheck;
	}

	return vec4(1.0f - fOcclusionFactor/numSamples, 0.0f, 0.0f, 0.0f);
}