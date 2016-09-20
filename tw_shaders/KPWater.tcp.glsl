//Tessellation control stage of rendering program implementing shading of a water surface

#version 430 core

#include "KPWaterCommonDefinitions.inc"
#include "KPWaterCommonUtils.inc"

layout (vertices = 4) out;	//output patches must have 4 vertices


/*in VS_DATA
{
	vec2 v2TessellationBilletTexCoord;	//texture coordinates of the vertices of the currently processed patch
}vs_in[];*/

/*out TCS_DATA
{
	vec2 v2TessellationBilletTexCoord;	//texture coordinates of the vertices of the currently processed patch
}tcs_out[];*/

uniform mat4 m4ModelViewTransform;	//Transforms coordinates from scaled object space to the view space
uniform mat4 m4ProjectionTransform;	//Transforms coordinates from the view space to the clip space

uniform float fLOD;	//level-of-detail factor controlling tessellation density
uniform uvec2 uv2ScreenSize;	//size of the screen represented in pixels
uniform vec3 v3Scale;	//scale factors transforming units from the nominal non-dimensional to the scaled object space

uniform sampler2D s2dTopographyHeightmap;	//topography height map
uniform sampler2D s2dFFTRipples;	//texture containing FFT-generated ripples
uniform float fMaxDeepWaterWaveAmplitude;	//maximal allowed amplitude of the deep water waves
uniform uvec2 uv2DeepWaterRippleMapTilingFactor;	//tiling factors applied to the deep water ripple map. The first component of the vector determines tiling used to shape geometrical waves, whilst the second component is employed for bump mapping
uniform float fMaxWaveHeightAsElevationFraction;	//fraction of depth of the shallow water surface that is used as the largest allowed deep water wave perturbation


//Transforms coordinates from the viewer space to the NDC
vec4 ontoNDC(vec4 v4Input)
{
	vec4 v4Output = m4ProjectionTransform * v4Input;
	v4Output.xyz /= v4Output.w;
	return v4Output;
}


//Transforms input vertex from NDC to the screen space
vec2 ontoScreen(vec4 v4Input)
{
	return clamp((v4Input.xy + 1.0f) * 0.5f, -0.3f, 1.3f) * uv2ScreenSize; 
}


//Transforms input vertex to the view space
vec4 toViewSpace(vec4 v4Input)
{
	return m4ModelViewTransform * vec4(v3Scale.x * v4Input.x, v4Input.y, v3Scale.z * v4Input.z, v4Input.w);
}


//Checks if given vertex represented in NDC lies outside the clip region
bool isOffscreen(vec4 v4Input)
{
	return (v4Input.z < -0.5f || any(lessThan(v4Input.xy, vec2(-1.3f)) || greaterThan(v4Input.xy, vec2(1.3f))));
}

//Checks if patch belongs to the dry area
bool isDry()
{
	float fAveragePatchWaterElevation = 0;
	for(uint i = 0; i < 4; ++i)
	{
		vec2 v2TexCoords = vec2(gl_in[i].gl_Position.x + 0.5f, 0.5f - gl_in[i].gl_Position.z);
		float fWaterElevation;
		float fDeepWaterWavePerturbation = computeDeepWaterWavePerturbation(v2TexCoords, uv2DeepWaterRippleMapTilingFactor.x,
			s2dTopographyHeightmap, gl_in[i].gl_Position.y, fMaxWaveHeightAsElevationFraction, fMaxDeepWaterWaveAmplitude, s2dFFTRipples, fWaterElevation);

		fAveragePatchWaterElevation = (fAveragePatchWaterElevation*i + fWaterElevation + fDeepWaterWavePerturbation)/(i+1.0f);
	}

	return fAveragePatchWaterElevation < TW__KPWATER_MIN_WATER_DEPTH__*v3Scale.y;
}


//Computes LOD factor based on the given section defined by two vertices represented in the view space
float getLODFactor(vec4 v4First, vec4 v4Second)
{
	float fRadius = distance(v4First.xyz, v4Second.xyz) * 0.5f;
	vec4 v4Center = vec4((v4First.xyz + v4Second.xyz) * 0.5f, 1.0f);

	vec4 p1 = ontoNDC(vec4(v4Center.x - fRadius, v4Center.yzw));
	vec4 p2 = ontoNDC(vec4(v4Center.x + fRadius, v4Center.yzw));

	float fProjectedLength = distance(ontoScreen(p1), ontoScreen(p2));

	return clamp(fProjectedLength / fLOD, 1.0f, 64.0f);
}


void main()
{
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
	//tcs_out[gl_InvocationID].v2TessellationBilletTexCoord = vs_in[gl_InvocationID].v2TessellationBilletTexCoord;

	if(gl_InvocationID == 0)
	{
		vec4 v4LL = toViewSpace(gl_in[0].gl_Position);	//lower-left corner of the current patch
		vec4 v4LR = toViewSpace(gl_in[1].gl_Position);	//lower-right corner of the current patch
		vec4 v4UR = toViewSpace(gl_in[2].gl_Position);	//upper-left corner of the current patch
		vec4 v4UL = toViewSpace(gl_in[3].gl_Position);	//upper-right corner of the current patch

		//Check if all four vertices of the current patch are outside the clip region. In practice this means
		//that the patch is fully invisible (until it is large enough to cross the view frustum)
		if(all(bvec4(
			isOffscreen(ontoNDC(v4LL)), 
			isOffscreen(ontoNDC(v4LR)), 
			isOffscreen(ontoNDC(v4UL)), 
			isOffscreen(ontoNDC(v4UR))
			)) || isDry())
		{
			gl_TessLevelOuter[0] = 0;
			gl_TessLevelOuter[1] = 0;
			gl_TessLevelOuter[2] = 0;
			gl_TessLevelOuter[3] = 0;

			gl_TessLevelInner[0] = 0;
			gl_TessLevelInner[1] = 0;
		}
		else
		{
			gl_TessLevelOuter[0] = getLODFactor(v4UL, v4LL);
			gl_TessLevelOuter[1] = getLODFactor(v4LL, v4LR);
			gl_TessLevelOuter[2] = getLODFactor(v4LR, v4UR);
			gl_TessLevelOuter[3] = getLODFactor(v4UR, v4UL);

			gl_TessLevelInner[0] = mix(gl_TessLevelOuter[1], gl_TessLevelOuter[2], 0.5f);
			gl_TessLevelInner[1] = mix(gl_TessLevelOuter[0], gl_TessLevelOuter[3], 0.5f);
		}
	}
}