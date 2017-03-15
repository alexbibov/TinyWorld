//Tessellation evaluation stage of the shader program implementing shading of a water surface

#version 430 core

#include "KPWaterCommonDefinitions.inc"
#include "KPWaterCommonUtils.inc"

layout(quads, fractional_even_spacing, ccw) in;

#define pi 3.1415926535897932384626433832795f

/*in TCS_DATA
{
	vec2 v2TessellationBilletTexCoord;	//texture coordinates of the vertices of the current patch
}tcs_in[];*/

out TEP_DATA
{
	vec2 v2TexCoord;	//texture coordinates of the current vertex output
	vec3 v3Normal;	//normal vector corresponding to the current vertex output 
}tep_out;

uniform vec3 v3Scale;	//scale factors transforming units from the nominal non-dimensional to the scaled object space

uniform sampler2D s2dWaterHeightMap;	//height map of the water surface
uniform sampler2D s2dTopographyHeightmap;	//topography height map
uniform sampler2D s2dFFTRipples;	//texture containing FFT-generated ripples
uniform sampler2D s2dFFTDisplacementMap;	//displacement map of the FFT-generated waves
uniform sampler2D s2dFFTRipplesNormalMapGlobalScale;	//normal map of the FFT-generated waves
uniform sampler2D s2dFractalNoiseMap;		//fractal noise map
uniform float fMaxDeepWaterWaveAmplitude;	//maximal allowed amplitude of the deep water waves
uniform uvec2 uv2DeepWaterRippleMapTilingFactor;	//tiling factors applied to the deep water ripple map. The first component of the vector determines tiling used to shape geometrical waves, whilst the second component is employed for bump mapping
uniform float fMaxWaveHeightAsElevationFraction;	//fraction of depth of shallow water surface that is used as the largest allowed deep water wave perturbation


//Transforms a point within the tessellation billet into corresponding height map texture coordinates
vec2 point2TexCoords(vec3 v3Point)
{
	return vec2(v3Point.x + 0.5f, 0.5f - v3Point.z);
}

vec2 point2TexCoords(vec4 v4Point)
{
	return vec2(v4Point.x + 0.5f, 0.5f - v4Point.z);
}


void main()
{
	//Generate tessellated vertex position
	vec2 v2TexSampleCoords = 
		vec2(mix(gl_in[0].gl_Position.x, gl_in[1].gl_Position.x, gl_TessCoord.x) + 0.5f,
		0.5f - mix(gl_in[1].gl_Position.z, gl_in[2].gl_Position.z, gl_TessCoord.y));

	//Retrieve B-spline approximation of the shallow water height map and compute the corresponding field of normal vectors
	vec3 v3ApproximativeWaterLevel = computeWaterLevel(s2dWaterHeightMap, v2TexSampleCoords, 0, v3Scale.xz, tep_out.v3Normal);

	float fWaterLevel = textureLod(s2dWaterHeightMap, v2TexSampleCoords, 0).r;
	vec4 v4P = vec4(v3ApproximativeWaterLevel.x /*v2TexSampleCoords.x -0.5f*/, min(v3ApproximativeWaterLevel.y, fWaterLevel) /*fWaterLevel*/, v3ApproximativeWaterLevel.z /*0.5f-v2TexSampleCoords.y*/, 1);
	gl_Position = v4P;

	//Extract deep water elevation data
	float fWaterElevation;
	vec2 v2NormalDisplacement, v2MapCoordinatesDisplacement;
	gl_Position.y += computeDeepWaterWavePerturbation(vec2(0.5f+v3ApproximativeWaterLevel.x, 0.5f-v3ApproximativeWaterLevel.z) /*v2TexSampleCoords*/, uv2DeepWaterRippleMapTilingFactor.x, 
		s2dTopographyHeightmap, v4P.y, fMaxWaveHeightAsElevationFraction, 
		fMaxDeepWaterWaveAmplitude, s2dFFTRipples, s2dFFTDisplacementMap, v3Scale.xz, s2dFFTRipplesNormalMapGlobalScale, fWaterElevation, v2NormalDisplacement, v2MapCoordinatesDisplacement);
	float fNoiseAddendum = textureLod(s2dFractalNoiseMap, v2TexSampleCoords, 0).r;
	//gl_Position.y += fNoiseAddendum*min(fWaterElevation, 1);
	//gl_Position.xz += v2MapCoordinatesDisplacement;


	//Generate normal
	/*vec2 v2TriangleSize = 
			vec2(distance(gl_in[0].gl_Position.xz, gl_in[1].gl_Position.xz), distance(gl_in[1].gl_Position.xz, gl_in[2].gl_Position.xz))
			/ min(min(gl_TessLevelOuter[0], gl_TessLevelOuter[1]), min(gl_TessLevelOuter[2], gl_TessLevelOuter[3])); 
	vec3 pos_dx = v4P.xyz + vec3(v2TriangleSize.x, 0, 0);
	vec3 pos_dz = v4P.xyz + vec3(0, 0, v2TriangleSize.y);
	pos_dx.y = textureLod(s2dWaterHeightMap, vec2(pos_dx.x + 0.5f, -(pos_dx.z - 0.5f)), 0).r;
	pos_dz.y = textureLod(s2dWaterHeightMap, vec2(pos_dz.x + 0.5f, -(pos_dz.z - 0.5f)), 0).r;
	tep_out.v3Normal = normalize(cross((pos_dz - v4P.xyz)*v3Scale, (pos_dx - v4P.xyz)*v3Scale));*/

	tep_out.v3Normal = normalize(tep_out.v3Normal - vec3(v2NormalDisplacement.x, 0, v2NormalDisplacement.y));

	tep_out.v2TexCoord = vec2(v4P.x + 0.5f, 0.5f - v4P.z);
}
