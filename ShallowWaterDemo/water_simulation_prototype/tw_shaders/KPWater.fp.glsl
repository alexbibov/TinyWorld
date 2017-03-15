//Implements fragment stage of the water surface shading program

#version 430 core

#include "KPWaterCommonDefinitions.inc"

in GS_DATA
{
	vec2 v2TexCoord;	//texture coordinates of the current vertex output
	float fWaterElevation;	//water level at the current vertex
}gs_in;

layout(location = 0) out vec4 v4FragmentColor; 
layout(location = 1) out vec4 v4BloomColor;

//uniform float fMaxWaterLevel;	//assumed maximal water level

uniform vec3 v3Scale;	//scale factors transforming units from the nominal non-dimensional to the scaled object space

//*********************Uniforms derived from the shaders provided by the lighting extension*********************
uniform bool bSupportsArrayNormalMaps;		//equals 'true' if bump-mapping is implemented using array textures
uniform bool bSupportsArraySpecularMaps;		//equals 'true' if specular reflection is described by array specular map
uniform bool bSupportsArrayEmissionMaps;		//equals 'true' if emission can be described by an array texture
uniform bool bSupportsArrayEnvironmentMaps;		//equals 'true' if environment mapping is implemented by an array texture

uniform sampler2DArray s2daNormalArrayMap;		//used instead of s2dNormalMap if bSupportsArrayNormalMaps = true
uniform sampler2DArray s2daSpecularArrayMap;	//used instead of s2dSpecularMap if bSupportsArraySpecularMaps = true
uniform sampler2DArray s2daEmissionArrayMap;	//used instead of s2dEmissionMap if bSupportsArrayEmissionMaps = true
uniform sampler2DArray s2daEnvironmentArrayMap;	//used instead of s2dEnvironmentMap if bSupportsArrayEnvironmentMaps = true
uniform samplerCubeArray scaEnvironmentArrayMap; //used instead of scEnvironmentMap if bSupportsArrayEnvironmentMaps = true
uniform uint uiEnvironmentMapType;	//determines type of the currently used environment map
//**************************************************************************************************************


//******************************Functions derived from the shaders provided by the lighting and HDR-Bloom extension******************************
vec4 computeLightContribution(vec4 v4DiffuseColor, vec2 v2NormalMapTexCoords, vec2 v2SpecularMapTexCoords, vec2 v2EmissionMapTexCoords, 
		float fNormalMapLayer, float fSpecularMapLayer, float fEmissionMapLayer, float fEnvironmentMapLayer);
vec4 computeBloomFragment(vec4 v4FragmentColor);
bool isCubicEnvironmentMap(uint uiEnvironmentMapType);
//***********************************************************************************************************************************************


void updateSelectionBuffer();	//selection buffer extension

void main()
{
	if(gs_in.fWaterElevation < TW__KPWATER_MIN_WATER_DEPTH__*1e2f * v3Scale.y) discard;

	vec4 v4OutputColor = computeLightContribution(vec4(1e-3f, 15e-4f, 3e-3f, 1.0f), gs_in.v2TexCoord, gs_in.v2TexCoord, gs_in.v2TexCoord, 0, 0, 0, 0);

	v4FragmentColor = v4OutputColor;
	v4BloomColor = computeBloomFragment(v4OutputColor);


	updateSelectionBuffer();
}