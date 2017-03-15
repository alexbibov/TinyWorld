#version 430

#define pi 3.1415926535897932384626433832795f
#include "KPWaterCommonDefinitions.inc"

uniform sampler2D source0;	//refraction map, which should be modulated by caustics

uniform sampler2D s2dFFTRipplesNormalMapGlobalScale;	//normal map of the water surface at the global scale
uniform sampler2D s2dFFTRipplesNormalMapCapillaryScale;	//normal map of the water surface at the capillary scale
uniform sampler2D s2dRefractionDepthMap;	//depth map corresponding to the refraction texture
uniform sampler2D s2dWaterHeightMap;		//height map of the water surface
uniform vec3 v3Scale;						//scale factors of the object
uniform uvec2 uv2DeepWaterRippleMapTilingFactor;	//tiling factors applied to the deep water ripple map. The first component of the vector determines tiling used to shape geometrical waves, whilst the second component is employed for bump mapping

uniform vec4 v4Viewport;		//parameters of the viewport packed into a 4D vector as x, y, width, and height in this order
uniform float fFocalDistance;	//distance between the observer and the near clipping plane
uniform vec4 v4FocalPlane;		//parameters of the focal plane (left, right, bottom, top) packed into a vec4
uniform mat4 m4VS2SOS;			//matrix transforming view-space coordinates to scaled object space coordinates

uniform vec3 v3LightIntensity;	//intensity of the light that generates caustics
uniform float fCausticsPower;	//value of the power, to which caustics modulation coefficient should be raised
uniform float fCausticsAmplification;	//amplification factor applied to caustics modulation coefficient
uniform float fCausticsSampleArea;	//defines length of the side of the square area from which to take the light samples that will contribute to caustics illumination. The length must be given in the non-dimensional model space, which is {X=[-0.5, 0.5], Y, Z=[0.5,-0.5]}


layout(pixel_center_integer) in vec4 gl_FragCoord;	//fragment coordinates should be aligned with integers

out vec4 v4ResultingFragmentColor;


void main()
{
	//Retrieve color of the currently processed fragment and write it into the shader's output
	v4ResultingFragmentColor = vec4(texelFetch(source0, ivec2(gl_FragCoord.xy), 0).rgb, 1);

	//Retrieve position of the current fragment represented in scaled object space
	float fFragDepth = texelFetch(s2dRefractionDepthMap, ivec2(gl_FragCoord.xy), 0).r;
	vec3 v3FragCoord_VS = vec3(((gl_FragCoord.xy - v4Viewport.xy) / v4Viewport.zw * (v4FocalPlane.yw - v4FocalPlane.xz) + v4FocalPlane.xz) * fFragDepth/(-fFocalDistance), fFragDepth);
	vec3 v3FragCoord_SOS = (m4VS2SOS*vec4(v3FragCoord_VS, 1)).xyz;

	vec2 v2TexCoords = v3FragCoord_SOS.xz / v3Scale.xz; v2TexCoords = vec2(v2TexCoords.x + 0.5f, 0.5f - v2TexCoords.y);
	float fWaterElevation = texture(s2dWaterHeightMap, v2TexCoords).r - v3FragCoord_SOS.y;



	if(fWaterElevation <= TW__KPWATER_MIN_WATER_DEPTH__*v3Scale.y) return;	//premature exit for dry areas

	vec3 v3LightDirection = vec3(0, 1, 0);
	uint uiNumCausticsSamples = 3;
	vec2 v2CausticsSampleArea = vec2(fCausticsSampleArea);
	float fCausticsModulation = 0;
	for(int i = 0; i < uiNumCausticsSamples; ++i)
	for(int j = 0; j < uiNumCausticsSamples; ++j)
	{
		vec3 v3Center = v3FragCoord_SOS + fWaterElevation*v3LightDirection;
		vec3 v3Sample; 
		v3Sample.xz = v3Center.xz - 0.5f*v2CausticsSampleArea*v3Scale.xz + v2CausticsSampleArea*v3Scale.xz/(uiNumCausticsSamples - 1.0f)*vec2(i, j);
		v3Sample.y = v3Center.y;
		vec2 v2SampleTexCoord = v3Sample.xz/v3Scale.xz; v2SampleTexCoord.x += 0.5f; v2SampleTexCoord.y = 0.5f - v2SampleTexCoord.y;
		float fTilingFactor = uv2DeepWaterRippleMapTilingFactor.x;
		vec2 v2Aux = 0.5f*texture(s2dFFTRipplesNormalMapGlobalScale, v2SampleTexCoord*fTilingFactor).rb / TW__KPWATER_FFT_SIZE__ + 
			0.5f*texture(s2dFFTRipplesNormalMapCapillaryScale, v2SampleTexCoord*fTilingFactor).rb / TW__KPWATER_FFT_SIZE__; 
		vec3 v3Normal = normalize(vec3(-v2Aux.x, 1, -v2Aux.y));



		vec3 v3InRay = -v3LightDirection;
		v3InRay.xz += 0.2f*v3Normal.xz;
		v3InRay = normalize(v3InRay);
		vec3 v3OpticalAxis = normalize(v3FragCoord_SOS - v3Sample);
		float fIncomingRadiosity = pow(max(dot(v3InRay, v3OpticalAxis), 1e-10f), fCausticsPower);


		vec3 v3OutRay = v3Sample - v3FragCoord_SOS;
		v3OutRay.xz += 0.2f*v3Normal.xz; 
		v3OutRay = normalize(v3OutRay);
		fCausticsModulation += fIncomingRadiosity*pow(max(dot(v3OutRay, v3LightDirection), 1e-10f), fCausticsPower);
	}
	//fCausticsModulation *= float(fWaterElevation > TW__KPWATER_MIN_WATER_DEPTH__*v3Scale.y)*smoothstep(TW__KPWATER_MIN_WATER_DEPTH__, 1e4f*TW__KPWATER_MIN_WATER_DEPTH__, fWaterElevation/v3Scale.y);
	vec3 v3CausticsImpact = 1 + fCausticsAmplification * smoothstep(1e-3f, 2e-2f, fWaterElevation / v3Scale.y) * v3LightIntensity * fCausticsModulation;
	v4ResultingFragmentColor.xyz *= v3CausticsImpact;
}











