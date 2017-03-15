#define pi 3.1415926535897932384626433832795f
#define max_high_frequency_wave_maps	50	//maximal number of high frequency wave maps that are used to simulate micro-scale water dynamics

#include "KPWaterCommonDefinitions.inc"

in GS_DATA
{
	vec2 v2TexCoord;	//texture coordinates of the current vertex output
	float fWaterElevation;	//water level at the current vertex
}gs_in;

uniform vec3 v3Scale;	//scale factors of the object
uniform float fMaxCapillaryWaveAmplitude;	//maximal allowed amplitude of the capillary waves
uniform float fMaxDeepWaterWaveAmplitude;	//maximal allowed amplitude of the deep water waves
uniform sampler2D s2dFFTRipplesNormalMapGlobalScale;	//normal map of the FFT-generated waves at the global scale
uniform sampler2D s2dFFTRipplesNormalMapCapillaryScale;	//normal map of the FFT-generated waves at the capillary scale
uniform uvec2 uv2DeepWaterRippleMapTilingFactor;	//tiling factors applied to the deep water ripple map. The first component of the vector determines tiling used to shape geometrical waves, whilst the second component is employed for bump mapping
uniform float fMaxWaveHeightAsElevationFraction;	//fraction of depth of shallow water surface that is used as the largest allowed deep water wave perturbation

uniform vec4 v4Viewport;	//parameters of the viewport packed into a 4D vector as (x, y, width, height)

float dSdU, dSdV;


subroutine(ProceduralNormalMapSampleRetriever) vec3 FFTRipples(vec2 v2TexCoord, float fArrayLayer)
{
	vec3 v3Normal = 
		    vec3(texture(s2dFFTRipplesNormalMapCapillaryScale, v2TexCoord * uv2DeepWaterRippleMapTilingFactor.y).rb / TW__KPWATER_FFT_SIZE__ / TW__KPWATER_FFT_SIZE__ * fMaxCapillaryWaveAmplitude, 1.0) + 
			vec3(texture(s2dFFTRipplesNormalMapGlobalScale, v2TexCoord * uv2DeepWaterRippleMapTilingFactor.y/8).rb / TW__KPWATER_FFT_SIZE__ / TW__KPWATER_FFT_SIZE__ * fMaxCapillaryWaveAmplitude, 0.0);
	v3Normal.xy *= 0.5f*smoothstep(TW__KPWATER_MIN_WATER_DEPTH__*1e2f, TW__KPWATER_MIN_WATER_DEPTH__*1e3f, gs_in.fWaterElevation/v3Scale.y);

	dSdU = v3Normal.x;
	dSdV = v3Normal.y;

	return normalize(v3Normal);
}

subroutine(ProceduralSpecularMapSampleRetriever) vec3 FFTRipplesSpecularModulation(vec2 v2TexCoord, float fArrayLayer)
{
	return vec3(length(vec2(dSdU, dSdV)));
}




uniform sampler2D s2dRefractionTexture;	//refraction texture sampler
uniform mat4 m4ModelViewTransform;	//Transforms coordinates from scaled object space to the view space
uniform mat4 m4ProjectionTransform;	//Transforms coordinates from the view space to the clip space
//uniform float fFresnelPower;	//exponent affecting behavior of the Fresnel effect
//uniform float fFarClipPlane;	//distances to the near and far clip planes stored in this order as a 2D vector
uniform float fMaxLightPenetrationDepth;	//assumed maximal water level
uniform vec3 v3ColorExtinctionFactors;	//identifies how fast red, green, and blue color channels get attenuated by water

#include "common/FogBuffer.include.glsl"	//Include definition of the light buffer



//Implements sample retrieving from environmental map applied to the KPWater
//Input arguments
//s2dEnvironmentMap — sampler used to retrieve data from environment map
//v3ReflectionVector — reflection vector represented in the world space
//
//Return value
//returns color value sampled from the given environment map
subroutine(CubeEnvironmentMapSampleRetriever) vec3 KPWaterCubicEnvironmentMapSampleRetriever(samplerCube scEnvironmentMap, vec3 v3ReflectionVector)
{   
	//Compute lighting parameters
	mat3 m3LightRotation = mat3(m4LightTransform[0].xyz, m4LightTransform[1].xyz, m4LightTransform[2].xyz); 
	vec3 v3LightDirection = m3LightRotation*normalize(10.0f*fog_buffer.v3SunDirection*smoothstep(fog_buffer.v3SunDirection.y, -1.0f, 1.0f) + fog_buffer.v3MoonDirection * float(fog_buffer.v3MoonDirection.y > 0));


	//Obtain refraction sample
	//Description: Here refraction vector is projected onto the NDC and coordinates of the current fragment are corrected in order to simulate
	//sum of the current fragment and refraction direction projected onto the NDC. In other words, given C as coordinates of the current fragment
	//represented in the scaled object space and R as displacement vector represented in the same space, we can obtain coordinates of the point
	//visible through refractive surface as C+R (here this point is also given in the scaled object space). Therefore, assuming that M is a homogeneous
	//matrix transforming the scaled object space into homogeneous clip space we obtain coordinates of the refraction sample in two steps
	//1) P = M*vec4(C,1)+M*vec4(R,0)
	//2) P.xyz /= P.w
	//The same steps can be simulated by means of gl_FragCoord and M*vec4(R,0), which is exactly what is done below
	float fN =  1.33f;
	vec3 v3Normal_SOS = transpose(m3ViewerRotation)*v3Normal;
	vec3 v3RefractionVector = refract(v3IncidentVector, v3Normal_SOS, 1.0f/fN);
	v3RefractionVector *= -gs_in.fWaterElevation/v3RefractionVector.y;
	vec4 v4ProjectedRefractionVector_HCS = m4ProjectionTransform*m4ModelViewTransform*vec4(v3RefractionVector, 0);

	vec2 v2ShoreSamplePointCoords = (gl_FragCoord.xy - 0.5f - v4Viewport.xy)/v4Viewport.zw;
	vec2 v2RefractedSamplePointTexCoords = (2*v2ShoreSamplePointCoords-1) / (1.0f + gl_FragCoord.w*v4ProjectedRefractionVector_HCS.w);

	vec2 v2RefractionDisplacement = clamp(v4ProjectedRefractionVector_HCS.xy / (1.0f/gl_FragCoord.w + v4ProjectedRefractionVector_HCS.w), 
		-1+1.0f/(1.0f + gl_FragCoord.w*v4ProjectedRefractionVector_HCS.w), 1-1.0f/(1.0f + gl_FragCoord.w*v4ProjectedRefractionVector_HCS.w));
	//vec2 v2RefractionDisplacement = v4ProjectedRefractionVector_HCS.xy / (1.0f/gl_FragCoord.w + v4ProjectedRefractionVector_HCS.w);
	

	v2RefractedSamplePointTexCoords += v2RefractionDisplacement;
	v2RefractedSamplePointTexCoords = (v2RefractedSamplePointTexCoords+1)*0.5f;

	vec3 v3RefractionColor = texture(s2dRefractionTexture, v2RefractedSamplePointTexCoords).rgb;


	//Incorporate color extinction
	float fRelativeWaterHeight = gs_in.fWaterElevation/fMaxLightPenetrationDepth;
	vec3 v3WaterAttenuation = exp(-v3ColorExtinctionFactors*fRelativeWaterHeight);
	v3RefractionColor *= v3WaterAttenuation;

	

	//Compute the Fresnel factor
	float fLV = dot(-v3IncidentVector, v3Normal_SOS);
	float fG = sqrt(fN*fN - 1.0f + fLV*fLV);
	float fFresnelWeight = clamp(0.5f * pow(fG - fLV, 2.0f)/pow(fG + fLV, 2.0f) * (pow(fLV*(fG + fLV) - 1.0f, 2.0f)/pow(fLV*(fG - fLV) + 1.0f, 2.0f) + 1.0f), 0, 1);
	fFresnelWeight *= smoothstep(TW__KPWATER_MIN_WATER_DEPTH__*1e2f, TW__KPWATER_MIN_WATER_DEPTH__*1e3f, gs_in.fWaterElevation/v3Scale.y);

	/*float fBias = 0.1f;
	float fFacing = 1.0f + dot(v3IncidentVector, v3Normal_SOS);
	float fFresnelWeight = max(fBias + (1.0f - fBias)*pow(fFacing, fFresnelPower), 0);
	fFresnelWeight = min((fFresnelWeight + max(abs(fLinearDepth)/fFarClipPlane, fFresnelWeight))*0.5f, 1.0f);*/



	//Combine reflection and refraction colors
	vec3 v3ReflectionColor = texture(scEnvironmentMap, normalize(v3ReflectionVector + 0.5f*v3IncidentVector)).rgb;
	vec3 v3WaterColor = fFresnelWeight*v3ReflectionColor + (1.0f - fFresnelWeight)*v3RefractionColor;


	//Add scattering (subject for update)
	float fScatteringModulation = 0.05f;
	fScatteringModulation *= dot(v3IncidentVector, v3LightDirection)*0.5f + 0.5f;
	fScatteringModulation *= dot(v3LightDirection, v3Normal_SOS)*0.5f + 0.5f;
	fScatteringModulation *= max(0, 1-abs(v3IncidentVector.y));
	fScatteringModulation *= max(0, gs_in.fWaterElevation);
	v3WaterColor += v3WaterAttenuation*fScatteringModulation;


	//Finally, make the edge of the water surface look fuzzy by blending the boundary water fragments with the colors of the shore line
	vec3 v3ShoreColor = texture(s2dRefractionTexture, v2ShoreSamplePointCoords, 0).rgb;
	float fShoreBlendingWeight = smoothstep(0, TW__KPWATER_MIN_WATER_DEPTH__*1e2f, gs_in.fWaterElevation/v3Scale.y);
	v3WaterColor = fShoreBlendingWeight*v3WaterColor + (1.0f - fShoreBlendingWeight)*v3ShoreColor;


	return v3WaterColor;
}