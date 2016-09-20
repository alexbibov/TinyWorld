//Fragment program used by default rendering mode in TessellatedTerrain

#version 430 core

uniform sampler2DArray terrain_tex_sampler;		//terrain diffuse texture sampler
uniform sampler2D s2dFractalNoise;	//fractal noise sampler object
uniform vec2 v2FractalNoiseScaling;	//scaling factors applied to the fractal noise

layout(location = 0)out vec4 fColor;		//output colour of the current fragment
layout(location = 1)out vec4 fBloomColor;	//output colour for the bloom filter

//Extra parameters outputted by geometry shader
in GS_DATA
{
	vec4 texcoord;
}gs_in;

//******************************Functions derived from the shaders provided by the lighting and HDR-Bloom extension******************************
vec4 computeLightContribution(vec4 v4DiffuseColor, vec2 v2NormalMapTexCoords, vec2 v2SpecularMapTexCoords, vec2 v2EmissionMapTexCoords, 
		float fNormalMapLayer, float fSpecularMapLayer, float fEmissionMapLayer, float fEnvironmentMapLayer);
vec4 computeBloomFragment(vec4 v4FragmentColor);
bool isCubicEnvironmentMap(uint uiEnvironmentMapType);
//***********************************************************************************************************************************************

void updateSelectionBuffer();	//selection buffer extension

void main()
{
	vec3 v3BaseColor = vec3(0);
	float fNoise = texture(s2dFractalNoise, gs_in.texcoord.st * v2FractalNoiseScaling).r;
	for(float q = 1; q <= 3; ++q)
	{
		float fPeriodMultiplier = sqrt(q); 
		vec4 v4Aux1 = texture(terrain_tex_sampler, vec3(gs_in.texcoord.st/fPeriodMultiplier, floor(gs_in.texcoord.p)));
		vec4 v4Aux2 = texture(terrain_tex_sampler, vec3(gs_in.texcoord.st/fPeriodMultiplier, ceil(gs_in.texcoord.p)));
		vec3 v3BaseColorUpdate = mix(v4Aux1.rgb, v4Aux2.rgb, fract(gs_in.texcoord.p));
		vec3 v3Cliff = vec3(v4Aux1.a);
		v3BaseColorUpdate = mix(v3BaseColorUpdate, v3Cliff, gs_in.texcoord.q);

		v3BaseColor = (v3BaseColor*(q-1) + v3BaseColorUpdate) / q;
	}
	v3BaseColor = 0.7f*v3BaseColor + 0.3f*fNoise;
	
	fColor = computeLightContribution(vec4(v3BaseColor, 1.0f), 
		gs_in.texcoord.st, gs_in.texcoord.st, gs_in.texcoord.st, 
		gs_in.texcoord.p, gs_in.texcoord.p, gs_in.texcoord.p, gs_in.texcoord.p);
		
	fBloomColor = computeBloomFragment(fColor);
	updateSelectionBuffer();
}

