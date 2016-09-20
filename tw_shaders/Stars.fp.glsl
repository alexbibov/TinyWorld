#version 430 core

//Implement fragment program for rendering of stars

in vec3 v3StarColour;	//colour of the current star

uniform float fScintillationParameter;		//parameter affecting scintillation of the star
uniform sampler2D s2dStarTexture;	//sampler of the star texture

layout(location = 0) out vec4 v4FragmentColour;	//colour of the output fragment
layout(location = 1) out vec4 v4BloomColour;	//colour of the output fragment forwarded to the bloom filter

vec4 computeBloomFragment(vec4 v4FragmentColor);

void main()
{
	//Compute attenuation factor of the atmospheric scattering and star scintillation modulation factor
	const vec3 v3RayleighScattering = vec3(0.16f, 0.37f, 0.91f);
	
	vec3 v3Attenuation = 10.0f * (1.0f + v3RayleighScattering);	//attenuation of the star colour
	
	//Compute resulting colour of the star
	float color_magnitude = texture(s2dStarTexture, gl_PointCoord).r;	//get magnitude of the current fragment's colour
	
	vec3 v3ScintillationModulation = 
	v3RayleighScattering / (fScintillationParameter * ((noise1(fScintillationParameter) + 1.0f)/2.0f*(1.0-1e-1) + 1e-1));
	
	if(color_magnitude < 1.3e-1)
		discard;
	else
	{
		v4FragmentColour = vec4(vec3(color_magnitude) * (v3StarColour * v3Attenuation * v3ScintillationModulation), 1.0f);
		v4BloomColour = computeBloomFragment(v4FragmentColour);
	}
}