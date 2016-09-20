#version 430 core

uniform bool bBloomEnabled;	//equals 'true' if bloom support is enabled
uniform float fBloomMinThreshold;	//minimal threshold of the bloom filter
uniform float fBloomMaxThreshold;	//maximal threshold of the bloom filter
uniform float fBloomIntensity;		//intensity of the bloom effect

vec4 computeBloomFragment(vec4 v4FragmentColor)
{
	if(bBloomEnabled)
	{
		//Compute luminance value of the current fragment
		float luminance = dot(v4FragmentColor.rgb, vec3(0.299f, 0.587f, 0.144f));
		vec3 v3BloomColor = v4FragmentColor.rgb * fBloomIntensity * smoothstep(fBloomMinThreshold, fBloomMaxThreshold, luminance);
		return vec4(v3BloomColor, v4FragmentColor.a);
	}
	else 
		return vec4(0.0f, 0.0f, 0.0f, v4FragmentColor.a);
}