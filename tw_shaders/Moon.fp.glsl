#version 430 core

//Implements per-fragment shading of moon
in vec3 v3MoonColour;	//moon colour modulation factor

uniform sampler2D s2dMoonTextureSampler;	//moon texture sampler


layout(location = 0) out vec4 v4FragmentColour;	//resulting colour of the currently processed fragment
layout(location = 1) out vec4 v4BloomColour;	//bloom color

vec4 computeBloomFragment(vec4 v4FragmentColor);

void main()
{
	vec4 v4AuxColour = texture(s2dMoonTextureSampler, gl_PointCoord);
	
	if(v4AuxColour.a < 0.5f)
		discard;
	else
	{
		v4FragmentColour = vec4(v4AuxColour.rgb * v3MoonColour, 1.0f);
		v4BloomColour = computeBloomFragment(v4FragmentColour);
	}
}
