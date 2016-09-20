//Fragment shader implementing visual volume accumulation in the output draw buffer

#version 430 core

//3D texture coordinates corresponding to the currently processed fragment
in vec3 v3TexCoord3D;

//Texture coordinates used to access data in the light buffer
in noperspective vec2 v2LightBufferTexCoords;

//sampler object of the texture containing light attenuation values
uniform sampler2D s2dLightAttenuationBuffer;

//Colour of the transparent medium containing the volumetric data to be rendered
uniform vec3 v3MediumColor; 

//3D texture sampler used to extract values from the volumetric data set
uniform sampler3D s3dMediumSampler;

//sampler of the colour map look-up texture
uniform sampler1D s1dColormapSampler; 

//Equals 'true' if shading should take into account RGB-channel of the 3D-texture 
uniform bool bUseRGBChannel;

//Equals 'true' if colour map is in use
uniform bool bUseColormap;

//Output fragment colour
layout(location = 0) out vec4 v4FragmentColor;
layout(location = 1) out vec4 v4BloomColor;


vec4 computeBloomFragment(vec4 v4FragmentColor);


void main()
{
	//Get current light colour
	vec4 v4CurrentLightColor = texture(s2dLightAttenuationBuffer, v2LightBufferTexCoords);
	
	//Extract sample from the 3D texture corresponding to the currently processed fragment
	vec4 v4TextureSample = texture(s3dMediumSampler, v3TexCoord3D);
	
	//Compute current reflective colour
	vec3 v3CurrentReflectiveColor = v3MediumColor;
	float fCurrentAbsorptionCoefficient;
	if (bUseRGBChannel)
	{
		v3CurrentReflectiveColor = v3CurrentReflectiveColor * v4TextureSample.rgb;
		fCurrentAbsorptionCoefficient = v4TextureSample.a;
	}
	else
	{
		fCurrentAbsorptionCoefficient = v4TextureSample.r;
	}
	if (bUseColormap) v3CurrentReflectiveColor = 
		v3CurrentReflectiveColor * texture(s1dColormapSampler, clamp(fCurrentAbsorptionCoefficient, 0, 1)).rgb;
	
	v4FragmentColor = vec4(v3CurrentReflectiveColor * v4CurrentLightColor.rgb, fCurrentAbsorptionCoefficient);
	v4BloomColor = computeBloomFragment(v4FragmentColor);
}

