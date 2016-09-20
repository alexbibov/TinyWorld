//Fragment shader implementing attenuation of light passing through the medium

#version 430 core

in vec3 v3TexCoord3D;	//3D texture coordinate corresponding to the currently processed fragment

//Colour of the transparent medium containing the volumetric data to be rendered
uniform vec3 v3MediumColor; 

//3D texture sampler used to extract values from the medium being visualized
uniform sampler3D s3dMediumSampler;

//sampler of the colormap look-up texture
uniform sampler1D s1dColormapSampler;

//Equals 'true' if shading should take into account RGB-channel of the 3D-texture 
uniform bool bUseRGBChannel;

//Equals 'true' if colormap is in use
uniform bool bUseColormap;


layout(location = 0) out vec4 v4FragmentColor;	//output color of the fragment


void main()
{
	vec4 v4TextureSample = texture(s3dMediumSampler, v3TexCoord3D);	//colour and light absorption of the medium corresponding to the currently processed fragment
	
	//Current colour component
	vec3 v3CurrentColor = v3MediumColor;
	float fCurrentLightAbsorption;
	if(bUseRGBChannel)
	{
		v3CurrentColor = v3CurrentColor * v4TextureSample.rgb;
		fCurrentLightAbsorption = v4TextureSample.a;
	}
	else
	{
		fCurrentLightAbsorption = v4TextureSample.r;
	}
	if(bUseColormap) v3CurrentColor = v3CurrentColor * texture(s1dColormapSampler, clamp(fCurrentLightAbsorption, 0, 1)).rgb;
	
	v4FragmentColor = vec4(v3CurrentColor, fCurrentLightAbsorption);
}