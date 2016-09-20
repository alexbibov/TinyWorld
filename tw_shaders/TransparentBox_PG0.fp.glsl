//Fragment shader implementing attenuation of light passing through the medium

#version 430 core

in vec3 v3TexCoord3D;	//3D texture coordinate corresponding to the currently processed fragment

layout(pixel_center_integer) in vec4 gl_FragCoord;	//output fragment coordinate redeclaration to 
													//guarantee that fragments are having integer coordinates

//Input colour buffer that contains light attenuation values computed at the previous rendering pass
layout(rgba32f) uniform image2D i2dInColorBuffer;

//Output colour buffer accumulating light attenuation values computed at the current rendering pass
uniform writeonly image2D i2dOutColorBuffer;

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

void main()
{
	vec4 v4AccumulatedColor = imageLoad(i2dInColorBuffer, ivec2(gl_FragCoord.xy));	//retrieve current light colour and attenuation
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
	
	
	vec3 v3UpdatedColor = v3CurrentColor + (1.0f - fCurrentLightAbsorption) * v4AccumulatedColor.rgb;
	float fUpdatedLightAbsorption = fCurrentLightAbsorption + (1.0f - fCurrentLightAbsorption) * v4AccumulatedColor.a;
	
	imageStore(i2dOutColorBuffer, ivec2(gl_FragCoord.xy), vec4(v3UpdatedColor, fUpdatedLightAbsorption));
}