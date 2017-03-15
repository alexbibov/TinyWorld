#version 430 core

uniform vec3 v3SunLightDirection;	//Direction to the sun light
uniform vec3 v3MoonLightDirection;	//Direction to the moon light
uniform float fMiePhaseFunctionParameter;	//parameter of the Mie phase function
uniform float fFarClipPlane;	//Distance from viewer to the far clip plane


//Data generated on the vertex shading stage
in VS_DATA
{
	vec3 v3SunScatteredColor1;	//Attenuated colour of the sun with Rayleigh scattering
	vec3 v3SunScatteredColor2;	//Attenuated colour of the sun with Mie scattering
	vec3 v3MoonScatteredColor1;	//Attenuated colour of the moon with Rayleigh scattering
	vec3 v3MoonScatteredColor2;	//Attenuated colour of the moon with Mie scattering
	vec3 v3CameraDirection;		//Direction from the current vertex to camera
}vs_in;

layout(location = 0)out vec4 v4Color;	//output fragment color
layout(location = 1)out vec4 v4BloomColor;	//bloom color
layout(location = 3)out float fLinearDepth; //linear depth the fragments belonging to the skydome

//Implements Mie phase function
float MiePhaseFunction(float c, float g)
{
	float g2 = g * g;
	return (3.0f * (1.0f - g2)) / (2.0f * (2.0f + g2)) * (1.0f + c * c) / pow(1.0f + g2 - 2.0f * g * c, 1.5f);
}


vec4 computeBloomFragment(vec4 v4FragmentColor);

void main()
{
	//Find cosine of angle between light direction and camera location
	vec3 normalized_camera_direction = normalize(vs_in.v3CameraDirection);	//normalized direction to camera
	float fCos1 = dot(v3SunLightDirection, normalized_camera_direction);
	float fCos2 = dot(v3MoonLightDirection, normalized_camera_direction);
	
	//The light reflected from moon appears to be more "scattered", therefore we
	//use this manually adjusted scaling factor to scale the Mie phase function parameter
	const float moon_mie_scale = 0.9f;	
	
	v4Color.rgb = (vs_in.v3SunScatteredColor1 * MiePhaseFunction(fCos1, 0.0f) + 
		vs_in.v3SunScatteredColor2 * MiePhaseFunction(fCos1, fMiePhaseFunctionParameter)) + 
		(vs_in.v3MoonScatteredColor1 * MiePhaseFunction(fCos2, 0.0f) + 
		vs_in.v3MoonScatteredColor2 * MiePhaseFunction(fCos2, moon_mie_scale * fMiePhaseFunctionParameter));
	v4Color.a = smoothstep(0.01f, 0.8f, length(v4Color.rgb));
	
	v4BloomColor = computeBloomFragment(v4Color);
	fLinearDepth = -fFarClipPlane;
}
