#version 430 core

//Implements rendering of moon
in vec4 v4MoonLocation;		//location of moon
in vec4 v4MoonLuminosity;	//luminosity of moon

uniform mat4 m4MVP;		//standard Model-View-Projection transform
uniform float fSkydomeRadius;	//radius of the skydome

out vec3 v3MoonColour;	//moon colour modulation factor

void main()
{
	v3MoonColour = v4MoonLuminosity.xyz;
	gl_PointSize = v4MoonLuminosity.w;
	
	vec4 v4ScaledMoonLocation = vec4(v4MoonLocation.xyz * fSkydomeRadius, v4MoonLocation.w);
	gl_Position = m4MVP * v4ScaledMoonLocation;
}