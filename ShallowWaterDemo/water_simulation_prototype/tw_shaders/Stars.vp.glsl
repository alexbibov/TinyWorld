#version 430 core

//Implements scintillating stars in the night sky

in vec4 v4StarLocation;		//location of the currently processed star
in vec4 v4StarLuminosity;	//luminosity (i.e. colour and brightness) of the currently processed star

uniform mat4 m4MVP;		//standard homogeneous Model-View-Transform matrix
uniform float fSkydomeRadius;		//radius of the sky dome to which the stars are "stick"


out vec3 v3StarColour;	//colour of the currently processed star


void main()
{
	//The vertex shader is a simple pass-through program with scale correction applied to positions of the vertices
	v3StarColour = v4StarLuminosity.xyz;
	gl_PointSize = v4StarLuminosity.w;
	
	vec4 v4ScaledStarLocation = 
		vec4(v4StarLocation.xyz * fSkydomeRadius, v4StarLocation.w);
	gl_Position = m4MVP * v4ScaledStarLocation;
}