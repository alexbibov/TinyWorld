//Vertex shader implementing projection of proxy geometry by the means of light projection volume

#version 430 core

in vec4 v4VertexPosition;	//location of the currently processed vertex position
in vec3 v3TextureCoordinate3D;	//3D texture coordinate associated with the currently processed vertex

uniform mat4 m4LightViewProjection;		//light Model-View-Projection transform

out vec3 v3TexCoord3D;	//3D texture coordinate corresponding to the currently processed vertex


void main()
{
	gl_Position = m4LightViewProjection * v4VertexPosition;
	v3TexCoord3D = v3TextureCoordinate3D;
} 