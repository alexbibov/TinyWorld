#version 430 core

in vec4 v4VertexPosition;	//location of the currently processed vertex position
in vec3 v3TextureCoordinate3D;	//3D texture coordinate associated with the currently processed vertex

uniform mat4 m4MVP;		//Standard Model-View-Projection transform

//Redeclaration of standard gl_PerVertex out block
//required by ARB_separate_shader_objects extension
out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};

out vec3 v3TexCoord3D;	//3D texture coordinate corresponding to the currently processed vertex


void main()
{
	gl_Position = m4MVP * v4VertexPosition;
	v3TexCoord3D = v3TextureCoordinate3D;
} 