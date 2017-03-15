//Vertex program implementing projection of proxy geometry by the means of viewer projection
//and computing texture coordinates used to access the data stored in light buffer

#version 430 core

in vec4 v4VertexPosition;	//location of the currently processed vertex position
in vec3 v3TextureCoordinate3D;	//3D texture coordinate associated with the currently processed vertex

uniform mat4 m4MVP;		//standard Model-View-Projection transform
uniform mat4 m4LightViewProjection;		//light Model-View-Projection transform employed to compute texture 
										//coordinates used to access the light buffer

out vec3 v3TexCoord3D;	//3D texture coordinate corresponding to the currently processed vertex
out noperspective vec2 v2LightBufferTexCoords;	//texture coordinates needed to access data in the light buffer

void main()
{
	gl_Position = m4MVP * v4VertexPosition;

	vec4 v4Aux = m4LightViewProjection * v4VertexPosition;
	v2LightBufferTexCoords = (v4Aux.xy / v4Aux.w) * 0.5f + 0.5f;

	v3TexCoord3D = v3TextureCoordinate3D;
}