#version 430 core

in vec4 vertex_position;		//position of the currently processed vertex of the transparent cube

uniform mat4 m4MVP;		//standard Model-View-Projection transform

void main()
{	
	//Compute projected position of the vertex
	gl_Position = m4MVP * vertex_position;
}