//Vertex stage of rendering program implementing shading of a water surface

#version 430 core


in vec4 v4TessellationBilletVertex;		//position of a vertex belonging to the tessellation billet
//in vec2 v2TessellationBilletTexCoord;	//texture coordinates of the currently processed point in the tessellation billet

uniform sampler2D s2dWaterHeightMap;	//height map of the water surface

void main()
{
	vec2 v2SampleTexCoords = vec2(v4TessellationBilletVertex.x + 0.5f, 0.5f - v4TessellationBilletVertex.z);
	float fHeight = textureLod(s2dWaterHeightMap, v2SampleTexCoords, 0).r;
	gl_Position = vec4(v4TessellationBilletVertex.x, fHeight, v4TessellationBilletVertex.z, 1.0f);
}