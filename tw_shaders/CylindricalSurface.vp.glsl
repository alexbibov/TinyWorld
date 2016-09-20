//Cylindrical surface vertex processing program

#version 430 core


#define pi 3.1415926535897932384626433832795f

in vec4 v4Position;		//position of the vertex
in vec2 v2TexCoord;		//texture coordinate of the vertex

uniform sampler2D surface_map;	//surface map texture sampler

out VS_DATA
{
	vec4 v4Vertex;
	vec2 v2TextureCoordinate;
}vs_out;


void main()
{
	//Check if vertex lies on the surface
	if(v4Position.w == 0.0f)
	{
		vs_out.v4Vertex.xy = textureLod(surface_map, v4Position.xy, 0).rg;
		vs_out.v4Vertex.zw = vec2(v4Position.z, 1.0f);
	}
	else
	{
		vs_out.v4Vertex = v4Position;
	}

	vs_out.v2TextureCoordinate = v2TexCoord;
}


