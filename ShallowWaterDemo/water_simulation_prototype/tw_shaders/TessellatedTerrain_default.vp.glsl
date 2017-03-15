//Default vertex processing program for tessellated terrain

#version 430 core

in vec4 tess_billet_vertex_position;		//position of a vertex from tessellation billet
in vec2 tess_billet_texcoord;	//texture coordinates aligned to the tessellation billet

//Contains data associated with each vertex
out VS_DATA
{
	vec2 raw_texcoord;	//raw texture coordinates aligned to the tessellation billet
}vs_out;

uniform sampler2D height_map_sampler;	//texture sampler associated with displacement map



void main()
{
	//Pass raw texture coordinates through
	vs_out.raw_texcoord = tess_billet_texcoord;

	//Extrude current tessellation billet vertex using data from displacement map
	float u = tess_billet_vertex_position.x + 0.5f;		//get u-coordinate of displacement map (default range of tess_billet_vertex_position.x is [-0.5, 0.5])
	float v = -(tess_billet_vertex_position.z - 0.5f);		//get v-coordinate of displacement map (default range of tess_billet_vertex_position.z is [0.5, -0.5])
	
	float height = textureLod(height_map_sampler, vec2(u, v), 0).r;	//get height of the current tessellation billet vertex
	
	//gl_Position stores displaced vertices from the tessellation billet represented in model space
	gl_Position = vec4(tess_billet_vertex_position.x, height, tess_billet_vertex_position.z, 1.0f);	
}