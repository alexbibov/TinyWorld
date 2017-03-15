//Tessellation evaluation program used by default rendering mode of TessellatedTerrain

#version 430 core

layout (quads) in;		//Tessellation is done in quads-mode
layout (fractional_even_spacing) in;		//Tessellation subdivision is odd-spacing
layout (ccw) in;	//Front face orientation used by tessellation is counter clock-wise order

uniform sampler2D height_map_sampler;	//texture sampler associated with displacement map
//uniform uint num_u_base_nodes;		//horizontal resolution of the tessellation billet
//uniform uint num_v_base_nodes;		//vertical resolution of the tessellation billet

uniform vec3 Scale;						//model scale factors

//Data passed from the tessellation control shader
in TCS_DATA
{
	vec2 raw_texcoord;	//raw texture coordinates aligned to the tessellation billet
}tcs_in[];

//Data passed to the next shader stage
out TES_DATA
{
	vec2 texcoord;		//terrain texture coordinates
	vec3 normal;		//terrain vertex normals
}tes_out;

void main()
{
	//Compute position of the current vertex
	vec4 p1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
	vec4 p2 = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x);
	vec4 p = mix(p1, p2, gl_TessCoord.y);
	
	//Compute displacement of the current vertex
	p.y = textureLod(height_map_sampler, vec2(p.x + 0.5f, -(p.z - 0.5f)), 0).r;
	
	gl_Position = p;

	//Generate normal
	vec2 v2TriangleSize = 
			vec2(distance(gl_in[0].gl_Position.xz, gl_in[1].gl_Position.xz), distance(gl_in[1].gl_Position.xz, gl_in[2].gl_Position.xz))
			/ max(max(gl_TessLevelOuter[0], gl_TessLevelOuter[1]), max(gl_TessLevelOuter[2], gl_TessLevelOuter[3])); 
	vec3 pos_dx = p.xyz + vec3(v2TriangleSize.x, 0, 0);
	vec3 pos_dz = p.xyz + vec3(0, 0, v2TriangleSize.y);
	pos_dx.y = textureLod(height_map_sampler, vec2(pos_dx.x + 0.5f, -(pos_dx.z - 0.5f)), 0).r;
	pos_dz.y = textureLod(height_map_sampler, vec2(pos_dz.x + 0.5f, -(pos_dz.z - 0.5f)), 0).r;
	tes_out.normal = normalize(cross((pos_dz - p.xyz)*Scale, (pos_dx - p.xyz)*Scale));

	
	//Generate texture coordinates
	vec2 aux_tc1 = mix(tcs_in[0].raw_texcoord, tcs_in[1].raw_texcoord, gl_TessCoord.x);
	vec2 aux_tc2 = mix(tcs_in[3].raw_texcoord, tcs_in[2].raw_texcoord, gl_TessCoord.x);
	tes_out.texcoord = mix(aux_tc1, aux_tc2, gl_TessCoord.y);
	
	//tes_out.texcoord = vec2((p.x + 0.5f) * (num_u_base_nodes - 1), -(p.z - 0.5f) * (num_v_base_nodes - 1));
}