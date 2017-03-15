//Implements geometry stage of the water surface shading program

#version 430 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in TEP_DATA
{
	vec2 v2TexCoord;	//texture coordinates of the current vertex output
	vec3 v3Normal;	//normal vector corresponding to the current vertex output 
}tep_in[];

out GS_DATA
{
	vec2 v2TexCoord;	//texture coordinates of the current vertex output
	float fWaterElevation;	//water level at the current vertex
}gs_out;

uniform mat4 m4ModelViewTransform;	//Transforms coordinates from scaled object space to the view space
uniform mat4 m4ProjectionTransform;	//Transforms coordinates from the view space to the clip space
uniform vec3 v3Scale;	//scale factors transforming units from the nominal non-dimensional to the scaled object space
uniform sampler2D s2dTopographyHeightmap;	//topography heightmap


void computeLightModelVertexData(vec4 v4Vertex, vec3 v3Normal, vec3 v3Tangent, vec3 v3Binormal);
mat3x3 computeTangentDirections(vec3 v3P0, vec3 v3P1, vec3 v3P2, vec2 v2P0_bm_coords, vec2 v2P1_bm_coords, vec2 v2P2_bm_coords);

void main()
{
	vec3 v3P0 = gl_in[0].gl_Position.xyz; v3P0.xz *= v3Scale.xz;
	vec3 v3P1 = gl_in[1].gl_Position.xyz; v3P1.xz *= v3Scale.xz;
	vec3 v3P2 = gl_in[2].gl_Position.xyz; v3P2.xz *= v3Scale.xz;

	mat3x3 m3x3TB = computeTangentDirections(v3P0, v3P1, v3P2, 
		tep_in[0].v2TexCoord, tep_in[1].v2TexCoord, tep_in[2].v2TexCoord);
	vec3 v3T = vec3(m3x3TB[0][0], m3x3TB[0][1], m3x3TB[0][2]);
	vec3 v3B = vec3(m3x3TB[1][0], m3x3TB[1][1], m3x3TB[1][2]);


	for(int i = 0; i < gl_in.length; ++i)
	{
		computeLightModelVertexData(vec4(v3Scale.x, 1.0f, v3Scale.z, 1.0f) * gl_in[i].gl_Position, tep_in[i].v3Normal, v3T, v3B);
		gs_out.v2TexCoord = tep_in[i].v2TexCoord;

		gs_out.fWaterElevation = gl_in[i].gl_Position.y - textureLod(s2dTopographyHeightmap, vec2(gl_in[i].gl_Position.x + 0.5f, 0.5f - gl_in[i].gl_Position.z), 0).r;
		gl_Position = m4ProjectionTransform * (m4ModelViewTransform * vec4(v3Scale.x * gl_in[i].gl_Position.x, gl_in[i].gl_Position.y, v3Scale.z * gl_in[i].gl_Position.z, gl_in[i].gl_Position.w));
		EmitVertex();
	}
	EndPrimitive();
}  