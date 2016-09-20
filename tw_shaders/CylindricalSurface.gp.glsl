#version 430 core


layout(triangles) in;
layout(triangle_strip) out;
layout(max_vertices=3) out;


uniform vec3 v3Scale;	//object scale factors
uniform mat4 m4ModelView;		//Model-View transform (non-dimensional object space => view space)
uniform mat4 m4Projection;	//Projection matrix (view space => NDC)


in VS_DATA
{
	vec4 v4Vertex;
	vec2 v2TextureCoordinate;
}vs_in[];


out GS_DATA
{
	vec2 v2TextureCoordinate;
}gs_out;


void computeLightModelVertexData(vec4 v4Vertex, vec3 v3Normal, vec3 v3Tangent, vec3 v3Binormal);
mat3x3 computeTangentDirections(vec3 v3P0, vec3 v3P1, vec3 v3P2, vec2 v2P0_bm_coords, vec2 v2P1_bm_coords, vec2 v2P2_bm_coords);


void main()
{
	//Compute TBN transform of the currently processed triangle
	vec3 P0 = v3Scale*vs_in[0].v4Vertex.xyz;
	vec3 P1 = v3Scale*vs_in[1].v4Vertex.xyz;
	vec3 P2 = v3Scale*vs_in[2].v4Vertex.xyz;
	mat3x3 TB = computeTangentDirections(P0, P1, P2, vs_in[0].v2TextureCoordinate, vs_in[1].v2TextureCoordinate, vs_in[2].v2TextureCoordinate);
	vec3 T = vec3(TB[0][0],TB[0][1],TB[0][2]);
	vec3 B = vec3(TB[1][0],TB[1][1],TB[1][2]);
	vec3 N = normalize(cross(P1-P0, P2-P0));

	for(int i = 0; i < vs_in.length; ++i)
	{
		computeLightModelVertexData(vec4(v3Scale, 1.0f)*vs_in[i].v4Vertex, N, T, B);
		gl_Position = m4Projection*(m4ModelView*vs_in[i].v4Vertex);
		gs_out.v2TextureCoordinate = vs_in[i].v2TextureCoordinate;
		EmitVertex();
	}
	EndPrimitive();
}