#version 430 core

//This program describes common parameters and per-vertex transforms needed to support lighting model
//This shader code must be attached to the last stage of vertex processing of the target object


//**************************************************Global parameters**************************************************
const uint max_directional_lights = 100;		//maximal number of directional light sources
const uint max_point_lights = 100;			//maximal number of point light sources
const uint max_spot_lights = 100;				//maximal number of spot light sources
//*********************************************************************************************************************

//******************************************Parameters related to the lit object***************************************
uniform vec3 v3ViewerLocation;	//position of the viewer in global space
uniform mat3 m3ViewerRotation;	//rotation component of the transform converting scaled object space into the viewer space
uniform mat4 m4LightTransform;	//light transformation matrix (transforms lights from the global to scaled model space)
//*********************************************************************************************************************



//Interface block that declares extra output data from the vertex shader
out LIGHT_MODEL_VERTEX_DATA
{
	//Normal vector represented in the viewer space
	//vec3 v3Normal;

	//Linear depth of the vertex represented in the viewer space
	float fLinearDepth;	

	//Tangent, bi-normal, and normal vectors corresponding to the current vertex
	vec3 v3T, v3B, v3N;

	//Position of a vertex being lit in scaled object space
	vec3 v3VertexLocation_SOS;

	//Relative location of the viewer represented in scaled object space
	vec3 v3ViewerRelativeLocation_SOS;
}light_model_vertex_data;




//This function computes per-vertex data that is needed to support lighting model.
//DETAILED DESCRIPTION:
//v4Vertex — currently processed vertex
//v3Normal — normal vector of the current vertex, must be of unit length
//v3Tangent — tangent vector
//v3Binormal — bi-normal vector
//All inputs must be represented in scaled object space
void computeLightModelVertexData(vec4 v4Vertex, vec3 v3Normal, vec3 v3Tangent, vec3 v3Binormal)
{
	//Rotation part of the light transform
	mat3 m3LightTransform3D = mat3(m4LightTransform[0].xyz, m4LightTransform[1].xyz, m4LightTransform[2].xyz);
	
	//Transition part of the light transform
	vec3 v3LightTransformShift = m4LightTransform[3].xyz;

	//Store normal vector of the current vertex represented in the viewer space
	//light_model_vertex_data.v3Normal = m3ViewerRotation * v3Normal;

	//Store linear depth of the current vertex as it appears in the viewer space
	vec3 v3ViewerLocation_sos = m3LightTransform3D * v3ViewerLocation + v3LightTransformShift;	//location of the viewer represented in scaled object space
	vec3 v3ViewerTransformZ = vec3(m3ViewerRotation[0][2], m3ViewerRotation[1][2], m3ViewerRotation[2][2]);
	light_model_vertex_data.fLinearDepth = dot(v3ViewerTransformZ, v4Vertex.xyz - v3ViewerLocation_sos);
	


	//Store TBN basis vectors to be processed on the fragment stage
	light_model_vertex_data.v3T = v3Tangent;
	light_model_vertex_data.v3B = v3Binormal;
	light_model_vertex_data.v3N = v3Normal;

	light_model_vertex_data.v3VertexLocation_SOS = v4Vertex.xyz;
	light_model_vertex_data.v3ViewerRelativeLocation_SOS = v3ViewerLocation_sos - v4Vertex.xyz;
}




//Computes tangent direction T and bi-normal vector B based on vertex positions and corresponding bump map coordinates.
//P0, P1, and P2 — vertices of the target triangle. Note that the vertices must be represent in SCALED object space
//P0_bm_coords, P1_bm_coords, and P2_bm_coords — coordinates of the bump map assigned to the points P0, P1, and P2.
//
//The return value is packed into 3-by-3 matrix with the first column containing the vector T, and the second column
//containing the vector B. The third column is not in use.
mat3x3 computeTangentDirections(vec3 v3P0, vec3 v3P1, vec3 v3P2, vec2 v2P0_bm_coords, vec2 v2P1_bm_coords, vec2 v2P2_bm_coords)
{
	vec3 v3Q1 = v3P1 - v3P0;
	vec3 v3Q2 = v3P2 - v3P0;

	vec2 v2ST1 = v2P1_bm_coords - v2P0_bm_coords;
	vec2 v2ST2 = v2P2_bm_coords - v2P0_bm_coords;

	float det = (v2ST1.s * v2ST2.t - v2ST2.s * v2ST1.t);
	vec3 v3T = (v2ST2.t*v3Q1 - v2ST1.t*v3Q2) / det;
	vec3 v3B = (-v2ST2.s*v3Q1 + v2ST1.s*v3Q2) / det;

	mat3x3 m3x3RV;
	m3x3RV[0][0] = v3T[0]; m3x3RV[0][1] = v3T[1]; m3x3RV[0][2] = v3T[2];
	m3x3RV[1][0] = v3B[0]; m3x3RV[1][1] = v3B[1]; m3x3RV[1][2] = v3B[2];

	return m3x3RV;
}