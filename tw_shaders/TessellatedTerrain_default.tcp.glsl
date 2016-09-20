//Tessellation control program for default rendering mode of object TessellatedTerrain

#version 430 core

layout (vertices = 4) out;			//Output patches composed of 4 vertices

//Data passed from the vertex shader stage
in VS_DATA
{
	vec2 raw_texcoord;	//raw texture coordinates aligned to the tessellation billet
}vs_in[];

//Data passed to the next shader stage
out TCS_DATA
{
	vec2 raw_texcoord;	//raw texture coordinates aligned to the tessellation billet
}tcs_out[];


uniform mat4 ModelViewTransform;		//Model-View transform (includes scaling transform)
uniform mat4 ProjectionTransform;		//Projection transform

//uniform uint num_u_base_nodes;		//horizontal resolution of the tessellation billet
//uniform uint num_v_base_nodes;		//vertical resolution of the tessellation billet

uniform float lod;	//Level-Of-Detail controlling adaptive tessellation
uniform uvec2 screen_size;	//Size of the canvas of the target rendering device represented in pixels


//Projects given vertex onto NDC
vec4 ontoNDC(vec4 v)
{
	vec4 NDC_v = ProjectionTransform * v;
	NDC_v.xyz /= NDC_v.w;
	return NDC_v;
}


//Projects given vertex to the screen space
vec2 ontoScreen(vec4 v)
{
	return clamp((v.xy + 1.0f) * 0.5f, -0.3f, 1.3f) * screen_size;
}


//Checks if the vertex represented in NDC lies outside the clip space
bool isOffscreen(vec4 v)
{
	if(v.z < -0.5f)
		return true;
		
	return any(lessThan(v.xy, vec2(-1.3f)) || greaterThan(v.xy, vec2(1.3f)));
}


//Transforms given vertex to the Model-View space
vec4 toViewSpace(vec4 v)
{
	return (ModelViewTransform * v);
}

//Computes automatic level-of-detail for the given edge represented by coupled pair of 
//vertices transformed to the viewer's frame
float getLODFactor(vec4 v1, vec4 v2)
{
	float radius = distance(v1.xyz, v2.xyz) * 0.5f;
	vec4 centre = vec4((v1.xyz + v2.xyz) * 0.5f, 1.0f);
	
	vec4 p1 = ontoNDC(centre - vec4(radius, 0.0f, 0.0f, 0.0f));
	vec4 p2 = ontoNDC(centre + vec4(radius, 0.0f, 0.0f, 0.0f));
	return clamp(distance(ontoScreen(p1), ontoScreen(p2)) / lod, 1.0f, 64.0f);
}


void main()
{
	//Pass raw texture coordinates through
	tcs_out[gl_InvocationID].raw_texcoord = vs_in[gl_InvocationID].raw_texcoord;
	
	//Pass untransformed tessellation billet vertex positions through
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
	
	if(gl_InvocationID == 0)
	{
		vec4 ll = toViewSpace(gl_in[0].gl_Position);	//lower-left corner of patch represented in viewer's frame
		vec4 lr = toViewSpace(gl_in[1].gl_Position);	//lower-right corner of patch represented in viewer's frame
		vec4 ur = toViewSpace(gl_in[2].gl_Position);	//upper-right corner of patch represented in viewer's frame
		vec4 ul = toViewSpace(gl_in[3].gl_Position);	//upper-left corner of patch represented in viewer's frame
		
		
		//If all vertices of the current patch lie outside clip space, the patch is guaranteed to be invisible and gets discarded
		if(all(bvec4(
			isOffscreen(ontoNDC(ll)), 
			isOffscreen(ontoNDC(lr)), 
			isOffscreen(ontoNDC(ur)), 
			isOffscreen(ontoNDC(ul))
			)))
		{
			gl_TessLevelOuter[0] = 0;
			gl_TessLevelOuter[1] = 0;
			gl_TessLevelOuter[2] = 0;
			gl_TessLevelOuter[3] = 0;
			
			gl_TessLevelInner[0] = 0;
			gl_TessLevelInner[1] = 0;
		}
		else
		{
			//Set outer tessellation levels
			//float diameter = max(1.0f / num_u_base_nodes, 1.0f / num_v_base_nodes);
			gl_TessLevelOuter[0] =  getLODFactor(ul, ll);
			gl_TessLevelOuter[1] =  getLODFactor(ll, lr);
			gl_TessLevelOuter[2] =  getLODFactor(lr, ur);
			gl_TessLevelOuter[3] =  getLODFactor(ur, ul);
			
			//Set inner tessellation levels
			gl_TessLevelInner[0] = mix(gl_TessLevelOuter[1], gl_TessLevelOuter[2], 0.5f);
			gl_TessLevelInner[1] = mix(gl_TessLevelOuter[0], gl_TessLevelOuter[3], 0.5f);
		}
		
	}
}

