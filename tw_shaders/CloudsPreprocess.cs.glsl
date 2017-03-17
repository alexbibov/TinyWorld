#version 430 core

#define ray_marching_steps 3
#define pi 3.1415926535897932384626433832795f

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform image3D writeonly i3dOutScatteringValues;
uniform samplerBuffer sbDensityPatternRandomShiftsTexture;
uniform vec3 v3LightDirection;
uniform uvec3 uv3DomainResolution;
uniform vec3 v3Scale;

uniform float fAlbedo;
uniform float fGamma;
uniform float fDensityScale;


int getVertexId(uvec3 vertex_coord)
{
    ivec3 problem_size = ivec3(gl_NumWorkGroups*gl_WorkGroupSize);
    int currentVertexId = int(problem_size.x*problem_size.y*vertex_coord.z + problem_size.y*vertex_coord.x + vertex_coord.y);
	return currentVertexId;
}

void main()
{
    if(any(greaterThanEqual(gl_GlobalInvocationID, uv3DomainResolution))) return;
    
    ivec3 problem_size = ivec3(gl_NumWorkGroups*gl_WorkGroupSize);
    
    vec3 currentVertexPosition = (vec3(gl_GlobalInvocationID)/(problem_size - 1.f) - .5f) * v3Scale;
    
    
    // locate point where current vector exits the domain
    float t_max = intBitsToFloat(0x7F800000);    //+Infinity
    float t_min = intBitsToFloat(0xFF800000);    //-Infinity
    
    
    vec3 v3Aux1 = (v3Scale/2.f - currentVertexPosition)/v3LightDirection;
    vec3 v3Aux2 = (-v3Scale/2.f - currentVertexPosition)/v3LightDirection;
    vec3 aux_t_min = min(v3Aux1, v3Aux2);
    vec3 aux_t_max = max(v3Aux1, v3Aux2);
    
    float t_max = min(min(min(t_max, aux_t_max.x), aux_t_max.y), aux_t_max.z);
    float t_min = max(max(max(max(t_min, aux_t_min.x), aux_t_min.y), aux_t_min.z), 0);
    t_max = max(t_max, t_min);
    
    if(t_max > t_min)
    {
	    vec3 v3TraversalStart = currentVertexPosition + v3LightDirection*t_max;
        float fStepSize = -1.f/(ray_marching_steps - 1.f)*(t_max-t_min);
		
		float I_i = 1.f;
        for(int i = 1; i < ray_marching_steps; ++i)
        {
            vec3 v3CurrentPoint = v3TraversalStart + v3LightDirection*fStepSize*i;
			vec3 v3CurrentPointInGridSpace = (v3CurrentPoint / v3Scale + .5f)*(problem_size + 1.f); 
			uvec3 v3ClosestVertex = uvec3(round(v3CurrentPointInGridSpace));
			
			v3ClosestVertex
			
			float density = texture(s2dDensityPatternTexture, texture_coordinates).r*fDensityScale;
			
			float g = -fAlbedo*density*1.5f*I_i*fGamma/(4*pi);

            I_i = g + I_i*exp(-density);			
		}
		
		imageStore(i3dOutScatteringValues, ivec3(gl_GlobalInvocationID), vec4(I_i,0,0,0));
    }
    else
    {
        imageStore(i3dOutScatteringValues, ivec3(gl_GlobalInvocationID), vec4(1,0,0,0));
    }
    
}


