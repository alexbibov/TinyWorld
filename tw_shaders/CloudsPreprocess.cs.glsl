#version 430 core

#define ray_marching_steps 3
#define pi 3.1415926535897932384626433832795f

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform image3D writeonly i3dOutScatteringValues;
uniform sampler2D s2dDensityPatternTexture;
uniform samplerBuffer sbDensityPatternRandomShiftsTexture;
uniform vec3 v3LightDirection;
uniform uvec3 uv3DomainResolution;
uniform vec3 v3DomainSize;
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
    
    vec3 currentVertexPosition = (vec3(gl_GlobalInvocationID)/(problem_size - 1.f) - .5f) * v3DomainSize * v3Scale;
    
    
    // locate point where current vector exits the domain
    float t_max = intBitsToFloat(0x7F800000);    //+Infinity
    float t_min = intBitsToFloat(0xFF800000);    //-Infinity
    
    float aux1 = (v3DomainSize.x/2.f - currentVertexPosition.x)/v3LightDirection.x;
    float aux2 = (-v3DomainSize.x/2.f - currentVertexPosition.x)/v3LightDirection.x;
    float aux_t_max = max(aux1, aux2);
    float aux_t_min = min(aux1, aux2);
    t_max = min(t_max, aux_t_max);
    t_min = max(t_min, aux_t_min);
    
    aux1 = (v3DomainSize.y/2.f - currentVertexPosition.y)/v3LightDirection.y;
    aux2 = (-v3DomainSize.y/2.f - currentVertexPosition.y)/v3LightDirection.y;
    aux_t_max = max(aux1, aux2);
    aux_t_min = min(aux1, aux2);
    t_max = min(t_max, aux_t_max);
    t_min = max(t_min, aux_t_min);
    
    aux1 = (v3DomainSize.z/2.f - currentVertexPosition.z)/v3LightDirection.z;
    aux2 = (-v3DomainSize.z/2.f - currentVertexPosition.z)/v3LightDirection.z;
    aux_t_max = max(aux1, aux2);
    aux_t_min = min(aux1, aux2);
    t_max = min(t_max, aux_t_max);
    t_min = max(t_min, aux_t_min);
    t_min = max(t_min, 0.f);
    t_max = max(t_min, t_max);
    
    if(t_max > t_min)
    {
	    vec3 v3TraversalStart = currentVertexPosition + v3LightDirection*t_max;
        float fStepSize = -1.f/(ray_marching_steps - 1.f)*(t_max-t_min);
		
		float I_i = 1.f;
        for(int i = 1; i < ray_marching_steps; ++i)
        {
            vec3 v3CurrentPoint = v3TraversalStart + v3LightDirection*fStepSize*i;
			vec3 v3CurrentPointInGridSpace = (v3CurrentPoint / v3Scale / v3DomainSize + .5f)*(problem_size + 1.f); 
			uvec3 v3ClosestVertex = uvec3(round(v3CurrentPointInGridSpace));
			
			vec2 texture_coordinates = v3ClosestVertex.xy - v3CurrentPointInGridSpace.xy;
			    //+ texelFetch(sbDensityPatternRandomShiftsTexture, getVertexId(v3ClosestVertex)).rg;
			
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


