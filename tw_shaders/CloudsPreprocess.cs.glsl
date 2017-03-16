#version 430 core

#define ray_marching_steps 3

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform image3D i3dOutScatteringValues;
uniform sampler2D s2dDensityPatternTexture;
uniform samplerBuffer sbDensityPatternRandomShiftsTexture;
uniform vec3 v3LightDirection;
uniform uvec3 uv3DomainResolution;
uniform vec3 v3DomainSize;

void main()
{
    if(any(greaterThanEqual(gl_GlobalInvocationID, uv3DomainResolution))) return;
    
    ivec3 problem_size = gl_NumWorkGroups*gl_WorkGroupSize;
    
    vec3 currentVertexPosition = (vec3(gl_GlobalInvocationID)/(problem_size - 1.f) - .5f) * v3DomainSize;
    int currentVertexId = problem_size.x*problem_size.y*gl_GlobalInvocationID.z + problem_size.y*gl_GlobalInvocationID.x + gl_GlobalInvocationID.y;
    
    // locate point where current vector exits the domain
    float t_max = intBitsToFloat(0x7F800000);    //+Infinity
    float t_min = intBitsToFloat(0xFF800000);    //-Infinity
    
    float aux_t_max = (v3DomainSize.x/2.f - currentVertexPosition.x)/v3LightDirection.x;
    float aux_t_min = (-v3DomainSize.x/2.f - currentVertexPosition.x)/v3LightDirection.x;
    t_max = min(t_max, aux_t_max);
    t_min = max(t_min, aux_t_min);
    
    aux_t_max = (v3DomainSize.y/2.f - currentVertexPosition.y)/v3LightDirection.y;
    aux_t_min = (-v3DomainSize.y/2.f - currentVertexPosition.y)/v3LightDirection.y;
    t_max = min(t_max, aux_t_max);
    t_min = max(t_min, aux_t_min);
    
    aux_t_max = (v3DomainSize.z/2.f - currentVertexPosition.z)/v3LightDirection.z;
    aux_t_min = (-v3DomainSize.z/2.f - currentVertexPosition.z)/v3LightDirection.z;
    t_max = min(t_max, aux_t_max);
    t_min = max(t_min, aux_t_min);
    t_min = max(t_min, 0.f);
    t_max = max(t_min, t_max);
    
    if(t_max > t_min)
    {
        float fStepSize = 1.f/(ray_marching_steps - 1.f)*(t_max-t_min);
        for(int i = 0; i < ray_marching_steps; ++i)
        {
            vec3 v3CurrentPoint = currentVertexPosition + v3LightDirection*fStepSize*i;
        }
    }
    else
    {
        vec2 texture_coordinates = fract(vec2(.5f) + texelFetch(sbDensityPatternRandomShiftsTexture, currentVertexId).xy);
        float density = texture(s2dDensityPatternTexture, texture_coordinates);
    }
    
}


