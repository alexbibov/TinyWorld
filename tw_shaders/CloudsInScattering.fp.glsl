#version 430 core

#define ray_marching_steps 5
#define pi 3.1415926535897932384626433832795f

layout(location = 0) out vec4 v4FragIntensityScale;

uniform sampler3D s3dOutScatteringValues; 
uniform sampler2D s2dDensityPatternTexture;
uniform samplerBuffer sbDensityPatternRandomShiftsTexture;
uniform sampler2D s2dCelestialBodyInScattering;
uniform vec3 v3LightDirection;
uniform vec3 v3Scale;

uniform float fAlbedo;
uniform float fGamma;
uniform float fDensityScale;

uniform mat4 m4ProjectionToScaledObjectTransform;
uniform vec3 v3ObserverLocation;
uniform vec4 v4Viewport;


int getVertexId(uvec3 vertex_coord)
{
    ivec3 problem_size = textureSize(s3dOutScatteringValues, 0);
    int currentVertexId = int(problem_size.x*problem_size.y*vertex_coord.z + problem_size.y*vertex_coord.x + vertex_coord.y);
	return currentVertexId;
}


void main()
{
    //Reconstruct position of the fragment in the world space
	vec2 v2FragFocalPlaneCoord = 2*(gl_FragCoord.xy - v4Viewport.xy)/v4Viewport.zw - 1;
	float fFragDepth = 2*gl_FragCoord.z - 1.f;
    vec3 v3FragNDC = vec3(v2FragFocalPlaneCoord, fFragDepth) / gl_FragCoord.w;	
	
	vec4 v4FragScaledObjectSpace = m4ProjectionToScaledObjectTransform*vec4(v3FragNDC, 1.f/gl_FragCoord.w);
	vec3 v3FragCoordInScaledObjectSpace = v4FragScaledObjectSpace.xyz / v4FragScaledObjectSpace.w;
	vec3 v3ViewDirection = normalize(v3FragCoordInScaledObjectSpace - v3ObserverLocation);
	
	//Locate the furthermost intersection between the fragment and the cloud bounding box
	float t_max = intBitsToFloat(0x7F800000);    //+Infinity
    float t_min = intBitsToFloat(0xFF800000);    //-Infinity
	
    vec3 v3Aux1 = (v3Scale/2.f - v3FragCoordInScaledObjectSpace)/v3ViewDirection;
    vec3 v3Aux2 = (-v3Scale/2.f - v3FragCoordInScaledObjectSpace)/v3ViewDirection;
    vec3 aux_t_min = min(v3Aux1, v3Aux2);
    vec3 aux_t_max = max(v3Aux1, v3Aux2);
    
    t_max = min(min(min(t_max, aux_t_max.x), aux_t_max.y), aux_t_max.z);
    t_min = max(max(max(max(t_min, aux_t_min.x), aux_t_min.y), aux_t_min.z), 0);
	
	if(t_max > t_min)
	{
	    vec3 v3TraversalStart = v3FragCoordInScaledObjectSpace + t_max*v3ViewDirection;
		float fStepSize = -1.f/(ray_marching_steps - 1.f)*(t_max - t_min);
		
		float E_i = 1.f;
		for(int i = 1; i < ray_marching_steps; ++i)
		{
		    vec3 v3CurrentPoint = v3TraversalStart + v3ViewDirection*fStepSize*i;
			vec3 v3CurrentPointInTextureSpace = (v3CurrentPoint / v3Scale + .5f);
			vec3 v3CurrentPointInGridSpace = v3CurrentPointInTextureSpace * (textureSize(s3dOutScatteringValues, 0) - 1.f);
			vec3 v3ClosestGridVertex = round(v3CurrentPointInGridSpace);
			//vec2 texture_coordinates = v3ClosestGridVertex.xy - v3CurrentPointInGridSpace.xy;
			    //+ texelFetch(sbDensityPatternRandomShiftsTexture, getVertexId(uvec3(v3ClosestGridVertex))).rg;
			float density = texture(s2dDensityPatternTexture, gl_PointCoord).r*fDensityScale;
			float I_i = texture(s3dOutScatteringValues, v3CurrentPointInTextureSpace).r;
			
			float c = dot(v3ViewDirection, v3LightDirection);
			float S_i = -fAlbedo*density*(3.f/4.f*(1f + c*c))*I_i*fGamma/(4*pi);

            E_i = S_i + E_i*exp(-density);			
		}
		
		v4FragIntensityScale = vec4(0.f, 0.f, 0.f, 1 - E_i);
	}
	else
	{
	    vec3 v3FragInTextureSpace = (v3FragCoordInScaledObjectSpace / v3Scale + .5f);
		v4FragIntensityScale = vec4(1.f, 1.f, 1.f, texture(s3dOutScatteringValues, v3FragInTextureSpace).r*fDensityScale);
	}
}