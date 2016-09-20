//Implements sample retrieval from an environment map stored in equirectangular format given environment map sampler and reflection vector represented in viewer space.


//Input arguments
//s2dEnvironmentMap — sampler used to retrieve data from environment map
//v3ReflectionVector — reflection vector represented in the world space
//
//Return value
//returns color value sampled from the given environment map
subroutine(EnvironmentMapSampleRetriever) vec3 EquirectangularEnvironmentMapSampleRetriever(sampler2D s2dEnvironmentMap, vec3 v3ReflectionVector)
{
	vec3 v3NormalizedReflectionVector = normalize(v3ReflectionVector);
	vec2 tc;

	tc.y = 0.5f + v3NormalizedReflectionVector.y * 0.5f;
	v3NormalizedReflectionVector.y = 0;

	tc.x = normalize(v3NormalizedReflectionVector).x * 0.25f;
	float s = sign(v3NormalizedReflectionVector.z) + float(v3NormalizedReflectionVector.z == 0.0f);
	tc.x = s * tc.x + 0.25f + float(s > 0) * 0.5f;
	
	return texture(s2dEnvironmentMap, tc).rgb;
}


//Input arguments
//s2daEnvironmentMap — sampler used to retrieve data from array environment map
//v3ReflectionVector — reflection vector represented in the world space
//layer — array layer, which will be the source for the data sampling
//
//Return value
//returns color value sampled from the given environment map
subroutine(ArrayEnvironmentMapSampleRetriever) vec3 EquirectangularArrayEnvironmentMapSampleRetriever(sampler2DArray s2daEnvironmentMap, vec3 v3ReflectionVector, uint layer)
{
	vec3 v3NormalizedReflectionVector = normalize(v3ReflectionVector);
	vec2 tc;

	tc.y = 0.5f + v3NormalizedReflectionVector.y * 0.5f;
	v3NormalizedReflectionVector.y = 0;

	tc.x = normalize(v3NormalizedReflectionVector).x * 0.25f;
	float s = sign(v3NormalizedReflectionVector.z) + float(v3NormalizedReflectionVector.z == 0.0f);
	tc.x = s * tc.x + 0.25f + float(s > 0) * 0.5f;
	
	return texture(s2daEnvironmentMap, vec3(tc, layer)).rgb;
}