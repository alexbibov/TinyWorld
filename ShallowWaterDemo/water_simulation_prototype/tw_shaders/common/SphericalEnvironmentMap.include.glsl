//Implements function that retrieves samples from a spherical environment map based on a given reflection vector represented in the world space


//Input arguments
//s2dEnvironmentMap — sampler used to retrieve data from environment map
//v3ReflectionVector — reflection vector represented in the world space
//
//Return value
//returns color value sampled from the given environment map
subroutine(EnvironmentMapSampleRetriever) vec3 SphericalEnvironmentMapSampleRetriever(sampler2D s2dEnvironmentMap, vec3 v3ReflectionVector)
{
	vec3 v3NormalizedReflectionVector = normalize(v3ReflectionVector);
	v3NormalizedReflectionVector.z += 1.0f;

	float m = 0.5f * inversesqrt(dot(v3NormalizedReflectionVector, v3NormalizedReflectionVector));
	float s = v3NormalizedReflectionVector.x*m + 0.5f;
	float t = v3NormalizedReflectionVector.y*m + 0.5f;

	return texture(s2dEnvironmentMap, vec2(s, t)).rgb;
}


//Input arguments
//s2daEnvironmentMap — sampler used to retrieve data from array environment map
//v3ReflectionVector — reflection vector represented in the world space
//layer — array layer, which will be the source for the data sampling
//
//Return value
//returns color value sampled from the given environment map
subroutine(ArrayEnvironmentMapSampleRetriever) vec3 SphericalArrayEnvironmentMapSampleRetriever(sampler2DArray s2daEnvironmentMap, vec3 v3ReflectionVector, uint layer)
{
	vec3 v3NormalizedReflectionVector = normalize(v3ReflectionVector);
	v3NormalizedReflectionVector.z += 1.0f;

	float m = 0.5f * inversesqrt(dot(v3NormalizedReflectionVector, v3NormalizedReflectionVector));
	float s = v3NormalizedReflectionVector.x*m + 0.5f;
	float t = v3NormalizedReflectionVector.y*m + 0.5f;

	return texture(s2daEnvironmentMap, vec3(s, t, layer)).rgb;
}
