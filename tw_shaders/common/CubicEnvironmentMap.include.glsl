//Implements sample retrieval from a cubic environment map given sampler object and reflection vector represented in view space

//Input arguments
//s2dEnvironmentMap — sampler used to retrieve data from environment map
//v3ReflectionVector — reflection vector represented in the world space
//
//Return value
//returns color value sampled from the given environment map
subroutine(CubeEnvironmentMapSampleRetriever) vec3 CubicEnvironmentMapSampleRetriever(samplerCube scEnvironmentMap, vec3 v3ReflectionVector)
{   
	return texture(scEnvironmentMap, v3ReflectionVector).rgb;
}


//Input arguments
//s2daEnvironmentMap — sampler used to retrieve data from array environment map
//v3ReflectionVector — reflection vector represented in the world space
//layer — array layer, which will be the source for the data sampling
//
//Return value
//returns color value sampled from the given environment map
subroutine(CubeArrayEnvironmentMapSampleRetriever) vec3 CubicArrayEnvironmentMapSampleRetriever(samplerCubeArray scaEnvironmentMap, vec3 v3ReflectionVector, uint layer)
{
	return texture(scaEnvironmentMap, vec4(v3ReflectionVector, layer)).rgb;
}