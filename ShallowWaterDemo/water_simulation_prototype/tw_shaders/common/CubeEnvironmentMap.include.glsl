//Implements sample retrieval from a cube environment map based on given sampler and on reflection vector represented in viewer space

//Input parameters
//scEnvironmentMap — sampler object used to extract values from environment map
//v3ReflectionVector — reflection vector represented in viewer space
//Return value
//returns color value corresponding to the sample extracted from the given environment map
subroutine(EnvironmentMapSampleRetriever) vec4 CubeEnvironmentMapSampleRetriever(samplerCube scEnvironmentMap, vec3 v3ReflectionVector)
{

}