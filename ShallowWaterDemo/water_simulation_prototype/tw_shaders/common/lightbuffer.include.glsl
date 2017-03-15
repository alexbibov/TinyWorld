//Declaration of uniform buffer containing description of the light sources



//**************************************************Global parameters**************************************************
const uint max_directional_lights = 100;		//maximal number of directional light sources
const uint max_point_lights = 100;			//maximal number of point light sources
const uint max_spot_lights = 100;				//maximal number of spot light sources
//*********************************************************************************************************************



layout(std140) uniform LightBuffer
{
	//********************************************************************Ambient light description********************************************************************
	vec3 v3AmbientLightIntensity;	//intensity of the ambient light affecting the object
	//*****************************************************************************************************************************************************************

	//**************************************************************Directional light sources description**************************************************************
	uint nDirectionalLights;	//factual number of directional lights that affect the object
	vec3 v3DirectionalLightDirections[max_directional_lights];	//direction of each directional light source
	vec3 v3DirectionalLightIntensities[max_directional_lights];	//intensity of each directional light source
	//*****************************************************************************************************************************************************************

	//*****************************************************************Point light sources description*****************************************************************
	uint nPointLights;	//factual number of point lights that affect the object
	vec3 v3PointLightLocations[max_point_lights];	//locations of point light sources
	vec3 v3PointLightAttenuationFactors[max_point_lights];	//attenuation factors of the point light sources. 
															//Here each triplet (x, y, z) in the array defines 
															//polynomial attenuation by the factor of x + y*d + z*d^2
	vec3 v3PointLightIntensities[max_point_lights];	//intensities of the point light sources
	float fPointLightHazeIntensityFactors[max_point_lights];	//intensity factors applied when computing atmospheric haze for point lights
	float fPointLightHazeLocationDecayFactors[max_point_lights];	//haze location decay factors of the point lights
	//*****************************************************************************************************************************************************************

	//******************************************************************Spot light sources description*****************************************************************
	uint nSpotLights;	//factual number of spot lights that affect the object
	vec3 v3SpotLightLocations[max_spot_lights];	//locations of spot light sources
	vec3 v3SpotLightDirections[max_spot_lights];	//directions of spot light sources
	vec3 v3SpotLightAttenuationFactors[max_spot_lights];	//polynomial attenuation factors of spot light sources
	float fSpotLightExponents[max_spot_lights];	//attenuation exponent of each spot light that affects the object
	vec3 v3SpotLightIntensities[max_spot_lights];	//intensity of each spot light that affects the object
	float fSpotLightHazeIntensityFactors[max_spot_lights];	//intensity factors applied when computing atmospheric haze for spot lights
	float fSpotLightHazeLocationDecayFactors[max_spot_lights];	//haze location decay factors of the spot lights
	float fSpotLightHazeDirectionDecayFactors[max_spot_lights];	//haze direction decay factors of the spot lights
	//*****************************************************************************************************************************************************************
}light_buffer;