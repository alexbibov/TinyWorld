//Declaration of uniform buffer containing description of atmospheric fog


layout(std140) uniform FogBuffer
{
	//**********************************Parameters related to the atmospheric fog**************************************
	vec3 v3AtmosphericFogColor;		//color of the atmospheric fog
	float fAtmosphericFogGlobalDensity;	//global density of the atmospheric fog
	vec3 v3SunDirection;	//direction to the sun
	float fAtmosphericFogHeightFallOff;	//height fall-off coefficient of the atmospheric fog
	vec3 v3MoonDirection;	//direction to the moon
	float fMiePhaseFunctionParameter;	//parameter of the Mie phase function
	float light_haze_attenuation_factor;	//global light haze attenuation factor
	//*****************************************************************************************************************
}fog_buffer;