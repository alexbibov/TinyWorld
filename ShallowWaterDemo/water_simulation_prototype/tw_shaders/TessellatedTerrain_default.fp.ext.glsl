
uniform sampler2D s2dFractalNoise;	//fractal noise sampler object
uniform vec2 v2FractalNoiseScaling;	//scaling factors applied to the fractal noise


subroutine(ProceduralNormalMapSampleRetriever) vec3 TessTerrainNormalMapFunc(vec2 v2TexCoords, float fArrayLayer)
{
	vec3 v3Normal = vec3(0);
	float fNoise = texture(s2dFractalNoise, v2TexCoords * v2FractalNoiseScaling + 0.1f).r;
	if(bSupportsArrayNormalMaps)
	{
		for(float q = 1; q <= 3; ++q)
		{
			float fPeriodMultiplier = sqrt(q);
			vec3 v3Aux1 = texture(s2daNormalArrayMap, vec3(v2TexCoords/fPeriodMultiplier, floor(fArrayLayer))).rgb * 2.0f - 1.0f;
			vec3 v3Aux2 = texture(s2daNormalArrayMap, vec3(v2TexCoords/fPeriodMultiplier, ceil(fArrayLayer))).rgb * 2.0f - 1.0f;
			v3Normal = mix(v3Aux1, v3Aux2, fract(fArrayLayer));
		}
	}
	else
	{
		for(float q = 1; q <= 3; ++q)
		{
			v3Normal = texture(s2dNormalMap, v2TexCoords/sqrt(q)).rgb * 2.0f - 1.0f;
		}
	}

	return normalize(v3Normal + 2*fNoise-1);
}


subroutine(ProceduralSpecularMapSampleRetriever) vec3 TessTerrainSpecularMapFunc(vec2 v2TexCoords, float fArrayLayer)
{
	vec3 v3SpecularModulation = vec3(0);
	float fNoise = texture(s2dFractalNoise, v2TexCoords * v2FractalNoiseScaling + 0.2f).r;
	if(bSupportsArraySpecularMaps)
	{
		for(float q = 1; q <= 3; ++q)
		{
			float fPeriodMultiplier = sqrt(q);
			vec3 v3Aux1 = texture(s2daSpecularArrayMap, vec3(v2TexCoords/fPeriodMultiplier, floor(fArrayLayer))).rgb;
			vec3 v3Aux2 = texture(s2daSpecularArrayMap, vec3(v2TexCoords/fPeriodMultiplier, ceil(fArrayLayer))).rgb;
			v3SpecularModulation = (v3SpecularModulation*(q-1) + mix(v3Aux1, v3Aux2, fract(fArrayLayer))) / q;
		}
	}
	else
	{
		for(float q = 1; q <= 3; ++q)
		{
			v3SpecularModulation = (v3SpecularModulation*(q-1) + texture(s2dSpecularMap, v2TexCoords/sqrt(q)).rgb) / q;
		}
	}

	return normalize(v3SpecularModulation + fNoise)*length(v3SpecularModulation);
}


subroutine(ProceduralEmissionMapSampleRetriever) vec3 TessTerrainEmissionMapFunc(vec2 v2TexCoords, float fArrayLayer)
{
	vec3 v3EmissionModulation = vec3(0);
	float fNoise = texture(s2dFractalNoise, v2TexCoords * v2FractalNoiseScaling + 0.3f).r;
	if(bSupportsArrayEmissionMaps)
	{
		for(float q = 1; q <= 3; ++q)
		{
			float fPeriodMultiplier = sqrt(q);
			vec3 v3Aux1 = texture(s2daEmissionArrayMap, vec3(v2TexCoords/fPeriodMultiplier, floor(fArrayLayer))).rgb;
			vec3 v3Aux2 = texture(s2daEmissionArrayMap, vec3(v2TexCoords/fPeriodMultiplier, ceil(fArrayLayer))).rgb;
			v3EmissionModulation = (v3EmissionModulation*(q-1) + mix(v3Aux1, v3Aux2, fract(fArrayLayer))) / q;
		}
	}
	else
	{
		for(float q = 1; q <= 3; ++q)
		{
			v3EmissionModulation = (v3EmissionModulation*(q-1) + texture(s2dEmissionMap, v2TexCoords/sqrt(q)).rgb) / q;
		}
	}

	return normalize(v3EmissionModulation + fNoise)*length(v3EmissionModulation);
}
