//Implements fast evaluation a cubic polynomial with given coefficients, where the coefficients are sorted
//starting from the highest polynomial degree (i.e. the polynomial reads as a0*t^3+a1*t^2+a2*t+a3)
float evaluatePolynomial(vec4 a, float t)
{
	float v = a[1] + t*a[0];
	v = a[2] + t*v;
	v = a[3] + t*v;
	return v;
}

//Implements fast evaluation of a quadratic polynomial with given coefficients, where the coefficients are sorted
//starting from the highest degree of the polynomial (i.e. the polynomial reads as a0*t^2+a1*t+a0)
float evaluatePolynomial(vec3 a, float t)
{
	float v = a[1] + t*a[0];
	v = a[2] + t*v;
	return v;
}

//Implements global B-spline weighting function
float BSplineWeight(int i, float u)
{
	float fa3 = u - (i-2);
	float fa2 = u - (i-1);
	float fa1 = u - i;
	float fa0 = u - (i+1);

	vec4 v4B;
	v4B[3] = evaluatePolynomial(vec4(1, 0, 0, 0), fa3);
	v4B[2] = evaluatePolynomial(vec4(-3, 3, 3, 1), fa2);
	v4B[1] = evaluatePolynomial(vec4(3, -6, 0, 4), fa1);
	v4B[0] = evaluatePolynomial(vec4(-1, 3, -3, 1), fa0);
	v4B /= 6.0f;

	vec4 v4c;
	v4c[3] = float(u>=i-2 && u<i-1);
	v4c[2] = float(u>=i-1 && u<i);
	v4c[1] = float(u>=i && u<i+1);
	v4c[0] = float(u>=i+1 && u<i+2);

	return dot(v4c, v4B);
}

//Implements derivative of the global B-spline weighting function
float DBSplineWeightDU(int i, float u)
{
	float fa3 = u - (i-2.0f);
	float fa2 = u - (i-1.0f);
	float fa1 = u - i;
	float fa0 = u - (i+1.0f);

	vec4 v4B;
	v4B[3] = evaluatePolynomial(vec3(1, 0, 0), fa3);
	v4B[2] = evaluatePolynomial(vec3(-3, 2, 1), fa2);
	v4B[1] = evaluatePolynomial(vec3(3, -4, 0), fa1);
	v4B[0] = evaluatePolynomial(vec3(-1, 2, -1), fa0);
	v4B /= 2.0f;

	vec4 v4c;
	v4c[3] = float(u>=i-2 && u<i-1);
	v4c[2] = float(u>=i-1 && u<i);
	v4c[1] = float(u>=i && u<i+1);
	v4c[0] = float(u>=i+1 && u<i+2);

	return dot(v4c, v4B);
}

//Approximates water surface by a uniform global B-spline
vec3 computeWaterLevel(sampler2D s2dWaterHeightMap, vec2 v2TexCoords, int iMipmapLevel, vec2 v2BilletScale, out vec3 v3Normal)
{
	vec2 v2TexSize = textureSize(s2dWaterHeightMap, iMipmapLevel) - 1;
	vec2 v2ScaledTexCoords = v2TexSize*v2TexCoords;

	ivec2 iv2Imin = ivec2(ceil(v2ScaledTexCoords)-2);
	ivec2 iv2Imax = ivec2(floor(v2ScaledTexCoords)+2);

	vec3 v3AuxControlPoints1[5];
	vec3 v3AuxControlPoints2[5];
	for(int i = iv2Imin.y; i <= iv2Imax.y; ++i)
	{
		v3AuxControlPoints1[i-iv2Imin.y] = vec3(0);
		v3AuxControlPoints2[i-iv2Imin.y] = vec3(0);
		for(int j = iv2Imin.x; j <= iv2Imax.x; ++j)
		{
			vec3 v3P = vec3(j/v2TexSize.x-0.5f, 
				texelFetch(s2dWaterHeightMap, ivec2(j,i), 0).r, 
				0.5f-i/v2TexSize.y);
			v3AuxControlPoints1[i-iv2Imin.y] += BSplineWeight(j, v2ScaledTexCoords.x)*v3P;
			v3AuxControlPoints2[i-iv2Imin.y] += DBSplineWeightDU(j, v2ScaledTexCoords.x)*v3P;
		}
	}


	vec3 v3S = vec3(0);
	vec3 v3DSDU = vec3(0);
	vec3 v3DSDV = vec3(0);
	for(int i = iv2Imin.y; i <= iv2Imax.y; ++i)
	{
		v3S += BSplineWeight(i, v2ScaledTexCoords.y)*v3AuxControlPoints1[i-iv2Imin.y];
		v3DSDU += BSplineWeight(i, v2ScaledTexCoords.y)*v3AuxControlPoints2[i-iv2Imin.y];
		v3DSDV += DBSplineWeightDU(i, v2ScaledTexCoords.y)*v3AuxControlPoints1[i-iv2Imin.y];
	}
	v3DSDU *= vec3(v2BilletScale.x, 1, v2BilletScale.y);
	v3DSDV *= vec3(v2BilletScale.x, 1, v2BilletScale.y);

	v3Normal = normalize(cross(v3DSDU, v3DSDV));
	v3Normal /= v3Normal.y;

	return v3S;
}




//Approximates water surface based on B-splines
vec3 computeWaterLevel(sampler2D s2dWaterHeightMap, vec2 v2TexCoords, int iMipmapLevel)
{
	vec2 v2TexSize = textureSize(s2dWaterHeightMap, iMipmapLevel) - 1;
	vec2 v2ScaledTexCoords = v2TexSize*v2TexCoords;

	ivec2 iv2Imin = ivec2(ceil(v2ScaledTexCoords)-2);
	ivec2 iv2Imax = ivec2(floor(v2ScaledTexCoords)+2);

	vec3 v3AuxControlPoints[5];
	for(int i = iv2Imin.y; i <= iv2Imax.y; ++i)
	{
		v3AuxControlPoints[i-iv2Imin.y] = vec3(0);
		for(int j = iv2Imin.x; j <= iv2Imax.x; ++j)
		{
			vec3 v3P = vec3(j/v2TexSize.x-0.5f, 
				texelFetch(s2dWaterHeightMap, ivec2(j,i), 0).r, 
				0.5f-i/v2TexSize.y);
			v3AuxControlPoints[i-iv2Imin.y] += BSplineWeight(j, v2ScaledTexCoords.x)*v3P;
		}
	}


	vec3 v3S = vec3(0);
	for(int i = iv2Imin.y; i <= iv2Imax.y; ++i)
		v3S += BSplineWeight(i, v2ScaledTexCoords.y)*v3AuxControlPoints[i-iv2Imin.y];

	return v3S;
}




//Computes perturbation of the water surface based on the water depth at the given map coordinates.
//Input arguments:
//v2MapCoordinates	coordinates of the point on the map for which to compute the perturbation
//fTilingFactor	tiling factor applied to the map coordinates
//s2dTopographyMap	sampler used to retrieve topography surface elevation values
//fWaterLevel	water level at the given point on the map
//fMaxWaveHeightAsElevationFraction	fraction of the water depth at the current point, which determines the highest allowed perturbation.
//fMaxWaveHeight	maximal allowed absolute value of the perturbation
//s2dFFTRipples		sampler used to retrieve values from the deep water wave surface map
//s2dFFTDisplacementMap	sampler used to retrieve displacement values used to produce the choppy waves
//v2XZScale	X- and Z- scaling factors of the water surface
//s2dFFTRipplesNormalMap	sampler used to retrieve values from the normal map corresponding to the deep water wave surface
//v2NormalDisplacement	output parameter receiving normal displacement vector corresponding to the computed perturbation
//fWaterElevation	water elevation at the given map coordinates (including the perturbation)
//
//Return value:	perturbation of the water surface at the given map coordinates



float computeDeepWaterWavePerturbation(vec2 v2MapCoordinates, float fTilingFactor, sampler2D s2dTopographyMap, float fWaterLevel, float fMaxWaveHeightAsElevationFraction, 
	float fMaxWaveHeight, sampler2D s2dFFTRipples, sampler2D s2dFFTDisplacementMap, vec2 v2XZScale, 
	sampler2D s2dFFTRipplesNormalMap, out float fWaterElevation, out vec2 v2NormalDisplacement, out vec2 v2MapCoordinatesDisplacement)
{
	fWaterElevation = max(fWaterLevel - textureLod(s2dTopographyMap, v2MapCoordinates, 0).r, 0);
	float fCorrectionFactor = min(fMaxWaveHeight, fMaxWaveHeightAsElevationFraction*fWaterElevation);

	vec2 v2TiledMapCoordinates = v2MapCoordinates * fTilingFactor;
	v2MapCoordinatesDisplacement = textureLod(s2dFFTDisplacementMap, v2TiledMapCoordinates, 0).rb / TW__KPWATER_FFT_SIZE__ / TW__KPWATER_FFT_SIZE__ / v2XZScale;
	v2MapCoordinatesDisplacement *= fCorrectionFactor;

	float fPerturbation = textureLod(s2dFFTRipples, v2TiledMapCoordinates, 0).r / TW__KPWATER_FFT_SIZE__ / TW__KPWATER_FFT_SIZE__; 
	fPerturbation *= fCorrectionFactor;
	

	v2NormalDisplacement = textureLod(s2dFFTRipplesNormalMap, v2TiledMapCoordinates, 0).rb / TW__KPWATER_FFT_SIZE__ / TW__KPWATER_FFT_SIZE__ * fCorrectionFactor;


	return max(fPerturbation, -fWaterElevation);
}



float computeDeepWaterWavePerturbation(vec2 v2MapCoordinates, float fTilingFactor, sampler2D s2dTopographyMap, float fWaterLevel, float fMaxWaveHeightAsElevationFraction, 
	float fMaxWaveHeight, sampler2D s2dFFTRipples, out float fWaterElevation)
{
	vec2 v2TiledMapCoordinates = v2MapCoordinates * fTilingFactor;

	float fPerturbation = textureLod(s2dFFTRipples, v2TiledMapCoordinates, 0).r / TW__KPWATER_FFT_SIZE__ / TW__KPWATER_FFT_SIZE__; 

	fWaterElevation = max(fWaterLevel - textureLod(s2dTopographyMap, v2MapCoordinates, 0).r, 0);
	float fCorrectionFactor = min(fMaxWaveHeight, fMaxWaveHeightAsElevationFraction*fWaterElevation);
	fPerturbation *= fCorrectionFactor;

	return max(fPerturbation, -fWaterElevation);
}