#version 430 core

//Implements atmospheric scattering for object Skydome. This is a pretty rough approximation,
//which assumes that camera's height is negligible compared to the atmosphere thickness (i.e camera is located on earth),
//and camera position is fixed to (0, 0, R), where R is radius of the planet and coordinate system is aligned to the centre of the planet.
//The latter assumption is possible when camera movements are infinitesimals compared to the radius of the planet

#define pi 3.1415926535897932384626433832795f

in vec4 v4VertexPosition;	//position of the currently processed vertex
in vec2 v2TexCoords;		//texture coordinates corresponding to the currently processed vertex

uniform mat4 m4MVP;	//Standard Model-View-Projection transform
uniform float fSkydomeRadius;	//Radius of the skydome
uniform float fPlanetRadius;		//Non-dimensional radius of the planet
uniform float fLengthScale;			//Main length scale of the atmospheric layer
uniform vec3 v3CameraPosition;		//Position of camera in non-dimensional coordinates
uniform vec3 v3RayleighCoefficient;	//Rayleigh coefficient corresponding to the wavelengths of red, green and blue channels
uniform float fMieCoefficient;	//Mie scattering coefficient
uniform float fH0;				//Non-dimensional height at which atmosphere has its average density

uniform vec3 v3SunLightDirection;	//Direction to the sun light
uniform vec3 v3SunLightIntensity;	//Intensity of the light source
uniform vec3 v3MoonLightDirection;	//Direction to the moon light
uniform vec3 v3MoonLightIntensity;	//Intensity of the moon light

uniform writeonly image2DArray in_scattering_sun;	//Texture containing in-scattering values contributed by the sun light
uniform writeonly image2DArray in_scattering_moon;	//Texture containing in-scattering values contributed by the moon light


//Common constants
const int nIntegrationSamples = 5;	//Number of samples approximating in-scattering integral equation


//Shader output interface structure
out VS_DATA
{
	vec3 v3SunScatteredColor1;	//Attenuated colour of the sun with Rayleigh scattering
	vec3 v3SunScatteredColor2;	//Attenuated colour of the sun with Mie scattering
	vec3 v3MoonScatteredColor1;	//Attenuated colour of the moon with Rayleigh scattering
	vec3 v3MoonScatteredColor2;	//Attenuated colour of the moon with Mie scattering
	vec3 v3CameraDirection;		//Direction from the current vertex to camera
}vs_out;



//Scale function approximating optical depth as defined by Sean O'Neil (see GPU Gems 2, ch. 16 for details)
//float scale(float fCos)
//{
//	float x = 1.0f - fCos;
//	return fH0 * exp(-0.00287f + x * (0.459f + x * (3.83f + x * (-6.80f + x * 5.25f))));
//}


//Scale function approximating optical depth as defined by Sean O'Neil (see GPU Gems 2, ch. 16 for details)
float scale(float fCos)
{
	float x = 0.5f - 0.5f*fCos;
	return fH0*exp(-0.0023f + x * (0.9920f + x * (13.8833f + x * (-48.3980f + x * 75.4683))));
}


void main()
{
	//Currently processed vertex represented in the scattering space
	vec3 v3Pos = v4VertexPosition.xyz + vec3(0.0f, fPlanetRadius, 0.0f);
	
	//Get ray from camera to the currently processed vertex of the skydome
	vec3 v3Ray = v3Pos - v3CameraPosition;	//Ray cast from camera toward the current vertex of the skydome
	if(v3Ray.y < 0) v3Ray.y = -v3Ray.y + 1e-5f;
	
	//Compute optical depth of the ray cast from camera toward the current vertex of the skydome
	float fFar = length(v3Ray);		//compute distance between camera and the current vertex
	v3Ray /= fFar;	//normalize direction from camera towards the current vertex
	float fCameraHeight = length(v3CameraPosition);
	float fStartDepth = exp((fPlanetRadius - fCameraHeight) * fLengthScale / fH0);
	float fStartAngle = dot(v3Ray, v3CameraPosition) / fCameraHeight;
	float fStartOffset = fStartDepth * scale(fStartAngle) ;	//scaled optical depth
	
	
	//Initialize loop that computes in-scattering equation integral
	float fSampleLength = fFar / nIntegrationSamples;	//length of a single integration step
	float fScaledSampleLength = fSampleLength * fLengthScale;	//step length scaled by the main length scaling factor
	vec3 v3SampleRay = v3Ray * fSampleLength;	//ray connecting two consequent samples
	vec3 v3Sample = v3CameraPosition + 0.5f * v3SampleRay;	//sample point
	vec3 v3SunScatteredColor = vec3(0.0f);	//scattered colour of the sun light
	vec3 v3MoonScatteredColor = vec3(0.0f);	//scattered colour of the moon light
	
	for(int i = 0; i < nIntegrationSamples; ++i)
	{
		float fSampleHeight = length(v3Sample);	//height of the current sample in the atmosphere
		float fSampleUnscaledDepth = exp((fPlanetRadius - fSampleHeight) * fLengthScale / fH0);
		float fSampleAngle = dot(v3Sample, v3Ray) / fSampleHeight;		
		float fSunLightAngle = dot(v3Sample, v3SunLightDirection) / fSampleHeight;
		float fMoonLightAngle = dot(v3Sample, v3MoonLightDirection) / fSampleHeight;
		
		//Compute out-scattering integral corresponding to the current sample
		float fSunAttenuation = 
			4.0f * pi * (fStartOffset + fSampleUnscaledDepth * (scale(fSunLightAngle) - scale(fSampleAngle)));
		
		float fMoonAttenuation = 
			4.0f * pi * (fStartOffset + fSampleUnscaledDepth * (scale(fMoonLightAngle) - scale(fSampleAngle)));
		
		vec3 v3SunOutScatter = vec3((v3RayleighCoefficient + fMieCoefficient) * fSunAttenuation);	
		vec3 v3MoonOutScatter = vec3((v3RayleighCoefficient + fMieCoefficient) * fMoonAttenuation);
		
		v3SunScatteredColor += fSampleUnscaledDepth * exp(-v3SunOutScatter) * fScaledSampleLength;
		v3MoonScatteredColor += fSampleUnscaledDepth * exp(-v3MoonOutScatter) * fScaledSampleLength;
		v3Sample += v3SampleRay;
	}
	
	
	//Apply Rayleigh and Mie coefficients to the attenuated light value
	vs_out.v3SunScatteredColor1 = (v3SunLightIntensity * v3RayleighCoefficient) * v3SunScatteredColor;
	vs_out.v3SunScatteredColor2 = (v3SunLightIntensity * fMieCoefficient) * v3SunScatteredColor;
	
	vec3 v3EarthShine = vec3(0.65f, 0.7f, 1.0f);
	vs_out.v3MoonScatteredColor1 = (v3MoonLightIntensity * v3RayleighCoefficient) * v3MoonScatteredColor * v3EarthShine;
	vs_out.v3MoonScatteredColor2 = (v3MoonLightIntensity * fMieCoefficient) * v3MoonScatteredColor * v3EarthShine;


	if(v2TexCoords.s >= 0 && v2TexCoords.t >= 0)
	{
		imageStore(in_scattering_sun, ivec3(v2TexCoords,0), vec4(vs_out.v3SunScatteredColor1, 0));
		imageStore(in_scattering_sun, ivec3(v2TexCoords,1), vec4(vs_out.v3SunScatteredColor2, 0));

		imageStore(in_scattering_moon, ivec3(v2TexCoords,0), vec4(vs_out.v3MoonScatteredColor1, 0));
		imageStore(in_scattering_moon, ivec3(v2TexCoords,1), vec4(vs_out.v3MoonScatteredColor2, 0));
	}
	
	
	//Transform vertices
	vec4 v4Pos = v4VertexPosition;
	v4Pos.xyz *= fSkydomeRadius;
	gl_Position = m4MVP * v4Pos;
	
	//Set direction from the current vertex to camera to be forwarded to the fragment shader stage
	vs_out.v3CameraDirection = v3CameraPosition - v3Pos;
}