#version 430 core

#define pi 3.1415926535897932384626433832795f

const uint MAX_DIRECTIONAL_LIGHTS = 4;		//maximal number of directional light sources
const float INFINITY = 1.0f / 0.0f;		//constant containing "the value of infinity"

uniform vec4 v4Viewport;		//client area defined by tuple (x, y, width, height) passed to glViewport
uniform vec2 v2DepthRange;		//pair of values (near, far) passed to glDepthRange
uniform vec3 v3Scale;			//scaling factors of the object-space
uniform mat4 m4InvProj;			//inverse projection matrix
uniform mat4 m4CameraToDimObj;	//homogeneous transform from the camera space to dimensional object space
uniform vec3 v3BoxSize;			//dimensions of the transparent box stored as (width, height, depth)
uniform uint uiNumPrimarySamples;		//number of samples along each ray
uniform uint uiNumSecondarySamples;	//number of samples along each scattering ray
uniform float fSolidAngle;		//small solid angle within which we approximate multiple forward scattering integral

//Equals 'true' if shading should take into account RGB-channel of the 3D-texture 
uniform bool bUseRGBChannel;

//Equals 'true' if colormap is in use
uniform bool bUseColormap;

//Colour of the medium represented by 3D-texture.
uniform vec3 v3MediumColor;	

//sampler of the 3D-texture representing medium contained in the transparent box
uniform sampler3D s3dMediumSampler;	

//sampler of the colormap look-up texture
uniform sampler1D s1dColormapSampler;

//Directions and intensities of the light sources taken into account by shading
//NOTE: light directions must be represented in TEXTURE space!
uniform vec3 v3LightSourceDirections[MAX_DIRECTIONAL_LIGHTS];
uniform vec3 v3LightSourceIntensities[MAX_DIRECTIONAL_LIGHTS];

//Number of light sources taken into account by shading
uniform uint uiNumLightSources;

//Define subroutine uniform type responsible for ray traversing optical model
subroutine vec4 OpticalModel(vec3, vec3, vec3);

//Subroutine uniform responsible for storage of currently selected optical model
subroutine uniform OpticalModel func_optical_model;

layout(location = 0)out vec4 v4FragmentColor;	//output colour of the fragment
layout(location = 1)out vec4 v4BloomColor;	//output colour for bloom effect



//Casts ray represented in camera frame towards the currently processed fragment
//NOTE: the returned ray is NOT normalized
vec3 castRay(vec4 frag_coord)
{
	//Convert fragment coordinates from window-space to normalized device coordinates
	vec4 v4NDC;
	v4NDC.xy = 2.0f * (frag_coord.xy - v4Viewport.xy) / v4Viewport.zw - 1.0f;
	v4NDC.z = (2.0f * frag_coord.z - v2DepthRange.y - v2DepthRange.x) / (v2DepthRange.y - v2DepthRange.x);
	v4NDC.w = 1.0f;
	
	//Convert coordinates from NDC-space to clip space
	vec4 v4ClipCoords = v4NDC / frag_coord.w;
	
	//Convert coordinates from clip space to eye space
	vec4 v4EyeCoords = m4InvProj * v4ClipCoords;
	
	return v4EyeCoords.xyz / v4EyeCoords.w;
}


//Computes intersection of the given ray and AABB residing in dimensional object space. The return value is
//a pair of "entry" and "exit" points represented by parametric values packed into a 2D-vector.
//For instance, if V is returned value, then ray_start + V.x * ray_direction is the 
//point where the given ray "enters" the transparent box and ray_start + V.y * ray_direction is
//the point where it leaves the box. In addition if V.x >= V.y then the ray does not intersect the box.
vec2 getRayBoxIntersection(vec3 ray_start, vec3 ray_direction, vec3 aabb)
{
	vec3 p_t1 = (-aabb / 2.0f - ray_start) / ray_direction;
	vec3 p_t2 = (aabb / 2.0f - ray_start) / ray_direction;
	
	//Filter out possible NaN's
	p_t1 = max(p_t1, -INFINITY);
	p_t2 = min(p_t2, INFINITY);
	
	vec3 pp_t1 = min(p_t1, p_t2);
	vec3 pp_t2 = max(p_t1, p_t2);
	
	float t1 = max(max(max(-INFINITY, pp_t1.x), pp_t1.y), pp_t1.z);
	float t2 = min(min(min( pp_t2.x, INFINITY), pp_t2.y), pp_t2.z);
	
	return (vec2(t1, t2));
}


//Converts coordinates represented in camera space to dimensional object-space frame
vec3 fromCameraToDimObject(vec3 point)
{
	//Convert given camera-space point to the object space
	vec4 v4ObjectSpaceCoords = m4CameraToDimObj * vec4(point, 1.0f);
		
	return v4ObjectSpaceCoords.xyz / v4ObjectSpaceCoords.w;
}


//Converts coordinates represented in dimensional object space into STP texture space
vec3 fromDimObjectToSTP(vec3 point)
{
	return (point / (v3Scale * v3BoxSize) + 0.5f);
}


//Implements Rayleigh scattering phase function for unpolarized incident light sources
float Phase(float c)
{
	return 3.0f / 4.0f * (1.0f + c*c);
}





subroutine(OpticalModel) vec4 GasScattering(vec3 ray_entry, vec3 ray_exit, vec3 view_ray_normalized)
{
	//Compute light scattering for each of the active lights
	vec3 v3PerceivedColor = vec3(0.0f);
	//Compute a single "step" ray (note: the ray is traversed backwards from the "exit" point toward the observer)
	vec3 v3Step1 = (ray_entry - ray_exit) / uiNumPrimarySamples;
	float fStep1Length = length(v3Step1);
	for(uint light = 0; light < uiNumLightSources; ++light)
	{
		vec3 v3CurrentLightColor = vec3(0.0f);
		vec3 v3CurrentLightDirection = normalize(-v3LightSourceDirections[light]);	
			
		//Cosine of the angle between the currently processed light and direction to the viewer
		float fCos = dot(v3CurrentLightDirection, -view_ray_normalized);
			
		//Start traversing the view ray
		for(uint sstep1 = 0; sstep1 < uiNumPrimarySamples; ++sstep1)
		{
			//Compute current point position
			vec3 v3CurrentPoint = ray_exit + (0.5f + sstep1) * v3Step1;
			
			//Calculate amount of light incident on the currently 
			//processed point from the currently processed light source
			vec3 I = v3LightSourceIntensities[light];
			
			//Get intersection between the ray starting at the current point and
			//oriented toward the light source
			vec2 v2LightIntersection = 
				getRayBoxIntersection(v3CurrentPoint, v3CurrentLightDirection, v3BoxSize * v3Scale);
							
			//Locate the point where the light ray enters the box
			vec3 v3LightEntry = v3CurrentPoint + v2LightIntersection.y * v3CurrentLightDirection;
			
			
			//Finally, compute the light incident on the currently processed point
			vec3 v3Step2 = (v3CurrentPoint - v3LightEntry) / uiNumSecondarySamples;
			float fStep2Length = length(v3Step2);
			if(v2LightIntersection.y > 0)
			{
				for(uint sstep2 = 0; sstep2 < uiNumSecondarySamples; ++sstep2)
				{
					vec3 v3Sample = v3LightEntry + (0.5f + sstep2) * v3Step2;
					vec4 v4TextureSample = texture(s3dMediumSampler, fromDimObjectToSTP(v3Sample));
					
					float fCurrentLightAbsorption;
					vec3 v3G = Phase(-1) * fSolidAngle / (4.0f * pi) * I * v3MediumColor;
					vec3 v3Aux = v3MediumColor;
					if(bUseRGBChannel)
					{
						fCurrentLightAbsorption = v4TextureSample.a;
						v3Aux = v3Aux * v4TextureSample.rgb;
						v3G = v3G * v4TextureSample.rgb;
					}
					else
					{
						fCurrentLightAbsorption = v4TextureSample.r;
					}
					
					vec3 v3ColormapModulation = vec3(1.0f);
					if(bUseColormap) v3ColormapModulation = texture(s1dColormapSampler, clamp(fCurrentLightAbsorption, 0, 1)).rgb;
					float fAlbedo = length(v3Aux * v3ColormapModulation) / sqrt(3);
					v3G = v3G * fAlbedo * v3ColormapModulation;
				
					float tau = fStep2Length;
					tau = tau * fCurrentLightAbsorption;
				
					v3G = v3G * tau;
					I = I * exp(-tau) + v3G;
				}
			}
			
			
			vec4 v4TextureSample = texture(s3dMediumSampler, fromDimObjectToSTP(v3CurrentPoint));
			
			float fCurrentLightAbsorption;
			vec3 v3G = Phase(fCos) * fSolidAngle / (4.0f * pi) * I * v3MediumColor;
			vec3 v3Aux = v3MediumColor;
			if(bUseRGBChannel)
			{
				fCurrentLightAbsorption = v4TextureSample.a;
				v3Aux = v3Aux * v4TextureSample.rgb;
				v3G = v3G * v4TextureSample.rgb;
			}
			else
			{
				fCurrentLightAbsorption = v4TextureSample.r;
			}
			
			vec3 v3ColormapModulation = vec3(1.0f);
			if (bUseColormap) v3ColormapModulation = texture(s1dColormapSampler, clamp(fCurrentLightAbsorption, 0, 1)).rgb;
			float fAlbedo = length(v3Aux * v3ColormapModulation) / sqrt(3);
			v3G = v3G * fAlbedo * v3ColormapModulation;
				
			float tau = fStep1Length;
			tau = tau * fCurrentLightAbsorption;
				
			v3G = v3G * tau;
			v3CurrentLightColor = v3CurrentLightColor * exp(-tau) + v3G;
		}
	
		v3PerceivedColor = v3PerceivedColor + v3CurrentLightColor;
	}
	
	return vec4(v3PerceivedColor, length(v3PerceivedColor) / length(v3Scale));
}




subroutine(OpticalModel) vec4 EmissionAbsorption(vec3 ray_entry, vec3 ray_exit, vec3 view_ray_normalized)
{
	vec3 v3PerceivedColor = vec3(0.0f);
	float fCurrentTransparency = 1.0f;
	
	vec3 v3Step1 = (ray_entry - ray_exit) / (uiNumPrimarySamples - 1.0f);
	float fSamplingRate = 1.0f / length(v3Step1);
	
	for(uint light = 0; light < uiNumLightSources; ++light)
	{
		vec3 v3CurrentLightPerceivedColor = vec3(0.0f);
		vec3 v3CurrentLightColor = v3LightSourceIntensities[light];
		vec3 v3CurrentLightDirection = normalize(-v3LightSourceDirections[light]);
	
		for(uint sstep1 = 0; sstep1 < uiNumPrimarySamples; ++sstep1)
		{
			vec3 v3Sample = ray_exit + sstep1 * v3Step1;
		
		
			//Retrieve amount of light arriving to the current ray sample
			vec2 v2LightRayIntersection = getRayBoxIntersection(v3Sample, v3CurrentLightDirection, v3BoxSize * v3Scale); 
			vec3 v3LightEntry = v3Sample + v2LightRayIntersection.y * v3CurrentLightDirection;
			vec3 v3ReceivedLight = v3CurrentLightColor;
			if (v2LightRayIntersection.y > 0)
			{
				vec3 v3AccumulatedColor = vec3(0.0f);
				vec3 v3Step2 = (v3Sample - v3LightEntry) / uiNumSecondarySamples;
				for(uint sstep2 = 0; sstep2 < uiNumSecondarySamples; ++sstep2)
				{
					vec3 v3LitPoint = v3LightEntry + (sstep2 + 0.5f) * v3Step2;
					vec4 v4TextureSample = texture(s3dMediumSampler, fromDimObjectToSTP(v3LitPoint));
					vec3 v3ColorSample = v3MediumColor;
					float fCurrentLightAbsorption;
					if (bUseRGBChannel)
					{
						fCurrentLightAbsorption = v4TextureSample.a;
						v3ColorSample = v3ColorSample * v4TextureSample.rgb;
					}
					else
					{
						fCurrentLightAbsorption = v4TextureSample.r;
					}
					if (bUseColormap) v3ColorSample = v3ColorSample * texture(s1dColormapSampler, clamp(fCurrentLightAbsorption, 0, 1)).rgb;
					v3AccumulatedColor = v3AccumulatedColor + v3ColorSample * (1.0f - fCurrentLightAbsorption);
				}
				v3ReceivedLight = v3ReceivedLight * v3AccumulatedColor / uiNumSecondarySamples ;
			}
			
		
			//Compute colour emitted by the current sample along the view ray
			vec4 v4TextureSample = texture(s3dMediumSampler, fromDimObjectToSTP(v3Sample));
			vec3 v3ColorSample = v3MediumColor * v3ReceivedLight;
			float fAlphaSample;
			if(bUseRGBChannel)
			{
				v3ColorSample = v3ColorSample * v4TextureSample.rgb;
				fAlphaSample = v4TextureSample.a;
			}
			else
			{
				fAlphaSample = v4TextureSample.r;
			}
			if (bUseColormap) v3ColorSample = v3ColorSample * texture(s1dColormapSampler, clamp(fAlphaSample, 0, 1)).rgb;
		
			fCurrentTransparency = fCurrentTransparency * (1.0f - fAlphaSample);
			v3CurrentLightPerceivedColor = (1.0f - fAlphaSample) * v3CurrentLightPerceivedColor + v3ColorSample;
		}
		
		v3PerceivedColor = v3PerceivedColor + v3CurrentLightPerceivedColor / uiNumPrimarySamples;
	}	
	
	ivec3 iv3TextureScale = textureSize(s3dMediumSampler, 0);
	float fReferenceSamplingRate = max(max(iv3TextureScale.x, iv3TextureScale.y), iv3TextureScale.z);
	return vec4(v3PerceivedColor, pow(1.0f - fCurrentTransparency, fReferenceSamplingRate / fSamplingRate));
}


vec4 computeBloomFragment(vec4 v4FragmentColor);


void main()
{
	//Cast ray towards the currently processed fragment
	vec3 v3ViewRayEnd = castRay(gl_FragCoord);
	
	//Convert the cast ray end-point from camera space to dimensional object space
	v3ViewRayEnd = fromCameraToDimObject(v3ViewRayEnd);
	
	//Get ray start point in dimensional object space
	vec3 v3ViewRayStart = fromCameraToDimObject(vec3(0.0f));
	
	//Define the view ray
	vec3 v3ViewRay = normalize(v3ViewRayEnd - v3ViewRayStart);
	
	//Compute intersection between the ray and the transparent box
	vec2 v2Intersection = getRayBoxIntersection(v3ViewRayStart, v3ViewRay, v3BoxSize * v3Scale);
	
	vec3 v3RayEntry = v3ViewRayStart + max(v2Intersection.x, 0.0f) * v3ViewRay;
	vec3 v3RayExit = v3ViewRayStart + v2Intersection.y * v3ViewRay;
	v4FragmentColor = func_optical_model(v3RayEntry, v3RayExit, v3ViewRay);
	v4FragmentColor.rgb = normalize(v4FragmentColor.rgb);
	v4FragmentColor.a = clamp(v4FragmentColor.a, 0, 1);
		
	v4BloomColor = computeBloomFragment(v4FragmentColor);
	//v4BloomColor.rgb = normalize(v4BloomColor.rgb);
	v4BloomColor.a = clamp(v4BloomColor.a, 0, 1);
}