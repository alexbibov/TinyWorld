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
uniform uint uiNumSamples;		//number of samples along each ray
uniform float fSolidAngle;		//small solid angle within which we approximate multiple forward scattering integral

//Equals 'true' if shading should take into account RGB-channel of the 3D-texture 
uniform bool bUseRGBChannel;

//Equals 'true' if colormap is in use
uniform bool bUseColormap;

//Colour of the medium represented by 3D-texture. This parameter is active only
//when RGB-channel of the texture is not taken into account by shading algorithm
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


out vec4 v4FragmentColor;	//output colour of the fragment


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


//Converts coordinates represented in camera space to dimensional object-space frame
vec3 fromCameraToDimObject(vec3 point)
{
	//Convert given camera-space point to the object space
	vec4 v4ObjectSpaceCoords = m4CameraToDimObj * vec4(point, 1.0f);
		
	return v4ObjectSpaceCoords.xyz / v4ObjectSpaceCoords.w;
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
	if(v2Intersection.x >= v2Intersection.y)
	{
		//If the current ray does not intersect the box, the output colour of the fragment should be black
		v4FragmentColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
		return;
	}
	else
	{	
		//Compute light scattering for each of the active lights
		vec3 v3PerceivedColor = vec3(0.0f);
		for(uint light = 0; light < uiNumLightSources; ++light)
		{
			vec3 v3ColormapModulation = vec3(1.0f);
			vec3 v3CurrentLightColor = vec3(0.0f);
			vec3 v3CurrentLightDirection = normalize(-v3LightSourceDirections[light]);
			
			//Compute "entry" point of the view ray
			vec3 v3Entry = v3ViewRayStart + v2Intersection.x * v3ViewRay;
			
			//Compute "exit" point of the view ray
			vec3 v3Exit = v3ViewRayStart + v2Intersection.y * v3ViewRay;
			
			//Compute a single "step" ray (note: the ray is traversed backwards from the "exit" point
			//toward the observer)
			vec3 v3Step1 = (v3Entry - v3Exit) / uiNumSamples;
			float fStep1Length = length(v3Step1);
			
			//Cosine of angle between the currently processed light and direction to the viewer
			
			float fCos = dot(v3CurrentLightDirection, -v3ViewRay);
			
			//Start traversing the view ray
			for(uint sstep1 = 0; sstep1 < uiNumSamples; ++sstep1)
			{
				//Compute current point position
				vec3 v3CurrentPoint = v3Exit + (0.5f + sstep1) * v3Step1;
				
				//Calculate amount of light incident on the currently 
				//processed point from the currently processed light source
				vec3 I = v3LightSourceIntensities[light];
				
				//Get intersection between the ray starting at the current point and
				//oriented toward the light source
				vec2 v2LightIntersection = 
					getRayBoxIntersection(v3CurrentPoint, v3CurrentLightDirection, v3BoxSize * v3Scale);
				v2LightIntersection.x = 0;
								
				//Locate the point where the light ray enters the box
				vec3 v3LightEntry = v3CurrentPoint + v2LightIntersection.y * v3CurrentLightDirection;
				
				//Finally, compute the light incident on the currently processed point
				
				vec3 v3Step2 = (v3CurrentPoint - v3LightEntry) / uiNumSamples;
				float fStep2Length = length(v3Step2);
				for(uint sstep2 = 0; sstep2 < uiNumSamples; ++sstep2)
				{
					vec3 v3Sample = v3LightEntry + (0.5f + sstep2) * v3Step2;
						
					vec4 v4TextureSample = texture(s3dMediumSampler, fromDimObjectToSTP(v3Sample));
					
						
					float fAlbedo = 1.0f;
					vec3 v3G = Phase(-1) * fSolidAngle / (4.0f * pi) * I * v3MediumColor;
					if(bUseRGBChannel)
					{
						fAlbedo = length(v3MediumColor * v4TextureSample.rgb) / sqrt(3);
						
						if(bUseColormap) 
							v3ColormapModulation = texture(s1dColormapSampler, clamp(v4TextureSample.a, 0, 1)).rgb;
						
						v3G = fAlbedo * v4TextureSample.rgb * v3ColormapModulation * v3G;
					}
					else
					{
						fAlbedo = length(v3MediumColor) / sqrt(3);
						
						if(bUseColormap) v3ColormapModulation = texture(s1dColormapSampler, clamp(v4TextureSample.r, 0, 1)).rgb;
						
						v3G = fAlbedo * v3ColormapModulation * v3G;
					}
					
					float tau = fStep2Length;
					if(bUseRGBChannel)
						tau = tau * v4TextureSample.a;
					else
						tau = tau * v4TextureSample.r;
					
					v3G = v3G * tau;
					I = I * exp(-tau) + v3G;
				}
				
				vec4 v4TextureSample = texture(s3dMediumSampler, fromDimObjectToSTP(v3CurrentPoint));
				
				float fAlbedo = 1.0f;
				vec3 v3G = Phase(fCos) * fSolidAngle / (4.0f * pi) * I * v3MediumColor;
				if(bUseRGBChannel)
				{
					fAlbedo = length(v3MediumColor * v4TextureSample.rgb) / sqrt(3);
					
					if(bUseColormap) v3ColormapModulation = 
							texture(s1dColormapSampler, clamp(v4TextureSample.a, 0, 1)).rgb;
					
					v3G = fAlbedo * v4TextureSample.rgb * v3ColormapModulation * v3G;
				}
				else
				{
					fAlbedo = length(v3MediumColor) / sqrt(3);
					
					if(bUseColormap) v3ColormapModulation = 
							texture(s1dColormapSampler, clamp(v4TextureSample.r, 0, 1)).rgb;
					
					v3G = fAlbedo * v3ColormapModulation * v3G;
				}
					
				float tau = fStep1Length;
				if(bUseRGBChannel)
					tau = tau * v4TextureSample.a;
				else
					tau = tau * v4TextureSample.r;
					
				v3G = v3G * tau;
				v3CurrentLightColor = v3CurrentLightColor * exp(-tau) + v3G;
			}
		
			v3PerceivedColor = v3PerceivedColor + v3CurrentLightColor;
		}
		
		
		v4FragmentColor = vec4(v3PerceivedColor, length(v3PerceivedColor) / sqrt(3));
	}
}