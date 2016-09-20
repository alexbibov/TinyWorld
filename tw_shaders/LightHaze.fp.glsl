//Implements light haze effect applied in screen-space

#version 430 core

layout(pixel_center_integer) in vec4 gl_FragCoord;


uniform float fFocalDistance;	//distance between the observer and the near plane of the view frustum
uniform vec4 v4FocalPlane;		//contains left, right, bottom, and top boundaries in this order of the focal plane 
uniform vec4 v4Viewport;		//contains viewport coordinates x and y and viewport dimensions width and height in this order
uniform mat4 m4ViewTransform;	//view transform that converts world coordinates into view-space coordinates


#include "common/LightBuffer.include.glsl"	//Include definition of the light buffer
#include "common/FogBuffer.include.glsl"	//Include definition of the fog buffer



//source0 — color texture defined in screen space
//source1 — linear depth texture defined in screen space
vec4 do_filtering(in sampler2D source0, in sampler2D source1)
{
	//Begin by reconstructing position of the current fragment in the world space
	float L = v4FocalPlane.x, R = v4FocalPlane.y, B = v4FocalPlane.z, T = v4FocalPlane.w;
	vec2 v2FragFocalPlaneCoord = (gl_FragCoord.xy - v4Viewport.xy)/v4Viewport.zw * vec2(R-L, T-B) + vec2(L, B);

	mat3 m3ViewRotation = mat3(m4ViewTransform[0].xyz, m4ViewTransform[1].xyz, m4ViewTransform[2].xyz);
	vec3 v3ViewShift = m4ViewTransform[3].xyz;

	float fFragDepth = texelFetch(source1, ivec2(gl_FragCoord.xy), 0).r;	//linear depth of the current fragment
	vec3 v3FragCoord_VS = vec3(-v2FragFocalPlaneCoord * fFragDepth/fFocalDistance, fFragDepth);	//coordinates of the current fragment in view space
	vec3 v3FragCoord_WS = transpose(m3ViewRotation) * (v3FragCoord_VS - v3ViewShift);	//coordinates of the current fragment in the world space

	vec3 v3ViewerLocation_WS = -transpose(m3ViewRotation) * v3ViewShift;	//location of the viewer in the world space


	vec4 v4FragmentColor = texelFetch(source0, ivec2(gl_FragCoord.xy), 0);



	vec3 v3d = v3FragCoord_WS - v3ViewerLocation_WS;	//vector starting at the location of the observer and pointing towards position of the current fragment
	float fFragmentRemoteness = length(v3d);	//distance from observer to the current fragment



	//Compute haze effect for the point light sources
	for(int i = 0; i < light_buffer.nPointLights; ++i)
	{
		float fHazeAttenuationFactor = light_buffer.fPointLightHazeLocationDecayFactors[i] + fog_buffer.light_haze_attenuation_factor;	//haze location attenuation factor applied to the point light source
		vec3 v3l = light_buffer.v3PointLightLocations[i];	//location of the currently processed point light source

		//Compute "amount" of light along the cast ray
		vec3 v3LO = v3ViewerLocation_WS - v3l;
		float u = fHazeAttenuationFactor * fFragmentRemoteness * fFragmentRemoteness;
		float v = 2*fHazeAttenuationFactor * dot(v3d, v3LO);
		float w = fHazeAttenuationFactor * dot(v3LO, v3LO) + 1;
		float fAux = sqrt(4*u*w - v*v);

		float fI = 2*fFragmentRemoteness * (atan((2*u+v)/fAux) - atan(v/fAux))/fAux;
		fI *= light_buffer.fPointLightHazeIntensityFactors[i];

		v4FragmentColor += vec4(light_buffer.v3PointLightIntensities[i]*fI, 0);
	}


	//Compute haze effect for the spot light sources
	for(int i = 0; i < light_buffer.nSpotLights; ++i)
	{
		vec3 v3D = light_buffer.v3SpotLightDirections[i];	//direction of the spot light
		vec3 v3l = light_buffer.v3SpotLightLocations[i];	//location of the spot light
		float fHazeAttenuationFactor1 = light_buffer.fSpotLightHazeLocationDecayFactors[i] + fog_buffer.light_haze_attenuation_factor;	//haze location attenuation factor applied to the spot light source
		float fHazeAttenuationFactor2 = light_buffer.fSpotLightHazeDirectionDecayFactors[i] + fog_buffer.light_haze_attenuation_factor;   //haze direction attenuation factor applied to the spot light source


		//Ensure that the ray only appears in the lit half-space
		float fdot_v3D_v3d = dot(v3D, v3d);
		vec3 v3LO = v3ViewerLocation_WS - v3l;	//vector pointing from the light source position towards location of the observer
		float fdot_v3LO_v3D = dot(v3LO, v3D);

		float t1 = -fdot_v3LO_v3D/fdot_v3D_v3d;
		float t2 = (1.0f / 0.0f) / fdot_v3D_v3d;

		float t_start = max(min(t1, t2), 0);
		float t_end = min(max(t1, t2), 1);
		if(t_start >= t_end) continue;

		vec3 v3Start = v3ViewerLocation_WS + t_start*v3d;
		vec3 v3End = v3ViewerLocation_WS + t_end*v3d;

		v3d = v3End -v3Start;
		fFragmentRemoteness = length(v3d);
		fdot_v3D_v3d = dot(v3D, v3d);
		v3LO = v3Start - v3l;
		fdot_v3LO_v3D = dot(v3LO, v3D);

		
		//Compute "amount of light" along the cast ray
		float A = (fHazeAttenuationFactor1 + fHazeAttenuationFactor2) * fFragmentRemoteness * fFragmentRemoteness - fHazeAttenuationFactor2 * fdot_v3D_v3d * fdot_v3D_v3d;
		float B = (fHazeAttenuationFactor1 + fHazeAttenuationFactor2) * dot(v3LO, v3d) - fHazeAttenuationFactor2 * fdot_v3D_v3d * fdot_v3LO_v3D;
		float C = (fHazeAttenuationFactor1 + fHazeAttenuationFactor2) * dot(v3LO, v3LO) + 1 - fHazeAttenuationFactor2 * fdot_v3LO_v3D * fdot_v3LO_v3D;

		float fAux = sqrt(A*C-B*B);
		float fI = fFragmentRemoteness * (atan((A+B)/fAux) - atan(B/fAux))/fAux;
		fI *= light_buffer.fSpotLightHazeIntensityFactors[i];

		v4FragmentColor += vec4(light_buffer.v3SpotLightIntensities[i]*fI, 0);
	}


	return v4FragmentColor;
}


