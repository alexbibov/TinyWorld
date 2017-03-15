//Implements volumetric fog applied in screen space

#version 430 core

#define pi 3.1415926535897932384626433832795f
#define MAX_FLOAT_VALUE 3.402823466e38f


uniform float fFocalDistance;	//distance between the observer and the near plane of the view frustum
uniform float fFarPlane;		//distance between the observer and the far clip plane
uniform vec4 v4FocalPlane;		//contains left, right, bottom, and top boundaries in this order of the focal plane 
uniform vec4 v4Viewport;		//contains viewport coordinates x and y and viewport dimensions width and height in this order
uniform mat4 m4ViewTransform;	//view transform that converts world coordinates into view-space coordinates

uniform sampler2DArray s2daInScatteringSun;		//in-scattering texture induced by the sun light
uniform sampler2DArray s2daInScatteringMoon;	//in-scattering texture induced by the moon light

uniform float fFogDistanceCutOff;	//maximal allowed distance when performing fog calculations


#include "common/FogBuffer.include.glsl"


layout(pixel_center_integer) in vec4 gl_FragCoord;



//Implements Mie phase function
float MiePhaseFunction(float c, float g)
{
	float g2 = g * g;
	return (3.0f * (1.0f - g2)) / (2.0f * (2.0f + g2)) * (1.0f + c * c) / pow(1.0f + g2 - 2.0f * g * c, 1.5f);
}

//Modified signum function
float msign(float x)
{
	return float(x>=0) - float(x<0);
}

//Computes luminance corresponding to the given color
float computeLuminance(vec3 v3Color)
{
	return dot(v3Color, vec3(0.299f, 0.587f, 0.144f));
}

//Computes exposure that should be applied to a color with given luminance
float computeExposure(float fLuminance, float bias)
{
	return min(sqrt(1.0f / (fLuminance + bias)), 3.402823466e38);
}

//Modified smoothstep between -1 and 1
float msmoothstep(float edge1, float edge2, float x)
{
	return smoothstep(edge1, edge2, x)*2 - 1;
}


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


	//Next, compute amount of fog along the ray cast from observation point towards position of the currently processed fragment
	float b = fog_buffer.fAtmosphericFogGlobalDensity, c = fog_buffer.fAtmosphericFogHeightFallOff;

	vec3 v3D = v3FragCoord_WS - v3ViewerLocation_WS;
	float fDLength = length(v3D);

	//Clamp the ray by the "box with open cap" centered at the world space's origin 
	float t1, t2, e1, e2;

	//Calmp by the X-sides of the cube
	e1 = (-fFogDistanceCutOff - v3ViewerLocation_WS.x)/v3D.x;
	e2 = ( fFogDistanceCutOff - v3ViewerLocation_WS.x)/v3D.x;
	t1 = min(e1, e2); t2 = max(e1, e2);

	//Clamp by the Z-sides of the cube
	e1 = (-fFogDistanceCutOff - v3ViewerLocation_WS.z)/v3D.z;
	e2 = ( fFogDistanceCutOff - v3ViewerLocation_WS.z)/v3D.z;
	t1 = max(min(e1, e2), t1); t2 = min(max(e1, e2), t2);

	//Clamp by the -Y-side of the cube
	e1 = (-v3ViewerLocation_WS.y)/v3D.y;
	e2 = sign(v3D.y) / 0;
	t1 = max(min(e1, e2), t1); t2 = min(max(e1,e2), t2);

	t1 = max(0, t1);
	t2 = min(1, t2);

	//Compute resulting amount of fog
	float fAux;
	if(abs(v3D.y)>1e-5f)
	{
		fAux = (exp(-c*(v3D.y*t1 + v3ViewerLocation_WS.y)*float(t1<t2)) - exp(-c*(v3D.y*t2 + v3ViewerLocation_WS.y)*float(t1<t2)) ) / (c*v3D.y);
	}
	else
	{
		fAux = exp(-c*v3ViewerLocation_WS.y);
	}
	float fFogAmount = b*max(t2-t1, 0)*fDLength*fAux;
	float fFogImpact = 1.0f - exp(-fFogAmount);		//blend factor based on the fog amount along the ray


	//Retrieve color of the current fragment
	vec3 v3FogColor = fog_buffer.v3AtmosphericFogColor;
	vec3 v3SkydomeSample = normalize(0.925f*fog_buffer.v3SunDirection + 0.075f*fog_buffer.v3MoonDirection);


	float fCos1 = dot(-v3SkydomeSample, fog_buffer.v3SunDirection);
	float fCos2 = dot(-v3SkydomeSample, fog_buffer.v3MoonDirection);

	vec2 v2ProjectedReflection = vec2(v3SkydomeSample.x, -v3SkydomeSample.z);
	float fProjectedReflectionLength = length(v2ProjectedReflection);
	v2ProjectedReflection = float(fProjectedReflectionLength>0)*v2ProjectedReflection/fProjectedReflectionLength + float(fProjectedReflectionLength==0)*vec2(1,0);

	float COS = clamp(v2ProjectedReflection.x, -1, 1), SIN = clamp(v2ProjectedReflection.y, -1, 1);
	vec2 v2InScatteringTexCoords = vec2(float(SIN < 0) / 2.0f + acos(msign(SIN)*COS) / (pi*2.0f), asin(v3SkydomeSample.y)/pi*2.0f);

	v3FogColor *=
		(texture(s2daInScatteringSun, vec3(v2InScatteringTexCoords, 0)).rgb * MiePhaseFunction(fCos1, 0) + 
		texture(s2daInScatteringSun, vec3(v2InScatteringTexCoords, 1)).rgb * MiePhaseFunction(fCos1, fog_buffer.fMiePhaseFunctionParameter)) + 
		(texture(s2daInScatteringMoon, vec3(v2InScatteringTexCoords, 0)).rgb * MiePhaseFunction(fCos2, 0) + 
		texture(s2daInScatteringMoon, vec3(v2InScatteringTexCoords, 1)).rgb * MiePhaseFunction(fCos2, fog_buffer.fMiePhaseFunctionParameter));

	float fLuminance = computeLuminance(v3FogColor);
	float fExposure = computeExposure(fLuminance, 1e-2);
	v3FogColor = 1.0f - exp(-fExposure*v3FogColor);


	//Compute resulting color
	vec4 v4Result = texelFetch(source0, ivec2(gl_FragCoord.xy), 0);
	v4Result.xyz = (1.0f - fFogImpact)*v4Result.xyz + fFogImpact*v3FogColor;

	return v4Result;
}